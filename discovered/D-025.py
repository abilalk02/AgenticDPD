import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear


class DPD(DPDModule):
    """KAN-augmented complex FIR with smooth power conditioning.

    Backbone: a complex linear FIR (CLinear over the tap window) provides the
    linear/memory correction cheaply. A KAN on bounded envelope features adds
    the nonlinearity (memory-polynomial-like, but with smooth spline bases that
    degrade gracefully at high drive). Power p only drives a tiny KAN that emits
    a global gain and a modulation on the nonlinear branch -> the high-drive
    point gets a dedicated correction shape without touching the x-path raw.
    """

    def __init__(self, window: int, nl_hidden: int = 12, kan_grid: int = 8):
        super().__init__(window)
        self.win = window
        c = window // 2

        # --- Linear complex FIR backbone: window taps -> 1 centre correction ---
        self.fir = CLinear(window, 1, bias=True)

        # --- Nonlinear branch on bounded envelope features ---
        # Features: |x| powers at each tap (bounded via tanh squash) + centre re/im.
        # KAN maps these to a hidden real representation with smooth splines.
        d_feat = window * 3 + 2   # a, a^2, a^3 per tap + centre real/imag
        self.kan1 = KANLayer(d_feat, nl_hidden, grid=kan_grid, order=3,
                             x_min=-1.5, x_max=1.5)
        # Second KAN stage folds to 2 (real, imag) correction.
        self.kan2 = KANLayer(nl_hidden, 2, grid=kan_grid, order=3,
                             x_min=-1.5, x_max=1.5)

        # --- Power conditioner: p -> [global gain, nl modulation, fir modulation]
        self.pcond = KANLayer(1, 3, grid=kan_grid, order=3,
                             x_min=-1.5, x_max=1.5)

        self.c = c

    def _squash(self, x, a=1.2):
        # map into KAN spline support [-1.5, 1.5]
        return a * torch.tanh(x / a)

    def forward(self, xw, p):
        B = xw.shape[0]
        a = xw.abs()

        # ---- Power conditioning (smooth, bounded) ----
        ps = 1.2 * torch.tanh((2.0 * p - 1.0) * 1.5).unsqueeze(-1)  # (B,1)
        cond = self.pcond(ps)                         # (B,3)
        gain = torch.tanh(cond[:, 0])                 # global scale
        nl_mod = 1.0 + 0.5 * torch.tanh(cond[:, 1])   # nonlinear branch modulation
        fir_mod = 1.0 + 0.5 * torch.tanh(cond[:, 2])  # linear branch modulation

        # ---- Linear FIR backbone ----
        fir_out = self.fir(xw)[:, 0]                  # (B,) complex

        # ---- Nonlinear envelope branch ----
        a1 = self._squash(a)
        a2 = self._squash(a * a)
        a3 = self._squash(a * a * a)
        cr = self._squash(xw.real[:, self.c])
        ci = self._squash(xw.imag[:, self.c])
        feat = torch.cat([a1, a2, a3, cr.unsqueeze(-1), ci.unsqueeze(-1)], dim=-1)

        h = torch.tanh(self.kan1(feat))               # (B, nl_hidden)
        h = self._squash(h)                            # keep in spline support
        nl = self.kan2(h)                              # (B, 2)
        nl_r = nl[:, 0] * nl_mod
        nl_i = nl[:, 1] * nl_mod
        nl_c = torch.complex(nl_r, nl_i)

        # ---- Combine ----
        corr = fir_mod * fir_out + gain * nl_c
        return xw[:, self.c] + corr