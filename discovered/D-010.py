import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """Lean power-conditioned GMP + 8b KAN static-NL correction (low-BOP niche)."""

    def __init__(self, window: int, kan_hidden: int = 8):
        super().__init__(window)
        c = window // 2

        # --- GMP memory-polynomial core (cheap: 4 mult/term complex coeffs) ---
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=5, cross=1)

        # --- power -> complex gain modulation of the GMP output (1-D learned) ---
        # KAN maps normalized power in [0,1] to a 2-vector (re, im) gain.
        self.pgain = KANLayer(1, 2, grid=6, order=3, x_min=0.0, x_max=1.0)

        # --- static-NL correction on centre-tap envelope, conditioned on power ---
        # inputs: |x|, |x|^2 (squashed), p  -> KAN spline -> small linear head.
        self.knl = KANLayer(3, kan_hidden, grid=7, order=3,
                            x_min=-1.5, x_max=1.5, quant_bits=8)
        self.head = RLinear(kan_hidden, 2, quant_bits=16)

    def forward(self, xw, p):
        c = self.window // 2
        xc = xw[:, c]

        # ---- GMP core, power-gated ----
        g = self.gmp(xw)
        if g.dim() > 1:
            g = g[:, c] if g.shape[-1] == self.window else g.reshape(g.shape[0], -1)[:, 0]
        pin = p.unsqueeze(-1)
        gate = self.pgain(pin)                       # (B,2)
        gmp_res = g * torch.complex(gate[:, 0], gate[:, 1])

        # ---- static NL correction (envelope features + power) ----
        a = xc.abs()
        # squash into KAN spline support [-1.5,1.5]
        f0 = 1.5 * torch.tanh(a / 1.5)
        f1 = 1.5 * torch.tanh((a * a) / 1.5)
        f2 = 3.0 * (p - 0.5)                          # power centred, in support
        feat = torch.stack([f0, f1, f2], dim=1)
        h = torch.tanh(self.knl(feat))
        o = self.head(h)
        # correction rotates/scales the centre sample by an envelope-dependent factor
        corr = torch.complex(o[:, 0], o[:, 1]) * xc

        return xc + gmp_res + corr