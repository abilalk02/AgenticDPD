import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Power-conditioned envelope KAN.

    A dedicated 1-D KAN on p learns (i) a smooth bounded gain that scales the
    whole nonlinear residual and (ii) a small additive conditioning bias fed to
    the envelope KAN. The main KAN sees only power-agnostic envelope features,
    so the p-path owns cross-sweep robustness. Residual multiplies the linear
    FIR estimate; near-identity init keeps seed spread tight at high drive.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 16,
                 grid: int = 12, order: int = 4,
                 p_hidden: int = 8, p_grid: int = 8, p_order: int = 3,
                 quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.n_taps = window

        lo = max(0, c - 3)
        hi = min(window, c + 4)
        self.register_buffer("mem_idx",
                             torch.arange(lo, hi, dtype=torch.long))

        # complex FIR over all taps -> centre linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # power path: smooth learned 1-D function of p
        # outputs: [residual_gain_scalar, cond_bias(n_feat)]
        self.n_feat = 5  # env, env^2, env^3, mem_env, mem_env^2 (power-free)
        self.pkan = KANLayer(1, 1 + self.n_feat, p_grid, p_order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # main envelope KAN stack
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            # near-identity residual, tight seed spread
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.08)
                l.wb.mul_(0.08)
            # p-path starts as a gentle, near-neutral modulator
            self.pkan.coef.mul_(0.05)
            self.pkan.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)

        lin = self.fir(x)[:, 0]                           # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        # power-agnostic envelope features, squashed into [-1,1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env * mem_env * 3.0)
        feat = torch.stack([e, e2, e3, m, m2], dim=-1)    # (B,5)

        # power path: safe at p==0 and p==1 (maps to -1 / +1, in grid)
        pp = (p * 2.0 - 1.0).reshape(B, 1)                # (B,1)
        pout = self.pkan(pp)                              # (B, 1+n_feat)
        # bounded residual-scaling gain in [0,2], neutral ~1 at init
        p_gain = 2.0 * torch.sigmoid(pout[:, 0])         # (B,)
        # additive conditioning bias, kept inside grid
        cond = 0.5 * torch.tanh(pout[:, 1:])             # (B,5)

        feat_c = 1.0 * torch.tanh(feat + cond)           # re-squash into [-1,1]

        h = self.kan1(feat_c)
        h = self.kan2(h)
        g = self.kan3(h)                                  # (B,2)

        # residual scaled by the smooth p-gain
        gr = g[:, 0] * p_gain
        gi = g[:, 1] * p_gain
        gain = torch.complex(1.0 + gr, gi)               # residual gain

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)