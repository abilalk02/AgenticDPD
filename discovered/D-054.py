import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """KAN-scaled, power-conditioned DPD for the 10^5.5-10^6.0 BOP niche.

    A complex tap-mixer builds a compact feature set; a two-stage KAN
    (envelope + squashed power feature) drives BOTH a complex power-gain
    and additive spline corrections. A light GMP backbone (power-gated)
    provides classical robustness. Pure spline core -> stable at high drive.
    """

    def __init__(self, window: int, feat: int = 10, hidden: int = 12):
        super().__init__(window)
        self.window = window
        self.F = feat

        # complex tap mixer -> compact complex feature set
        self.mix = CLinear(window, feat, quant_bits=16)

        # KAN stage 1: [|feat| (F), p, p^2] -> hidden  (real spline core)
        self.kan1 = KANLayer(feat + 2, hidden, grid=8, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=8)
        # KAN stage 2: hidden -> gain(2) + additive corr basis (2*feat)
        self.kan2 = KANLayer(hidden, 2 + 2 * feat, grid=8, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # combine additive complex corrections back to a single sample
        self.corr_head = CLinear(feat, 1, quant_bits=16)

        # classical GMP backbone, power-gated
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=5, cross=2)
        self.pgmp = KANLayer(1, 2, grid=6, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2

        # complex feature mix
        z = self.mix(xw)                        # (B, F) complex
        env = z.abs()                           # (B, F) real, free

        # smooth power features (safe at p=0 and p=1)
        pc = p.reshape(B, 1)                    # (B,1)
        ps = 3.0 * (pc - 0.5)                   # centered, in-support
        pfeat = torch.cat([env, pc, ps], dim=1) # (B, F+2)
        pfeat = 1.5 * torch.tanh(pfeat / 1.5)   # squash into grid

        h = torch.tanh(self.kan1(pfeat))        # (B, hidden)
        h = 1.5 * torch.tanh(h / 1.5)           # keep in-support
        out = self.kan2(h)                      # (B, 2 + 2F)

        # complex power-modulated gain
        gr = 1.0 + torch.tanh(out[:, 0])
        gi = torch.tanh(out[:, 1])
        gain = torch.complex(gr, gi)            # (B,)

        # additive spline corrections on complex path
        cr = out[:, 2:2 + self.F]               # (B, F)
        ci = out[:, 2 + self.F:2 + 2 * self.F]  # (B, F)
        corr_c = torch.complex(cr, ci)          # (B, F)

        # gated feature path + additive correction
        gated = z * gain.unsqueeze(-1)          # (B, F)
        corr_nn = self.corr_head(gated + corr_c).squeeze(-1)  # (B,)

        # classical GMP backbone, power-gated
        gmp_out = self.gmp(xw)                   # (B,) complex
        gp = self.pgmp(ps)                       # (B,2)
        ggr = 1.0 + torch.tanh(gp[:, 0])
        ggi = torch.tanh(gp[:, 1])
        gmp_gain = torch.complex(ggr, ggi)
        corr_gmp = gmp_out * gmp_gain

        y = xw[:, c] + corr_nn + corr_gmp
        return y.to(torch.complex64)