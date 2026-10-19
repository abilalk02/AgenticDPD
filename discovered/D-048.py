import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear


class DPD(DPDModule):
    """Envelope-conditioned KAN, scaled up for the 10^6.0-10^6.5 BOP niche.

    Complex FIR near-identity anchor + complex tap mixer -> envelope basis,
    fed with a smooth learned 1-D function of drive p into a wide two-layer
    KAN whose spline inputs are hard-squashed into grid support. KAN emits a
    complex multiplicative gain plus additive correction. One model spans the
    whole power sweep; p enters only through a smooth KAN embedding, never raw
    division, so p==0 and p==1 are stable.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int,
                 n_mix: int = 12,
                 hidden: int = 24,
                 grid: int = 9,
                 order: int = 3,
                 quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        self.c = window // 2

        # ---- complex FIR backbone (full window, near-identity init) ----
        self.fir = CLinear(window, 1, bias=False, quant_bits=32)

        # ---- complex tap mixer -> compact complex basis ----
        self.mix = CLinear(window, n_mix, bias=True, quant_bits=quant_bits)
        self.n_mix = n_mix

        # ---- smooth 1-D power embedding (learned function of p) ----
        self.pkan = KANLayer(1, 3, grid=7, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=32)

        # feature vector into main KAN: [env(n_mix) , env^2(n_mix) , p_emb(3)]
        self.kan_in = 2 * n_mix + 3

        # ---- wide two-layer KAN core ----
        self.kan1 = KANLayer(self.kan_in, hidden, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)

        # ---- head: hidden -> [gr, gi, cr, ci] ----
        self.head = RLinear(hidden, 4, bias=True, quant_bits=32)

        with torch.no_grad():
            # near-identity FIR anchor at centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, self.c] = 1.0
            # gentle KAN init so gain starts ~1, corr ~0
            for l in (self.kan1, self.kan2, self.pkan):
                if hasattr(l, "coef"):
                    l.coef.mul_(0.1)
                if hasattr(l, "wb"):
                    l.wb.mul_(0.1)
            self.head.weight.mul_(0.03)
            self.head.bias.zero_()

    @staticmethod
    def _squash(z, a=1.5):
        return a * torch.tanh(z / a)

    def forward(self, xw, p):
        x = xw * self.FEAT_SCALE
        B = x.shape[0]

        # linear backbone anchor (complex)
        lin = self.fir(x)[:, 0]

        # complex basis from tap mixer
        z = self.mix(x)                     # (B, n_mix) complex
        env = z.abs()                       # (B, n_mix) real, free
        env2 = env * env                    # (B, n_mix) real, free

        # smooth power embedding
        pc = p.reshape(B, 1)                # (B,1) safe at 0 and 1
        p_sq = self._squash(2.0 * pc - 1.0)  # map [0,1]->~[-1,1] then squash
        p_emb = self.pkan(p_sq)            # (B,3) real

        # assemble & squash into grid support
        feat = torch.cat([env, env2, p_emb], dim=1)     # (B, kan_in)
        feat = self._squash(feat)

        h = self.kan1(feat)
        h = self._squash(h)
        h = self.kan2(h)                    # (B, hidden) real

        out = self.head(h)                  # (B,4)
        gr = out[:, 0]
        gi = out[:, 1]
        cr = out[:, 2]
        ci = out[:, 3]

        gain = torch.complex(1.0 + gr, gi)
        corr = torch.complex(cr, ci)

        o = (lin * gain + corr) / self.FEAT_SCALE
        return o.to(torch.complex64)