import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear


class DPD(DPDModule):
    """Deep envelope-conditioned KAN scaled into the high-BOP niche (10^6.5-10^7.5).

    Complex near-identity FIR anchor + complex tap mixer produce an envelope
    basis; a smooth learned 1-D KAN embedding of drive p is appended. A wide
    three-stage KAN core (float precision -> high BOP) with hard-squashed
    spline inputs emits a complex multiplicative gain + additive correction.
    One model spans the whole sweep; p never enters as raw division/log, so
    p==0 and p==1 are numerically safe. Deep KAN scales up without the CNN/MLP
    high-drive collapse.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int,
                 n_mix: int = 16,
                 hidden: int = 32,
                 grid: int = 11,
                 order: int = 3,
                 quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        self.c = window // 2

        # ---- complex FIR backbone (full window, near-identity init) ----
        self.fir = CLinear(window, 1, bias=False, quant_bits=32)

        # ---- complex tap mixer -> compact complex basis ----
        self.mix = CLinear(window, n_mix, bias=True, quant_bits=quant_bits)
        self.n_mix = n_mix

        # ---- smooth 1-D power embedding (learned function of p) ----
        self.pkan = KANLayer(1, 4, grid=7, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=32)

        # feature vector: [env(n_mix), env^2(n_mix), env^3(n_mix), p_emb(4)]
        self.kan_in = 3 * n_mix + 4

        # ---- wide three-stage KAN core (float -> high BOP) ----
        self.kan1 = KANLayer(self.kan_in, hidden, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, hidden, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)

        # ---- head: hidden -> [gr, gi, cr, ci] ----
        self.head = RLinear(hidden, 4, bias=True, quant_bits=32)

        with torch.no_grad():
            # near-identity FIR anchor at centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, self.c] = 1.0
            # gentle KAN init so gain starts ~1, corr ~0
            for l in (self.kan1, self.kan2, self.kan3, self.pkan):
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
        env2 = env * env                    # (B, n_mix) real
        env3 = env2 * env                   # (B, n_mix) real

        # smooth power embedding (safe at p=0 and p=1)
        pc = p.reshape(B, 1)
        p_sq = self._squash(2.0 * pc - 1.0)  # map [0,1] -> ~[-1,1], squashed
        p_emb = self.pkan(p_sq)             # (B,4) real

        # assemble & hard-squash into grid support (high-drive stability)
        feat = torch.cat([env, env2, env3, p_emb], dim=1)
        feat = self._squash(feat)

        h = self.kan1(feat)
        h = self._squash(h)
        h = self.kan2(h)
        h = self._squash(h)
        h = self.kan3(h)                    # (B, hidden) real

        out = self.head(h)                  # (B,4)
        gr = out[:, 0]
        gi = out[:, 1]
        cr = out[:, 2]
        ci = out[:, 3]

        gain = torch.complex(1.0 + gr, gi)
        corr = torch.complex(cr, ci)

        o = (lin * gain + corr) / self.FEAT_SCALE
        return o.to(torch.complex64)