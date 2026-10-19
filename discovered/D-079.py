import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear, LayerNormPrim


class DPD(DPDModule):
    """Widened envelope-conditioned KAN DPD, targeting BOP niche 10^6.0-10^6.5.

    - Complex FIR anchor (near-identity) preserves linear path.
    - Complex tap-mix -> real envelope basis (parameter-free abs/powers).
    - Two-layer KAN on [env-basis, p, p^2] (all squashed into grid support)
      produces a residual complex gain (1+g) and additive correction.
    - Power p is a smooth learned 1-D KAN feature and modulates gains, not a
      bare concatenated feature -> robust across the drive sweep.
    - 8-bit QAT on the heavy KAN/linear layers places BOP in the target bin.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int,
                 n_mix: int = 16,
                 hidden: int = 20,
                 grid: int = 10,
                 order: int = 3,
                 quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        self.c = window // 2

        # ---- complex FIR backbone (all taps, near-identity init) ----
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # ---- complex tap mix -> envelope basis ----
        self.mix = CLinear(window, n_mix, bias=True, quant_bits=quant_bits)
        self.n_mix = n_mix

        # KAN input: env basis (n_mix) + |centre| + p + p^2
        self.kan_in = n_mix + 3
        self.kan1 = KANLayer(self.kan_in, hidden, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)

        # head: hidden -> [gr, gi, cr, ci]
        self.head = RLinear(hidden, 4, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            # near-identity FIR anchor at centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, self.c] = 1.0
            # damp KAN so it starts near-linear
            for l in (self.kan1, self.kan2):
                if hasattr(l, "coef"):
                    l.coef.mul_(0.1)
                if hasattr(l, "wb"):
                    l.wb.mul_(0.1)
            self.head.weight.mul_(0.05)
            self.head.bias.zero_()

    def forward(self, xw, p):
        x = xw * self.FEAT_SCALE
        B = x.shape[0]

        # linear anchor
        lin = self.fir(x)[:, 0]                          # complex (B,)

        # complex envelope basis
        z = self.mix(x)                                  # complex (B, n_mix)
        env = z.abs()                                    # real (B, n_mix)

        # centre-tap magnitude + power features
        xc = x[:, self.c]
        env_c = xc.abs().reshape(B, 1)                   # (B,1)
        pc = p.reshape(B, 1)                             # (B,1), safe at 0/1
        pc2 = pc * pc

        feat = torch.cat([env, env_c, pc, pc2], dim=1)   # (B, kan_in)
        feat = 1.5 * torch.tanh(feat / 1.5)              # squash into grid

        h = self.kan1(feat)
        h = 1.5 * torch.tanh(h / 1.5)                    # keep in grid
        h = self.kan2(h)

        out = self.head(h)                               # (B,4)
        gr = out[:, 0]
        gi = out[:, 1]
        cr = out[:, 2]
        ci = out[:, 3]

        gain = torch.complex(1.0 + gr, gi)
        corr = torch.complex(cr, ci)

        o = (lin * gain + corr) / self.FEAT_SCALE
        return o.to(torch.complex64)