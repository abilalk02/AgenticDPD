import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear


class DPD(DPDModule):
    """Envelope-conditioned KAN DPD sized for the EMPTY BOP niche
    10^4.5-10^5.0 (~200 RMpS, ~8e4 BOP at 8-bit KAN precision).

    Fills an empty low-BOP bin whose neighbouring ultra-low band is
    3dBFS-capped at only 30.9 dB. Uses the proven non-collapsing recipe:
      - near-identity complex FIR anchor preserves the linear path,
      - small complex tap-mix -> real envelope basis (free abs),
      - single squashed KAN on [env, |centre|, p, p^2] (all in grid),
      - residual (1+g) gain + additive correction head,
      - p enters only as smooth features (never inverted / logged).
    8-bit quant on KAN + mix cuts BOP ~4x to hit the target niche.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int,
                 n_mix: int = 6,
                 hidden: int = 8,
                 grid: int = 10,
                 order: int = 3,
                 kan_bits: int = 8,
                 lin_bits: int = 8):
        super().__init__(window)
        self.window = window
        self.c = window // 2

        # complex FIR backbone (all taps, near-identity init)
        self.fir = CLinear(window, 1, bias=False, quant_bits=lin_bits)

        # complex tap mix -> envelope basis
        self.mix = CLinear(window, n_mix, bias=True, quant_bits=lin_bits)
        self.n_mix = n_mix

        # KAN input: env basis (n_mix) + |centre| + p + p^2
        self.kan_in = n_mix + 3
        self.kan = KANLayer(self.kan_in, hidden, grid=grid, order=order,
                            x_min=-1.5, x_max=1.5, quant_bits=kan_bits)

        # head: hidden -> [gr, gi, cr, ci]
        self.head = RLinear(hidden, 4, bias=True, quant_bits=lin_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, self.c] = 1.0
            if hasattr(self.kan, "coef"):
                self.kan.coef.mul_(0.1)
            if hasattr(self.kan, "wb"):
                self.kan.wb.mul_(0.1)
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

        h = self.kan(feat)                               # (B, hidden)

        out = self.head(h)                               # (B,4)
        gr = out[:, 0]
        gi = out[:, 1]
        cr = out[:, 2]
        ci = out[:, 3]

        gain = torch.complex(1.0 + gr, gi)
        corr = torch.complex(cr, ci)

        o = (lin * gain + corr) / self.FEAT_SCALE
        return o.to(torch.complex64)