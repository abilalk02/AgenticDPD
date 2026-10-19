import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear


class DPD(DPDModule):
    """Envelope-conditioned KAN DPD scaled for BOP niche 10^6.5-10^7.0.

    Target ~7.5k RMpS at float (32-bit) KAN precision -> ~7-8M BOP,
    filling the weak bin whose current best (g015c3) is 3dBFS-capped at
    32.6 dB. Builds on the g019c3 topology (no high-drive collapse):
      - complex FIR anchor (near-identity) preserves linear path,
      - complex tap-mix -> real envelope basis (parameter-free),
      - two-layer widened KAN on [env, |centre|, p, p^2] squashed to grid,
      - head produces residual complex gain (1+g) + additive correction,
      - p enters as a smooth learned 1-D KAN feature modulating gains.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int,
                 n_mix: int = 20,
                 hidden: int = 28,
                 grid: int = 12,
                 order: int = 3,
                 quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        self.c = window // 2

        # complex FIR backbone (all taps, near-identity init), float
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # complex tap mix -> envelope basis, float
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
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, self.c] = 1.0
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