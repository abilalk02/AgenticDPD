import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear


class DPD(DPDModule):
    """Ultra-lean envelope-conditioned KAN DPD for the EMPTY low-BOP niche
    (10^4.0-10^4.5 BOP, ~230 RMpS). Builds on the g019 family topology
    (complex FIR anchor + complex tap-mix envelope basis + single squashed
    KAN + residual complex gain/correction), shrunk and 8-bit quantized so
    BOP lands in the empty bin while keeping the 3 dBFS point up.

    RMpS estimate (float count):
      fir   CLinear(w->1):   4*w*1
      mix   CLinear(w->6):   4*w*6
      kan   KAN(9->6,g6,o3): 9*6*(2*(6+3)+1) = 9*6*19 = 1026
      head  RLinear(6->4):   24
    For window=13: 52 + 312 + 1026 + 24 ~= 1414 float RMpS... too big,
    so grid/hidden trimmed below to hit ~230 RMpS band.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int,
                 n_mix: int = 5,
                 hidden: int = 5,
                 grid: int = 4,
                 order: int = 3,
                 quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        self.c = window // 2

        # complex FIR backbone (near-identity init), 8-bit
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # complex tap mix -> envelope basis, 8-bit
        self.mix = CLinear(window, n_mix, bias=True, quant_bits=quant_bits)
        self.n_mix = n_mix

        # KAN input: env basis (n_mix) + |centre| + p + p^2
        self.kan_in = n_mix + 3
        self.kan = KANLayer(self.kan_in, hidden, grid=grid, order=order,
                            x_min=-1.5, x_max=1.5, quant_bits=quant_bits)

        # head: hidden -> [gr, gi, cr, ci]
        self.head = RLinear(hidden, 4, bias=True, quant_bits=quant_bits)

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

        # centre-tap magnitude + smooth power features
        xc = x[:, self.c]
        env_c = xc.abs().reshape(B, 1)                   # (B,1)
        pc = p.reshape(B, 1).to(env.dtype)               # (B,1), safe at 0/1
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