import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Ultra-lean envelope-conditioned KAN for the empty low-BOP niches (bins 0/1).

    A near-identity complex FIR provides the linear memory path; a compact
    parameter-free 8-D feature vector (tanh-squashed envelope powers, a memory
    envelope, centered p and env*p / env^2*p cross-terms) drives a single
    hidden-layer KAN that produces a complex residual gain on the centre sample.
    One model linearizes the whole power sweep — p enters multiplicatively to
    bend AM/AM at high drive without a bare-feature concat. All spline inputs
    are tanh-squashed into [-1.2, 1.2] so nothing escapes grid support at 3 dBFS
    (the binding limiter). 8-bit fake-quant keeps BOP in the empty ultra-low
    bins while leaving RMpS ~250. Residual gain starts at ~1 for tight seeds.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 10,
                 grid: int = 8, order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.n_taps = window

        lo1 = max(0, c - 2)
        hi1 = min(window, c + 3)
        self.register_buffer("mem_idx",
                             torch.tensor(list(range(lo1, hi1)), dtype=torch.long))

        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        self.n_feat = 8
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.2, x_max=1.2, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.2, x_max=1.2, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.05)
                l.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)

        lin = self.fir(x)[:, 0]                            # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        pv = p.reshape(B)                                 # (B,), safe at 0 and 1

        # bounded envelope powers squashed into spline support
        e  = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m1 = torch.tanh(mem_env * 2.0)

        pp  = pv * 2.0 - 1.0                              # center p -> [-1,1]
        ep  = torch.tanh(env * pv * 2.5)                  # env * p
        e2p = torch.tanh(env * env * pv * 3.0)            # env^2 * p
        mp  = torch.tanh(mem_env * pv * 2.5)              # mem * p

        feat = torch.stack([e, e2, e3, m1, pp, ep, e2p, mp], dim=-1)  # (B,8)
        feat = 1.2 * torch.tanh(feat / 1.2)               # hard clamp into grid

        h = self.kan1(feat)
        h = torch.tanh(h)
        h = 1.2 * torch.tanh(h / 1.2)                     # keep stage-2 in grid
        g = self.kan2(h)                                  # (B, 2)

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])      # residual gain ~1

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)