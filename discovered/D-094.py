import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Ultra-low-complexity envelope-conditioned KAN targeting the empty
    10^4.5-10^5.0 BOP niche.

    RMpS budget (~200):
      * FIR CLinear(window,1): 4*window
      * kan1: 10*8*(2*(8+3)+1) = 10*8*23 = 1840  -> dominant, shrunk
      * kan2: 8*2*23 = 368
      * pkan: 1*4*23 = 92
    8-bit fake-quant on KANs cuts BOP ~4x -> lands in target band.
    All spline inputs tanh-squashed into [-1,1]; residual gain ~1 for
    tight seed spread and high-drive stability.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8,
                 grid: int = 8, order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.n_taps = window

        lo1 = max(0, c - 2)
        hi1 = min(window, c + 3)
        self.register_buffer("mem_idx",
                             torch.tensor(list(range(lo1, hi1)), dtype=torch.long))
        lo2 = max(0, c - 5)
        hi2 = min(window, c + 6)
        self.register_buffer("mem_idx2",
                             torch.tensor(list(range(lo2, hi2)), dtype=torch.long))

        # FIR at higher precision (cheap, keeps linear path clean)
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=16)

        self.n_feat = 10
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.pkan = KANLayer(1, 4, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.05)
                l.wb.mul_(0.05)
            self.pkan.coef.mul_(0.05)
            self.pkan.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)

        lin = self.fir(x)[:, 0]                            # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)

        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)    # (B,)
        mem_env2 = torch.abs(x[:, self.mem_idx2]).mean(dim=-1)  # (B,)

        pv = p.reshape(B)                                 # (B,), safe at 0/1

        # bounded envelope-power squashes into spline support [-1,1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m1 = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env2 * 2.0)

        pp = pv * 2.0 - 1.0                               # centre p to [-1,1]
        ep = torch.tanh(env * pv * 2.5)
        e2p = torch.tanh(env * env * pv * 3.0)
        mp = torch.tanh(mem_env * pv * 2.5)

        feat = torch.stack(
            [e, e2, e3, m1, m2, pp, ep, e2p, mp, pp * pp], dim=-1)  # (B,10)

        h = self.kan1(feat)
        h = torch.tanh(h)
        g = self.kan2(h)                                  # (B, 2)

        # smooth learned 1-D function of p -> multiplicative conditioning
        pfeat = torch.tanh(pp.reshape(B, 1))
        pg = self.pkan(pfeat)                             # (B,4)
        p_re = 1.0 + torch.tanh(pg[:, 0]) * 0.5
        p_im = torch.tanh(pg[:, 1]) * 0.5
        pmod = torch.complex(p_re, p_im)                  # (B,) ~1

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])      # residual gain ~1
        gain = gain * pmod

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)