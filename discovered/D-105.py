import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """6-9k RMpS envelope-conditioned KAN (bin 10^6.5-10^7.0 BOP).

    Fills the weakest-3dBFS niche (currently g015c3 @32.6, a retired capped
    branch) with a g019/parent-style backbone scaled down: near-identity
    complex FIR + 14-D parameter-free feature vector (envelope powers, two
    memory-envelope scales, centered p, p cross-terms) feeding a three-stage
    KAN that emits a complex residual gain on the centre sample. One model
    linearizes the whole sweep; p bends AM/AM via multiplicative cross-terms.
    All spline inputs tanh-squashed into [-1,1]; residual gain starts ~1 for
    tight seed spread. Mixed quant (8/16 bit) keeps BOP in the target bin.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 32,
                 grid: int = 14, order: int = 4):
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

        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=8)

        self.n_feat = 14
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=16)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=8)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=16)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.05)
                l.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)

        lin = self.fir(x)[:, 0]                            # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)

        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)
        mem_env2 = torch.abs(x[:, self.mem_idx2]).mean(dim=-1)

        pv = p.reshape(B)                                 # safe at 0 and 1

        # bounded squashes into spline support [-1, 1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e4 = torch.tanh(env * env * env * env * 5.0)
        m1 = torch.tanh(mem_env * 2.0)
        m1b = torch.tanh(mem_env * mem_env * 3.0)
        m2 = torch.tanh(mem_env2 * 2.0)
        m2b = torch.tanh(mem_env2 * mem_env2 * 3.0)

        pp = pv * 2.0 - 1.0                               # center p to [-1,1]
        ep = torch.tanh(env * pv * 2.5)
        e2p = torch.tanh(env * env * pv * 3.0)
        mp = torch.tanh(mem_env * pv * 2.5)
        ep2 = torch.tanh(env * pv * pv * 2.5)
        e3p = torch.tanh(env * env * env * pv * 3.5)

        feat = torch.stack(
            [e, e2, e3, e4, m1, m1b, m2, m2b,
             pp, ep, e2p, mp, ep2, e3p], dim=-1)          # (B,14)

        h = self.kan1(feat)
        h = torch.tanh(h)
        h = self.kan2(h)
        h = torch.tanh(h)
        g = self.kan3(h)                                  # (B, 2)

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])      # residual gain

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)