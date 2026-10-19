import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Ultra-low-BOP envelope-conditioned KAN DPD (targets EMPTY niche bin 0,
    10^4.0-10^4.5 BOP, ~236 RMpS). Single compact KAN over free squashed
    envelope-power features; residual-1 complex gain; tiny 1-D p-KAN for
    smooth multiplicative power conditioning. Heavy 6-bit quant on splines
    to place BOP in the target bin at unchanged RMpS. All spline inputs
    tanh-squashed into grid support -> no high-drive collapse, tight seeds."""

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8,
                 grid: int = 6, order: int = 3):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.n_taps = window

        lo1 = max(0, c - 2); hi1 = min(window, c + 3)
        self.register_buffer("mem_idx",
                             torch.tensor(list(range(lo1, hi1)), dtype=torch.long))
        lo2 = max(0, c - 5); hi2 = min(window, c + 6)
        self.register_buffer("mem_idx2",
                             torch.tensor(list(range(lo2, hi2)), dtype=torch.long))

        # FIR linear path (8-bit) : 4*window*1 RMpS
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=8)

        # feature vector (all parameter-free, squashed):
        self.n_feat = 11
        # single-stage KAN backbone, 6-bit
        self.kan = KANLayer(self.n_feat, hidden, grid, order,
                            x_min=-1.0, x_max=1.0, quant_bits=6)
        # gain head : real KAN mapping hidden -> 2, 6-bit
        self.ghead = KANLayer(hidden, 2, grid, order,
                              x_min=-1.0, x_max=1.0, quant_bits=6)
        # tiny 1-D power branch, 6-bit
        self.pkan = KANLayer(1, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=6)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan, self.ghead, self.pkan):
                l.coef.mul_(0.05)
                l.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE

        lin = self.fir(x)[:, 0]                            # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)

        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)
        mem_env2 = torch.abs(x[:, self.mem_idx2]).mean(dim=-1)

        pv = p.reshape(B)                                 # (B,), safe 0..1
        pp = pv * 2.0 - 1.0

        # bounded squashed envelope-power features (all in [-1,1])
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e4 = torch.tanh(env * env * env * env * 5.0)
        m1 = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env2 * 2.0)
        ep = torch.tanh(env * pv * 2.5)
        e2p = torch.tanh(env * env * pv * 3.0)

        feat = torch.stack(
            [e, e2, e3, e4, m1, m2, pp, ep, e2p, e * e2, m1 * m2], dim=-1)  # (B,11)

        h = self.kan(feat)
        h = torch.tanh(h)
        g = self.ghead(h)                                 # (B,2)

        # smooth 1-D power conditioning
        pfeat = torch.tanh(pp).reshape(B, 1)
        pg = self.pkan(pfeat)                             # (B,2)
        p_re = 1.0 + torch.tanh(pg[:, 0]) * 0.5
        p_im = torch.tanh(pg[:, 1]) * 0.5
        pmod = torch.complex(p_re, p_im)

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])      # residual ~1
        gain = gain * pmod

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)