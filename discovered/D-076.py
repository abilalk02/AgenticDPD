import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Shrunk envelope-conditioned KAN (parent g018c2 lineage).

    Cost reduction >=3x vs parent by:
      * single compact hidden KAN stage (16 -> hidden) instead of two wide
        48-unit stages, plus a tiny KAN head (hidden -> 2)
      * grid 20->10, order 4->3
    Retained for robustness (all parameter-free, so free):
      * near-identity complex FIR (linear memory)
      * 16-D envelope/memory/p cross-term feature vector, tanh-squashed
      * dedicated smooth 1-D p-branch multiplicatively conditioning the gain
      * residual gain init ~1 for tight seed-to-seed spread at high drive
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 12,
                 grid: int = 10, order: int = 3, quant_bits: int = 32):
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
        lo3 = max(0, c - 8)
        hi3 = min(window, c + 9)
        self.register_buffer("mem_idx3",
                             torch.tensor(list(range(lo3, hi3)), dtype=torch.long))

        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        self.n_feat = 16
        # single compact hidden stage + tiny head
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # dedicated smooth 1-D function of p -> modulation gains
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
        mem_env3 = torch.abs(x[:, self.mem_idx3]).mean(dim=-1)  # (B,)

        pv = p.reshape(B)                                 # (B,), safe at 0 and 1

        # bounded envelope-power squashes into spline support [-1,1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e4 = torch.tanh(env * env * env * env * 5.0)
        m1 = torch.tanh(mem_env * 2.0)
        m1b = torch.tanh(mem_env * mem_env * 3.0)
        m2 = torch.tanh(mem_env2 * 2.0)
        m3 = torch.tanh(mem_env3 * 2.0)

        pp = pv * 2.0 - 1.0                               # centre p to [-1,1]
        ep = torch.tanh(env * pv * 2.5)
        e2p = torch.tanh(env * env * pv * 3.0)
        mp = torch.tanh(mem_env * pv * 2.5)
        ep2 = torch.tanh(env * pv * pv * 2.5)
        e2p2 = torch.tanh(env * env * pv * pv * 3.0)
        e3p = torch.tanh(env * env * env * pv * 3.5)

        feat = torch.stack(
            [e, e2, e3, e4, m1, m1b, m2, m3,
             pp, ep, e2p, mp, ep2, e2p2, e3p, pp * pp], dim=-1)  # (B,16)

        h = self.kan1(feat)
        h = torch.tanh(h)
        g = self.kan2(h)                                  # (B, 2)

        # smooth learned 1-D function of p -> multiplicative conditioning
        pfeat = pp.reshape(B, 1)
        pfeat = torch.tanh(pfeat)
        pg = self.pkan(pfeat)                             # (B,4)
        p_re = 1.0 + torch.tanh(pg[:, 0]) * 0.5
        p_im = torch.tanh(pg[:, 1]) * 0.5
        pmod = torch.complex(p_re, p_im)                  # (B,) ~1

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])      # residual gain ~1
        gain = gain * pmod

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)