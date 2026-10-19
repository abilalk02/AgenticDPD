import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """6-9k RMpS niche (10^6.5-10^7.0 BOP). Replaces the 3dBFS-CAPPED g015c3
    incumbent (ACLR 32.6) with the native-3dBFS g019-family KAN core.
    Full-window complex FIR (near-identity init) feeds a parameter-free
    envelope/memory/power basis (incl. explicit odd-order cross powers) into a
    three-stage fine-grid KAN emitting a complex residual gain. A small separate
    1-D KAN on p alone provides a smooth learned power-conditioning envelope on
    the correction strength -- robust at p=0 and p=1, no concat fragility. All
    spline inputs are hard-squashed into [-1,1]; small-coef init keeps seeds
    tight. Sized (hidden=12, grid=9, order=3, 14 feats) ~7k RMpS."""

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 12,
                 grid: int = 9, order: int = 3, quant_bits: int = 32):
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

        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        self.n_feat = 14
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # smooth learned 1-D power-conditioning envelope on correction strength
        self.pkan = KANLayer(1, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.06)
                l.wb.mul_(0.06)
            self.pkan.coef.mul_(0.05)
            self.pkan.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)

        lin = self.fir(x)[:, 0]                            # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)

        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)
        mem_env2 = torch.abs(x[:, self.mem_idx2]).mean(dim=-1)

        pv = p.reshape(B).clamp(0.0, 1.0)                 # safe at 0 and 1

        # base envelope powers (odd-order emphasis for high drive)
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e5 = torch.tanh((env ** 5) * 6.0)
        e7 = torch.tanh((env ** 7) * 8.0)

        m1 = torch.tanh(mem_env * 2.0)
        m1b = torch.tanh(mem_env * mem_env * 3.0)
        m2 = torch.tanh(mem_env2 * 2.0)
        m2b = torch.tanh(mem_env2 * mem_env2 * 3.0)

        # power cross-terms
        pp = pv * 2.0 - 1.0
        ep = torch.tanh(env * pv * 2.5)
        e2p = torch.tanh(env * env * pv * 3.0)
        mp = torch.tanh(mem_env * pv * 2.5)
        ep2 = torch.tanh(env * pv * pv * 2.5)

        feat = torch.stack(
            [e, e2, e3, e5, e7, m1, m1b, m2, m2b, pp, ep, e2p, mp, ep2],
            dim=-1)

        h = self.kan1(feat)
        h = torch.tanh(h)                                 # keep stage-2 in grid
        h = self.kan2(h)
        h = torch.tanh(h)                                 # keep stage-3 in grid
        g = self.kan3(h)                                  # (B, 2)

        # smooth 1-D power-conditioning of correction strength
        pcoord = (pv * 2.0 - 1.0).reshape(B, 1)           # (B,1) in [-1,1]
        pmod = self.pkan(pcoord)                          # (B,2)
        s_re = 1.0 + pmod[:, 0]
        s_im = 1.0 + pmod[:, 1]

        gr = g[:, 0] * s_re
        gi = g[:, 1] * s_im
        gain = torch.complex(1.0 + gr, gi)

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)