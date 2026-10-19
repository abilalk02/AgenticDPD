import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Envelope-conditioned KAN with memory-cross basis and smooth p-modulation.

    Targets the EMPTY 6-9k RMpS band (~7.5k RMpS). Grows the g023c2-derived
    parent (which clears the 3 dBFS ceiling) by widening spline stages and
    enriching the memory-cross feature bank. All spline inputs are squashed
    into [-1,1]; residual gain ~1 for tight seeds and high-drive stability.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 24,
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
        lo3 = max(0, c - 8)
        hi3 = min(window, c + 9)
        self.register_buffer("mem_idx3",
                             torch.tensor(list(range(lo3, hi3)), dtype=torch.long))

        # FIR at higher precision (cheap, keeps linear path clean)
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=16)

        self.n_feat = 13
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
        mem_env3 = torch.abs(x[:, self.mem_idx3]).mean(dim=-1)  # (B,)

        pv = p.reshape(B)                                 # (B,), safe at 0/1

        # bounded envelope-power squashes into spline support [-1,1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m1 = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env2 * 2.0)
        m3 = torch.tanh(mem_env3 * 2.0)

        # memory-cross terms: centre envelope modulated by longer memory
        # (catches high-drive memory effects that cap 3 dBFS ACLR)
        cross = torch.tanh(env * mem_env2 * 3.0)
        cross3 = torch.tanh(env * mem_env3 * 3.0)

        pp = pv * 2.0 - 1.0                               # centre p to [-1,1]
        ep = torch.tanh(env * pv * 2.5)
        e2p = torch.tanh(env * env * pv * 3.0)
        mp = torch.tanh(mem_env * pv * 2.5)

        feat = torch.stack(
            [e, e2, e3, m1, m2, m3, pp, ep, e2p, mp, pp * pp, cross, cross3],
            dim=-1)                                       # (B,13)

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