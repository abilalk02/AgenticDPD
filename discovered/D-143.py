import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """g005c2-core envelope-conditioned KAN grown into the 10^6.5-10^7.0 BOP
    niche (currently held by the 3dBFS-capped g015c3 @32.6 dB).

    Strategy: wide two-layer KAN modulating a residual complex gain on a
    clean 16-bit FIR spine, plus a smooth learned 1-D function of p. All
    spline inputs tanh-squashed into [-1.2,1.2] for high-drive stability
    and tight seed spread. KANs at 12-bit to place BOP in-band.

    RMpS (float mults):
      * fir  CLinear(window,1): 4*window
      * kan1 KANLayer(14,16,grid=10,order=3): 14*16*(2*13+1) = 14*16*27 = 6048
      * kan2 KANLayer(16,2, grid=10,order=3): 16*2*27           =  864
      * pkan KANLayer(1,4,  grid=8, order=3): 1*4*(2*11+1)      =   92
      ~ 7000+ RMpS -> lands in the 6.5-7.0 band; 12-bit KANs set BOP.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 16,
                 grid: int = 10, order: int = 3, quant_bits: int = 12):
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

        # clean linear spine at higher precision
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=16)

        self.n_feat = 14
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.2, x_max=1.2, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.2, x_max=1.2, quant_bits=quant_bits)
        self.pkan = KANLayer(1, 4, 8, order,
                             x_min=-1.2, x_max=1.2, quant_bits=quant_bits)

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

        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)
        mem_env2 = torch.abs(x[:, self.mem_idx2]).mean(dim=-1)
        mem_env3 = torch.abs(x[:, self.mem_idx3]).mean(dim=-1)

        pv = p.reshape(B)                                 # (B,), safe at 0/1

        # bounded envelope-power squashes strictly inside [-1.2,1.2]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e4 = torch.tanh(env * env * env * env * 5.0)
        m1 = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env2 * 2.0)
        m3 = torch.tanh(mem_env3 * 2.0)

        pp = pv * 2.0 - 1.0                               # centre p to [-1,1]
        ep = torch.tanh(env * pv * 2.5)
        e2p = torch.tanh(env * env * pv * 3.0)
        mp = torch.tanh(mem_env * pv * 2.5)
        m2p = torch.tanh(mem_env2 * pv * 2.5)

        feat = torch.stack(
            [e, e2, e3, e4, m1, m2, m3,
             pp, ep, e2p, mp, m2p, pp * pp, pp * e], dim=-1)   # (B,14)

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