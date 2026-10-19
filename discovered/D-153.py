import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Envelope-conditioned KAN with memory-cross basis and smooth p-modulation.

    Targets the 10^6.5-10^7.0 BOP niche (~7e6 BOP, ~7-8k RMpS) to REPLACE the
    g015c3 incumbent (3 dBFS-capped at 32.6 dB). Built on the g005c2-descended
    parent core that natively clears the 3 dBFS ceiling (parent 36.4 dB),
    widened into the band via KAN width/grid growth (the sanctioned scale-up
    route). Residual-gain-on-FIR topology preserved; all spline inputs squashed
    into support for tight high-drive seed spread.

    RMpS budget (~7-8k, float KANs):
      * FIR CLinear(window,1) 16-bit : 4*window
      * kan1 : 13*24*(2*(10+3)+1) = 13*24*27 = 8424
        (wait: too big alone -> use grid 8) recomputed below in code
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 20,
                 grid: int = 8, order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.n_taps = window

        lo1 = max(0, c - 3)
        hi1 = min(window, c + 4)
        self.register_buffer("mem_idx",
                             torch.tensor(list(range(lo1, hi1)), dtype=torch.long))
        lo2 = max(0, c - 6)
        hi2 = min(window, c + 7)
        self.register_buffer("mem_idx2",
                             torch.tensor(list(range(lo2, hi2)), dtype=torch.long))
        lo3 = max(0, c - 1)
        hi3 = min(window, c + 2)
        self.register_buffer("mem_idx3",
                             torch.tensor(list(range(lo3, hi3)), dtype=torch.long))

        # FIR at higher precision (cheap, keeps linear path clean)
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=16)

        self.n_feat = 13
        # kan1 : 13*20*(2*(8+3)+1) = 13*20*23 = 5980
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        # kan2 : 20*12*23 = 5520 -> keep second stage moderate
        self.kan2 = KANLayer(hidden, 8, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        # kan3 : 8*2*23 = 368
        self.kan3 = KANLayer(8, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        # pkan : 1*6*23 = 138
        self.pkan = KANLayer(1, 6, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan1, self.kan2, self.kan3):
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
        e4 = torch.tanh(env * env * env * env * 5.0)
        m1 = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env2 * 2.0)
        m3 = torch.tanh(mem_env3 * 2.0)

        # memory-cross terms: catch high-drive memory effects that cap 3 dBFS
        cross = torch.tanh(env * mem_env2 * 3.0)
        cross2 = torch.tanh(env * env * mem_env * 4.0)

        pp = pv * 2.0 - 1.0                               # centre p to [-1,1]
        ep = torch.tanh(env * pv * 2.5)
        e2p = torch.tanh(env * env * pv * 3.0)
        mp = torch.tanh(mem_env2 * pv * 2.5)

        feat = torch.stack(
            [e, e2, e3, e4, m1, m2, m3, ep, e2p, mp, cross, cross2, pp],
            dim=-1)                                       # (B,13)

        h = self.kan1(feat)
        h = torch.tanh(h)
        h = self.kan2(h)
        h = torch.tanh(h)
        g = self.kan3(h)                                  # (B,2)

        # smooth learned 1-D function of p -> multiplicative conditioning
        pfeat = torch.tanh(pp.reshape(B, 1))
        pg = self.pkan(pfeat)                             # (B,6)
        p_re = 1.0 + torch.tanh(pg[:, 0]) * 0.5
        p_im = torch.tanh(pg[:, 1]) * 0.5
        pmod = torch.complex(p_re, p_im)                  # (B,) ~1

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])      # residual gain ~1
        gain = gain * pmod

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)