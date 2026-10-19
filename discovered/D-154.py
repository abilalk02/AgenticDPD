import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Envelope-conditioned KAN with a smooth learned 1-D power gate.

    Targets the 10^6.5-10^7.0 BOP niche (currently held by the 3dBFS-capped
    g015c3 @ 32.6 dB). Built on the g005c2/g028c3 KAN lineage that clears the
    3 dBFS ceiling natively. Wide complex FIR provides robust linear memory;
    a two-stage fine-grid KAN bends the AM/AM curve from a rich parameter-free
    feature vector (env powers + memory env + p cross terms). Power p enters
    both as squashed cross features AND through a dedicated small KAN that
    produces a smooth 1-D gain modulation (variance-gate style), which helped
    seed robustness on clean inserts. All spline inputs squashed into [-1,1];
    near-identity init for tight seed spread. Full 32-bit to sit in the band.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 18,
                 grid: int = 12, order: int = 4, quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.n_taps = window

        # short symmetric neighbourhood for memory-envelope feature (free)
        lo = max(0, c - 3)
        hi = min(window, c + 4)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # complex FIR over all taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # main feature-conditioned KAN (9-D features, see forward)
        self.n_feat = 9
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # smooth learned 1-D power gate: p -> small 2-D modulation of the
        # residual gain (variance-gate style power conditioning)
        self.pkan = KANLayer(1, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            # identity FIR at centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            # small-signal init: near identity, tight seed spread
            for l in (self.kan1, self.kan2, self.kan3, self.pkan):
                l.coef.mul_(0.08)
                l.wb.mul_(0.08)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                         # (B, window) complex

        lin = self.fir(x)[:, 0]                           # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)

        # local memory envelope (mean |x| over short neighbourhood), free
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        # bounded squashes into spline support [-1, 1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env * mem_env * 3.0)

        pp = p * 2.0 - 1.0                                # center p to [-1,1]
        ep = torch.tanh(env * p * 2.5)                    # env*p cross term
        e2p = torch.tanh(env * env * p * 3.0)             # env^2*p cross term
        mp = torch.tanh(mem_env * p * 2.5)                # mem*p cross term

        feat = torch.stack([e, e2, e3, m, m2, pp, ep, e2p, mp], dim=-1)  # (B,9)

        h = self.kan1(feat)
        h = self.kan2(h)
        g = self.kan3(h)                                  # (B, 2) residual gain

        # smooth 1-D power conditioning (safe at p=0 and p=1)
        pcol = (p * 2.0 - 1.0).reshape(B, 1)              # (B,1) in [-1,1]
        pmod = self.pkan(pcol)                            # (B, 2)

        gr = 1.0 + g[:, 0] + pmod[:, 0]
        gi = g[:, 1] + pmod[:, 1]
        gain = torch.complex(gr, gi)                      # residual gain

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)