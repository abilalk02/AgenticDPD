import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Bounded two-stage KAN residual DPD (g003c1/g005c2 lineage) placed in
    the 10^6.5-10^7.0 BOP band at float32 precision.

    A complex FIR over the full window produces the dominant linear/memory
    estimate. A rich, tanh-squashed real feature basis (envelope powers,
    memory envelopes, p cross-terms) is bent by two stacked bounded KAN
    stages into a SMALL complex correction gain (1+delta). The linear path
    always dominates, so high-drive (3 dBFS) never diverges -- this core
    natively clears the 3 dBFS ceiling that g015c3 cannot. All spline inputs
    are squashed into the grid support; p enters only via centered / cross
    terms (safe at p=0 and p=1). Float32 weights sit the BOP in the target
    band while RMpS stays well under g015c3's 7700.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int,
                 grid: int = 10, order: int = 3, hidden: int = 16,
                 quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.window = window

        # memory taps around centre for envelope memory features
        lo = max(0, c - 3)
        hi = min(window, c + 4)
        self.register_buffer("mem_idx", torch.tensor(list(range(lo, hi)),
                                                     dtype=torch.long))

        # full-window complex FIR -> dominant linear/memory estimate
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # feature basis: [e, e2, e3, e5, m0, m1, pc, ep, mp]  -> 9 feats
        self.n_feat = 9
        # stage 1 KAN: features -> hidden (bounded)
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        # stage 2 KAN: hidden -> (re, im) correction
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            # FIR starts as identity at centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            # start near identity gain: tiny KAN perturbation
            self.kan1.coef.mul_(0.1)
            self.kan1.wb.mul_(0.1)
            self.kan2.coef.mul_(0.05)
            self.kan2.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)
        lin = self.fir(x)[:, 0]                            # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)
        mem = torch.abs(x[:, self.mem_idx])               # (B, k)
        m0 = mem.mean(dim=-1)                              # (B,)
        m1 = mem.amax(dim=-1)                              # (B,)

        # bounded, free feature construction squashed into [-1,1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e5 = torch.tanh((env ** 5) * 6.0)                 # deep-drive term
        mm0 = torch.tanh(m0 * 2.0)
        mm1 = torch.tanh(m1 * 2.0)
        pc = p * 2.0 - 1.0                                # (B,) in [-1,1]
        ep = torch.tanh(env * p * 2.5)                    # env*p cross
        mp = torch.tanh(m0 * p * 2.5)                     # mem*p cross

        feat = torch.stack([e, e2, e3, e5, mm0, mm1, pc, ep, mp], dim=-1)

        h = self.kan1(feat)                               # (B, hidden)
        h = torch.tanh(h)                                 # keep in stage-2 grid
        g = self.kan2(h)                                  # (B, 2)

        # small residual correction: gain = 1 + delta, delta bounded
        dr = 0.6 * torch.tanh(g[:, 0])
        di = 0.6 * torch.tanh(g[:, 1])
        gain = torch.complex(1.0 + dr, di)                # (B,) complex

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)                      # (B,)