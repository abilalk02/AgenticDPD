import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Ultra-low-BOP envelope-conditioned KAN corrector with power-variance gate.

    Targets the empty bin-0 niche (10^4.0-10^4.5 BOP, ~170 RMpS) whose current
    best (g021c3) is 3dBFS-capped at 30.9 dB. A decimated complex FIR gives the
    linear/memory estimate; a single 8-bit KAN bends a bounded 8-feature
    envelope/power vector (incl. an explicit high-drive env^3 term) into a
    complex multiplicative gain. A separate tiny 1-D KAN of p forms a smooth
    variance-gate (0..1) modulating correction depth across the sweep. All
    spline inputs are tanh-squashed into [-1,1] for high-drive seed stability.
    p enters only via centred / cross terms (safe at p==0 and p==1).
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2,
                 grid: int = 7, order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2
        self.c = c

        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)

        lo = max(0, c - 2)
        hi = min(window, c + 3)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # decimated complex FIR -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # main envelope+power KAN: 8 features -> (re, im) complex gain
        self.kan = KANLayer(8, 2, grid, order, x_min=-1.0, x_max=1.0,
                            quant_bits=quant_bits)

        # smooth 1-D power-variance gate: p_c -> depth scalar (pre-sigmoid)
        self.pgate = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # start near identity gain: tiny KAN perturbation
            self.kan.coef.mul_(0.05)
            self.kan.wb.mul_(0.05)
            # gate starts near open (depth ~1)
            self.pgate.coef.mul_(0.05)
            self.pgate.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                       # (B, window)
        taps = x[:, self.tap_idx]                      # (B, n_taps)
        lin = self.fir(taps)[:, 0]                     # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                            # (B,)
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        pc = (p * 2.0 - 1.0).reshape(B)                # (B,) in [-1,1]

        # bounded, free feature construction squashed into [-1,1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)         # explicit high-drive term
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env * mem_env * 3.0)
        ep = torch.tanh(env * p * 2.5)                 # env*p cross term
        e3p = torch.tanh(env * env * env * p * 3.0)    # high-drive*p cross

        feat = torch.stack([e, e2, e3, m, m2, pc, ep, e3p], dim=-1)  # (B, 8)
        g = self.kan(feat)                             # (B, 2)

        # smooth power-variance gate in (0,1), full depth at p=1 by init
        gate = torch.sigmoid(self.pgate(pc.reshape(B, 1))[:, 0] + 2.0)  # (B,)

        gr = 1.0 + g[:, 0] * gate
        gi = g[:, 1] * gate
        gain = torch.complex(gr, gi)                   # (B,) complex

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)                   # (B,)