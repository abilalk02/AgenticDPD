import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Single-KAN power-modulated envelope corrector on a decimated complex FIR.

    ~3x cheaper than the parent: one KANLayer (5 -> 2) replaces the stacked
    pair. A cheap decimated complex FIR gives the linear/memory estimate; the
    KAN bends a bounded feature vector [env, env^2, p_centered, env*p, mem_env]
    into a complex gain that multiplies the FIR output. p enters via centered
    and cross terms only (safe at p==0 and p==1). All spline inputs are
    tanh-squashed into the grid support for high-drive seed stability.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2,
                 grid: int = 6, order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2

        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        lo = max(0, c - 2)
        hi = min(window, c + 3)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # decimated complex FIR -> centre-sample linear/memory estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # single bounded KAN: [env, env^2, p_c, env*p, mem_env] -> (re,im) gain
        self.kan = KANLayer(5, 2, grid, order, x_min=-1.0, x_max=1.0,
                            quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # start near identity gain: tiny KAN perturbation
            self.kan.coef.mul_(0.05)
            self.kan.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                       # (B, window)
        taps = x[:, self.tap_idx]                      # (B, n_taps)
        lin = self.fir(taps)[:, 0]                     # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                            # (B,)
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        # bounded, free feature construction squashed into [-1,1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        pc = p * 2.0 - 1.0                             # (B,) in [-1,1]
        ep = torch.tanh(env * p * 2.5)                 # env*p cross term
        m = torch.tanh(mem_env * 2.0)

        feat = torch.stack([e, e2, pc, ep, m], dim=-1)  # (B, 5)

        g = self.kan(feat)                             # (B, 2)
        gain = torch.complex(1.0 + g[:, 0], g[:, 1])   # (B,) complex

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)                   # (B,)