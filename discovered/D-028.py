import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Decimated weight-shared complex FIR + power-modulated envelope KAN.

    Backbone: cheap complex FIR over a decimated tap window (linear memory).
    Nonlinearity: a single bounded KAN maps a rich, parameter-free feature
    vector [env, env^2, p, env*p, mem_env] -> complex gain, multiplying the
    centre sample. Power p enters both directly and as an env*p cross term so
    one model spans the whole drive sweep and can bend the AM/AM curve at high
    drive (the ACLR bottleneck). mem_env is a decimated envelope-average that
    injects a touch of memory nonlinearity lost to tap decimation.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2, hidden: int = 6,
                 grid: int = 8, order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2

        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        # short symmetric neighbourhood for a memory-envelope feature (free)
        lo = max(0, c - 2)
        hi = min(window, c + 3)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # complex FIR over decimated taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # feature layout: [env, env^2, p, env*p, mem_env]  (5-D, all free)
        self.kan1 = KANLayer(5, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def forward(self, xw, p):
        x = xw * self.FEAT_SCALE                      # (B, window)
        taps = x[:, self.tap_idx]                     # (B, n_taps)
        lin = self.fir(taps)[:, 0]                    # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                           # (B,)

        # local memory envelope (mean |x| over short neighbourhood), free
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        # bounded squashes into spline support
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        m = torch.tanh(mem_env * 2.0)
        pp = p * 2.0 - 1.0                            # center p to [-1,1]
        ep = torch.tanh(env * p * 2.5) * 2.0 - 0.0    # env*p cross term
        ep = torch.tanh(ep)

        feat = torch.stack([e, e2, pp, ep, m], dim=-1)  # (B, 5)

        g = self.kan2(self.kan1(feat))                # (B, 2)
        gain = torch.complex(1.0 + g[:, 0], g[:, 1])

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)