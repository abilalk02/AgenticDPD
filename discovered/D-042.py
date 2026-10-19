import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Scaled envelope-conditioned KAN with power modulation (1-6k RMpS niche).

    Backbone: cheap complex FIR over a decimated tap window (linear memory).
    Nonlinearity: a WIDE two-layer bounded KAN maps a parameter-free feature
    vector [env, env^2, env^3, p, env*p, mem_env, mem_env*p] -> complex gain
    that multiplies the centre-sample linear estimate. All KAN inputs are
    squashed into the spline support so high drive (3 dBFS) stays stable and
    seed-to-seed variance is low. Power p enters directly, via cross terms,
    and via a smooth learned function, so one model spans the whole sweep and
    is safe at p==0 and p==1.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2, hidden: int = 16,
                 grid: int = 10, order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2
        self.c = c

        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)

        lo = max(0, c - 3)
        hi = min(window, c + 4)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # complex FIR over decimated taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # feature layout (7-D, all parameter-free):
        # [env, env^2, env^3, p_centered, env*p, mem_env, mem_env*p]
        n_feat = 7
        self.kan1 = KANLayer(n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                       # (B, window)
        taps = x[:, self.tap_idx]                      # (B, n_taps) complex
        lin = self.fir(taps)[:, 0]                     # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                            # (B,)

        # local memory envelope (mean |x| over short neighbourhood), free
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        pv = p.reshape(B)                              # (B,), safe at 0 and 1

        # bounded squashes into [-1,1] spline support
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        pp = pv * 2.0 - 1.0                            # center p to [-1,1]
        ep = torch.tanh(env * pv * 2.5)
        m = torch.tanh(mem_env * 2.0)
        mp = torch.tanh(mem_env * pv * 2.5)

        feat = torch.stack([e, e2, e3, pp, ep, m, mp], dim=-1)  # (B, 7)

        h = self.kan1(feat)
        h = self.kan2(h)
        g = self.kan3(h)                               # (B, 2)

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)