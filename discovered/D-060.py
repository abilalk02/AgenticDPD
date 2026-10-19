import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Pruned g007c0-lineage DPD: decimated complex FIR + single power-modulated
    envelope KAN. One bounded KAN maps a compact free feature vector
    [env, env^2, p_centered, env*p, mem_env] -> complex gain on the centre
    sample. Envelope is compressed before splining so high drive stays in-grid,
    giving tight seed spread at 3 dBFS. Power enters directly and as an env*p
    cross term so one model spans the whole sweep with no unsafe ops at p=0/1.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 3, hidden: int = 8,
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

        lo = max(0, c - 2)
        hi = min(window, c + 3)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # cheap complex FIR over decimated taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # single bounded KAN: 5 free features -> complex gain (2 outputs)
        self.kan = KANLayer(5, hidden, grid, order, x_min=-1.0, x_max=1.0,
                            quant_bits=quant_bits)
        self.head = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            for l in (self.kan, self.head):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)
        taps = x[:, self.tap_idx]                         # (B, n_taps)
        lin = self.fir(taps)[:, 0]                        # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)

        # local memory envelope (free)
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        pc = p.reshape(B)                                 # (B,) in [0,1]

        # bounded compressive squashes into spline support [-1,1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        m = torch.tanh(mem_env * 2.0)
        pp = pc * 2.0 - 1.0                               # center p -> [-1,1]
        ep = torch.tanh(env * pc * 2.5)                   # env*p cross term

        feat = torch.stack([e, e2, pp, ep, m], dim=-1)    # (B, 5)

        h = self.kan(feat)
        h = torch.tanh(h)                                 # keep head in-grid
        g = self.head(h)                                  # (B, 2)

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])
        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)