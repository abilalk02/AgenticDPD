import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Ultra-low-BOP KAN DPD for the empty 10^4.0-10^4.5 BOP niche.

    Decimated weight-shared complex FIR (linear memory) + a single bounded,
    heavily-quantized envelope/power KAN that bends the AM/AM curve across the
    whole drive sweep. 6-bit fake-quant on FIR + KAN drops BOP into the empty
    ultra-low bin while a modest RMpS (~180) keeps the spline backbone intact,
    preserving the high-drive ACLR that dense/conv cores lose.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 3, hidden: int = 5,
                 grid: int = 7, order: int = 3, quant_bits: int = 6):
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

        # complex FIR over decimated taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # feature layout: [env, env^2, p_centered, env*p, mem_env]  (all free)
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
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                       # (B, window)
        taps = x[:, self.tap_idx]                      # (B, n_taps)
        lin = self.fir(taps)[:, 0]                     # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                            # (B,)

        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        pc = p.reshape(B)                              # safe at 0 and 1
        # bounded squashes into spline support [-1, 1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        m = torch.tanh(mem_env * 2.0)
        pp = torch.tanh((pc * 2.0 - 1.0))              # centered p in [-1,1]
        ep = torch.tanh(env * pc * 2.5)                # env*p cross term

        feat = torch.stack([e, e2, pp, ep, m], dim=-1)  # (B, 5)

        g = self.kan2(self.kan1(feat))                 # (B, 2)
        gain = torch.complex(1.0 + g[:, 0], g[:, 1])

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)