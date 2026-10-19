import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Decimated complex-FIR backbone + power-modulated envelope KAN gain.

    Aimed at BOP bin 1 (10^4.5-10^5.0). Keeps the parent's clean
    decimated-FIR + envelope-only KAN structure (flat all-drive, no
    high-drive collapse) but:
      - feeds a smooth learned function of p into BOTH KAN stages so a
        single model spans p=0..1 with tight seeds,
      - uses two envelope powers (|x|, |x|^2 squashed) to better resolve
        the compression knee at 3 dBFS,
      - quantizes FIR (8b) and KAN (10b) to sit inside the BOP band while
        holding RMpS ~300.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2, hidden: int = 8,
                 grid: int = 7, order: int = 3):
        super().__init__(window)
        c = window // 2
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        # complex FIR over decimated taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=8)

        # envelope + power nonlinearity: (|x|, |x|^2, p) -> hidden -> (gr, gi)
        self.kan1 = KANLayer(3, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=10)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=10)

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

        xc = x[:, self.c]                              # centre sample
        env = torch.abs(xc)                            # (B,)

        # two envelope features (free ops), squashed into spline support
        e1 = torch.tanh(env * 2.0)                     # (B,)
        e2 = torch.tanh(env * env * 3.0)               # (B,) compression knee

        # smooth power feature into [-1,1]; safe at p=0 and p=1
        pc = p.reshape(B)                              # (B,)
        p_s = torch.tanh(1.5 * (pc - 0.5))             # (B,)

        feat = torch.stack([e1, e2, p_s], dim=-1)      # (B, 3)
        feat = torch.tanh(feat)                        # keep strictly inside

        h = self.kan1(feat)                            # (B, hidden)
        h = torch.tanh(h)                              # squash for kan2 grid
        g = self.kan2(h)                               # (B, 2)

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])   # gain near 1

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)