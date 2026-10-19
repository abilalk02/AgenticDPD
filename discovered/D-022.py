import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear


class DPD(DPDModule):
    """Decimated weight-shared complex FIR backbone + envelope-only KAN gain.

    - Linear memory: CLinear over a strided (decimated) subset of the tap
      window gives cheap complex FIR memory compensation.
    - Nonlinearity: a compact KAN maps (|x_c|, p) -> complex gain (real, imag)
      that multiplies the centre sample, capturing AM/AM & AM/PM compression.
      Envelope-only input keeps the KAN 2-D and bounded (best mid-niche
      quality/RMpS pattern), and power p enters directly so one model spans
      the whole drive sweep.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2, hidden: int = 8,
                 grid: int = 7, order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        # decimated tap indices, always including the centre
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        # complex FIR over decimated taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # envelope-only nonlinearity: (|x|, p) -> hidden -> (gr, gi)
        self.kan1 = KANLayer(2, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        with torch.no_grad():
            # init FIR near a centre passthrough
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # small nonlinear correction to start
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def forward(self, xw, p):
        x = xw * self.FEAT_SCALE                      # (B, window)
        taps = x[:, self.tap_idx]                     # (B, n_taps)
        lin = self.fir(taps)[:, 0]                    # (B,) complex

        xc = x[:, self.c]                             # centre sample (complex)
        env = torch.abs(xc)                           # (B,)
        # squash env into spline support
        env_s = torch.tanh(env * 2.0)
        feat = torch.stack([env_s, p], dim=-1)        # (B, 2)

        g = self.kan2(self.kan1(feat))                # (B, 2)
        gain = torch.complex(1.0 + g[:, 0], g[:, 1])  # gain near 1

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)