import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear


class DPD(DPDModule):
    """Compact power-aware predistorter: short complex FIR + tiny GMP-style
    nonlinear correction whose strength is modulated by RF drive power p."""

    def __init__(self, window: int, taps: int = 9, ntaps: int = 5):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1

        # Linear complex FIR (dominant response)
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # Nonlinear correction: complex-weighted odd-order envelope terms.
        # Feature vector per tap: x * |x|^2 and x * |x|^4  (memory nonlinear)
        # Built parameter-free; combined by a small CLinear.
        self.nl = CLinear(2 * self.ntaps, 1, bias=False)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()

        # Power modulation: map p -> real gain for the nonlinear branch.
        self.pgain = RLinear(2, 1, bias=True)
        with torch.no_grad():
            self.pgain.weight.zero_()
            self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2

        # Linear path
        lin = self.fir(xw[:, c - h : c + h + 1]).squeeze(-1)

        # Nonlinear features (parameter-free)
        xn = xw[:, c - hn : c + hn + 1]           # (B, ntaps) complex
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2                               # 3rd order
        t5 = xn * a2 * a2                          # 5th order
        feats = torch.cat([t3, t5], dim=1)         # (B, 2*ntaps) complex
        nl = self.nl(feats).squeeze(-1)            # (B,) complex

        # Power-dependent real gain on nonlinear term
        pf = torch.stack([p, p * p], dim=1)        # (B, 2)
        g = self.pgain(pf).squeeze(-1)             # (B,) real
        nl = nl * g.to(nl.dtype)

        return lin + nl