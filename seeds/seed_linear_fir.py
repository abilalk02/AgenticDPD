
import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear


class DPD(DPDModule):
    """Complex linear FIR predistorter (cheapest | ignores power)."""

    def __init__(self, window: int, taps: int = 21):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

    def forward(self, xw, p):
        c, h = self.window // 2, self.taps // 2
        return self.fir(xw[:, c - h : c + h + 1]).squeeze(-1)
