
import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, RLinear


class DPD(DPDModule):
    """DPD-KAN paper model -> memoryless [I, Q, p] (grid 7, order 3, hidden 8,
    grid range [-1, 1]) around an identity-initialized learnable 2x2 linear
    path. FEAT_SCALE maps this pipeline's unit-RMS waveform onto the
    analyzer scale (~0.21 RMS)"""

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8, grid: int = 7,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.kan1 = KANLayer(3, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.linear_path = RLinear(2, 2, bias=False, quant_bits=quant_bits)
        with torch.no_grad():
            self.linear_path.weight.copy_(torch.eye(2))
            # lstsq-noise spline init + identity residual path
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def forward(self, xw, p):
        c = self.window // 2
        x = xw[:, c] * self.FEAT_SCALE
        iq = torch.stack([x.real, x.imag], dim=-1)
        f = torch.cat([iq, p.unsqueeze(-1)], dim=-1)
        corr = self.kan2(self.kan1(f))
        lin = self.linear_path(iq)
        o = (lin + corr) / self.FEAT_SCALE
        return torch.complex(o[:, 0], o[:, 1])
