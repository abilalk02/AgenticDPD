
import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RLinear


class DPD(DPDModule):
    """Memoryless power-conditioned MLP: [I, Q, p] of the centre sample ->
    two ReLU hidden layers (16, 16) -> complex correction added to the
    centre sample. RM = 3*16 + 16*16 + 16*2 = 336."""

    def __init__(self, window: int, hidden: int = 16, depth: int = 2):
        super().__init__(window)
        d = 3
        layers = []
        for _ in range(depth):
            layers += [RLinear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [RLinear(d, 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, xw, p):
        x = xw[:, self.window // 2]
        f = torch.stack([x.real, x.imag, p], dim=-1)
        o = self.net(f)
        return x + torch.complex(o[:, 0], o[:, 1])
