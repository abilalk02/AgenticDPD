
import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RConv1d, RLinear


class DPD(DPDModule):
    """1-D CNN over the tap window; p is a constant 4th input channel."""

    def __init__(self, window: int, ch: int = 24, k: int = 7, depth: int = 3,
                 head_hidden: int = 32):
        super().__init__(window)
        convs, c_in = [], 4
        for _ in range(depth):
            convs += [RConv1d(c_in, ch, k, padding=k // 2), nn.Tanh()]
            c_in = ch
        self.convs = nn.Sequential(*convs)
        self.head = nn.Sequential(RLinear(ch, head_hidden), nn.Tanh(),
                                  RLinear(head_hidden, 2))

    def forward(self, xw, p):
        pc = p.unsqueeze(-1).expand(-1, self.window)
        f = torch.stack([xw.real, xw.imag, xw.abs(), pc], dim=1)
        h = self.convs(f)[:, :, self.window // 2]
        o = self.head(h)
        return xw[:, self.window // 2] + torch.complex(o[:, 0], o[:, 1])
