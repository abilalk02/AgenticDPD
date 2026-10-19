import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RConv1d, RLinear, KANLayer


class DPD(DPDModule):
    """8-bit quantized 1-D CNN trunk + 16-bit head + KAN power modulation."""

    def __init__(self, window: int, ch: int = 24, k: int = 7, depth: int = 3,
                 head_hidden: int = 32):
        super().__init__(window)
        # input channels: real, imag, |x|, |x|^2, p  -> 5
        convs, c_in = [], 5
        for _ in range(depth):
            convs += [RConv1d(c_in, ch, k, padding=k // 2, quant_bits=8),
                      nn.Tanh()]
            c_in = ch
        self.convs = nn.Sequential(*convs)

        # correction head kept at higher precision (cheap in BOP)
        self.head = nn.Sequential(
            RLinear(ch, head_hidden, quant_bits=16), nn.Tanh(),
            RLinear(head_hidden, 2, quant_bits=16))

        # learned smooth 1-D gain of drive power p in [0,1] -> real gain
        self.pgain = KANLayer(1, 2, grid=6, order=3,
                              x_min=-1.5, x_max=1.5, quant_bits=8)

    def forward(self, xw, p):
        env = xw.abs()
        pc = p.unsqueeze(-1).expand(-1, self.window)
        f = torch.stack([xw.real, xw.imag, env, env * env, pc], dim=1)
        h = self.convs(f)[:, :, self.window // 2]
        o = self.head(h)

        # power-dependent complex gain: map p to [-1.5,1.5] support
        ps = 3.0 * (p.unsqueeze(-1) - 0.5)          # (B,1) in [-0.75,0.75]
        g = self.pgain(ps)                          # (B,2)
        gr = 1.0 + torch.tanh(g[:, 0])
        gi = torch.tanh(g[:, 1])
        gain = torch.complex(gr, gi)

        corr = torch.complex(o[:, 0], o[:, 1]) * gain
        return xw[:, self.window // 2] + corr