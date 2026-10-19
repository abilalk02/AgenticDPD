import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RLinear, KANLayer


class DPD(DPDModule):
    """Power-conditioned mixture-of-experts with a smooth learned gain g(p).

    p never touches the x-path directly. A KAN maps p -> {expert mix weights,
    correction gain}. Each expert is a Tanh MLP on the envelope features; their
    outputs are blended per-power and scaled by g(p). This lets distinct drive
    levels get specialized correction shapes -> better worst-case over sweep.
    """

    def __init__(self, window: int, hidden: int = 40, experts: int = 4):
        super().__init__(window)
        self.experts = experts
        d = window * 5  # real, imag, |x|, |x|^2, |x|^3  (no raw p feature)

        # Shared trunk producing a common representation.
        self.trunk = nn.Sequential(
            RLinear(d, hidden), nn.Tanh(),
            RLinear(hidden, hidden), nn.Tanh(),
        )

        # Per-expert heads: each outputs a 2-vector (real, imag) correction.
        self.heads = nn.ModuleList([RLinear(hidden, 2) for _ in range(experts)])

        # Smooth 1-D power conditioner: p -> [mix logits (experts), gain (1)].
        # KAN gives a smooth, well-behaved function on [0,1] mapped into spline
        # support via a tanh squash.
        self.pcond = KANLayer(1, experts + 1, grid=8, order=3,
                              x_min=-1.5, x_max=1.5)

    def forward(self, xw, p):
        a = xw.abs()
        f = torch.cat([xw.real, xw.imag, a, a**2, a**3], dim=-1)
        h = self.trunk(f)  # (B, hidden)

        # Power conditioning path.
        ps = 1.2 * torch.tanh((2.0 * p - 1.0) * 1.5).unsqueeze(-1)  # -> spline range
        cond = self.pcond(ps)                                       # (B, experts+1)
        mix = torch.softmax(cond[:, :self.experts], dim=-1)         # (B, experts)
        gain = torch.tanh(cond[:, self.experts])                    # (B,) smooth scale

        # Blend expert corrections weighted by per-power mix.
        acc_r = torch.zeros_like(gain)
        acc_i = torch.zeros_like(gain)
        for k, head in enumerate(self.heads):
            o = head(h)                       # (B, 2)
            w = mix[:, k]
            acc_r = acc_r + w * o[:, 0]
            acc_i = acc_i + w * o[:, 1]

        corr = torch.complex(gain * acc_r, gain * acc_i)
        return xw[:, self.window // 2] + corr