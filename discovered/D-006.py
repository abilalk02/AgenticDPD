import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RConv1d, RLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """Quantized CNN (8b) + 16b head + power-modulated GMP memory-polynomial residual."""

    def __init__(self, window: int, ch: int = 24, k: int = 7, depth: int = 3,
                 head_hidden: int = 32):
        super().__init__(window)

        # --- 8-bit convolutional feature stack (dominant BOP -> 4x cheaper) ---
        convs, c_in = [], 4
        for _ in range(depth):
            convs += [RConv1d(c_in, ch, k, padding=k // 2, quant_bits=8), nn.Tanh()]
            c_in = ch
        self.convs = nn.Sequential(*convs)

        # --- head: first layer 8b, output layer kept 16b for residual fidelity ---
        self.head1 = RLinear(ch, head_hidden, quant_bits=8)
        self.head2 = RLinear(head_hidden, 2, quant_bits=16)

        # --- GMP memory-polynomial residual (robust complex coeffs, cheap) ---
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=5, cross=2)

        # --- power-conditioned scalar gain on the GMP contribution ---
        self.pgain = KANLayer(1, 2, grid=6, order=3, x_min=0.0, x_max=1.0)

    def forward(self, xw, p):
        c = self.window // 2

        # power broadcast as extra conv channel
        pc = p.unsqueeze(-1).expand(-1, self.window)
        f = torch.stack([xw.real, xw.imag, xw.abs(), pc], dim=1)
        h = self.convs(f)[:, :, c]

        o = self.head2(torch.tanh(self.head1(h)))
        cnn_res = torch.complex(o[:, 0], o[:, 1])

        # GMP residual, gated by a learned function of power
        g = self.gmp(xw)                         # complex (B,)
        if g.dim() > 1:
            g = g[:, c] if g.shape[-1] == self.window else g.reshape(g.shape[0], -1)[:, 0]
        gate = self.pgain(p.unsqueeze(-1))       # (B,2)
        gmp_res = g * torch.complex(gate[:, 0], gate[:, 1])

        return xw[:, c] + cnn_res + gmp_res