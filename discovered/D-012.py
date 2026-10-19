import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RConv1d, RLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """8b CNN features -> power-modulated KAN spline head + power-gated GMP skip.

    Hybridizes the parent's quantized-CNN/GMP topology with the KAN family:
    the dense tanh head is replaced by a KANLayer that learns per-channel
    spline nonlinearities, and a second KAN maps power -> per-channel gains
    so a single model linearizes the whole power sweep.
    """

    def __init__(self, window: int, ch: int = 24, k: int = 7, depth: int = 3,
                 kan_in: int = 16):
        super().__init__(window)
        self.kan_in = kan_in

        # --- 8-bit convolutional feature stack (dominant BOP -> 4x cheaper) ---
        convs, c_in = [], 4
        for _ in range(depth):
            convs += [RConv1d(c_in, ch, k, padding=k // 2, quant_bits=8), nn.Tanh()]
            c_in = ch
        self.convs = nn.Sequential(*convs)

        # --- project conv features down to KAN input width (8b) ---
        self.proj = RLinear(ch, kan_in, quant_bits=8)

        # --- power-conditioned per-channel gain on the KAN input ---
        self.pmod = KANLayer(1, kan_in, grid=6, order=3, x_min=0.0, x_max=1.0)

        # --- KAN spline head: learnable per-channel nonlinearity -> (real, imag) ---
        self.kan = KANLayer(kan_in, 2, grid=8, order=3, x_min=-1.5, x_max=1.5)

        # --- GMP memory-polynomial residual (cheap complex coeffs) ---
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=5, cross=2)

        # --- power-conditioned complex gain on the GMP contribution ---
        self.pgain = KANLayer(1, 2, grid=6, order=3, x_min=0.0, x_max=1.0)

    def forward(self, xw, p):
        c = self.window // 2
        pin = p.unsqueeze(-1)  # (B,1)

        # power broadcast as extra conv channel
        pc = pin.expand(-1, self.window)
        f = torch.stack([xw.real, xw.imag, xw.abs(), pc], dim=1)
        h = self.convs(f)[:, :, c]                     # (B, ch)

        # project + power-modulated gating, squash into spline support
        z = self.proj(h)                               # (B, kan_in)
        gmod = self.pmod(pin)                          # (B, kan_in)
        z = torch.tanh(z) * (1.0 + 0.5 * torch.tanh(gmod))

        o = self.kan(z)                                # (B, 2)
        cnn_res = torch.complex(o[:, 0], o[:, 1])

        # GMP residual, gated by a learned function of power
        g = self.gmp(xw)
        if g.dim() > 1:
            g = g[:, c] if g.shape[-1] == self.window else g.reshape(g.shape[0], -1)[:, 0]
        gate = self.pgain(pin)                         # (B,2)
        gmp_res = g * torch.complex(gate[:, 0], gate[:, 1])

        return xw[:, c] + cnn_res + gmp_res