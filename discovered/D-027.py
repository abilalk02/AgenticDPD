import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RConv1d, RLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """Compact quantized conv feature extractor -> power-modulated KAN spline head
    + power-gated GMP residual. One 8-bit conv (cheap features), 16-bit bounded
    spline bases for graceful high-drive behaviour, single model over full sweep.
    """

    def __init__(self, window: int, ch: int = 16, k: int = 7, kan_in: int = 12):
        super().__init__(window)
        self.kan_in = kan_in

        # --- single 8-bit conv feature extractor over [re, im, |x|, p] ---
        self.conv1 = RConv1d(4, ch, k, padding=k // 2, quant_bits=8)
        self.conv2 = RConv1d(ch, ch, 3, padding=1, quant_bits=8)

        # --- project conv centre-tap features down to KAN width (8b) ---
        self.proj = RLinear(ch, kan_in, quant_bits=8)

        # --- power -> per-channel gain (16b bounded spline) ---
        self.pmod = KANLayer(1, kan_in, grid=6, order=3,
                             x_min=0.0, x_max=1.0, quant_bits=16)

        # --- KAN spline head: learnable per-channel nonlinearity -> (re, im) ---
        self.kan = KANLayer(kan_in, 2, grid=8, order=3,
                            x_min=-1.5, x_max=1.5, quant_bits=16)

        # --- GMP memory-polynomial residual (cheap complex coeffs) ---
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=5, cross=2)

        # --- power -> complex gain on GMP residual (16b bounded spline) ---
        self.pgain = KANLayer(1, 2, grid=6, order=3,
                              x_min=0.0, x_max=1.0, quant_bits=16)

    def forward(self, xw, p):
        c = self.window // 2
        pin = p.unsqueeze(-1)                          # (B,1)

        # feature stack: real, imag, envelope, power channel
        pc = pin.expand(-1, self.window)
        f = torch.stack([xw.real, xw.imag, xw.abs(), pc], dim=1)
        h = torch.tanh(self.conv1(f))
        h = torch.tanh(self.conv2(h))[:, :, c]         # (B, ch)

        # project + power-modulated per-channel gate, squash into spline support
        z = self.proj(h)                               # (B, kan_in)
        gmod = self.pmod(pin)                          # (B, kan_in)
        z = torch.tanh(z) * (1.0 + 0.5 * torch.tanh(gmod))

        o = self.kan(z)                                # (B, 2)
        cnn_res = torch.complex(o[:, 0], o[:, 1])

        # GMP residual gated by learned function of power
        g = self.gmp(xw)
        if g.dim() > 1:
            g = g[:, c] if g.shape[-1] == self.window else g.reshape(g.shape[0], -1)[:, 0]
        gate = self.pgain(pin)                         # (B,2)
        gmp_res = g * torch.complex(gate[:, 0], gate[:, 1])

        return xw[:, c] + cnn_res + gmp_res