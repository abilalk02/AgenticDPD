import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """Wide dual-stage KAN core (float32) on complex-mixed envelope basis,
    with a smooth learned 1-D power gain and a GMP memory-polynomial residual.
    Targets empty high-BOP niche (bin 9, 10^7.5-10^8.0 BOP)."""

    def __init__(self, window: int, mix: int = 20, hid: int = 24,
                 grid: int = 10, order: int = 4):
        super().__init__(window)
        self.window = window

        # complex linear mixing of taps -> rich complex basis
        self.cmix = CLinear(window, mix)

        # feature dim: |z| envelope (mix) + real part (mix) + imag part (mix) + p
        d_in = 3 * mix + 1

        # --- wide, high-order KAN core (float32 -> high BOP niche) ---
        self.kan1 = KANLayer(d_in, hid, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5)
        self.kan2 = KANLayer(hid, mix, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5)

        # complex head on gated basis
        self.head = CLinear(mix, 1)

        # GMP memory-polynomial residual
        self.gmp = GMPBranch(window, orders=(1, 3, 5, 7), mem=5, cross=2)

        # smooth learned 1-D power functions: gains for kan-path and gmp-path
        self.pgain = KANLayer(1, 4, grid=6, order=3, x_min=0.0, x_max=1.0)

    def _sq(self, x):
        return 1.5 * torch.tanh(x / 1.5)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2

        z = self.cmix(xw)                       # complex (B, mix)
        env = z.abs()                           # real (B, mix)
        re = z.real
        im = z.imag

        pc = p.reshape(B, 1)                     # (B,1), safe at 0 and 1
        feat = torch.cat([env, re, im, pc], dim=1)
        feat = self._sq(feat)                    # keep in spline support

        h = torch.tanh(self.kan1(feat))         # (B, hid)
        g = self.kan2(h)                         # (B, mix)
        gate = torch.sigmoid(g)                  # free, bounded

        # power-modulated complex correction path
        kan_out = self.head((z * gate.to(z.dtype)))  # complex (B,1)
        kan_res = kan_out.squeeze(-1)

        # GMP residual
        gm = self.gmp(xw)
        if gm.dim() > 1:
            gm = gm[:, c] if gm.shape[-1] == self.window \
                else gm.reshape(B, -1)[:, 0]

        # smooth learned power gains
        pg = self.pgain(pc)                       # (B,4)
        kan_gain = torch.complex(pg[:, 0], pg[:, 1])
        gmp_gain = torch.complex(pg[:, 2], pg[:, 3])

        y = xw[:, c] + kan_gain * kan_res + gmp_gain * gm
        return y.to(torch.complex64)