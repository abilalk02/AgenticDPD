import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, GMPBranch


class DPD(DPDModule):
    """Wide power-modulated KAN DPD targeting the empty 10^7.0-10^7.5 BOP niche.

    A complex tap-mix produces a rich set of complex features; their envelope
    powers feed a wide 32-bit KAN core (high BOP, graceful high-drive scaling).
    Power p enters only through smooth learned 1-D KAN gain modulation, and a
    complex GMP backbone anchors the classical response. All spline inputs are
    tanh-squashed into the grid for seed-stable high-drive behaviour.
    """

    def __init__(self, window: int, nmix: int = 12, core: int = 20,
                 grid: int = 10, order: int = 3):
        super().__init__(window)
        self.nmix = nmix
        self.core = core

        # complex mix of taps -> nmix complex features (full precision)
        self.mix = CLinear(window, nmix, quant_bits=32)

        # wide KAN core: envelope-power features -> core real gains
        # feature vector = [|z|, |z|^2, |z|^3] per mix channel  => 3*nmix inputs
        self.kan1 = KANLayer(3 * nmix, core, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=32)
        self.kan2 = KANLayer(core, nmix, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=32)

        # power conditioner: smooth 1-D KAN(p) -> per-channel complex gain
        self.pcond = KANLayer(1, 2 * nmix, grid=8, order=3,
                              x_min=-1.5, x_max=1.5, quant_bits=32)

        # complex head: modulated features -> centre correction
        self.head = CLinear(nmix, 1, quant_bits=32)

        # classical GMP backbone, power-gated
        self.gmp = GMPBranch(window, orders=(1, 3, 5, 7), mem=5, cross=2)
        self.pgmp = KANLayer(1, 2, grid=6, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=32)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2

        # complex feature mix
        z = self.mix(xw)                                  # (B, nmix) complex
        e = z.abs()                                       # (B, nmix) real, free

        # envelope power features, squashed into grid support
        e1 = 1.5 * torch.tanh(e / 1.5)
        e2 = 1.5 * torch.tanh((e * e) / 1.5)
        e3 = 1.5 * torch.tanh((e * e * e) / 1.5)
        feat = torch.cat([e1, e2, e3], dim=1)             # (B, 3*nmix)

        # KAN core -> per-channel real gains
        h = torch.tanh(self.kan1(feat))                   # (B, core)
        g = self.kan2(h)                                  # (B, nmix)
        gate = torch.sigmoid(g)                           # (B, nmix), free

        # power-modulated complex per-channel gain (smooth 1-D function of p)
        ps = 3.0 * (p.reshape(B, 1) - 0.5)                # (B,1) in support
        pc = self.pcond(ps)                               # (B, 2*nmix)
        pr = 1.0 + torch.tanh(pc[:, :self.nmix])          # (B, nmix)
        pi = torch.tanh(pc[:, self.nmix:])                # (B, nmix)
        pgain = torch.complex(pr, pi)                     # (B, nmix)

        # apply gates and power modulation to complex features
        zmod = z * gate.to(z.dtype) * pgain               # (B, nmix) complex
        corr_nn = self.head(zmod).squeeze(-1)             # (B,) complex

        # classical GMP backbone, power-gated
        gmp_out = self.gmp(xw)                            # (B,) complex
        gp = self.pgmp(ps)                                # (B,2)
        ggr = 1.0 + torch.tanh(gp[:, 0])
        ggi = torch.tanh(gp[:, 1])
        gmp_gain = torch.complex(ggr, ggi)               # (B,)
        corr_gmp = gmp_out * gmp_gain

        y = xw[:, c] + corr_nn + corr_gmp                # (B,) complex
        return y.to(torch.complex64)