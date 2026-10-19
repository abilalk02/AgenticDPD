import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """Compact GMP-augmented KAN head with power conditioning.

    Replaces the parent's dominant CNN stack with parameter-free envelope/
    delay features -> small RLinear mixer -> KAN spline head. A power-driven
    KAN gates the spline inputs and a GMP residual (power-gated complex gain)
    covers the high-drive nonlinear bottleneck. One model, whole sweep.
    """

    def __init__(self, window: int, mem: int = 5, feat: int = 12,
                 kan_in: int = 10):
        super().__init__(window)
        self.mem = min(mem, window)
        self.kan_in = kan_in
        c = window // 2
        # symmetric memory taps around centre
        half = self.mem // 2
        idx = [max(0, min(window - 1, c + d)) for d in range(-half, half + 1)]
        self.register_buffer("taps", torch.tensor(idx, dtype=torch.long))
        n_taps = len(idx)

        # parameter-free feature width:
        #   centre I, centre Q, + |x|,|x|^2,|x|^3 over memory taps
        self.n_feat = 2 + 3 * n_taps

        # --- linear mixer down to KAN input width (8b) ---
        self.mix = RLinear(self.n_feat, kan_in, quant_bits=8)

        # --- power -> per-channel spline gate ---
        self.pmod = KANLayer(1, kan_in, grid=6, order=3, x_min=0.0, x_max=1.0)

        # --- KAN spline head: learnable per-channel nonlinearity -> (re, im) ---
        self.kan = KANLayer(kan_in, 2, grid=8, order=3, x_min=-1.5, x_max=1.5)

        # --- GMP memory-polynomial residual (cheap complex coeffs) ---
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=self.mem, cross=2)

        # --- power-conditioned complex gain on the GMP contribution ---
        self.pgain = KANLayer(1, 2, grid=6, order=3, x_min=0.0, x_max=1.0)

    def forward(self, xw, p):
        c = self.window // 2
        pin = p.unsqueeze(-1)  # (B,1)

        xc = xw[:, c]
        mtaps = xw[:, self.taps]           # (B, n_taps) complex
        a = mtaps.abs()                    # (B, n_taps)
        a2 = a * a
        a3 = a2 * a

        feats = torch.cat([
            xc.real.unsqueeze(-1),
            xc.imag.unsqueeze(-1),
            a, a2, a3,
        ], dim=-1)                         # (B, n_feat) real, free

        # linear mix -> squash into spline support
        z = self.mix(feats)                # (B, kan_in)
        gmod = self.pmod(pin)              # (B, kan_in)
        z = torch.tanh(z) * (1.0 + 0.5 * torch.tanh(gmod))

        o = self.kan(z)                    # (B, 2)
        main = torch.complex(o[:, 0], o[:, 1])

        # GMP residual gated by learned power function
        g = self.gmp(xw)
        if g.dim() > 1:
            g = g[:, c] if g.shape[-1] == self.window else g.reshape(g.shape[0], -1)[:, 0]
        gate = self.pgain(pin)            # (B,2)
        gmp_res = g * torch.complex(gate[:, 0], gate[:, 1])

        return xc + main + gmp_res