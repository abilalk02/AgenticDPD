import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Scaled envelope-conditioned KAN predistorter aimed at the
    10^7.0-10^7.5 BOP niche. Power p enters as a smooth gain/coefficient
    modulation (never a divisor); all spline inputs are squashed into the
    KAN grid support for high-drive robustness."""

    def __init__(self, window: int):
        super().__init__(window)
        self.window = window

        D = 16          # complex basis width
        H = 20          # KAN hidden width
        G = 10          # spline grid
        K = 3           # spline order

        # Linear anchor: complex FIR over the whole window (dominant response)
        self.fir = CLinear(window, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, window // 2] = 1.0

        # Complex tap mixer -> rich complex basis
        self.mix = CLinear(window, D, bias=True)

        # KAN core operating on real envelope features + power feature.
        # Feature dim = |z| (D) + z.real (D) + z.imag (D) + p features (2)
        d_feat = 3 * D + 2
        self.kan1 = KANLayer(d_in=d_feat, d_out=H, grid=G, order=K,
                             x_min=-1.5, x_max=1.5)
        self.kan2 = KANLayer(d_in=H, d_out=2 * D, grid=G, order=K,
                             x_min=-1.5, x_max=1.5)

        # Head: complex combine of the modulated basis
        self.head = CLinear(D, 1, bias=False)
        with torch.no_grad():
            self.head.wr.zero_(); self.head.wi.zero_()

    def forward(self, xw, p):
        B = xw.shape[0]

        # Linear anchor path
        lin = self.fir(xw).squeeze(-1)             # complex (B,)

        # Complex basis
        z = self.mix(xw)                           # complex (B,D)
        env = z.abs()                              # real (B,D), free
        zr = z.real
        zi = z.imag

        # Smooth power features (safe at p=0 and p=1)
        pc = p.reshape(B, 1)
        pf = torch.cat([pc, pc * pc], dim=1)       # (B,2)

        feat = torch.cat([env, zr, zi, pf], dim=1) # (B, 3D+2)
        feat = 1.5 * torch.tanh(feat / 1.5)        # squash into grid

        h = self.kan1(feat)                        # real (B,H)
        h = 1.5 * torch.tanh(h / 1.5)              # keep inside next grid
        out = self.kan2(h)                         # real (B, 2D)

        D = env.shape[1]
        gate = torch.sigmoid(out[:, :D])           # per-branch gain (B,D)
        corr = out[:, D:]                          # additive correction (B,D)

        # Power-modulated gate: at low drive gate stays near learned base;
        # correction scaled by a smooth (0..1) function of p for robustness.
        pmod = 0.5 + 0.5 * torch.tanh(pc)          # (B,1) in (0.5,1)

        # Modulate the complex basis: gain on magnitude/phase, plus a small
        # real-phase correction expressed back into complex domain.
        mod = gate + pmod * torch.tanh(corr)       # real (B,D)
        zc = z * mod.to(z.dtype)                    # complex (B,D)

        y = self.head(zc).squeeze(-1)              # complex (B,)

        return lin + y