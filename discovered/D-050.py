import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """Mid-scale envelope-conditioned dual-KAN with smooth power modulation.

    Backbone is pure spline (KAN) — scales up gracefully without the CNN's
    high-drive power-robustness penalty. Power p enters only through smooth
    learned 1-D KAN functions (gain / coeff modulation), never as a raw
    concatenated feature and never via divide/log, keeping every seed stable
    at 3.0 dBFS. Targets the sparse 950-6.9k RMpS band (~1.5k RMpS).
    """

    def __init__(self, window: int, d: int = 16, grid: int = 10, order: int = 3):
        super().__init__(window)
        self.d = d

        # complex tap mixer -> latent complex features
        self.mix = CLinear(window, d)

        # first KAN over envelope powers -> per-channel gate features
        self.kan_env = KANLayer(d, d, grid=grid, order=order,
                                x_min=-1.5, x_max=1.5)

        # smooth learned 1-D function of power -> per-channel gain modulation
        self.pmod = KANLayer(1, d, grid=6, order=order, x_min=0.0, x_max=1.0)

        # second KAN spline head: real & imag parts of the correction
        self.kan_head = KANLayer(d, 2, grid=grid, order=order,
                                 x_min=-1.5, x_max=1.5)

        # cheap complex memory-polynomial residual
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=5, cross=2)

        # power-conditioned complex gain on the GMP residual
        self.pgain = KANLayer(1, 2, grid=6, order=order, x_min=0.0, x_max=1.0)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        pc = p.reshape(B, 1)  # (B,1); safe at p==0 and p==1

        # --- complex latent + envelope features ---
        z = self.mix(xw)                        # complex (B,d)
        env = z.abs()                           # real (B,d), free
        env_sq = 1.5 * torch.tanh(env / 1.5)    # squash into spline support

        # first KAN -> gate features
        g_feat = self.kan_env(env_sq)           # (B,d)

        # smooth power modulation (multiplicative, no divide-by-p)
        pmod = self.pmod(pc)                     # (B,d)
        gate = torch.sigmoid(g_feat) * (1.0 + 0.5 * torch.tanh(pmod))

        # modulate complex latent, re-derive squashed real features
        zg = z * gate.to(z.dtype)               # complex (B,d)
        feat_r = 1.5 * torch.tanh(zg.real / 1.5)
        feat_i = 1.5 * torch.tanh(zg.imag / 1.5)
        feat = feat_r + feat_i                  # (B,d), both inside grid

        o = self.kan_head(feat)                 # (B,2)
        kan_res = torch.complex(o[:, 0], o[:, 1])

        # --- GMP memory residual, p-gated ---
        gm = self.gmp(xw)
        if gm.dim() > 1:
            if gm.shape[-1] == self.window:
                gm = gm[:, c]
            else:
                gm = gm.reshape(B, -1)[:, 0]
        pg = self.pgain(pc)                      # (B,2)
        gmp_res = gm * torch.complex(pg[:, 0], pg[:, 1])

        y = xw[:, c] + kan_res + gmp_res         # complex (B,)
        return y