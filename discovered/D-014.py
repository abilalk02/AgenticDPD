import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, RLinear, GMPBranch


class DPD(DPDModule):
    """Hybrid: parent's power-aware memoryless KAN corrector + a complex
    GMPBranch memory-polynomial skip path whose contribution is gain-
    modulated by a learned 1-D KAN function of the RF drive power p."""

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8, grid: int = 7,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        # --- memoryless power-aware KAN path (from parent) ---
        self.kan1 = KANLayer(3, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.linear_path = RLinear(2, 2, bias=False, quant_bits=quant_bits)

        # --- GMP memory-nonlinear skip path over full window ---
        self.gmp = GMPBranch(window, orders=(1, 3, 5, 7), mem=5, cross=2)

        # --- learned 1-D power->gain function for the GMP path ---
        self.pgain = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        with torch.no_grad():
            self.linear_path.weight.copy_(torch.eye(2))
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            # power-gain KAN starts near a constant ~1 so GMP passes through
            self.pgain.coef.mul_(0.05)
            self.pgain.wb.mul_(0.05)

    def forward(self, xw, p):
        c = self.window // 2
        x = xw[:, c] * self.FEAT_SCALE
        iq = torch.stack([x.real, x.imag], dim=-1)

        # power feature squashed into spline support (p in [0,1] -> [-1,1])
        p_feat = (2.0 * p - 1.0).unsqueeze(-1)

        # memoryless KAN correction + identity linear residual
        f = torch.cat([iq, p.unsqueeze(-1)], dim=-1)
        corr = self.kan2(self.kan1(f))
        lin = self.linear_path(iq)
        mless = (lin + corr) / self.FEAT_SCALE
        y_mless = torch.complex(mless[:, 0], mless[:, 1])

        # complex GMP memory path over full scaled window
        gmp_out = self.gmp(xw * self.FEAT_SCALE) / self.FEAT_SCALE
        if gmp_out.dim() > 1:
            gmp_out = gmp_out[:, 0] if gmp_out.shape[-1] == 1 else gmp_out.mean(-1)

        # power-dependent gain (learned 1-D function), near-unity offset
        g = 1.0 + self.pgain(p_feat)[:, 0]
        gmp_out = gmp_out * torch.complex(g, torch.zeros_like(g))

        return (y_mless + gmp_out).to(torch.complex64)