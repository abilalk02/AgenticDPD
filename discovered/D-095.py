import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """Ultra-low-BOP envelope-conditioned KAN DPD (targets empty bins 0/1).

    Target niche: 10^4.0-10^5.0 BOP, ~180-220 RMpS.
    Core: 8-bit-quantized compact KAN gain correction around an identity FIR,
    with power p entering both as a squashed KAN feature and a smooth learned
    1-D power gate (variance-gate style) for robust high-drive linearization.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 6, grid: int = 3,
                 order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        c = window // 2

        # static AM/AM + AM/PM branch: [r_-1, r_0, r_+1, env_pow, therm, p] -> hidden -> 2
        self.kan1 = KANLayer(6, hidden, grid, order, x_min=-1.5, x_max=1.5,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.5, x_max=1.5,
                             quant_bits=quant_bits)

        # smooth learned 1-D power gate on the nonlinear correction
        self.pkan = KANLayer(1, 1, grid, order, x_min=-1.5, x_max=1.5,
                             quant_bits=quant_bits)

        # static complex FIR carrying identity
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0 / self.FEAT_SCALE
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.pkan.coef.mul_(0.05)
            self.pkan.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        w = self.window

        xs = xw * self.FEAT_SCALE
        xc = xs[:, c]

        # free envelope features
        env = torch.abs(xs)                         # (B, window)
        r = env[:, c]
        idx_m = max(c - 1, 0)
        idx_p = min(c + 1, w - 1)
        rm = env[:, idx_m]
        rp = env[:, idx_p]
        env_pow = env.pow(2).mean(dim=-1)
        therm = env.mean(dim=-1)

        # squash into spline support [-1.5, 1.5]
        rf  = 1.5 * torch.tanh(r * 2.0)
        rmf = 1.5 * torch.tanh(rm * 2.0)
        rpf = 1.5 * torch.tanh(rp * 2.0)
        epf = 1.5 * torch.tanh(env_pow * 3.0)
        tf  = 1.5 * torch.tanh(therm * 2.0)
        pf  = 1.5 * (p.reshape(B) * 2.0 - 1.0)      # p in [0,1] -> [-1.5,1.5]

        feat = torch.stack([rf, rmf, rpf, epf, tf, pf], dim=-1)  # (B,6)
        gain = self.kan2(self.kan1(feat))                        # (B,2)

        # smooth power gate around 1
        gate = 1.0 + self.pkan(pf.unsqueeze(-1))[:, 0]           # (B,)

        gI = gain[:, 0] * gate
        gQ = gain[:, 1] * gate
        corr = torch.complex(gI, gQ)
        am_out = xc * corr

        # identity FIR path
        fir_out = self.fir(xs)[:, 0]

        out = fir_out + am_out
        return (out / self.FEAT_SCALE).to(torch.complex64)