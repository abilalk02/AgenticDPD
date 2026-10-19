import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """Envelope-conditioned KAN DPD, mid-BOP (10^5.5-10^6.0) fill.

    Core:
      * identity complex FIR carries the linear path (init exact identity)
      * dual-KAN static AM/AM+AM/PM law over local envelope context + p,
        quantized 8-bit to place BOP in the target niche
      * smooth learned 1-D KAN(p) gate sharpens the nonlinear correction at
        high drive without hurting low-drive linearity
      * envelope-power-modulated complex FIR for dynamic memory
    All spline inputs are hard-squashed into [-1,1] for seed robustness.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 16, grid: int = 8,
                 order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        c = window // 2

        # static AM/AM + AM/PM: [r_-2,r_-1,r_0,r_+1,r_+2, env_pow, therm, p] -> hidden -> 2
        self.kan1 = KANLayer(8, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # smooth 1-D power gate KAN(p) -> scalar (full precision, tiny)
        self.pkan = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=32)

        # identity complex FIR
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # dynamic envelope-modulated complex FIR
        self.fir_dyn = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0 / self.FEAT_SCALE
            self.fir_dyn.wr.mul_(0.02)
            self.fir_dyn.wi.mul_(0.02)
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

        env = torch.abs(xs)                        # (B, window), free
        r = env[:, c]
        therm = env.mean(dim=-1)
        env_pow = env.pow(2).mean(dim=-1)

        idx_m = max(c - 1, 0)
        idx_p = min(c + 1, w - 1)
        idx_m2 = max(c - 2, 0)
        idx_p2 = min(c + 2, w - 1)

        rm = env[:, idx_m]
        rp = env[:, idx_p]
        rm2 = env[:, idx_m2]
        rp2 = env[:, idx_p2]

        # squash all features into spline support [-1,1]
        rf   = torch.tanh(r * 2.0)
        rmf  = torch.tanh(rm * 2.0)
        rpf  = torch.tanh(rp * 2.0)
        rm2f = torch.tanh(rm2 * 2.0)
        rp2f = torch.tanh(rp2 * 2.0)
        epf  = torch.tanh(env_pow * 3.0)
        tf   = torch.tanh(therm * 2.0)
        pf   = (p * 2.0 - 1.0).clamp(-1.0, 1.0)

        feat = torch.stack([rm2f, rmf, rf, rpf, rp2f, epf, tf, pf], dim=-1)  # (B,8)

        gain = self.kan2(self.kan1(feat))          # (B,2)

        # smooth power gate around 1
        gate = 1.0 + self.pkan(pf.reshape(B, 1))[:, 0]   # (B,)

        gI = gain[:, 0] * gate
        gQ = gain[:, 1] * gate
        corr = torch.complex(gI, gQ)
        am_out = xc * corr                         # nonlinear residual

        # identity linear FIR
        fir_out = self.fir(xs)[:, 0]

        # dynamic envelope-modulated FIR
        mod = (env * env).to(xs.dtype)
        xs_mod = xs * mod
        fir_dyn_out = self.fir_dyn(xs_mod)[:, 0] * gate.to(torch.complex64)

        out = fir_out + am_out + fir_dyn_out

        return (out / self.FEAT_SCALE).to(torch.complex64)