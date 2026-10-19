import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """Ultra-low-BOP envelope-conditioned KAN DPD.

    Core: dual KAN (AM/AM + AM/PM) driven by squashed envelope features and
    a smooth 1-D power gate, wrapped around an identity complex FIR. All
    spline inputs are held inside [-1,1] so high-drive (3 dBFS) stays flat
    and seed-tight. KAN/MP layers are 8-bit quantized to land in the
    10^4.0-10^5.0 BOP niche at ~260 RMpS.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 10, grid: int = 7,
                 order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        c = window // 2

        # --- static AM/AM + AM/PM branch ---
        self.kan1 = KANLayer(6, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # --- smooth 1-D power gate ---
        self.pkan = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # --- identity complex FIR ---
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # --- small quantized memory-polynomial cross term ---
        self.mp = RLinear(5, 2, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0 / self.FEAT_SCALE
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.pkan.coef.mul_(0.05)
            self.pkan.wb.mul_(0.05)
            self.mp.weight.mul_(0.05)
            self.mp.bias.zero_()

    def forward(self, xw, p):
        c = self.window // 2
        w = self.window
        B = xw.shape[0]

        xs = xw * self.FEAT_SCALE
        xc = xs[:, c]

        # --- free envelope features ---
        env = torch.abs(xs)                       # (B, window)
        therm = env.mean(dim=-1)

        idx_m = max(c - 1, 0)
        idx_p = min(c + 1, w - 1)
        idx_m2 = max(c - 2, 0)
        idx_p2 = min(c + 2, w - 1)

        r  = env[:, c]
        rm = env[:, idx_m]
        rp = env[:, idx_p]
        env_pow = env.pow(2).mean(dim=-1)

        # squash into spline support [-1,1]
        rf   = torch.tanh(r * 2.0)
        rmf  = torch.tanh(rm * 2.0)
        rpf  = torch.tanh(rp * 2.0)
        epf  = torch.tanh(env_pow * 3.0)
        tf   = torch.tanh(therm * 2.0)
        pf   = (p * 2.0 - 1.0)                     # safe at p=0 and p=1

        feat = torch.stack([rf, rmf, rpf, epf, tf, pf], dim=-1)  # (B,6)
        gain = self.kan2(self.kan1(feat))                       # (B,2)

        gate = 1.0 + self.pkan(pf.reshape(B, 1))[:, 0]          # (B,)

        gI = gain[:, 0] * gate
        gQ = gain[:, 1] * gate
        corr = torch.complex(gI, gQ)
        am_out = xc * corr

        # --- identity linear FIR ---
        fir_out = self.fir(xs)[:, 0]

        # --- memory-polynomial cross term ---
        p2 = torch.stack([
            env[:, idx_m2] ** 2,
            env[:, idx_m] ** 2,
            env[:, c] ** 2,
            env[:, idx_p] ** 2,
            env[:, idx_p2] ** 2,
        ], dim=-1)                                # (B,5)
        mp = self.mp(p2)                          # (B,2)
        mp_c = torch.complex(mp[:, 0], mp[:, 1]) * xc

        out = fir_out + am_out + mp_c

        return (out / self.FEAT_SCALE).to(torch.complex64)