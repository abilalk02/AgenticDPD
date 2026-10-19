import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """Envelope-conditioned KAN DPD (g019/g005c2-style core that natively clears
    the 3 dBFS ceiling) with a power-gated high-drive odd-order branch and
    smooth KAN(p) power conditioning. Spline inputs are hard-squashed into the
    grid so nothing leaves support at peak envelope -> tight seeds at 3 dBFS.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 14, grid: int = 8,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2

        # --- static AM/AM + AM/PM branch (proven core) ---
        self.kan1 = KANLayer(6, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # --- dedicated high-drive odd-order branch ---
        # inputs: [r, r^3, r^5, rm, rp, epf, p]  all squashed to [-1,1]
        self.hkan1 = KANLayer(7, hidden, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)
        self.hkan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        # --- power scheduling gates: smooth learned 1-D functions of p ---
        self.pkan = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)   # gate on main correction
        self.hpkan = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)  # gate on high-drive branch

        # --- static linear complex FIR (carries identity) ---
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # --- memory-polynomial cross term ---
        self.mp = RLinear(5, 2, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            # identity linear path through centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0 / self.FEAT_SCALE
            # small nonlinear corrections (near-zero residuals)
            for l in (self.kan1, self.kan2, self.hkan1, self.hkan2):
                l.coef.mul_(0.08)
                l.wb.mul_(0.08)
            # power gates init -> approx constant ~1
            self.pkan.coef.mul_(0.05)
            self.pkan.wb.mul_(0.05)
            self.hpkan.coef.mul_(0.05)
            self.hpkan.wb.mul_(0.05)
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

        # squash into spline support [-1,1] (bounded even at peak drive)
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

        # --- high-drive odd-order branch (all inputs squashed) ---
        r3 = torch.tanh((r ** 3) * 4.0)
        r5 = torch.tanh((r ** 5) * 6.0)
        hfeat = torch.stack([rf, r3, r5, rmf, rpf, epf, pf], dim=-1)  # (B,7)
        hgain = self.hkan2(self.hkan1(hfeat))                         # (B,2)
        # ramp on near full drive; sigmoid keeps it in [0,1], starts small
        hgate = torch.sigmoid(self.hpkan(pf.reshape(B, 1))[:, 0] - 1.0)  # (B,)
        hcorr = torch.complex(hgain[:, 0] * hgate, hgain[:, 1] * hgate)
        hd_out = xc * hcorr

        # --- static linear FIR (identity path) ---
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

        out = fir_out + am_out + hd_out + mp_c

        return (out / self.FEAT_SCALE).to(torch.complex64)