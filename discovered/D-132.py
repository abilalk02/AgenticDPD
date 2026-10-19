import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """g019/g005c2-lineage envelope-conditioned KAN DPD, scaled up (wider hidden,
    higher grid, extra stacked KAN stage) to fill the empty 6-9k RMpS niche.
    Native 3 dBFS-clearing core. All spline inputs hard-squashed into grid
    support -> tight seeds at high drive. Power enters via 2p-1 (safe at 0/1)
    and smooth learned KAN(p) gates.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 28, grid: int = 12,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2

        # --- main AM/AM + AM/PM branch: 3 stacked KAN stages ---
        self.kan1 = KANLayer(7, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # --- dedicated high-drive odd-order branch ---
        self.hkan1 = KANLayer(8, hidden, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)
        self.hkan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        # --- smooth power-scheduling gates: learned 1-D functions of p ---
        self.pkan = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.hpkan = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        # --- static linear complex FIR (identity path + linear memory) ---
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # --- memory-polynomial cross term ---
        self.mp = RLinear(7, 2, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0 / self.FEAT_SCALE
            for l in (self.kan1, self.kan2, self.kan3, self.hkan1, self.hkan2):
                l.coef.mul_(0.06)
                l.wb.mul_(0.06)
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

        env = torch.abs(xs)                       # (B, window)
        therm = env.mean(dim=-1)

        idx_m = max(c - 1, 0)
        idx_p = min(c + 1, w - 1)
        idx_m2 = max(c - 2, 0)
        idx_p2 = min(c + 2, w - 1)
        idx_m3 = max(c - 3, 0)
        idx_p3 = min(c + 3, w - 1)

        r  = env[:, c]
        rm = env[:, idx_m]
        rp = env[:, idx_p]
        rm2 = env[:, idx_m2]
        rp2 = env[:, idx_p2]
        env_pow = env.pow(2).mean(dim=-1)

        # squash into spline support [-1,1]
        rf   = torch.tanh(r * 2.0)
        rmf  = torch.tanh(rm * 2.0)
        rpf  = torch.tanh(rp * 2.0)
        rm2f = torch.tanh(rm2 * 2.0)
        epf  = torch.tanh(env_pow * 3.0)
        tf   = torch.tanh(therm * 2.0)
        pf   = (p * 2.0 - 1.0)                     # safe at p=0 and p=1

        feat = torch.stack([rf, rmf, rpf, rm2f, epf, tf, pf], dim=-1)  # (B,7)
        h = self.kan1(feat)
        h = self.kan2(h)
        gain = self.kan3(h)                                            # (B,2)

        gate = 1.0 + self.pkan(pf.reshape(B, 1))[:, 0]                 # (B,)

        gI = gain[:, 0] * gate
        gQ = gain[:, 1] * gate
        corr = torch.complex(gI, gQ)
        am_out = xc * corr

        # --- high-drive odd-order branch ---
        r3 = torch.tanh((r ** 3) * 4.0)
        r5 = torch.tanh((r ** 5) * 6.0)
        r7 = torch.tanh((r ** 7) * 8.0)
        hfeat = torch.stack([rf, r3, r5, r7, rmf, rpf, epf, pf], dim=-1)  # (B,8)
        hgain = self.hkan2(self.hkan1(hfeat))                            # (B,2)
        hgate = torch.sigmoid(self.hpkan(pf.reshape(B, 1))[:, 0] - 1.0)  # (B,)
        hcorr = torch.complex(hgain[:, 0] * hgate, hgain[:, 1] * hgate)
        hd_out = xc * hcorr

        # --- static linear FIR (identity + linear memory) ---
        fir_out = self.fir(xs)[:, 0]

        # --- memory-polynomial cross term ---
        p2 = torch.stack([
            env[:, idx_m3] ** 2,
            env[:, idx_m2] ** 2,
            env[:, idx_m] ** 2,
            env[:, c] ** 2,
            env[:, idx_p] ** 2,
            env[:, idx_p2] ** 2,
            env[:, idx_p3] ** 2,
        ], dim=-1)                                # (B,7)
        mp = self.mp(p2)                          # (B,2)
        mp_c = torch.complex(mp[:, 0], mp[:, 1]) * xc

        out = fir_out + am_out + hd_out + mp_c

        return (out / self.FEAT_SCALE).to(torch.complex64)