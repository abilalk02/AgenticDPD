import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """Power-conditioned VCSEL IM/DD DPD.

    Power-conditioning mechanism (new vs parent):
      * A dedicated 1-D KAN reads drive power p ALONE and emits
          - a soft 2-way mixing weight m(p) in (0,1) over two AM/AM experts
          - a smooth correction scale s(p) that multiplies the ENTIRE
            nonlinear residual (expert mix + memory-polynomial term).
        This makes power a smooth learned gain/mixer rather than a flat
        feature competing inside the envelope spline.
      * Two compact AM/AM experts act on power-free envelope features
        [r, therm]; the p-gate blends them so capacity can specialize to
        the high-drive saturation regime (the ACLR bottleneck) without
        disturbing low-drive response.
      * A short complex FIR carries the linear identity path unchanged at
        all p (residual is scaled, identity is not).
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8, grid: int = 7,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2

        # --- power conditioner: p -> [gate_logit, scale, bias] ---
        self.pgate = KANLayer(1, 4, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)
        self.phead = RLinear(4, 3, bias=True, quant_bits=quant_bits)

        # --- two AM/AM experts on envelope features [r, therm] ---
        self.kanA1 = KANLayer(2, hidden, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)
        self.kanA2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)
        self.kanB1 = KANLayer(2, hidden, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)
        self.kanB2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        # --- short-memory complex FIR (identity + relaxation/TIA) ---
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # --- memory-polynomial cross term ---
        self.mp = RLinear(3, 2, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            # FIR passes the centre tap (identity path), undoing FEAT_SCALE
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0 / self.FEAT_SCALE

            for l in (self.kanA1, self.kanA2, self.kanB1, self.kanB2, self.pgate):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

            self.mp.weight.mul_(0.05)
            self.mp.bias.zero_()

            # phead: start near gate=0.5 (balanced), scale=1, bias=0
            self.phead.weight.mul_(0.05)
            self.phead.bias.zero_()

    def forward(self, xw, p):
        c = self.window // 2

        xs = xw * self.FEAT_SCALE
        xc = xs[:, c]

        # --- envelope features (free) ---
        r = torch.abs(xc)
        env = torch.abs(xs)
        therm = env.mean(dim=-1)

        rf = torch.tanh(r * 2.0)
        tf = torch.tanh(therm * 2.0)
        efeat = torch.stack([rf, tf], dim=-1)               # (B,2) power-free

        # --- power conditioner ---
        pf = (p * 2.0 - 1.0).unsqueeze(-1)                  # (B,1) in [-1,1]
        pc = self.phead(self.pgate(pf))                     # (B,3)
        m = torch.sigmoid(pc[:, 0])                         # mix weight (0,1)
        s = 1.0 + torch.tanh(pc[:, 1])                      # scale ~ (0,2), init 1
        bpow = torch.tanh(pc[:, 2]) * 0.3                   # small power bias

        # --- two AM/AM experts, blended by p-gate ---
        gA = self.kanA2(self.kanA1(efeat))                  # (B,2)
        gB = self.kanB2(self.kanB1(efeat))                  # (B,2)
        gmix = (1.0 - m).unsqueeze(-1) * gA + m.unsqueeze(-1) * gB  # (B,2)

        g = torch.complex(gmix[:, 0] + bpow, gmix[:, 1])    # complex gain corr
        am_corr = xc * g                                    # residual gain*centre

        # --- memory-polynomial cross term ---
        idx_m = max(c - 1, 0)
        idx_p = min(c + 1, self.window - 1)
        p2 = torch.stack([
            env[:, idx_m] ** 2,
            env[:, c] ** 2,
            env[:, idx_p] ** 2,
        ], dim=-1)
        mp = self.mp(p2)
        mp_c = torch.complex(mp[:, 0], mp[:, 1]) * xc

        # --- linear FIR identity path (unscaled) ---
        fir_out = self.fir(xs)[:, 0]

        # nonlinear residual is power-scaled; identity path preserved at all p
        residual = s * (am_corr + mp_c)
        out = fir_out + residual

        return (out / self.FEAT_SCALE).to(torch.complex64)