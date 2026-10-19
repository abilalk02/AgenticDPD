import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear, GMPBranch


class DPD(DPDModule):
    """VCSEL IM/DD DPD: KAN AM/AM + complex FIR + GMPBranch memory skip.

    Hybridization: the parent's neural/KAN static corrector and complex FIR
    are augmented with a GMPBranch (generalized memory polynomial) skip path
    — a physics-matched basis for relaxation-oscillation / TIA memory
    nonlinearity. A KAN-of-power gates the GMP contribution so a single model
    linearizes the whole drive-power sweep.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8, grid: int = 7,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2

        # --- static AM/AM branch ---
        self.kan1 = KANLayer(3, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # --- short-memory complex FIR ---
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # --- memory-polynomial cross term ---
        self.mp = RLinear(3, 2, bias=True, quant_bits=quant_bits)

        # --- NEW: GMPBranch skip path (memory nonlinearity) ---
        mem = min(5, window)
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=mem, cross=2)

        # --- power gate for the GMP path (learned 1-D fn of p) ---
        self.pgate = KANLayer(1, 2, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0 / self.FEAT_SCALE
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.mp.weight.mul_(0.05)
            self.mp.bias.zero_()
            # gate near unity around p (bias-ish): start small
            for l in (self.pgate,):
                l.coef.mul_(0.05)
                l.wb.mul_(0.05)

    def forward(self, xw, p):
        c = self.window // 2

        xs = xw * self.FEAT_SCALE
        xc = xs[:, c]

        # envelope features
        r = torch.abs(xc)
        env = torch.abs(xs)
        therm = env.mean(dim=-1)

        rf = torch.tanh(r * 2.0)
        tf = torch.tanh(therm * 2.0)
        pf = (p * 2.0 - 1.0)
        feat = torch.stack([rf, pf, tf], dim=-1)

        gain = self.kan2(self.kan1(feat))
        g = torch.complex(1.0 + gain[:, 0], gain[:, 1])
        am_out = xc * g

        # complex FIR (identity + short memory)
        fir_out = self.fir(xs)[:, 0]

        # memory-polynomial cross term
        idx_m = max(c - 1, 0)
        idx_p = min(c + 1, self.window - 1)
        p2 = torch.stack([
            env[:, idx_m] ** 2,
            env[:, c] ** 2,
            env[:, idx_p] ** 2,
        ], dim=-1)
        mp = self.mp(p2)
        mp_c = torch.complex(mp[:, 0], mp[:, 1]) * xc

        # --- GMPBranch skip (on scaled window) ---
        gmp_out = self.gmp(xs)
        if gmp_out.dim() > 1:
            gmp_out = gmp_out[:, 0] if gmp_out.shape[-1] == 1 else gmp_out.sum(dim=-1)

        # power gate (learned 1-D complex scalar around identity)
        pfeat = pf.unsqueeze(-1)
        gt = self.pgate(pfeat)
        gate = torch.complex(1.0 + gt[:, 0], gt[:, 1])

        out = fir_out + (am_out - xc) + mp_c + gate * gmp_out

        return (out / self.FEAT_SCALE).to(torch.complex64)