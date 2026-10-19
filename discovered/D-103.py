import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Decimated complex-FIR backbone + envelope KAN correction, with an
    explicit smooth 1-D power-conditioning path (KAN on p) that modulates
    both the envelope drive and the nonlinear-correction gate.

    Power conditioning:
      - kanp maps p -> (drive_scale, gate_bias) via a smooth learned 1-D
        function. drive_scale rescales the envelope handed to the nonlinear
        KAN (mimics compression moving with drive); gate scales the
        additive complex correction so the model reduces to a clean linear
        passthrough where correction is not needed. This is robust at all
        drives and tight across seeds because it decouples 'how much
        correction' (p-curve) from 'what shape' (envelope KAN).
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2, hidden: int = 8,
                 grid: int = 7, order: int = 3, pgrid: int = 8,
                 quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        # complex FIR over decimated taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # envelope-only nonlinearity: (env_drive, env_raw) -> hidden -> (gr, gi)
        self.kan1 = KANLayer(2, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # smooth 1-D power-conditioning: p -> (drive_scale, gate)
        self.kanp = KANLayer(1, 2, pgrid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        with torch.no_grad():
            # init FIR near a centre passthrough
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            # power path starts near neutral
            self.kanp.coef.mul_(0.1)
            self.kanp.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                      # (B, window)
        taps = x[:, self.tap_idx]                     # (B, n_taps)
        lin = self.fir(taps)[:, 0]                    # (B,) complex

        # ---- smooth power-conditioning path (safe at p=0 and p=1) ----
        pc = p.reshape(B, 1)                          # (B,1) in [0,1]
        pin = torch.tanh(2.0 * pc - 1.0)              # map to grid, monotone
        pout = self.kanp(pin)                         # (B,2)
        drive = 1.0 + 0.5 * torch.tanh(pout[:, 0])    # ~[0.5,1.5] scaling
        gate = torch.sigmoid(pout[:, 1])              # (B,) in (0,1)

        # ---- envelope nonlinearity, drive-modulated by power ----
        xc = x[:, self.c]                             # centre sample (complex)
        env = torch.abs(xc)                           # (B,)
        env_d = torch.tanh(env * 2.0 * drive)         # drive-modulated env
        env_r = torch.tanh(env * 2.0)                 # raw squashed env
        feat = torch.stack([env_d, env_r], dim=-1)    # (B,2)

        g = self.kan2(self.kan1(feat))                # (B,2) additive delta
        gr = g[:, 0] * gate
        gi = g[:, 1] * gate
        gain = torch.complex(1.0 + gr, gi)            # gain near 1

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)