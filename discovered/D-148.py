import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """g005c2/g030c0 lineage: decimated complex-FIR backbone + richer
    envelope KAN correction + smooth 1-D power conditioning + a complex
    GMPBranch memory-polynomial skip.

    Improvement over parent:
      - KAN nonlinearity now sees a richer squashed envelope-power basis
        (|x|, |x|^2, |x|^3) plus a drive-modulated envelope, giving the
        splines more physical handles on the compression curve without
        adding dense width.
      - A lightweight complex GMPBranch skip adds odd-order memory
        polynomial correction (4 RMpS/term) that lifts high-drive ACLR.
      - Smooth 1-D power-conditioning KAN(p) modulates both envelope drive
        and the correction gate for seed-tight robustness across the sweep.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2, hidden: int = 10,
                 grid: int = 8, order: int = 3, pgrid: int = 8,
                 gmp_orders=(1, 3, 5), gmp_mem: int = 3, gmp_cross: int = 1,
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

        # richer envelope-only nonlinearity:
        #   (env_drive, |x|, |x|^2, |x|^3) -> hidden -> (gr, gi)
        self.kan1 = KANLayer(4, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # smooth 1-D power-conditioning: p -> (drive_scale, gate)
        self.kanp = KANLayer(1, 2, pgrid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # complex GMP memory-polynomial skip (odd-order aware)
        self.gmp = GMPBranch(window, orders=gmp_orders, mem=gmp_mem,
                             cross=gmp_cross)

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
        x = xw * self.FEAT_SCALE                      # (B, window) complex
        taps = x[:, self.tap_idx]                     # (B, n_taps)
        lin = self.fir(taps)[:, 0]                    # (B,) complex

        # ---- smooth power-conditioning path (safe at p=0 and p=1) ----
        pc = p.reshape(B, 1)                          # (B,1) in [0,1]
        pin = torch.tanh(2.0 * pc - 1.0)              # map to grid, monotone
        pout = self.kanp(pin)                         # (B,2)
        drive = 1.0 + 0.5 * torch.tanh(pout[:, 0])    # ~[0.5,1.5] scaling
        gate = torch.sigmoid(pout[:, 1])              # (B,)

        # ---- richer envelope nonlinearity, drive-modulated by power ----
        xc = x[:, self.c]                             # centre sample (complex)
        env = torch.abs(xc)                           # (B,)
        env_d = torch.tanh(env * 2.0 * drive)         # drive-modulated env
        e1 = torch.tanh(env * 2.0)                    # |x| squashed
        e2 = torch.tanh(env * env * 4.0)              # |x|^2 squashed
        e3 = torch.tanh(env * env * env * 8.0)        # |x|^3 squashed
        feat = torch.stack([env_d, e1, e2, e3], dim=-1)   # (B,4)

        g = self.kan2(self.kan1(feat))                # (B,2) additive delta
        gr = g[:, 0] * gate
        gi = g[:, 1] * gate
        gain = torch.complex(1.0 + gr, gi)            # gain near 1

        # ---- complex GMP memory-polynomial skip, gated by power ----
        skip = self.gmp(x)                            # complex (B,) or (B,1)
        if skip.dim() > 1:
            skip = skip.reshape(B)
        skip = skip * gate.to(skip.dtype)

        o = (lin * gain + skip) / self.FEAT_SCALE
        return o.to(torch.complex64)