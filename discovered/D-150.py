import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """g005c2/g030c0-lineage envelope-KAN DPD, upgraded for high-drive.

    - Decimated complex FIR -> centre-sample linear estimate.
    - Richer envelope basis (env, env^2, env^3) hard-squashed into the
      spline grid so nothing escapes at 3 dBFS.
    - Short complex-FIR memory term whose contribution the KAN modulates
      (drive-dependent memory effect -> lifts high-drive ACLR).
    - Smooth 1-D power conditioning (p -> drive scale + gate) keeps seeds
      tight and collapses to clean linear passthrough at low drive.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2, hidden: int = 10,
                 grid: int = 8, order: int = 3, pgrid: int = 8,
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

        # short memory window around centre for the memory-modulated term
        m_lo = max(0, c - 2)
        m_hi = min(window, c + 3)
        midx = list(range(m_lo, m_hi))
        self.register_buffer("mem_idx", torch.tensor(midx, dtype=torch.long))
        self.n_mem = len(midx)

        # complex FIR over decimated taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)
        # complex FIR for the memory term (short) -> 1 complex output
        self.memfir = CLinear(self.n_mem, 1, bias=False, quant_bits=quant_bits)

        # envelope nonlinearity: (env_d, env2, env3) -> hidden -> (gr,gi,mr,mi)
        self.kan1 = KANLayer(3, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 4, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # smooth 1-D power conditioning: p -> (drive_scale, gate)
        self.kanp = KANLayer(1, 2, pgrid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            self.memfir.wr.zero_()
            self.memfir.wi.zero_()
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.kanp.coef.mul_(0.1)
            self.kanp.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                       # (B, window) complex

        taps = x[:, self.tap_idx]                      # (B, n_taps)
        lin = self.fir(taps)[:, 0]                     # (B,) complex

        mem = self.memfir(x[:, self.mem_idx])[:, 0]    # (B,) complex

        # ---- smooth power conditioning (safe at p=0 and p=1) ----
        pc = p.reshape(B, 1)                           # (B,1) in [0,1]
        pin = torch.tanh(2.0 * pc - 1.0)               # monotone -> grid
        pout = self.kanp(pin)                          # (B,2)
        drive = 1.0 + 0.5 * torch.tanh(pout[:, 0])     # ~[0.5,1.5]
        gate = torch.sigmoid(pout[:, 1])               # (B,) in (0,1)

        # ---- envelope basis, drive-modulated, hard-squashed into grid ----
        xc = x[:, self.c]                              # centre complex sample
        env = torch.abs(xc)                            # (B,)
        env_d = torch.tanh(env * 2.0 * drive)          # drive-modulated env
        env2 = torch.tanh(env * env * 3.0)             # squared basis
        env3 = torch.tanh(env * env * env * 4.0)       # cubic basis
        feat = torch.stack([env_d, env2, env3], dim=-1)  # (B,3) in [-1,1]

        g = self.kan2(self.kan1(feat))                 # (B,4)
        gr = g[:, 0] * gate
        gi = g[:, 1] * gate
        mr = g[:, 2] * gate
        mi = g[:, 3] * gate

        gain = torch.complex(1.0 + gr, gi)             # multiplicative, ~1
        mmod = torch.complex(mr, mi)                   # memory modulation

        o = (lin * gain + mem * mmod) / self.FEAT_SCALE
        return o.to(torch.complex64)                   # (B,) complex64