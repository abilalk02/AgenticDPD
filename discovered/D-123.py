import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Decimated complex FIR + bounded envelope-KAN AM/AM corrector with a
    smooth learned 1-D power gate (g005c2/g003c1-style lineage that clears
    the 3 dBFS ceiling natively).

    - FIR: cheap complex linear memory over decimated taps (identity init).
    - Corrector KAN: maps a rich, fully-bounded envelope feature vector to a
      complex gain that multiplies the FIR estimate -> bends AM/AM & AM/PM.
    - Power gate: a tiny KAN of p alone produces a smooth scalar that
      modulates the correction strength across the drive sweep (robust at
      p==0 and p==1, no divisions/logs).
    All spline inputs are squashed into the grid so high drive stays stable.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2, hidden: int = 7,
                 grid: int = 8, order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2

        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        # short symmetric neighbourhood for a memory-envelope feature (free)
        lo = max(0, c - 2)
        hi = min(window, c + 3)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # complex FIR over decimated taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # feature layout: [env, env^2, env^3, mem_env, p_c, env*p]  (6-D, free)
        self.kan1 = KANLayer(6, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # smooth learned 1-D power gate: p -> scalar strength (variance-gate)
        self.pgate = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        with torch.no_grad():
            # identity FIR: pass centre sample through
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # start correction near zero (clean linear pass-through)
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            # power gate starts small-positive-neutral
            self.pgate.coef.mul_(0.1)
            self.pgate.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                       # (B, window)
        taps = x[:, self.tap_idx]                      # (B, n_taps)
        lin = self.fir(taps)[:, 0]                     # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                            # (B,)

        # local memory envelope (mean |x| over short neighbourhood), free
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        # bounded squashes into spline support [-1, 1]
        e  = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m  = torch.tanh(mem_env * 2.0)
        pc = p.reshape(B) * 2.0 - 1.0                  # centre p to [-1,1]
        pc = torch.clamp(pc, -1.0, 1.0)               # safe at p==0 and p==1
        ep = torch.tanh(env * p.reshape(B) * 2.5)      # env*p cross term

        feat = torch.stack([e, e2, e3, m, pc, ep], dim=-1)  # (B, 6)

        g = self.kan2(self.kan1(feat))                 # (B, 2)

        # smooth 1-D power gate on correction strength
        gate = torch.sigmoid(self.pgate(pc.reshape(B, 1)))[:, 0]  # (B,) in (0,1)

        gr = 1.0 + gate * g[:, 0]
        gi = gate * g[:, 1]
        gain = torch.complex(gr, gi)                   # (B,) complex

        o = (lin * gain) / self.FEAT_SCALE
        return o.reshape(B).to(torch.complex64)