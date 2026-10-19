import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Lean g003c1-lineage DPD: complex FIR memory + envelope/p-conditioned
       KAN AM/AM-AM/PM + short GMP nonlinear residual with a smooth learned
       1-D power-gain function. All spline inputs squashed into grid support
       so high-drive (3 dBFS) stays stable and seed-tight."""

    def __init__(self, window: int, taps: int = 5, ntaps: int = 5,
                 kan_grid: int = 10, kan_order: int = 3):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1

        # ---- 1) Linear complex FIR: short bandwidth-limited memory ----
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) Static AM/AM & AM/PM via quantized KAN ----
        # inputs: centre env, mem-max env, mem-mean env, p operating point
        # outputs: gain corr, phase corr
        self.amam = KANLayer(4, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 3) Short GMP-style complex memory nonlinear residual ----
        # per tap: t3, t5, t7  -> 3 terms/tap
        self.nl = CLinear(3 * self.ntaps, 1, bias=False, quant_bits=8)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()

        # ---- 4) smooth learned 1-D power gain for the residual (KAN on p) ----
        # keeps p only through bounded features; safe at p==0 and p==1
        self.pkan = KANLayer(1, 1, grid=6, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2
        B = xw.shape[0]

        xc = xw[:, c]                                        # centre sample (complex)
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt() # (B, window)

        # ---- 1) Linear FIR path ----
        lin = self.fir(xw[:, c - h : c + h + 1]).squeeze(-1)

        # ---- 2) Static AM/AM correction via KAN, envelope + p conditioned ----
        ec = env[:, c]                                       # centre envelope
        emax = env.max(dim=1).values                         # mem peak (free)
        emean = env.mean(dim=1)                              # mem mean (free)

        ein = 1.5 * torch.tanh(ec)
        emx = 1.5 * torch.tanh(emax)
        emn = 1.5 * torch.tanh(emean)
        pin = 1.5 * torch.tanh(2.0 * (p - 0.5))              # operating point
        kin = torch.stack([ein, emx, emn, pin], dim=1)       # (B,4)
        kout = self.amam(kin)                                # (B,2)

        gain = 1.0 + torch.tanh(kout[:, 0])                  # AM/AM gain
        phase = 0.5 * torch.tanh(kout[:, 1])                 # small AM/PM
        cph = torch.cos(phase); sph = torch.sin(phase)
        rot = torch.complex(cph, sph)
        amam_out = xc * gain.to(xc.dtype) * rot

        # ---- 3) Short GMP memory nonlinear residual ----
        xn = xw[:, c - hn : c + hn + 1]
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = xn * a2 * a2
        t7 = xn * a2 * a2 * a2
        feats = torch.cat([t3, t5, t7], dim=1)
        nl = self.nl(feats).squeeze(-1)

        # ---- 4) smooth power modulation of residual (bounded, safe at 0/1) ----
        pin1 = (1.5 * torch.tanh(2.0 * (p - 0.5))).reshape(B, 1)  # (B,1) in grid
        g = 1.0 + torch.tanh(self.pkan(pin1).squeeze(-1))         # (B,) real, ~[0,2]
        nl = nl * g.to(nl.dtype)

        y = lin + amam_out + nl                              # complex (B,)
        return y