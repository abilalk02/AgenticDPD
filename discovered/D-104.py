import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Ultra-lean KAN DPD for the 10^4.0-10^4.5 BOP niche (~180 RMpS).
       Complex FIR memory + quantized KAN static AM/AM (envelope+p) with
       power-modulated gain + short GMP nonlinear residual.
       KAN inputs squashed into grid support; 8-bit KAN cuts BOP ~4x."""

    def __init__(self, window: int, taps: int = 5, ntaps: int = 3,
                 kan_grid: int = 8, kan_order: int = 3):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1

        # ---- 1) Linear complex FIR: short bandwidth-limited memory ----
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) Static compressive AM/AM via quantized KAN ----
        # inputs: squashed centre env, a wider-mem env descriptor, and p
        # outputs: gain corr, phase corr  (2 outputs)
        self.amam = KANLayer(3, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 3) Short GMP-style complex memory nonlinear residual ----
        self.nl = CLinear(2 * self.ntaps, 1, bias=False, quant_bits=8)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()

        # power-modulated residual gain: smooth 1-D function of p (safe at 0/1)
        self.pgain = RLinear(3, 1, bias=True)
        with torch.no_grad():
            self.pgain.weight.zero_(); self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2
        B = xw.shape[0]

        xc = xw[:, c]                                        # centre sample (complex)
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt() # (B, window)

        # ---- 1) Linear FIR path ----
        lin = self.fir(xw[:, c - h : c + h + 1]).squeeze(-1)

        # ---- 2) Static AM/AM correction via KAN, p-biased ----
        ec = env[:, c]                                       # centre envelope
        # wider memory-averaged envelope descriptor (free)
        emean = env.mean(dim=1)

        ein = 1.5 * torch.tanh(ec)                           # squash to grid
        emin = 1.5 * torch.tanh(emean)
        pin = 1.5 * torch.tanh(2.0 * (p - 0.5))              # p operating point
        kin = torch.stack([ein, emin, pin], dim=1)           # (B,3)
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
        feats = torch.cat([t3, t5], dim=1)
        nl = self.nl(feats).squeeze(-1)

        # smooth power modulation (no 1/p, safe at p==0 and p==1)
        pf = torch.stack([p, p * p, torch.tanh(2.0 * p)], dim=1)  # (B,3)
        g = self.pgain(pf).squeeze(-1)                       # (B,) real
        nl = nl * g.to(nl.dtype)

        y = lin + amam_out + nl                              # complex (B,)
        return y