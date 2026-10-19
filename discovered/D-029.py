import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Quantized ultra-low-BOP DPD for the empty low-complexity niches.
       linear complex FIR (bandwidth memory)
       + p-biased KAN static AM/AM & AM/PM (with free mean-envelope thermal feature)
       + short GMP complex memory-NL residual, power-gated.
       All learnable primitives fake-quantized to 4-6 bits to sink BOP."""

    def __init__(self, window: int, taps: int = 9, ntaps: int = 5,
                 kan_grid: int = 8, kan_order: int = 3):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1

        # ---- 1) Linear complex FIR: short bandwidth-limited memory (6-bit) ----
        self.fir = CLinear(self.taps, 1, bias=False, quant_bits=6)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) KAN static AM/AM + AM/PM, biased by p and mean envelope ----
        # inputs: centre |x|, drive power p, mean-envelope thermal descriptor
        # outputs: gain corr, phase corr  (bounded splines -> graceful high-drive)
        self.amam = KANLayer(3, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=6)

        # ---- 3) Short GMP complex memory nonlinear residual (4-bit) ----
        self.nl = CLinear(2 * self.ntaps, 1, bias=False, quant_bits=4)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()
        # power gate: learned quadratic function of p (4-bit)
        self.pgain = RLinear(2, 1, bias=True, quant_bits=4)
        with torch.no_grad():
            self.pgain.weight.zero_(); self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2

        xc = xw[:, c]                                      # centre sample (complex)
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt()  # (B, window)

        # ---- 1) Linear FIR path ----
        lin = self.fir(xw[:, c - h: c + h + 1]).squeeze(-1)

        # ---- free mean-envelope thermal descriptor (no params) ----
        tstate = torch.tanh(env.mean(dim=1))               # (B,)

        # ---- 2) Static AM/AM + AM/PM via KAN, biased by p ----
        ec = env[:, c]
        ein = 1.5 * torch.tanh(ec)
        pin = 1.5 * torch.tanh(2.0 * (p - 0.5))
        tin = 1.5 * tstate
        kin = torch.stack([ein, pin, tin], dim=1)          # (B,3)
        kout = self.amam(kin)                              # (B,2)
        gain = 1.0 + torch.tanh(kout[:, 0])                # AM/AM multiplicative gain
        phase = torch.tanh(kout[:, 1])                     # AM/PM correction

        cph = torch.cos(phase); sph = torch.sin(phase)
        rot = torch.complex(cph, sph)
        amam_out = xc * gain.to(xc.dtype) * rot

        # ---- 3) Short GMP memory nonlinear residual, power-gated ----
        xn = xw[:, c - hn: c + hn + 1]
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = xn * a2 * a2
        feats = torch.cat([t3, t5], dim=1)
        nl = self.nl(feats).squeeze(-1)
        pf = torch.stack([p, p * p], dim=1)
        g = self.pgain(pf).squeeze(-1)
        nl = nl * g.to(nl.dtype)

        return lin + amam_out + nl