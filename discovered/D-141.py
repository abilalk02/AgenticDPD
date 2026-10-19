import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Ultra-lean KAN DPD (g026c0 lineage) with a learned power-gate KAN.
       Complex FIR memory + quantized AM/AM-KAN (envelope+p) + short GMP
       residual, with all three correction strengths modulated by a smooth
       1-D KAN function of drive p. Spline inputs squashed into grid."""

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
        # inputs: centre env, mean env, p (all squashed) -> gain, phase
        self.amam = KANLayer(3, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 3) Short GMP-style complex memory nonlinear residual ----
        self.nl = CLinear(2 * self.ntaps, 1, bias=False, quant_bits=8)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()

        # ---- 4) Power-gate KAN: smooth 1-D f(p) -> 3 modulation gains ----
        # robust across sweep; safe at p==0 and p==1 (tanh-squashed input).
        self.pgate = KANLayer(1, 3, grid=6, order=3,
                              x_min=-1.5, x_max=1.5, quant_bits=8)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2
        B = xw.shape[0]

        xc = xw[:, c]                                        # centre (complex)
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt() # (B, window)

        # ---- power-gate: learned smooth function of drive p ----
        pin = 1.5 * torch.tanh(2.0 * (p.reshape(B, 1) - 0.5))  # (B,1) in grid
        gates = self.pgate(pin)                              # (B,3) real
        g_am = 1.0 + 0.5 * torch.tanh(gates[:, 0])          # AM/AM strength
        g_ph = 0.5 + 0.5 * torch.sigmoid(gates[:, 1])       # AM/PM strength
        g_nl = torch.sigmoid(gates[:, 2])                   # residual strength

        # ---- 1) Linear FIR path ----
        lin = self.fir(xw[:, c - h : c + h + 1]).squeeze(-1)

        # ---- 2) Static AM/AM correction via KAN, p-biased ----
        ec = env[:, c]                                       # centre envelope
        emean = env.mean(dim=1)                              # free descriptor

        ein = 1.5 * torch.tanh(ec)
        emin = 1.5 * torch.tanh(emean)
        pop = pin.squeeze(1)                                 # operating point
        kin = torch.stack([ein, emin, pop], dim=1)          # (B,3)
        kout = self.amam(kin)                                # (B,2)

        gain = 1.0 + g_am * torch.tanh(kout[:, 0])          # p-modulated gain
        phase = g_ph * 0.5 * torch.tanh(kout[:, 1])         # p-modulated AM/PM
        cph = torch.cos(phase); sph = torch.sin(phase)
        rot = torch.complex(cph, sph)
        amam_out = xc * gain.to(xc.dtype) * rot

        # ---- 3) Short GMP memory nonlinear residual, p-modulated ----
        xn = xw[:, c - hn : c + hn + 1]
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = xn * a2 * a2
        feats = torch.cat([t3, t5], dim=1)
        nl = self.nl(feats).squeeze(-1)
        nl = nl * g_nl.to(nl.dtype)

        y = lin + amam_out + nl                             # complex (B,)
        return y.to(torch.complex64)