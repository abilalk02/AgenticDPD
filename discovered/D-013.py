import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer, RConv1d


class DPD(DPDModule):
    """VCSEL IM/DD-inspired DPD:
       linear FIR (bandwidth memory) + KAN static AM/AM (L-I saturation, p-biased)
       + slow thermal envelope modulation + short GMP memory-NL residual."""

    def __init__(self, window: int, taps: int = 9, ntaps: int = 5,
                 kan_grid: int = 8, kan_order: int = 3):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1

        c = window // 2
        # ---- 1) Linear complex FIR: short bandwidth-limited memory ----
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) Static compressive AM/AM via KAN (splines) ----
        # inputs: squashed centre envelope |x|, and drive power p (operating point)
        # outputs: 2 -> (real gain corr, imag/phase corr) applied to centre sample
        self.amam = KANLayer(2, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5)

        # ---- 3) Slow thermal envelope: decimated LP conv over window envelope ----
        # captures slow average-power (self-heating) dynamics
        self.therm = RConv1d(1, 1, kernel_size=window, bias=True)
        with torch.no_grad():
            self.therm.weight.fill_(1.0 / window)  # start as mean (LP)
            self.therm.bias.zero_()
        # map thermal state + p -> scalar correction gain
        self.tgain = RLinear(2, 1, bias=True)
        with torch.no_grad():
            self.tgain.weight.zero_(); self.tgain.bias.fill_(1.0)

        # ---- 4) Short GMP-style complex memory nonlinear residual ----
        self.nl = CLinear(2 * self.ntaps, 1, bias=False)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()
        self.pgain = RLinear(2, 1, bias=True)
        with torch.no_grad():
            self.pgain.weight.zero_(); self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2
        B = xw.shape[0]

        xc = xw[:, c]                                       # centre sample (complex)
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt()  # (B, window) envelope

        # ---- 1) Linear FIR path ----
        lin = self.fir(xw[:, c - h : c + h + 1]).squeeze(-1)

        # ---- 3) Slow thermal state (mean-level, slow dynamics) ----
        tstate = self.therm(env.unsqueeze(1)).squeeze(1).squeeze(-1)  # (B,)
        tstate = torch.tanh(tstate)                        # bounded thermal descriptor

        # ---- 2) Static AM/AM correction via KAN, biased by p ----
        ec = env[:, c]                                     # centre envelope
        # squash into spline support [-1.5,1.5]
        ein = 1.5 * torch.tanh(ec)
        pin = 1.5 * torch.tanh(2.0 * (p - 0.5))            # centre p operating point
        kin = torch.stack([ein, pin], dim=1)               # (B,2)
        kout = self.amam(kin)                              # (B,2): gain, phase corr
        gain = 1.0 + torch.tanh(kout[:, 0])                # multiplicative AM/AM gain
        phase = torch.tanh(kout[:, 1])                     # small AM/PM correction

        # combine gain + phase as complex multiplier on centre sample
        cph = torch.cos(phase); sph = torch.sin(phase)
        rot = torch.complex(cph, sph)
        amam_out = xc * gain.to(xc.dtype) * rot

        # thermal modulation of the static correction (slow envelope influence)
        tfeat = torch.stack([tstate, p], dim=1)            # (B,2)
        tg = self.tgain(tfeat).squeeze(-1)                 # (B,) real
        amam_out = amam_out * tg.to(amam_out.dtype)

        # ---- 4) Short GMP memory nonlinear residual ----
        xn = xw[:, c - hn : c + hn + 1]
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = xn * a2 * a2
        feats = torch.cat([t3, t5], dim=1)
        nl = self.nl(feats).squeeze(-1)
        pf = torch.stack([p, p * p], dim=1)
        g = self.pgain(pf).squeeze(-1)
        nl = nl * g.to(nl.dtype)

        # AM/AM replaces the linear centre contribution as a corrected version;
        # combine: linear memory (off-centre taps handled by FIR) + static + residual
        return lin + amam_out + nl