import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """KAN-scaled DPD for the 10^5.0-10^5.5 BOP band.
       - Complex FIR mix -> multi-lag envelope basis
       - Smooth 1-D power schedule (KAN on p) modulates gain/phase & NL gate
       - Two-stage KAN static AM/AM + AM/PM
       - Short GMP complex memory-NL residual, power-gated
       All spline inputs squashed into [x_min,x_max] for high-drive stability."""

    def __init__(self, window: int, taps: int = 9, ntaps: int = 5,
                 nmix: int = 6, kan_grid: int = 10, kan_order: int = 3,
                 hidden: int = 8):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1
        self.nmix = nmix

        # ---- 1) Linear complex FIR: short bandwidth-limited memory ----
        self.fir = CLinear(self.taps, 1, bias=False, quant_bits=8)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) Complex mix producing a multi-tap envelope basis ----
        self.mix = CLinear(self.taps, self.nmix, bias=False, quant_bits=8)

        # ---- 3) Smooth 1-D power schedule via a small KAN on p ----
        # outputs modulation channels used across the model
        self.pkan = KANLayer(1, 4, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 4) Two-stage KAN static AM/AM + AM/PM ----
        # inputs: nmix envelope features + mean-env thermal + p schedule (1 ch)
        kin_dim = self.nmix + 1 + 1
        self.kan1 = KANLayer(kin_dim, hidden, grid=kan_grid, order=kan_order,
                            x_min=-1.5, x_max=1.5, quant_bits=8)
        self.kan2 = KANLayer(hidden, 2, grid=kan_grid, order=kan_order,
                            x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 5) Short GMP complex memory nonlinear residual ----
        self.nl = CLinear(2 * self.ntaps, 1, bias=False, quant_bits=6)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2

        xc = xw[:, c]                                        # centre (complex)
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt() # (B, window)

        xt = xw[:, c - h: c + h + 1]                         # (B, taps)

        # ---- 1) Linear FIR path ----
        lin = self.fir(xt).squeeze(-1)

        # ---- 2) multi-tap complex envelope basis ----
        zmix = self.mix(xt)                                  # (B, nmix) complex
        emix = 1.5 * torch.tanh(zmix.abs())                 # (B, nmix) squashed

        # free mean-envelope thermal descriptor
        tstate = 1.5 * torch.tanh(env.mean(dim=1))          # (B,)

        # ---- 3) smooth power schedule (KAN on p) ----
        pin = 1.5 * torch.tanh(2.0 * (p.reshape(B, 1) - 0.5))   # (B,1) in grid
        psched = self.pkan(pin)                             # (B,4)
        # channel 0: bias into KAN, 1: static-gain scale, 2: phase scale, 3: NL gate
        p_feat = 1.5 * torch.tanh(psched[:, 0])             # (B,)
        g_scale = 0.5 + torch.sigmoid(psched[:, 1])         # (B,) in (0.5,1.5)
        ph_scale = torch.tanh(psched[:, 2])                 # (B,)
        nl_gate = torch.sigmoid(psched[:, 3])               # (B,)

        # ---- 4) two-stage KAN static AM/AM + AM/PM ----
        kin = torch.cat([emix,
                         tstate.unsqueeze(1),
                         p_feat.unsqueeze(1)], dim=1)        # (B, nmix+2)
        hmid = self.kan1(kin)                                # (B, hidden)
        hmid = 1.5 * torch.tanh(hmid)                        # squash into grid
        kout = self.kan2(hmid)                               # (B, 2)

        gain = 1.0 + g_scale * torch.tanh(kout[:, 0])        # AM/AM
        phase = ph_scale * torch.tanh(kout[:, 1])            # AM/PM
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        amam_out = xc * gain.to(xc.dtype) * rot

        # ---- 5) short GMP memory nonlinear residual, power-gated ----
        xn = xw[:, c - hn: c + hn + 1]
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = xn * a2 * a2
        feats = torch.cat([t3, t5], dim=1)
        nl = self.nl(feats).squeeze(-1)
        nl = nl * nl_gate.to(nl.dtype)

        y = lin + amam_out + nl
        return y                                             # complex64 (B,)