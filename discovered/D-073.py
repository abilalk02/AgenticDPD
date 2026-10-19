import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """KAN-scaled DPD for the empty high-BOP niche (10^7.0-10^7.5).
       Wide complex FIR mix -> envelope basis + smooth p-function ->
       large power-modulated KAN AM/AM engine -> complex gain head.
       Spline inputs are hard-squashed; p enters only as bounded gains."""

    def __init__(self, window: int,
                 basis: int = 16, kan_out: int = 16,
                 kan_grid: int = 12, kan_order: int = 3):
        super().__init__(window)
        self.window = window
        c = window // 2

        # ---- 1) Linear complex FIR over full window (bandwidth memory) ----
        self.fir = CLinear(window, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0

        # ---- 2) Complex mix -> envelope basis for the KAN ----
        self.mix = CLinear(window, basis, bias=True)

        # smooth learned 1-D function of p (bounded, robust at 0 and 1)
        self.pfun = KANLayer(1, 4, grid=8, order=3, x_min=-1.5, x_max=1.5)

        # ---- 3) Large power-modulated KAN AM/AM engine ----
        # inputs: [envelope basis (basis), p-features(4)] squashed to grid
        self.amam = KANLayer(basis + 4, kan_out, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 4) Complex gain/phase head applied to mixed complex path ----
        # split KAN output into gain and phase halves
        self.kg = kan_out // 2
        self.gain_head = CLinear(basis, 1, bias=False)  # combines modulated taps
        with torch.no_grad():
            self.gain_head.wr.zero_(); self.gain_head.wi.zero_()
            self.gain_head.wr[0, 0] = 1.0

        # ---- 5) Short complex GMP-style nonlinear residual ----
        ntaps = min(7, window) | 1
        self.ntaps = ntaps
        self.nl = CLinear(2 * ntaps, 1, bias=False)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()
        self.pgain = RLinear(3, 1, bias=True)
        with torch.no_grad():
            self.pgain.weight.zero_(); self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2

        # ---- 1) linear FIR memory ----
        lin = self.fir(xw).squeeze(-1)                      # complex (B,)

        # ---- 2) complex mix -> envelope basis ----
        z = self.mix(xw)                                    # complex (B, basis)
        env = z.abs()                                       # (B, basis) real, free
        env = 1.5 * torch.tanh(env / 1.5)                   # squash to grid

        # smooth p features (bounded, safe at 0 and 1)
        pc = 1.5 * torch.tanh(2.0 * (p.reshape(B, 1) - 0.5))
        pf = self.pfun(pc)                                  # (B,4)
        pf = 1.5 * torch.tanh(pf / 1.5)

        feat = torch.cat([env, pf], dim=1)                  # (B, basis+4)
        k = self.amam(feat)                                 # (B, kan_out)

        # split into gain and phase modulation
        gpart = k[:, :self.kg]
        ppart = k[:, self.kg:]
        gain = 1.0 + torch.tanh(gpart)                      # (B, kg) mult gains
        phase = torch.tanh(ppart)                           # (B, kg) small rotations

        # pad/trim gain,phase to basis width and apply to complex mix z
        kg = self.kg
        basis = z.shape[1]
        if kg < basis:
            reps = (basis + kg - 1) // kg
            gain = gain.repeat(1, reps)[:, :basis]
            phase = phase.repeat(1, reps)[:, :basis]
        else:
            gain = gain[:, :basis]
            phase = phase[:, :basis]

        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        zmod = z * gain.to(z.dtype) * rot                   # complex (B, basis)

        amam_out = self.gain_head(zmod).squeeze(-1)         # complex (B,)

        # ---- 5) short GMP nonlinear residual ----
        hn = self.ntaps // 2
        xn = xw[:, c - hn: c + hn + 1]
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = xn * a2 * a2
        feats = torch.cat([t3, t5], dim=1)
        nl = self.nl(feats).squeeze(-1)
        pv = p.reshape(B, 1)
        pfeat = torch.cat([pv, pv * pv, torch.tanh(pv)], dim=1)
        g = self.pgain(pfeat).squeeze(-1)
        nl = nl * g.to(nl.dtype)

        y = lin + amam_out + nl
        return y                                            # complex64 (B,)