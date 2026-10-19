import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """KAN-scaled DPD for the empty 10^6.5-10^7.0 BOP niche.
       - complex FIR memory (bandwidth)
       - wide complex tap-mixer -> dual wide KAN AM/AM + AM/PM, p-biased
       - per-branch KAN-gated GMP complex nonlinear memory residual
       All spline inputs hard-squashed into grid support for high-drive stability.
    """

    def __init__(self, window: int, taps: int = 11, ntaps: int = 7,
                 nmix: int = 12, hidden: int = 16,
                 kan_grid: int = 10, kan_order: int = 3, qbits: int = 8):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1
        self.nmix = nmix
        self.hidden = hidden

        # ---- 1) Linear complex FIR: bandwidth-limited memory ----
        self.fir = CLinear(self.taps, 1, bias=False, quant_bits=qbits)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) complex tap-mixer producing envelope basis ----
        self.mix = CLinear(window, nmix, bias=True, quant_bits=qbits)

        # KAN stage 1: [nmix envelope feats, p-bias, mean-env thermal] -> hidden
        self.kan1 = KANLayer(nmix + 2, hidden, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=qbits)
        # KAN stage 2: hidden -> (gain, phase)
        self.kan2 = KANLayer(hidden, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=qbits)

        # ---- 3) GMP complex nonlinear memory residual ----
        # complex taps t3,t5 basis, mixed by CLinear, gated by KAN(p, env)
        self.nl = CLinear(2 * self.ntaps, hidden, bias=False, quant_bits=qbits)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()
        self.nlhead = CLinear(hidden, 1, bias=False, quant_bits=qbits)
        with torch.no_grad():
            self.nlhead.wr.zero_(); self.nlhead.wi.zero_()

        # KAN gate on residual, function of (p, p^2, mean-env)
        self.gatekan = KANLayer(3, hidden, grid=kan_grid, order=kan_order,
                                x_min=-1.5, x_max=1.5, quant_bits=qbits)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2

        xc = xw[:, c]                                      # complex (B,)
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt()  # (B, window)

        # ---- 1) Linear FIR path ----
        lin = self.fir(xw[:, c - h: c + h + 1]).squeeze(-1)

        # ---- free thermal descriptor ----
        tstate = torch.tanh(env.mean(dim=1))               # (B,)

        # p-bias feature (safe at 0 and 1)
        pin = 1.5 * torch.tanh(2.0 * (p - 0.5))            # (B,)

        # ---- 2) wide complex mixer -> envelope basis -> dual KAN ----
        z = self.mix(xw)                                   # complex (B, nmix)
        zenv = z.abs()                                     # (B, nmix)
        ein = 1.5 * torch.tanh(zenv)                       # squash into grid
        feat1 = torch.cat([ein, pin.reshape(B, 1),
                           (1.5 * tstate).reshape(B, 1)], dim=1)  # (B,nmix+2)
        hsig = self.kan1(feat1)                            # (B, hidden)
        hsig = 1.5 * torch.tanh(hsig)                      # keep in grid for kan2
        kout = self.kan2(hsig)                             # (B,2)

        gain = 1.0 + torch.tanh(kout[:, 0])                # AM/AM
        phase = torch.tanh(kout[:, 1])                     # AM/PM
        cph = torch.cos(phase); sph = torch.sin(phase)
        rot = torch.complex(cph, sph)
        amam_out = xc * gain.to(xc.dtype) * rot

        # ---- 3) GMP complex nonlinear memory residual ----
        xn = xw[:, c - hn: c + hn + 1]                     # complex (B,ntaps)
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = xn * a2 * a2
        feats = torch.cat([t3, t5], dim=1)                 # complex (B,2*ntaps)
        nlh = self.nl(feats)                               # complex (B,hidden)

        # KAN gate from power + thermal
        pquad = 1.5 * torch.tanh(2.0 * (p * p - 0.5))
        gfeat = torch.stack([pin, pquad, 1.5 * tstate], dim=1)  # (B,3)
        gate = torch.sigmoid(self.gatekan(gfeat))          # (B,hidden), free op
        nlg = nlh * gate.to(nlh.dtype)
        nl = self.nlhead(nlg).squeeze(-1)                  # complex (B,)

        y = lin + amam_out + nl
        return y