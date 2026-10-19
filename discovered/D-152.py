import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """6-9k RMpS DPD from the clean g005c2/g003c1 KAN lineage.
       Complex FIR memory + wide envelope/p-conditioned KAN AM/AM in the
       complex plane + deep GMP complex nonlinear residual with smooth
       power modulation. All spline inputs squashed into grid support to
       keep high-drive (3 dBFS) ACLR high and seeds tight."""

    def __init__(self, window: int,
                 taps: int = 9, ntaps: int = 7,
                 mix_d: int = 12, kan_h: int = 20,
                 kan_grid: int = 10, kan_order: int = 3):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1
        self.mix_d = mix_d
        self.kan_h = kan_h

        # ---- 1) Linear complex FIR (bandwidth-limited memory) ----
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) Complex tap mix -> real envelope descriptors for KAN ----
        self.mix = CLinear(self.taps, mix_d, bias=True)

        # feature dim into KAN: mix envelopes + centre env + mem env + p
        self.kin_d = mix_d + 3
        # first (wide) KAN layer
        self.kan1 = KANLayer(self.kin_d, kan_h, grid=kan_grid,
                             order=kan_order, x_min=-1.5, x_max=1.5)
        # second KAN layer -> gain + phase corrections (2 outputs)
        self.kan2 = KANLayer(kan_h, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5)

        # ---- 3) Deep GMP-style complex nonlinear residual ----
        # orders 3,5,7 across ntaps window = 3*ntaps complex features
        self.nl = CLinear(3 * self.ntaps, 1, bias=False, quant_bits=8)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()

        # smooth 1-D power modulation of the residual (safe at p=0/1)
        self.pgain = RLinear(3, 1, bias=True)
        with torch.no_grad():
            self.pgain.weight.zero_(); self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2
        B = xw.shape[0]

        xc = xw[:, c]                                            # centre (complex)
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt()     # (B, window)

        # ---- 1) Linear FIR path ----
        xt = xw[:, c - h : c + h + 1]                            # (B, taps)
        lin = self.fir(xt).squeeze(-1)

        # ---- 2) Envelope/p conditioned KAN AM/AM in complex plane ----
        z = self.mix(xt)                                         # complex (B, mix_d)
        zenv = z.abs()                                           # (B, mix_d) free

        ec = env[:, c]                                           # centre envelope
        emean = env.mean(dim=1)                                  # mem-avg envelope

        # squash all spline inputs into [x_min, x_max]
        zin = 1.5 * torch.tanh(zenv)
        ein = 1.5 * torch.tanh(ec).reshape(B, 1)
        emin = 1.5 * torch.tanh(emean).reshape(B, 1)
        pin = 1.5 * torch.tanh(2.0 * (p - 0.5)).reshape(B, 1)    # op point, safe

        kin = torch.cat([zin, ein, emin, pin], dim=1)           # (B, kin_d)
        hk = self.kan1(kin)                                     # (B, kan_h)
        hk = 1.5 * torch.tanh(hk / 1.5)                         # keep in grid
        kout = self.kan2(hk)                                    # (B, 2)

        gain = 1.0 + torch.tanh(kout[:, 0])                     # AM/AM
        phase = 0.6 * torch.tanh(kout[:, 1])                   # AM/PM
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        amam_out = xc * gain.to(xc.dtype) * rot

        # ---- 3) Deep GMP complex nonlinear residual ----
        xn = xw[:, c - hn : c + hn + 1]                         # (B, ntaps)
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = t3 * a2
        t7 = t5 * a2
        feats = torch.cat([t3, t5, t7], dim=1)                 # (B, 3*ntaps)
        nl = self.nl(feats).squeeze(-1)

        # smooth power modulation (no 1/p, log, or sqrt of negatives)
        pf = torch.stack([p, p * p, torch.tanh(2.0 * p)], dim=1)
        g = self.pgain(pf).squeeze(-1)
        nl = nl * g.to(nl.dtype)

        y = lin + amam_out + nl                                # complex (B,)
        return y.to(torch.complex64)