import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Mid-BOP seed-robust scaled-KAN DPD (niche 10^5.5-10^6.0 BOP).

    - 8-bit complex FIR anchor (passthrough init)
    - complex tap-mix -> envelope feature basis
    - smooth learned 1-D power warp (KAN on bounded p) modulating features
    - single wide envelope-conditioned KAN -> AM/AM gain + AM/PM phase
    - bounded 8-bit KAN complex nonlinear residual on squashed taps
    All spline inputs hard-squashed into grid -> high-drive stable across seeds.
    """

    def __init__(self, window: int, taps: int = 9, ntaps: int = 7,
                 kan_grid: int = 10, kan_order: int = 3,
                 nfeat: int = 14, nhid: int = 14):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1
        self.nfeat = nfeat
        self.nhid = nhid

        # ---- 1) Linear complex FIR memory (anchor) ----
        self.fir = CLinear(self.taps, 1, bias=False, quant_bits=8)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) Complex mix of taps -> feature basis ----
        self.mix = CLinear(window, nfeat, bias=True, quant_bits=8)

        # ---- 3) Smooth learned 1-D power warp (KAN on bounded p) ----
        self.pmod = KANLayer(1, nfeat, grid=6, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 4) Wide envelope-conditioned KAN -> hidden ----
        # input: nfeat warped envelopes + centre|x| + mean-env + p-scalar
        self.kan1 = KANLayer(nfeat + 3, nhid, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 5) Output KAN: hidden -> [gain, phase] ----
        self.kan2 = KANLayer(nhid, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 6) Bounded complex nonlinear residual via KAN AM shaping ----
        # KAN maps squashed envelope powers -> real per-tap weights; combine
        # complexly through an 8-bit CLinear on the weighted taps.
        self.rkan = KANLayer(self.ntaps, self.ntaps, grid=8, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=8)
        self.rmix = CLinear(self.ntaps, 1, bias=False, quant_bits=8)
        with torch.no_grad():
            self.rmix.wr.zero_(); self.rmix.wi.zero_()

        # ---- power gate for residual: smooth quadratic in p ----
        self.pgate = RLinear(2, 1, bias=True, quant_bits=8)
        with torch.no_grad():
            self.pgate.weight.zero_(); self.pgate.bias.fill_(0.5)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2

        xc = xw[:, c]                                          # centre complex
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt()   # (B, window)

        # ---- 1) linear FIR anchor ----
        lin = self.fir(xw[:, c - h: c + h + 1]).squeeze(-1)

        # ---- 2) complex feature mix -> envelope basis ----
        z = self.mix(xw)                                       # complex (B,nfeat)
        zenv = z.abs()                                         # (B,nfeat)

        # ---- 3) smooth 1-D power warp (safe at p=0 and p=1) ----
        pc = p.reshape(B, 1)
        pin = 1.5 * torch.tanh(2.0 * (pc - 0.5))               # (B,1) in grid
        pwarp = self.pmod(pin)                                 # (B,nfeat)
        pgate = torch.sigmoid(pwarp)                           # (B,nfeat)
        zmod = zenv * pgate                                    # (B,nfeat)

        # ---- free descriptors ----
        ec = env[:, c].reshape(B, 1)                           # (B,1)
        tstate = torch.tanh(env.mean(dim=1)).reshape(B, 1)     # (B,1)
        pscalar = torch.tanh(2.0 * (p - 0.5)).reshape(B, 1)    # (B,1)

        # ---- 4) wide feature KAN ----
        feat = torch.cat([zmod, ec, tstate, pscalar], dim=1)   # (B,nfeat+3)
        feat = 1.5 * torch.tanh(feat / 1.5)                    # squash into grid
        hid = self.kan1(feat)                                  # (B,nhid)
        hid = 1.5 * torch.tanh(hid / 1.5)                      # keep in grid

        # ---- 5) output KAN: AM/AM + AM/PM ----
        kout = self.kan2(hid)                                  # (B,2)
        gain = 1.0 + torch.tanh(kout[:, 0])                    # multiplicative
        phase = torch.tanh(kout[:, 1])                         # bounded phase
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        amam_out = xc * gain.to(xc.dtype) * rot

        # ---- 6) bounded KAN complex nonlinear residual ----
        xn = xw[:, c - hn: c + hn + 1]                         # complex (B,ntaps)
        aenv = (xn.real * xn.real + xn.imag * xn.imag).sqrt()  # (B,ntaps)
        aenv_s = 1.5 * torch.tanh(aenv / 1.5)                  # squash into grid
        wts = torch.tanh(self.rkan(aenv_s))                    # (B,ntaps) bounded
        xn_w = xn * wts.to(xn.dtype)                           # weighted taps
        nl = self.rmix(xn_w).squeeze(-1)                       # complex (B,)

        pf = torch.stack([p, p * p], dim=1)
        g = torch.sigmoid(self.pgate(pf).squeeze(-1))          # (B,) bounded
        nl = nl * g.to(nl.dtype)

        y = lin + amam_out + nl                                # complex (B,)
        return y