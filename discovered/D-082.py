import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """KAN-scaled power-conditioned DPD for the 10^6.5-10^7.0 BOP niche.
       Complex FIR memory + wide 2-layer KAN static AM/AM & AM/PM (p-biased)
       + residual GMP cross-memory polynomial. High-drive-weighted, KAN
       backbone keeps 3 dBFS stable across seeds. ~7-8k RMpS."""

    def __init__(self, window: int, taps: int = 7,
                 kan_grid: int = 8, kan_order: int = 3,
                 nenv: int = 6, hidden: int = 16):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.nenv = min(nenv, window)

        # ---- 1) Linear complex FIR: short bandwidth-limited memory ----
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # feature dim: nenv delayed envelopes + centre env + mean env + p
        self.feat_dim = self.nenv + 3

        # ---- 2) Wide two-layer KAN static nonlinearity ----
        self.kan1 = KANLayer(self.feat_dim, hidden, grid=kan_grid,
                             order=kan_order, x_min=-1.5, x_max=1.5)
        self.kan2 = KANLayer(hidden, 2, grid=kan_grid,
                             order=kan_order, x_min=-1.5, x_max=1.5)

        # ---- 3) Residual GMP cross-memory polynomial (high-drive terms) ----
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=3, cross=1)
        # small scale so it starts as a perturbation
        self.gmp_scale = CLinear(1, 1, bias=False)
        with torch.no_grad():
            self.gmp_scale.wr.fill_(0.1); self.gmp_scale.wi.zero_()

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        B = xw.shape[0]

        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt()  # (B, window)

        # ---- 1) Linear FIR path ----
        lin = self.fir(xw[:, c - h: c + h + 1]).squeeze(-1)   # complex (B,)

        # ---- feature construction (all free ops) ----
        ec = env[:, c]                                        # centre env
        em = env.mean(dim=1)                                  # mean env
        # nenv delayed envelope taps centred around c
        half = self.nenv // 2
        idx = [min(max(c - half + k, 0), self.window - 1) for k in range(self.nenv)]
        etaps = env[:, idx]                                   # (B, nenv)

        ein = 1.5 * torch.tanh(etaps)                         # squash into grid
        ecin = (1.5 * torch.tanh(ec)).reshape(B, 1)
        emin = (1.5 * torch.tanh(em)).reshape(B, 1)
        pin = (1.5 * torch.tanh(2.0 * (p - 0.5))).reshape(B, 1)  # p op point

        feat = torch.cat([ein, ecin, emin, pin], dim=1)      # (B, feat_dim)

        # ---- 2) KAN static correction (2-layer) ----
        hmid = self.kan1(feat)                               # (B, hidden)
        hmid = 1.5 * torch.tanh(hmid / 1.5)                  # keep inside grid
        kout = self.kan2(hmid)                               # (B, 2)

        gain = 1.0 + torch.tanh(kout[:, 0])                  # bounded AM/AM
        phase = torch.tanh(kout[:, 1])                       # bounded AM/PM
        rot = torch.complex(torch.cos(phase), torch.sin(phase))

        main = lin * gain.to(lin.dtype) * rot                # complex (B,)

        # ---- 3) Residual GMP high-drive polynomial ----
        g = self.gmp(xw)                                     # complex (B,) or (B,1)
        if g.dim() > 1:
            g = g.reshape(B, -1)[:, 0]
        gscaled = self.gmp_scale(g.reshape(B, 1)).squeeze(-1)

        y = main + gscaled                                   # complex (B,)
        return y