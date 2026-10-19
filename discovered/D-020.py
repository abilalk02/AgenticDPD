import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Power-aware predistorter: complex FIR + power-modulated GMP terms +
    a bounded-spline (KAN) complex-gain branch driven by local envelope and
    RF drive power. Bounded bases target the high-drive ACLR bottleneck."""

    def __init__(self, window: int, taps: int = 9, ntaps: int = 5, kgrid: int = 7):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1

        # --- Linear complex FIR (dominant response) ---
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # --- Classic GMP-style odd-order correction (3rd/5th) ---
        self.nl = CLinear(2 * self.ntaps, 1, bias=False)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()

        # Power modulation of GMP branch: p -> real gain
        self.pgain = RLinear(2, 1, bias=True)
        with torch.no_grad():
            self.pgain.weight.zero_()
            self.pgain.bias.fill_(1.0)

        # --- KAN branch: learned complex gain g(env, p) on centre tap ---
        # Input features (parameter-free, squashed into spline support):
        #   e0 = |x_center|,  e1 = mean local |x|,  and p.
        self.kan = KANLayer(3, 2, grid=kgrid, order=3, x_min=-1.5, x_max=1.5)
        # kan output -> (real_gain_perturb, imag_gain_perturb); we add to
        # identity so init acts as passthrough.

        self.hn = self.ntaps // 2

    def _squash(self, x, a=1.4):
        # map into (~-1.5, 1.5) spline support
        return a * torch.tanh(x / a)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        hn = self.hn

        # Linear path
        lin = self.fir(xw[:, c - h : c + h + 1]).squeeze(-1)

        # Nonlinear GMP features (parameter-free)
        xn = xw[:, c - hn : c + hn + 1]                 # (B, ntaps) complex
        a2 = xn.real * xn.real + xn.imag * xn.imag
        t3 = xn * a2                                    # 3rd order
        t5 = xn * a2 * a2                               # 5th order
        feats = torch.cat([t3, t5], dim=1)              # (B, 2*ntaps)
        nl = self.nl(feats).squeeze(-1)                 # (B,) complex

        # Power-dependent real gain on GMP term
        pf = torch.stack([p, p * p], dim=1)             # (B, 2)
        g = self.pgain(pf).squeeze(-1)                  # (B,) real
        nl = nl * g.to(nl.dtype)

        # --- KAN complex-gain branch on centre sample ---
        xc = xw[:, c]                                   # (B,) complex
        e0 = torch.sqrt(a2[:, hn] + 1e-12)              # centre envelope
        elocal = torch.sqrt(a2.mean(dim=1) + 1e-12)     # local mean envelope
        pin = (2.0 * p - 1.0)                           # center p to ~[-1,1]

        kin = torch.stack([
            self._squash(e0 * 2.0),
            self._squash(elocal * 2.0),
            self._squash(pin * 1.4),
        ], dim=1)                                       # (B,3)

        kout = self.kan(kin)                            # (B,2) real gain perturb
        gr = 1.0 + kout[:, 0]
        gi = kout[:, 1]
        gc = torch.complex(gr, gi)                      # complex gain ~1+0j init

        centre_corr = xc * gc                           # (B,) complex

        # Combine: linear FIR + GMP + KAN-gated centre correction.
        # Subtract xc so KAN adds only its correction over the FIR passthrough.
        return lin + nl + (centre_corr - xc)