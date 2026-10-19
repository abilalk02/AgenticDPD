import torch
import torch.nn as nn
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RLinear, CLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """VCSEL IM/DD physics-biased DPD.

    Three explicit branches mirror the impairment:
      * Static compressive AM/AM   -> power-conditioned KAN on the center envelope
      * Short memory (RO + TIA BW) -> GMPBranch + compact complex memory mixer
      * Slow thermal envelope       -> KAN( slow envelope stat, p ) -> complex gain
    """

    def __init__(self, window: int, mem: int = 5, kan_grid: int = 10):
        super().__init__(window)
        self.window = window
        c = window // 2
        self.c = c
        # short-memory taps centered on the middle sample
        half = min(mem // 2, c)
        self.lo = c - half
        self.hi = c + half + 1
        self.mlen = self.hi - self.lo

        # ---- Static compressive AM/AM branch -------------------------------
        # KAN maps [ |x_c|, |x_c|^2, p ] -> (dr, di) correction on center tap.
        # p shifts the compression operating point (bias/thermal knee).
        self.amam = KANLayer(3, 2, grid=kan_grid, order=3, x_min=-1.5, x_max=1.5)

        # ---- Short-memory branch (relaxation-osc + TIA bandwidth) ----------
        # GMPBranch gives a low-order complex Volterra kernel over short mem.
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=self.mlen, cross=1)
        # Compact learned complex mixer over the short memory window: models
        # the few-tap dispersive tail with power-dependent scaling handled below.
        self.memmix = CLinear(self.mlen, 1, bias=True)

        # ---- Slow thermal envelope branch ---------------------------------
        # Feeds a low-pass envelope statistic + p into a KAN producing a
        # complex gain (gr, gi) applied to the aggregate correction.
        self.thermal = KANLayer(2, 2, grid=8, order=3, x_min=-1.5, x_max=1.5)

    def _squash(self, x, a=1.4):
        # map into spline support [-1.5, 1.5]
        return a * torch.tanh(x / a)

    def forward(self, xw, p):
        B = xw.shape[0]
        a = xw.abs()                       # (B, window)
        xc = xw[:, self.c]                 # center complex sample
        ac = a[:, self.c]                  # center envelope

        p1 = 2.0 * p - 1.0                 # center p about 0

        # ---- Static AM/AM correction --------------------------------------
        amam_in = torch.stack([
            self._squash(ac * 2.0 - 1.0),
            self._squash(ac * ac * 2.0 - 1.0),
            self._squash(p1 * 1.5),
        ], dim=-1)                          # (B, 3)
        amam_out = self.amam(amam_in)       # (B, 2)
        corr_static = torch.complex(amam_out[:, 0], amam_out[:, 1])

        # ---- Short memory correction --------------------------------------
        gmp_out = self.gmp(xw)              # complex (B,) or (B,1)
        if gmp_out.dim() > 1:
            gmp_out = gmp_out[:, 0]
        xmem = xw[:, self.lo:self.hi]       # (B, mlen) complex
        mem_out = self.memmix(xmem)[:, 0]   # (B,) complex
        corr_mem = gmp_out + mem_out

        # ---- Slow thermal envelope gain -----------------------------------
        # Low-pass statistic of the window envelope (mean) is a proxy for the
        # slow thermal state; p is the steady drive component.
        env_slow = a.mean(dim=-1)           # (B,)
        th_in = torch.stack([
            self._squash(env_slow * 2.0 - 1.0),
            self._squash(p1 * 1.5),
        ], dim=-1)                          # (B, 2)
        th = self.thermal(th_in)            # (B, 2)
        gain = torch.complex(1.0 + 0.5 * torch.tanh(th[:, 0]),
                             0.5 * torch.tanh(th[:, 1]))  # complex ~1 gain

        corr = gain * (corr_static + corr_mem)
        return xc + corr