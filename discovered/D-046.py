import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear


class DPD(DPDModule):
    """Scaled envelope-conditioned KAN targeting BOP niche 10^7.0-10^7.5.

    Pure-KAN nonlinear core (robust at high drive, scales gracefully),
    complex FIR backbone for the linear path, smooth power conditioning.
    All KAN inputs are tanh-squashed into the spline support so no seed
    diverges at 3 dBFS.
    """

    FEAT_SCALE = 0.21
    GRID = 12
    ORDER = 3

    def __init__(self, window: int, hidden: int = 20,
                 quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2
        self.c = c

        # ---- complex FIR backbone over decimated taps ----
        decim = 2
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.fir = CLinear(self.n_taps, 2, bias=False, quant_bits=quant_bits)

        # ---- free envelope feature construction ----
        # center + 4 neighbour envelope taps, plus powers -> real features
        # features: [e0, e0^2, e0^3, eL, eR, eLL, eRR, e0*eL, e0*eR, pw]
        self.n_feat = 10

        # ---- wide two-stage KAN core (float, high BOP) ----
        self.kan1 = KANLayer(self.n_feat, hidden, self.GRID, self.ORDER,
                             x_min=-1.3, x_max=1.3, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, self.GRID, self.ORDER,
                             x_min=-1.3, x_max=1.3, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 6, self.GRID, self.ORDER,
                             x_min=-1.3, x_max=1.3, quant_bits=quant_bits)

        # head: KAN out -> complex gain(2) + correction(2)
        self.head = RLinear(6, 4, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            # FIR: first output = centre passthrough, second = small
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.head.weight.mul_(0.05)
            self.head.bias.zero_()

    def _sq(self, t):
        return 1.3 * torch.tanh(t / 1.3)

    def forward(self, xw, p):
        x = xw * self.FEAT_SCALE
        B = x.shape[0]
        c = self.c

        # linear backbone
        taps = x[:, self.tap_idx]
        firout = self.fir(taps)            # (B,2) complex
        lin = firout[:, 0]                 # (B,) complex

        # neighbour indices (clamped, free)
        iL = max(c - 1, 0)
        iR = min(c + 1, self.window - 1)
        iLL = max(c - 2, 0)
        iRR = min(c + 2, self.window - 1)

        e0 = torch.abs(x[:, c])
        eL = torch.abs(x[:, iL])
        eR = torch.abs(x[:, iR])
        eLL = torch.abs(x[:, iLL])
        eRR = torch.abs(x[:, iRR])

        # scale envelopes into a comfortable range then squash
        s = 2.0
        f0 = e0 * s
        feat = torch.stack([
            f0,
            f0 * f0,
            f0 * f0 * f0,
            eL * s,
            eR * s,
            eLL * s,
            eRR * s,
            f0 * eL * s,
            f0 * eR * s,
            p,                             # smooth power feature, safe at 0/1
        ], dim=-1)                         # (B, 10)
        feat = self._sq(feat)

        h = self.kan1(feat)
        h = self._sq(h)
        h = self.kan2(h)
        h = self._sq(h)
        kg = self.kan3(h)                  # (B,6)

        out = self.head(kg)                # (B,4)
        gr = out[:, 0]
        gi = out[:, 1]
        cr = out[:, 2]
        ci = out[:, 3]

        gain = torch.complex(1.0 + gr, gi)
        corr = torch.complex(cr, ci)

        o = (lin * gain + corr) / self.FEAT_SCALE
        return o.to(torch.complex64)