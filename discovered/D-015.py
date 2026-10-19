import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Power-aware predistorter: complex FIR backbone + widened GMP-style
    nonlinear correction whose per-order complex strength is a learned
    smooth 1-D function of RF drive power p (via KAN)."""

    def __init__(self, window: int, taps: int = 11, ntaps: int = 7):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1

        # Linear complex FIR (dominant response)
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # Nonlinear features per tap: x*|x|^2, x*|x|^4, x*|x|^6  (3 orders)
        # plus lagged cross-envelope terms x[c]*|x[c±1]|^2.
        self.n_orders = 3
        self.n_cross = 2
        nl_in = self.n_orders * self.ntaps + self.n_cross
        self.nl = CLinear(nl_in, 1, bias=False)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()

        # Power -> per-order complex modulation coefficients.
        # KAN learns a smooth 1-D function of p (squashed into spline support),
        # producing 2*n_orders reals = complex gain per nonlinear order.
        self.pkan = KANLayer(1, 2 * self.n_orders, grid=6, order=3,
                             x_min=-1.5, x_max=1.5)
        with torch.no_grad():
            # bias-like init: encourage near-unity real gains initially
            pass

        # small refinement of p features feeding kan
        self.pin = RLinear(2, 1, bias=True)
        with torch.no_grad():
            self.pin.weight.zero_()
            self.pin.weight[0, 0] = 1.0
            self.pin.bias.zero_()

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2

        # Linear path
        lin = self.fir(xw[:, c - h : c + h + 1]).squeeze(-1)

        # Nonlinear window
        xn = xw[:, c - hn : c + hn + 1]            # (B, ntaps) complex
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = xn * (a2 * a2)
        t7 = xn * (a2 * a2 * a2)

        # cross-memory: centre sample modulated by neighbour envelopes
        xc = xn[:, hn:hn + 1]                       # (B,1) complex
        aL = a2[:, hn - 1:hn]                        # left neighbour env
        aR = a2[:, hn + 1:hn + 2]                    # right neighbour env
        cL = xc * aL
        cR = xc * aR

        # Power-dependent complex per-order gains via KAN
        pf = torch.stack([p, p * p], dim=1)          # (B,2) real
        pk = self.pin(pf)                            # (B,1)
        pk = 1.4 * torch.tanh(pk)                    # squash into spline range
        g = self.pkan(pk)                           # (B, 2*n_orders) real
        gr = g[:, 0::2]                             # (B, n_orders)
        gi = g[:, 1::2]
        gc = torch.complex(gr, gi).to(t3.dtype)     # (B, n_orders) complex

        # apply per-order gain (broadcast across taps)
        t3 = t3 * gc[:, 0:1]
        t5 = t5 * gc[:, 1:2]
        t7 = t7 * gc[:, 2:3]

        feats = torch.cat([t3, t5, t7, cL, cR], dim=1)  # (B, nl_in) complex
        nl = self.nl(feats).squeeze(-1)

        return lin + nl