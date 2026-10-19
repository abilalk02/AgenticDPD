import torch
import torch.nn as nn
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Power-modulated KAN-augmented complex FIR.

    Design:
      * Linear complex FIR (CLinear) provides the memory backbone.
      * A KAN maps per-tap envelope magnitude -> complex nonlinear correction
        coefficients (bounded spline basis = graceful high-drive behaviour).
      * p enters ONLY through a small 1-D KAN conditioner that outputs smooth,
        bounded scalar modulators (global gain, nonlinear gain, phase tilt).
        These scale/rotate the nonlinear correction per drive level so the
        worst-case (high-drive) point gets its own specialized shaping.
    """

    def __init__(self, window: int, nl_hidden: int = 6):
        super().__init__(window)
        self.window = window
        c = window // 2

        # --- Linear complex FIR backbone (memory) ---
        self.fir = CLinear(window, 1, bias=True)

        # --- Nonlinear correction path ---
        # KAN on envelope magnitude features -> real hidden channels.
        # Bounded spline basis -> robust across drive.
        self.nl_kan = KANLayer(window, nl_hidden, grid=8, order=3,
                               x_min=-1.5, x_max=1.5)
        # Map hidden nonlinear features to a complex correction via CLinear.
        # We feed nl features as real part; CLinear handles the complex mix.
        self.nl_out = CLinear(nl_hidden, 1, bias=False)

        # --- Power conditioner: p -> [global_gain, nl_gain, phase_tilt] ---
        self.pcond = KANLayer(1, 3, grid=8, order=3,
                              x_min=-1.5, x_max=1.5)

    def forward(self, xw, p):
        c = self.window // 2
        a = xw.abs()  # (B, window) envelope

        # Linear backbone output (complex).
        lin = self.fir(xw)[:, 0]  # (B,)

        # Nonlinear correction: bounded-magnitude envelope into KAN.
        # squash envelope into spline support smoothly.
        a_sq = 1.2 * torch.tanh(a)                      # (B, window) in ~[-1.2,1.2]
        nlf = self.nl_kan(a_sq)                          # (B, nl_hidden) real
        nlf = torch.tanh(nlf)                            # bound the basis output
        nlc = self.nl_out(torch.complex(nlf, torch.zeros_like(nlf)))[:, 0]  # (B,) complex

        # Power conditioning (smooth 1-D functions of p).
        ps = 1.2 * torch.tanh((2.0 * p - 1.0) * 1.5).unsqueeze(-1)  # (B,1) in spline range
        cond = self.pcond(ps)                            # (B, 3)
        g_glob = 1.0 + 0.5 * torch.tanh(cond[:, 0])      # ~[0.5,1.5]
        g_nl = torch.tanh(cond[:, 1])                    # ~[-1,1] nonlinear scale
        phase = (math.pi / 2) * torch.tanh(cond[:, 2])   # bounded phase tilt

        # Apply per-power phase rotation to the nonlinear correction.
        rot = torch.complex(torch.cos(phase), torch.sin(phase))  # (B,)
        nlc = nlc * (g_nl.to(nlc.dtype)) * rot

        # Combine: global gain scales linear backbone, add modulated nonlinear.
        out = g_glob.to(lin.dtype) * lin + nlc

        # Residual passthrough of centre sample keeps early training stable.
        return xw[:, c] + out