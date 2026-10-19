import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Compact power-conditioned KAN DPD:
       short complex FIR (linear memory) + p-biased KAN static AM/AM & AM/PM.
       Aggressively pruned from the GMP+thermal parent for >=3x lower RMpS
       while keeping the robust KAN backbone for high-drive stability."""

    def __init__(self, window: int, taps: int = 5,
                 kan_grid: int = 5, kan_order: int = 3):
        super().__init__(window)
        self.taps = min(taps, window) | 1

        # ---- 1) Linear complex FIR: short bandwidth-limited memory ----
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) Static AM/AM + AM/PM via KAN (splines), p-biased ----
        # inputs: squashed centre envelope, squashed local mean envelope, p-feature
        # outputs: 2 -> (gain correction, phase correction)
        self.amam = KANLayer(3, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        B = xw.shape[0]

        xc = xw[:, c]                                       # centre sample (complex)
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt()  # (B, window) envelope

        # ---- 1) Linear FIR path ----
        lin = self.fir(xw[:, c - h : c + h + 1]).squeeze(-1)  # complex (B,)

        # ---- feature construction (all free ops) ----
        ec = env[:, c]                                     # centre envelope
        em = env.mean(dim=1)                               # local mean envelope
        ein = 1.5 * torch.tanh(ec)                         # squash into grid
        emin = 1.5 * torch.tanh(em)
        pin = 1.5 * torch.tanh(2.0 * (p - 0.5))            # smooth p operating point
        kin = torch.stack([ein, emin, pin], dim=1)         # (B,3)

        # ---- 2) KAN static correction ----
        kout = self.amam(kin)                              # (B,2)
        gain = 1.0 + torch.tanh(kout[:, 0])                # bounded AM/AM gain
        phase = torch.tanh(kout[:, 1])                     # bounded AM/PM
        rot = torch.complex(torch.cos(phase), torch.sin(phase))

        amam_out = lin * gain.to(lin.dtype) * rot          # complex (B,)

        return amam_out