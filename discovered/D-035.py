import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """Cheap power-conditioned KAN-augmented FIR DPD.

    - Complex FIR over a centred sub-window supplies linear memory.
    - A SINGLE KAN(3->hidden) + RLinear(hidden->2) forms a bounded
      nonlinear correction on the centre sample envelope features.
    - A tiny 1-D KAN maps p -> smooth scalar gain modulating the
      nonlinear correction (graceful high-drive behaviour).

    Target: >3x RMpS reduction vs the 500-RMpS parent, EVM held within
    0.3 pts by retaining bounded KAN basis + linear memory.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 4, grid: int = 5,
                 order: int = 3, mem: int = 7, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2
        # effective centred memory span
        half = min(mem // 2, c)
        self.lo = c - half
        self.hi = c + half + 1
        self.weff = self.hi - self.lo

        # linear memory: complex FIR over centred sub-window
        self.fir = CLinear(self.weff, 1, bias=False, quant_bits=quant_bits)

        # single-layer bounded nonlinear correction
        self.kan = KANLayer(3, hidden, grid, order, x_min=-1.0, x_max=1.0,
                            quant_bits=quant_bits)
        self.mix = RLinear(hidden, 2, bias=True, quant_bits=quant_bits)

        # power-conditioning: smooth 1-D gain function of p
        self.pgain = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        with torch.no_grad():
            # FIR identity on centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cc = self.weff // 2
            self.fir.wr[0, cc] = 1.0
            # small nonlinear correction at init
            self.kan.coef.mul_(0.1)
            self.kan.wb.mul_(0.1)
            self.mix.weight.mul_(0.1)
            self.mix.bias.zero_()
            # gain KAN near neutral
            self.pgain.coef.mul_(0.1)
            self.pgain.wb.mul_(0.1)

    def forward(self, xw, p):
        c = self.window // 2
        xs = xw * self.FEAT_SCALE                       # (B, window)

        # --- linear memory branch (centred sub-window) ---
        lin_c = self.fir(xs[:, self.lo:self.hi])[:, 0]  # (B,) complex

        # --- nonlinear correction features (centre sample) ---
        xc = xs[:, c]
        amp = torch.abs(xc)
        feats = torch.stack([xc.real, xc.imag, amp], dim=-1)  # (B,3)
        h = torch.tanh(self.kan(feats))
        corr = self.mix(h)                              # (B,2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])

        # --- power-conditioning gain: smooth 1-D function of p ---
        pin = (p.unsqueeze(-1) * 2.0 - 1.0)             # [0,1]->[-1,1]
        g = self.pgain(pin)[:, 0]
        gain = 1.0 + torch.tanh(g)                      # (0,2)

        out_c = (lin_c + gain * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)