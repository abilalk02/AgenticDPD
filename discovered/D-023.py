import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """Power-conditioned KAN-augmented FIR DPD.

    - Complex FIR (CLinear over the whole tap window) provides linear memory.
    - A KAN on [I, Q, |x|, |x|^2] of the centre sample provides a bounded
      nonlinear correction.
    - A tiny KAN maps p -> smooth scalar gain that MODULATES the amplitude
      of the nonlinear correction (learned 1-D function of drive power),
      so the network learns how hard to push the nonlinearity per power
      level -- targeting graceful behaviour at the high-drive ACLR bottleneck.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8, grid: int = 7,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window

        # linear memory: complex FIR over the whole window -> 1 complex out
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # nonlinear correction KAN on envelope-augmented centre sample
        self.kan1 = KANLayer(4, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # power-conditioning: smooth 1-D gain function of p
        # p in [0,1] mapped into spline support, -> scalar modulation gain
        self.pgain1 = KANLayer(1, 4, grid, order, x_min=-1.0, x_max=1.0,
                               quant_bits=quant_bits)
        self.pgain2 = KANLayer(4, 1, grid, order, x_min=-1.0, x_max=1.0,
                               quant_bits=quant_bits)

        with torch.no_grad():
            # FIR: identity on centre tap (pass-through linear start)
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            c = window // 2
            self.fir.wr[0, c] = 1.0
            # small nonlinear correction at init
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            # gain KAN near neutral (~1.0) at init: keep small so tanh->~1
            for l in (self.pgain1, self.pgain2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def forward(self, xw, p):
        c = self.window // 2
        xs = xw * self.FEAT_SCALE                     # (B, window)

        # --- linear memory branch ---
        lin_c = self.fir(xs)[:, 0]                     # (B,) complex

        # --- nonlinear correction features (centre sample envelope) ---
        xc = xs[:, c]
        amp = torch.abs(xc)
        feats = torch.stack([xc.real, xc.imag, amp, amp * amp], dim=-1)
        corr = self.kan2(self.kan1(feats))             # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])

        # --- power-conditioning gain: smooth 1-D function of p ---
        pin = (p.unsqueeze(-1) * 2.0 - 1.0)            # [0,1] -> [-1,1]
        g = self.pgain2(self.pgain1(pin))[:, 0]        # (B,)
        gain = 1.0 + torch.tanh(g)                     # smooth in (0, 2)

        out_c = (lin_c + gain * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)