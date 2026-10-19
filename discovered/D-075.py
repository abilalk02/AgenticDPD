import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """Envelope-conditioned KAN DPD, scaled into the 7-9k RMpS / 10^6.5-10^7.0 BOP niche.

    - Complex FIR bank projects the whole tap window into K complex features
      (linear memory + cross-tap mixing).
    - A two-layer wide KAN operates on real envelope-power features of that
      bank plus a smooth power feature, producing a complex nonlinear
      correction. Spline inputs are tanh-squashed into the grid.
    - Two smooth 1-D KAN gains (functions of p) modulate the linear FIR pass
      and the nonlinear correction amplitude respectively, so drive is
      handled gracefully across the whole sweep (esp. the 3 dBFS bottleneck).
    - Heavy KAN layers use 8-bit fake-quant to keep BOP in-band.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, K: int = 12, hidden: int = 24,
                 grid: int = 8, order: int = 3):
        super().__init__(window)
        self.window = window
        self.K = K

        # --- complex FIR bank: window -> K complex features (linear memory) ---
        self.fir = CLinear(window, K, bias=False, quant_bits=8)
        # single complex "linear pass" tap (identity-ish start)
        self.linpass = CLinear(window, 1, bias=False, quant_bits=8)

        # nonlinear feature dim: [re, im, |.|, |.|^2] per K feature + 1 power
        feat_in = 4 * K + 1
        self.kan1 = KANLayer(feat_in, hidden, grid, order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # --- power-conditioning: smooth 1-D functions of p ---
        # nonlinear-correction amplitude gain
        self.pg1 = KANLayer(1, 6, grid, order, x_min=-1.5, x_max=1.5, quant_bits=8)
        self.pg2 = KANLayer(6, 1, grid, order, x_min=-1.5, x_max=1.5, quant_bits=8)
        # linear-pass gain
        self.plg1 = KANLayer(1, 6, grid, order, x_min=-1.5, x_max=1.5, quant_bits=8)
        self.plg2 = KANLayer(6, 1, grid, order, x_min=-1.5, x_max=1.5, quant_bits=8)

        with torch.no_grad():
            c = window // 2
            # linear pass: identity on centre tap
            self.linpass.wr.zero_()
            self.linpass.wi.zero_()
            self.linpass.wr[0, c] = 1.0
            # FIR bank: small random-ish start (centre-tap seeded lightly)
            self.fir.wr.mul_(0.3)
            self.fir.wi.mul_(0.3)
            # nonlinear KANs start small (near pass-through)
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            # gain KANs near neutral
            for l in (self.pg1, self.pg2, self.plg1, self.plg2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        xs = xw * self.FEAT_SCALE                          # (B, window) complex

        # --- linear pass branch ---
        lin_c = self.linpass(xs)[:, 0]                     # (B,) complex

        # --- complex FIR feature bank ---
        z = self.fir(xs)                                   # (B, K) complex
        amp = z.abs()                                      # (B, K)
        feats = torch.cat([z.real, z.imag, amp, amp * amp], dim=1)  # (B,4K)

        # smooth power feature
        pc = p.reshape(B, 1) * 2.0 - 1.0                   # [0,1]->[-1,1]
        feat_all = torch.cat([feats, pc], dim=1)           # (B, 4K+1)
        feat_all = 1.5 * torch.tanh(feat_all / 1.5)        # squash into grid

        h = self.kan1(feat_all)
        h = 1.5 * torch.tanh(h / 1.5)
        h = self.kan2(h)
        h = 1.5 * torch.tanh(h / 1.5)
        corr = self.kan3(h)                                # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])

        # --- power-conditioned gains ---
        pin = 1.5 * torch.tanh(pc / 1.5)                   # (B,1) into grid
        g = self.pg2(self.pg1(pin))[:, 0]                  # (B,)
        gain = 1.0 + torch.tanh(g)                         # (0,2)

        lg = self.plg2(self.plg1(pin))[:, 0]               # (B,)
        lgain = 1.0 + 0.5 * torch.tanh(lg)                 # (0.5,1.5) around 1

        out_c = (lgain.to(lin_c.dtype) * lin_c
                 + gain.to(corr_c.dtype) * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)