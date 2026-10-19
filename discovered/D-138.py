import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """g005c2-core DPD grown into the 6-9k RMpS band.

    - Complex FIR over the whole window -> linear memory (identity init).
    - A second complex FIR maps taps -> a small complex feature vector whose
      real/imag/envelope drive a widened 2-layer KAN nonlinearity.
    - A smooth 1-D KAN function of p modulates the nonlinear correction gain
      (power conditioning), clearing the high-drive ACLR ceiling.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, feat: int = 6, hidden: int = 24,
                 grid: int = 10, order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        self.feat = feat

        # linear memory: complex FIR over the whole window -> 1 complex out
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # feature FIR: whole window -> small complex feature bank
        self.ffir = CLinear(window, feat, bias=False, quant_bits=quant_bits)

        # nonlinear KAN: [Re,Im,|.| per feature] -> hidden -> 2 (complex out)
        d_in = 3 * feat
        self.kan1 = KANLayer(d_in, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # power-conditioning: smooth 1-D gain function of p
        self.pgain1 = KANLayer(1, 6, grid, order,
                               x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.pgain2 = KANLayer(6, 1, grid, order,
                               x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            # FIR: identity on centre tap (pass-through linear start)
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            c = window // 2
            self.fir.wr[0, c] = 1.0
            # feature FIR: small random start (keep it modest)
            self.ffir.wr.mul_(0.3)
            self.ffir.wi.mul_(0.3)
            # small nonlinear correction at init
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            # gain KAN near neutral (~1.0) at init
            for l in (self.pgain1, self.pgain2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        xs = xw * self.FEAT_SCALE                       # (B, window)

        # --- linear memory branch ---
        lin_c = self.fir(xs)[:, 0]                      # (B,) complex

        # --- nonlinear feature bank ---
        z = self.ffir(xs)                               # (B, feat) complex
        amp = z.abs()                                   # (B, feat) real
        feats = torch.cat([z.real, z.imag, amp], dim=1) # (B, 3*feat)
        feats = torch.tanh(feats)                       # squash into grid

        h = torch.tanh(self.kan1(feats))
        h = torch.tanh(self.kan2(h))
        corr = self.kan3(h)                             # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])  # (B,) complex

        # --- power-conditioning gain: smooth 1-D function of p ---
        pin = (p.reshape(B, 1) * 2.0 - 1.0)             # [0,1] -> [-1,1]
        g = self.pgain2(self.pgain1(pin))[:, 0]         # (B,)
        gain = 1.0 + torch.tanh(g)                      # (0, 2), smooth

        out_c = (lin_c + gain * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)