import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, GMPBranch


class DPD(DPDModule):
    """GMP-physics-prior + envelope-conditioned-KAN residual DPD.

    - Complex FIR (CLinear over window) -> clean linear memory / passthrough.
    - GMPBranch -> generalized-memory-polynomial physics prior (structured
      memory nonlinearity, cheap complex coeffs, robust at high drive).
    - KAN on [I,Q,|x|,|x|^2] of centre sample -> learned nonlinear residual
      that cooperates with (corrects) the GMP prior.
    - Two smooth learned 1-D functions of p (KAN) modulate the GMP and KAN
      amplitudes separately, so drive-dependent push is controlled per power.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8, grid: int = 7,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window

        # linear memory branch
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # physics prior: generalized memory polynomial
        mem = min(5, window)
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=mem, cross=2)

        # nonlinear residual KAN on envelope-augmented centre sample
        self.kan1 = KANLayer(4, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # power-conditioning: two smooth 1-D gains of p
        self.pg1 = KANLayer(1, 4, grid, order, x_min=-1.0, x_max=1.0,
                            quant_bits=quant_bits)
        self.pg2 = KANLayer(4, 2, grid, order, x_min=-1.0, x_max=1.0,
                            quant_bits=quant_bits)

        with torch.no_grad():
            # FIR: identity on centre tap (linear pass-through start)
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            c = window // 2
            self.fir.wr[0, c] = 1.0
            # small nonlinear corrections at init
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            for l in (self.pg1, self.pg2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def _gmp_center(self, out):
        # normalize GMPBranch output to a (B,) complex tensor
        if out.dim() == 2:
            if out.shape[1] == 1:
                out = out[:, 0]
            else:
                out = out[:, out.shape[1] // 2]
        if not torch.is_complex(out):
            out = out.to(torch.complex64)
        return out

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        xs = xw * self.FEAT_SCALE                      # (B, window)

        # --- linear memory branch ---
        lin_c = self.fir(xs)[:, 0]                     # (B,) complex

        # --- physics prior branch (GMP) ---
        gmp_c = self._gmp_center(self.gmp(xs))         # (B,) complex

        # --- nonlinear residual KAN (centre sample envelope) ---
        xc = xs[:, c]
        amp = torch.abs(xc)
        feats = torch.stack([xc.real, xc.imag, amp, amp * amp], dim=-1)
        feats = torch.tanh(feats)                      # squash into grid
        corr = self.kan2(self.kan1(feats))             # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])

        # --- power-conditioning: two smooth 1-D gains of p ---
        pin = (p.reshape(B, 1) * 2.0 - 1.0)            # [0,1] -> [-1,1]
        g = self.pg2(self.pg1(pin))                    # (B, 2)
        gain_gmp = 1.0 + torch.tanh(g[:, 0])           # (0, 2)
        gain_kan = 1.0 + torch.tanh(g[:, 1])           # (0, 2)

        out_c = (lin_c
                 + gain_gmp * gmp_c
                 + gain_kan * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)