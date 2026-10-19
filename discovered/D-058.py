import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """High-BOP power-conditioned KAN DPD (targets empty niche bin 9, >1e7 BOP).

    - Wide complex FIR over the full tap window: linear memory.
    - Multi-tap envelope feature stack -> two wide float32 KAN layers
      (dense grids -> large BOP) producing a bounded complex nonlinear
      correction.
    - Smooth 1-D KAN gain function of p modulates the correction amplitude
      (learned per-drive scaling) for graceful high-drive behaviour.
    """

    FEAT_SCALE = 0.21
    GRID_LIM = 1.5

    def __init__(self, window: int, hidden: int = 32, grid: int = 12,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2
        # neighbour taps used for the envelope feature stack (centre +/- up to 2)
        offs = [o for o in (-2, -1, 0, 1, 2) if 0 <= c + o < window]
        self.tap_idx = [c + o for o in offs]
        nt = len(self.tap_idx)
        # feature layout per tap: [re, im, amp, amp^2]; plus 1 for p
        d_feat = nt * 4 + 1

        # linear memory: complex FIR over the whole window -> 1 complex out
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # wide nonlinear KAN core (float32 -> high BOP)
        self.kan1 = KANLayer(d_feat, hidden, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)

        # smooth 1-D power-conditioning gain
        self.pgain1 = KANLayer(1, 8, grid, order,
                               x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                               quant_bits=quant_bits)
        self.pgain2 = KANLayer(8, 1, grid, order,
                               x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                               quant_bits=quant_bits)

        with torch.no_grad():
            # FIR: identity on centre tap (pass-through linear start)
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            # small nonlinear correction at init
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            for l in (self.pgain1, self.pgain2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def _squash(self, x):
        return self.GRID_LIM * torch.tanh(x / self.GRID_LIM)

    def forward(self, xw, p):
        B = xw.shape[0]
        xs = xw * self.FEAT_SCALE                       # (B, window)

        # --- linear memory branch ---
        lin_c = self.fir(xs)[:, 0]                      # (B,) complex

        # --- multi-tap envelope feature stack ---
        cols = []
        for i in self.tap_idx:
            xc = xs[:, i]
            amp = torch.abs(xc)
            cols.append(xc.real)
            cols.append(xc.imag)
            cols.append(amp)
            cols.append(amp * amp)
        pin = (p.reshape(B, 1) * 2.0 - 1.0)             # [0,1] -> [-1,1]
        feats = torch.stack(cols, dim=-1)               # (B, nt*4)
        feats = torch.cat([feats, pin], dim=1)          # (B, d_feat)
        feats = self._squash(feats)

        h = self._squash(self.kan1(feats))
        h = self._squash(self.kan2(h))
        corr = self.kan3(h)                             # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])

        # --- smooth power-conditioning gain ---
        g = self.pgain2(self._squash(self.pgain1(self._squash(pin))))[:, 0]
        gain = 1.0 + torch.tanh(g)                      # smooth in (0, 2)

        out_c = (lin_c + gain * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)