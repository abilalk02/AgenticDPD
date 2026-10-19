import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """Ultra-low-BOP power-conditioned KAN DPD for the EMPTY niche
    10^4.0-10^4.5 BOP (current best g021c3 weak at 3dBFS, ACLR 30.9).

    - passthrough-init complex FIR linear branch
    - single-stage envelope-conditioned KAN correction (8-bit quant to
      land BOP in the target band at ~180 RMpS)
    - smooth learned 1-D power gate (KAN of drive p) modulating correction
    All learnable ops are KAN/CLinear; spline inputs squashed into grid
    support for zero high-drive collapse and tight seed spread @3dBFS.
    """

    FEAT_SCALE = 0.21
    GRID_LIM = 1.5

    def __init__(self, window: int, hidden: int = 8, grid: int = 6,
                 order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        c = window // 2
        # compact tap set around centre
        offs = [o for o in (-2, -1, 0, 1, 2) if 0 <= c + o < window]
        self.tap_idx = [c + o for o in offs]
        nt = len(self.tap_idx)
        # per tap: [re, im, amp^2]; plus p feature column
        d_feat = nt * 3 + 1

        # linear memory branch (float, cheap in BOP but keep 8-bit too)
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # single-stage KAN correction core, quantized to land in band
        self.kan1 = KANLayer(d_feat, hidden, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)

        # smooth 1-D power-conditioning gain
        self.pgain = KANLayer(1, 1, grid, order,
                              x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                              quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan1, self.kan2, self.pgain):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def _squash(self, x):
        return self.GRID_LIM * torch.tanh(x / self.GRID_LIM)

    def forward(self, xw, p):
        B = xw.shape[0]
        xs = xw * self.FEAT_SCALE                       # (B, window)

        # --- linear memory branch ---
        lin_c = self.fir(xs)[:, 0]                      # (B,) complex

        # --- envelope feature stack ---
        cols = []
        for i in self.tap_idx:
            xc = xs[:, i]
            amp = torch.abs(xc)
            cols.append(xc.real)
            cols.append(xc.imag)
            cols.append(amp * amp)
        pin = (p.reshape(B, 1) * 2.0 - 1.0)             # [0,1]->[-1,1], safe
        feats = torch.stack(cols, dim=-1)               # (B, nt*3)
        feats = torch.cat([feats, pin], dim=1)          # (B, d_feat)
        feats = self._squash(feats)

        h = self._squash(self.kan1(feats))
        corr = self.kan2(h)                             # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])  # (B,) complex

        # --- smooth power-conditioning gain in (0, 2) ---
        g = self.pgain(self._squash(pin))[:, 0]
        gain = 1.0 + torch.tanh(g)

        out_c = (lin_c + gain.to(lin_c.dtype) * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)                # (B,) complex64