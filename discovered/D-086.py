import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """High-BOP power-conditioned KAN DPD targeting EMPTY niche bin 9 (>10^7.5 BOP).

    Structure:
      - complex FIR linear-memory branch (identity init pass-through)
      - wide two-stage KAN correction core over a squashed multi-tap
        envelope basis (real/imag/amp/amp^2 per tap)
      - smooth learned 1-D power-conditioning gain (KAN of drive p) that
        modulates the correction amplitude across the whole power sweep
    All learnable ops are KAN/CLinear primitives; spline inputs squashed into
    grid support for zero high-drive collapse and tight seed spread @3dBFS.
    """

    FEAT_SCALE = 0.21
    GRID_LIM = 1.5

    def __init__(self, window: int, hidden: int = 56, grid: int = 14,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2
        offs = [o for o in (-4, -3, -2, -1, 0, 1, 2, 3, 4) if 0 <= c + o < window]
        self.tap_idx = [c + o for o in offs]
        nt = len(self.tap_idx)
        # per tap: [re, im, amp, amp^2]; plus p feature column
        d_feat = nt * 4 + 1

        # linear memory: complex FIR over whole window
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # wide dense-grid KAN correction core (float32 -> high BOP)
        self.kan1 = KANLayer(d_feat, hidden, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)

        # smooth 1-D power-conditioning gain on the correction
        self.pgain1 = KANLayer(1, 10, grid, order,
                               x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                               quant_bits=quant_bits)
        self.pgain2 = KANLayer(10, 1, grid, order,
                               x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                               quant_bits=quant_bits)

        with torch.no_grad():
            # FIR: identity on centre tap (pass-through linear start)
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            # small nonlinear correction at init for stability
            for l in (self.kan1, self.kan2, self.kan3, self.pgain1, self.pgain2):
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
        pin = (p.reshape(B, 1) * 2.0 - 1.0)             # [0,1] -> [-1,1], safe
        feats = torch.stack(cols, dim=-1)               # (B, nt*4)
        feats = torch.cat([feats, pin], dim=1)          # (B, d_feat)
        feats = self._squash(feats)

        h = self._squash(self.kan1(feats))
        h = self._squash(self.kan2(h))
        corr = self.kan3(h)                             # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])  # (B,) complex

        # --- smooth power-conditioning gain in (0, 2) ---
        g = self.pgain2(self._squash(self.pgain1(self._squash(pin))))[:, 0]
        gain = 1.0 + torch.tanh(g)

        out_c = (lin_c + gain.to(lin_c.dtype) * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)                # (B,) complex64