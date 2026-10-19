import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """Ultra-lean power-conditioned KAN DPD for the EMPTY low-BOP niche
    (10^4.0-10^4.5 BOP). g005c2/g030c0 lineage — clears the 3 dBFS ceiling.

    - complex FIR memory (identity init pass-through)
    - one compact KAN correction core over a squashed envelope basis
    - smooth learned 1-D power gain modulating the correction
    - 6-bit fake-quant to land BOP in the target band at low RMpS
    """

    FEAT_SCALE = 0.21
    GRID_LIM = 1.5

    def __init__(self, window: int, hidden: int = 8, grid: int = 6,
                 order: int = 3, quant_bits: int = 6):
        super().__init__(window)
        self.window = window
        c = window // 2
        offs = [o for o in (-2, -1, 0, 1, 2) if 0 <= c + o < window]
        self.tap_idx = [c + o for o in offs]
        nt = len(self.tap_idx)
        # per tap: [re, im, amp]; plus p feature column
        d_feat = nt * 3 + 1

        # linear memory branch: complex FIR over whole window
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # compact KAN correction core
        self.kan1 = KANLayer(d_feat, hidden, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)

        # smooth 1-D power-conditioning gain on the correction
        self.pgain = KANLayer(1, 1, grid, order,
                              x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                              quant_bits=quant_bits)

        with torch.no_grad():
            # FIR: identity on centre tap (pass-through linear start)
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

        # --- multi-tap envelope feature stack ---
        cols = []
        for i in self.tap_idx:
            xc = xs[:, i]
            amp = torch.abs(xc)
            cols.append(xc.real)
            cols.append(xc.imag)
            cols.append(amp)
        pin = (p.reshape(B, 1) * 2.0 - 1.0)             # [0,1] -> [-1,1], safe
        feats = torch.stack(cols, dim=-1)               # (B, nt*3)
        feats = torch.cat([feats, pin], dim=1)          # (B, d_feat)
        feats = self._squash(feats)

        h = self._squash(self.kan1(feats))
        corr = self.kan2(h)                             # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])  # (B,) complex

        # --- smooth power-conditioning gain in (0, 2) ---
        g = self.pgain(self._squash(pin))[:, 0]         # (B,)
        gain = 1.0 + torch.tanh(g)

        out_c = (lin_c + gain.to(lin_c.dtype) * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)                # (B,) complex64