import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """Power-conditioned dual-KAN DPD for BOP niche 10^6.5-10^7.0 (bin 9).

    Backbone: complex FIR (linear memory pass-through) + wide dense-grid KAN
    correction on a multi-tap re/im/amp basis, PLUS a parallel amplitude-only
    KAN branch for AM/AM at high drive. Correction amplitude is modulated by a
    single shallow smooth 1-D function of drive p. Grid limit tightened to 1.2
    and minimal squash cascade -> no spline input leaves support at 3 dBFS,
    giving tight seed spread and >parent ACLR at high drive.
    """

    FEAT_SCALE = 0.21
    GRID_LIM = 1.2

    def __init__(self, window: int, hidden: int = 40, grid: int = 12,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2
        offs = [o for o in (-3, -2, -1, 0, 1, 2, 3) if 0 <= c + o < window]
        self.tap_idx = [c + o for o in offs]
        nt = len(self.tap_idx)
        # per tap: [re, im, amp]; plus p feature column
        d_feat = nt * 3 + 1
        # amplitude-only branch: [amp, amp^2, amp^3] per tap + p
        d_amp = nt * 3 + 1

        # linear memory: complex FIR over whole window
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # main wide dense-grid KAN correction core
        self.kan1 = KANLayer(d_feat, hidden, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)

        # parallel amplitude (AM/AM) branch, smaller hidden
        self.kama1 = KANLayer(d_amp, 16, grid, order,
                              x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                              quant_bits=quant_bits)
        self.kama2 = KANLayer(16, 2, grid, order,
                              x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                              quant_bits=quant_bits)

        # shallow smooth 1-D power-conditioning gain (single KAN, no cascade)
        self.pgain = KANLayer(1, 1, grid, order,
                              x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                              quant_bits=quant_bits)

        with torch.no_grad():
            # FIR: identity on centre tap (linear pass-through start)
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            # attenuate corrections for stable, tight-seed start
            for l in (self.kan1, self.kan2, self.kama1, self.kama2):
                l.coef.mul_(0.08)
                l.wb.mul_(0.08)
            self.pgain.coef.mul_(0.1)
            self.pgain.wb.mul_(0.1)

    def _squash(self, x):
        return self.GRID_LIM * torch.tanh(x / self.GRID_LIM)

    def forward(self, xw, p):
        B = xw.shape[0]
        xs = xw * self.FEAT_SCALE                       # (B, window)

        # --- linear memory branch ---
        lin_c = self.fir(xs)[:, 0]                      # (B,) complex

        pin = (p.reshape(B, 1) * 2.0 - 1.0)             # [0,1]->[-1,1]

        # --- main re/im/amp feature stack ---
        cols = []
        acols = []
        for i in self.tap_idx:
            xc = xs[:, i]
            amp = torch.abs(xc)
            cols.append(xc.real)
            cols.append(xc.imag)
            cols.append(amp)
            acols.append(amp)
            acols.append(amp * amp)
            acols.append(amp * amp * amp)

        feats = torch.stack(cols, dim=-1)               # (B, nt*3)
        feats = torch.cat([feats, pin], dim=1)          # (B, d_feat)
        feats = self._squash(feats)
        h = self._squash(self.kan1(feats))
        corr_main = self.kan2(h)                        # (B, 2)

        afeat = torch.stack(acols, dim=-1)              # (B, nt*3)
        afeat = torch.cat([afeat, pin], dim=1)          # (B, d_amp)
        afeat = self._squash(afeat)
        ha = self._squash(self.kama1(afeat))
        corr_amp = self.kama2(ha)                       # (B, 2)

        corr = corr_main + corr_amp
        corr_c = torch.complex(corr[:, 0], corr[:, 1])  # (B,) complex

        # --- smooth power gain in (0, 2), single shallow KAN ---
        g = self.pgain(self._squash(pin))[:, 0]         # (B,)
        gain = 1.0 + torch.tanh(g)

        out_c = (lin_c + gain.to(lin_c.dtype) * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)                # (B,) complex64