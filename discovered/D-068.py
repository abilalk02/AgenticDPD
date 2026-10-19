import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """Envelope-conditioned dual-KAN DPD with amplitude-normalised phase
    features and a monotone high-drive-only correction gate.

    Backbone = complex FIR (linear memory, owns low/mid drive) + a KAN
    correction whose amplitude is gated by a smooth learned 1-D function of
    drive power p that vanishes near p=0. Phase geometry is fed normalised so
    spline inputs stay bounded and seed spread stays tight at 3 dBFS.
    """

    FEAT_SCALE = 0.21
    GRID_LIM = 1.5

    def __init__(self, window: int, hidden: int = 32, grid: int = 10,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2
        offs = [o for o in (-3, -2, -1, 0, 1, 2, 3) if 0 <= c + o < window]
        self.tap_idx = [c + o for o in offs]
        nt = len(self.tap_idx)
        # per tap: [cos, sin (unit-phase), amp, amp^2]  + p feature
        d_feat = nt * 4 + 1

        # linear memory: complex FIR over whole window
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # KAN correction core
        self.kan1 = KANLayer(d_feat, hidden, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)

        # smooth 1-D power-conditioning gate (monotone, ~0 at low drive)
        self.pgain1 = KANLayer(1, 8, grid, order,
                               x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                               quant_bits=quant_bits)
        self.pgain2 = KANLayer(8, 1, grid, order,
                               x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                               quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan1, self.kan2, self.pgain1, self.pgain2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def _squash(self, x):
        return self.GRID_LIM * torch.tanh(x / self.GRID_LIM)

    def forward(self, xw, p):
        B = xw.shape[0]
        xs = xw * self.FEAT_SCALE                       # (B, window)

        # --- linear memory branch (owns low/mid drive) ---
        lin_c = self.fir(xs)[:, 0]                       # (B,) complex

        # --- multi-tap envelope + normalised-phase feature stack ---
        cols = []
        for i in self.tap_idx:
            xc = xs[:, i]
            amp = torch.abs(xc)
            inv = 1.0 / (amp + 1e-6)                     # safe: amp>=0
            cols.append(xc.real * inv)                  # cos(phase), bounded
            cols.append(xc.imag * inv)                  # sin(phase), bounded
            cols.append(amp)
            cols.append(amp * amp)
        pin = (p.reshape(B, 1) * 2.0 - 1.0)             # [0,1] -> [-1,1]
        feats = torch.stack(cols, dim=-1)               # (B, nt*4)
        feats = torch.cat([feats, pin], dim=1)          # (B, d_feat)
        feats = self._squash(feats)

        h = self._squash(self.kan1(feats))
        corr = self.kan2(h)                             # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])  # (B,) complex

        # --- monotone power gate: ~0 at low drive, grows at high drive ---
        g = self.pgain2(self._squash(self.pgain1(self._squash(pin))))[:, 0]
        # sigmoid keeps gate in (0,1); learned bias lets it near-vanish low
        gate = torch.sigmoid(g)

        out_c = (lin_c + gate.to(lin_c.dtype) * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)                # (B,) complex64