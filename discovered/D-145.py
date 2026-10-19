import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """Envelope-conditioned KAN DPD with VCSEL IM/DD inductive bias.

    - Short complex FIR memory (relaxation-oscillation / TIA bandwidth).
    - Compressive AM/AM envelope features (L-I saturation) + slow thermal
      envelope, fed to a deep well-squashed KAN correction.
    - Smooth 1-D KAN gain of p modulates correction amplitude (robust sweep).
    Lineage: envelope-conditioned KAN core that clears the 3 dBFS ceiling.
    """

    FEAT_SCALE = 0.21
    GRID_LIM = 1.5

    def __init__(self, window: int, hidden: int = 28, grid: int = 12,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2
        self.c = c
        # short-memory taps: centre +/- 2 (bounded, physics-motivated)
        offs = [o for o in (-2, -1, 0, 1, 2) if 0 <= c + o < window]
        self.tap_idx = [c + o for o in offs]
        nt = len(self.tap_idx)
        # per tap: [re, im, amp(compressed), amp^2]; + thermal env + p
        d_feat = nt * 4 + 2

        # linear memory over full window
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # nonlinear KAN correction core
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
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0  # identity pass-through start
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

        # --- compressive envelope feature stack (L-I saturation bias) ---
        cols = []
        for i in self.tap_idx:
            xc = xs[:, i]
            amp = torch.abs(xc)
            amp_comp = torch.tanh(amp)                  # soft compression
            cols.append(xc.real)
            cols.append(xc.imag)
            cols.append(amp_comp)
            cols.append(amp * amp)

        # --- slow thermal envelope: mean amplitude over window ---
        thermal = torch.tanh(torch.abs(xs).mean(dim=1, keepdim=True))  # (B,1)

        pin = (p.reshape(B, 1) * 2.0 - 1.0)             # [0,1]->[-1,1]
        feats = torch.stack(cols, dim=-1)               # (B, nt*4)
        feats = torch.cat([feats, thermal, pin], dim=1) # (B, d_feat)
        feats = self._squash(feats)

        h = self._squash(self.kan1(feats))
        h = self._squash(self.kan2(h))
        corr = self.kan3(h)                             # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])

        # --- smooth power-conditioning gain ---
        g = self.pgain2(self._squash(self.pgain1(self._squash(pin))))[:, 0]
        gain = 1.0 + torch.tanh(g)                      # (0, 2)

        out_c = (lin_c + gain * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)