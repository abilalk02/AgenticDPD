import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """VCSEL-physics-biased DPD.

    - Short complex FIR  -> relaxation-oscillation / TIA short memory.
    - Envelope-driven compressive AM/AM correction via a KAN that outputs a
      complex-valued (gain, rotation) applied multiplicatively to the centre
      sample: phase-preserving core that lifts high-drive ACLR without
      injecting asymmetric spurs.
    - Slow thermal envelope (window-mean |x|) + drive power p jointly modulate
      the correction through a smooth 1-D learned power function (KAN gains),
      never as a raw concat feature -> tight seed spread across the sweep.
    """

    FEAT_SCALE = 0.21
    GRID_LIM = 1.5

    def __init__(self, window: int, hidden: int = 24, grid: int = 10,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2
        self.c = c
        # short memory taps only (relaxation oscillation is fast)
        offs = [o for o in (-2, -1, 0, 1, 2) if 0 <= c + o < window]
        self.tap_idx = [c + o for o in offs]
        nt = len(self.tap_idx)

        # per tap: envelope amp + amp^2  (compressive basis, phase-free)
        # plus: centre re/im (for small phase memory) + thermal env + p
        d_feat = nt * 2 + 2 + 2

        # --- linear short memory branch (complex FIR over central taps) ---
        self.fir = CLinear(nt, 1, bias=False, quant_bits=quant_bits)

        # --- compressive AM/AM + AM/PM correction core ---
        self.kan1 = KANLayer(d_feat, hidden, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)

        # --- smooth 1-D power+thermal conditioning of correction gain ---
        self.pg1 = KANLayer(2, 8, grid, order,
                            x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                            quant_bits=quant_bits)
        self.pg2 = KANLayer(8, 1, grid, order,
                            x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                            quant_bits=quant_bits)

        with torch.no_grad():
            # FIR identity on centre tap -> pass-through linear start
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            ci = self.tap_idx.index(c)
            self.fir.wr[0, ci] = 1.0
            for l in (self.kan1, self.kan2, self.pg1, self.pg2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def _squash(self, x):
        return self.GRID_LIM * torch.tanh(x / self.GRID_LIM)

    def forward(self, xw, p):
        B = xw.shape[0]
        xs = xw * self.FEAT_SCALE                       # (B, window)

        # --- gather short-memory taps ---
        taps = xs[:, self.tap_idx]                      # (B, nt) complex

        # --- linear short memory ---
        lin_c = self.fir(taps)[:, 0]                    # (B,) complex

        # --- compressive envelope basis (phase-preserving) ---
        amp = taps.abs()                                # (B, nt)
        cols = [amp, amp * amp]
        xc = xs[:, self.c]                              # centre sample
        cols.append(xc.real.reshape(B, 1))
        cols.append(xc.imag.reshape(B, 1))

        # slow thermal envelope: window-mean amplitude (free op)
        therm = xs.abs().mean(dim=1, keepdim=True)      # (B,1)
        pin = (p.reshape(B, 1) * 2.0 - 1.0)             # [-1,1], safe @0/1
        cols.append(therm)
        cols.append(pin)

        feats = torch.cat(cols, dim=1)                  # (B, d_feat)
        feats = self._squash(feats)

        h = self._squash(self.kan1(feats))
        corr = self.kan2(h)                             # (B,2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])  # (B,) complex

        # --- smooth power+thermal gain in (0,2) on the correction ---
        pcond = self._squash(torch.cat([pin, self._squash(therm)], dim=1))
        g = self.pg2(self._squash(self.pg1(pcond)))[:, 0]
        gain = 1.0 + torch.tanh(g)

        out_c = (lin_c + gain.to(lin_c.dtype) * corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)                # (B,) complex64