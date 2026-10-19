import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """g005c2/g030c0 core grown into the 6-9k RMpS band.

    Backbone: full-window decimated complex FIR (centre-anchored) gives a
    strong linear memory estimate. Nonlinearity: several learned complex
    mixes produce an envelope basis which, together with a smooth power
    feature, drives a 2-layer KAN that outputs a complex gain modulating
    the linear estimate. Envelope + power features are tanh-squashed into
    the spline grid so nothing escapes support at 3 dBFS drive.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 1, n_env: int = 12,
                 hidden: int = 20, grid: int = 9, order: int = 3,
                 quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c

        # decimated tap indices, always including the centre
        idx = list(range(0, window, max(1, decim)))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)

        # complex FIR over taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # learned complex mixes for an envelope basis (multi-projection)
        self.env_mix = CLinear(self.n_taps, n_env, bias=True,
                               quant_bits=quant_bits)
        self.n_env = n_env

        # KAN: [n_env envelopes, centre env, power feat] -> hidden -> (gr, gi)
        d_in = n_env + 2
        self.kan1 = KANLayer(d_in, hidden, grid, order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)

        with torch.no_grad():
            # init FIR near centre passthrough
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # small nonlinear correction to start
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                       # (B, window) complex
        taps = x[:, self.tap_idx]                      # (B, n_taps) complex

        # linear memory estimate
        lin = self.fir(taps)[:, 0]                     # (B,) complex

        # envelope basis from learned complex mixes
        z = self.env_mix(taps)                         # (B, n_env) complex
        env = z.abs()                                  # (B, n_env) real, free
        env_s = 1.5 * torch.tanh(env * 3.0 / 1.5)      # into grid support

        # centre-sample envelope (dominant AM/AM feature)
        xc = x[:, self.c]                              # (B,) complex
        cenv = torch.abs(xc).reshape(B, 1)             # (B,1)
        cenv_s = 1.5 * torch.tanh(cenv * 3.0 / 1.5)

        # smooth power feature (safe at p=0 and p=1, no div/log/sqrt-neg)
        pc = p.reshape(B, 1)                           # (B,1) float32
        pf = 1.5 * torch.tanh((pc - 0.5) * 2.0)        # centred, in grid

        feat = torch.cat([env_s, cenv_s, pf], dim=1)   # (B, n_env+2) real

        h = self.kan1(feat)                            # (B, hidden)
        h = 1.5 * torch.tanh(h / 1.5)                  # keep 2nd KAN in grid
        g = self.kan2(h)                               # (B, 2) real

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])   # (B,) complex, near 1

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)