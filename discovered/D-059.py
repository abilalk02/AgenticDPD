import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear


class DPD(DPDModule):
    """Envelope-conditioned scaled-KAN DPD with smooth 1-D power modulation.

    Niche target: 10^6.0-10^6.5 BOP band (~2-3k RMpS), filled with a KAN
    backbone (robust up- and down-scaling) rather than a dense/conv core.

    - Complex FIR anchor (near-identity) provides the linear backbone.
    - Free real envelope features (|x|^k across a few centre taps) drive a
      two-layer KAN gain, all inputs squashed into the spline grid.
    - Power p feeds a separate small KAN producing a smooth 1-D gain/shift
      vector that modulates the main KAN state (robust across the sweep).
    - Head maps KAN state -> complex gain + additive correction on anchor.
    """

    FEAT_SCALE = 0.21
    SQ = 1.5  # spline squash amplitude (matches grid support)

    def __init__(self, window: int,
                 hidden: int = 16,
                 grid: int = 10, order: int = 3,
                 quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2
        self.c = c

        # ---- complex FIR anchor (decimated taps, identity init) ----
        decim = 2
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # ---- envelope feature taps (centre neighbourhood) ----
        mem = min(5, window)
        half = mem // 2
        env_idx = [c + d for d in range(-half, mem - half) if 0 <= c + d < window]
        self.register_buffer("env_idx", torch.tensor(env_idx, dtype=torch.long))
        self.n_env = len(env_idx)
        # features: for each env tap -> [|x|, |x|^2, |x|^3]  (all free)
        self.feat_in = self.n_env * 3

        # ---- main KAN backbone on envelope features ----
        self.kan1 = KANLayer(self.feat_in, hidden, grid, order,
                             x_min=-self.SQ, x_max=self.SQ,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-self.SQ, x_max=self.SQ,
                             quant_bits=quant_bits)

        # ---- smooth 1-D power modulation via its own KAN ----
        # produces (gain, shift) applied to the KAN state (hidden channels)
        self.pkan = KANLayer(1, 2 * hidden, grid, order,
                             x_min=-self.SQ, x_max=self.SQ,
                             quant_bits=quant_bits)
        self.hidden = hidden

        # ---- head: KAN state -> complex gain + additive correction ----
        self.head = RLinear(hidden, 4, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            for l in (self.kan1, self.kan2, self.pkan):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.head.weight.mul_(0.05)
            self.head.bias.zero_()

    def _squash(self, t):
        return self.SQ * torch.tanh(t / self.SQ)

    def forward(self, xw, p):
        x = xw * self.FEAT_SCALE
        B = x.shape[0]

        # linear anchor
        taps = x[:, self.tap_idx]
        lin = self.fir(taps)[:, 0]                       # complex (B,)

        # free envelope features
        env = torch.abs(x[:, self.env_idx])              # (B, n_env)
        e1 = env
        e2 = env * env
        e3 = e2 * env
        feat = torch.cat([e1, e2, e3], dim=-1)           # (B, feat_in)
        feat = self._squash(feat * 4.0)                  # into grid

        # main KAN backbone
        h = self.kan1(feat)                              # (B, hidden)
        h = self._squash(h)
        h = self.kan2(h)                                 # (B, hidden)

        # smooth 1-D power modulation
        pc = p.reshape(B, 1)                             # (B,1) safe at 0,1
        pc = self._squash(pc * 2.0 - self.SQ)            # map [0,1]->grid
        pm = self.pkan(pc)                               # (B, 2*hidden)
        pg = pm[:, :self.hidden]
        ps = pm[:, self.hidden:]
        # gate in [0,1] * state + smooth additive shift; bounded => stable
        h = torch.sigmoid(pg) * h + torch.tanh(ps)
        h = self._squash(h)

        # head -> complex gain + correction
        out = self.head(h)                               # (B,4)
        gr = out[:, 0]
        gi = out[:, 1]
        cr = out[:, 2]
        ci = out[:, 3]

        gain = torch.complex(1.0 + gr, gi)
        corr = torch.complex(cr, ci)

        o = (lin * gain + corr) / self.FEAT_SCALE
        return o.to(torch.complex64)