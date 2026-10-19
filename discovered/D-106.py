import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """Fresh KAN-backbone DPD with per-channel power-conditioned gating.

    Targets the EMPTY low-BOP niches (bins 0/1). NOT a g015c3 descendant.

    Structure:
      - complex FIR linear-memory branch (identity init pass-through)
      - single wide KAN correction core over a squashed multi-tap
        envelope basis (re/im/amp/amp^2 per tap)
      - a smooth 1-D KAN of drive p producing a PER-CHANNEL modulation
        vector m(p) in (0,2) applied to the hidden correction features,
        i.e. a learned mix between low- and high-drive correction experts
      - small linear head folds modulated features into the complex output
    Every learnable op is KAN/CLinear; all spline inputs squashed into grid.
    8-bit correction core keeps BOP in the empty low bins.
    """

    FEAT_SCALE = 0.21
    GRID_LIM = 1.5

    def __init__(self, window: int, hidden: int = 16, grid: int = 8,
                 order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        c = window // 2
        offs = [o for o in (-3, -2, -1, 0, 1, 2, 3) if 0 <= c + o < window]
        self.tap_idx = [c + o for o in offs]
        nt = len(self.tap_idx)
        d_feat = nt * 4                      # re, im, amp, amp^2 per tap

        # linear memory: complex FIR over whole window
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # wide KAN correction core (8-bit -> low BOP, RMpS unchanged)
        self.kan1 = KANLayer(d_feat, hidden, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)

        # per-channel smooth power-conditioning modulation m(p)
        self.pmod1 = KANLayer(1, hidden, grid, order,
                              x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                              quant_bits=quant_bits)

        # head: modulated hidden features -> complex correction (2 real)
        self.head = KANLayer(hidden, 2, grid, order,
                             x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                             quant_bits=quant_bits)

        with torch.no_grad():
            # FIR identity on centre tap (pass-through start)
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            # small correction at init for stability / tight seeds
            for l in (self.kan1, self.pmod1, self.head):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def _squash(self, x):
        return self.GRID_LIM * torch.tanh(x / self.GRID_LIM)

    def forward(self, xw, p):
        B = xw.shape[0]
        xs = xw * self.FEAT_SCALE                        # (B, window)

        # --- linear memory branch ---
        lin_c = self.fir(xs)[:, 0]                       # (B,) complex

        # --- multi-tap envelope feature stack (parameter-free) ---
        cols = []
        for i in self.tap_idx:
            xc = xs[:, i]
            amp = torch.abs(xc)
            cols.append(xc.real)
            cols.append(xc.imag)
            cols.append(amp)
            cols.append(amp * amp)
        feats = torch.stack(cols, dim=-1)                # (B, nt*4)
        feats = self._squash(feats)

        # correction hidden features
        h = self._squash(self.kan1(feats))               # (B, hidden)

        # --- per-channel power modulation (mix of drive experts) ---
        pin = (p.reshape(B, 1) * 2.0 - 1.0)              # [0,1]->[-1,1], safe
        m = self.pmod1(self._squash(pin))                # (B, hidden)
        gate = 1.0 + torch.tanh(m)                       # in (0,2)
        h = h * gate                                     # power-conditioned

        corr = self.head(self._squash(h))                # (B, 2)
        corr_c = torch.complex(corr[:, 0], corr[:, 1])   # (B,) complex

        out_c = (lin_c + corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)                 # (B,) complex64