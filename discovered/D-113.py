import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Decimated complex-FIR backbone + memory-envelope KAN correction with
    drive-aware spline squashing and smooth 1-D power conditioning.

    Improvements over parent (aimed squarely at the 3 dBFS ceiling):
      - The nonlinear KAN now sees a small MEMORY envelope basis (centre +
        lagging |taps| and a lagging power term), so high-drive memory
        compression is representable instead of collapsed to one scalar.
      - Spline support is tightened and every input is hard-squashed into it
        with a power-dependent drive scale, so at peak envelope nothing ever
        leaves the grid (removes the seed-variance / high-drive collapse).
      - Power path outputs a per-feature drive scale, a correction gate, and
        an extra high-drive emphasis term that sharpens the correction near
        p=1 while collapsing to clean passthrough at low drive.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2, hidden: int = 10,
                 grid: int = 8, order: int = 3, pgrid: int = 8,
                 quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        # memory envelope taps: centre plus symmetric neighbours (free feats)
        mem_off = []
        for d in (1, 2, 3):
            if c - d >= 0:
                mem_off.append(c - d)
            if c + d < window:
                mem_off.append(c + d)
        mem_off = sorted(set(mem_off))
        self.register_buffer("mem_idx", torch.tensor(mem_off, dtype=torch.long))
        self.n_mem = len(mem_off)

        # number of nonlinear features: env_centre, env_pow(centre^3-ish),
        # mem envelopes, and one aggregated lagging power
        self.n_feat = 2 + self.n_mem + 1

        # complex FIR over decimated taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # envelope-only nonlinearity -> hidden -> (gr, gi)
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # smooth 1-D power-conditioning: p -> (drive_scale, gate, hd_emph)
        self.kanp = KANLayer(1, 3, pgrid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            # init FIR near a centre passthrough
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.kanp.coef.mul_(0.1)
            self.kanp.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                       # (B, window)
        taps = x[:, self.tap_idx]                      # (B, n_taps)
        lin = self.fir(taps)[:, 0]                     # (B,) complex

        # ---- smooth power-conditioning path (safe at p=0 and p=1) ----
        pc = p.reshape(B, 1)                           # (B,1) in [0,1]
        pin = torch.tanh(2.0 * pc - 1.0)               # monotone map to grid
        pout = self.kanp(pin)                          # (B,3)
        drive = 1.0 + 0.6 * torch.tanh(pout[:, 0:1])   # ~[0.4,1.6] per-sample
        gate = torch.sigmoid(pout[:, 1])               # (B,)
        hd = 0.5 * (torch.tanh(pout[:, 2]) + 1.0)      # (B,) high-drive emph

        # ---- memory envelope feature basis (all free ops) ----
        xc = x[:, self.c]                              # centre sample (complex)
        env = torch.abs(xc)                            # (B,)
        # drive-modulated envelope + a raw and a stronger (odd-order) term
        env_d = torch.tanh(env * 2.0 * drive[:, 0])    # (B,)
        env3 = torch.tanh((env ** 3) * 4.0)            # captures compression

        mem = torch.abs(x[:, self.mem_idx])            # (B, n_mem)
        mem_sq = torch.tanh(mem * 2.0)                 # squashed mem envelopes

        # aggregated lagging power (mean of neighbour env^2), squashed
        lag_pow = torch.tanh((mem ** 2).mean(dim=1) * 4.0)   # (B,)

        feat = torch.cat([
            env_d.unsqueeze(1),
            env3.unsqueeze(1),
            mem_sq,
            lag_pow.unsqueeze(1),
        ], dim=1)                                      # (B, n_feat)

        # hard-squash EVERYTHING into the spline grid support [-1,1]
        feat = torch.tanh(feat)

        h = torch.tanh(self.kan1(feat))                # keep kan2 input in grid
        g = self.kan2(h)                               # (B,2) additive delta

        # high-drive emphasis sharpens correction near p=1, gate scales it
        scale = gate * (1.0 + hd)                      # (B,)
        gr = g[:, 0] * scale
        gi = g[:, 1] * scale
        gain = torch.complex(1.0 + gr, gi)             # gain near 1

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)