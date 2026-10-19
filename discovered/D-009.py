import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """Aggressively pruned DPD: decimated taps + shared complex linear +
    cheap envelope KAN nonlinearity + power-gated GMP residual."""

    def __init__(self, window: int, decim: int = 2, kan_grid: int = 8,
                 kan_order: int = 3):
        super().__init__(window)
        c = window // 2

        # --- decimate taps: symmetric subset around centre ---
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_tap = len(idx)
        self.c = c

        # --- shared complex linear mixing the decimated taps (memory linear) ---
        self.lin = CLinear(self.n_tap, 1, bias=True)

        # --- cheap envelope nonlinearity: 1-D learned function of |x_centre| ---
        # outputs a complex multiplicative correction (2 reals) on the linear term
        self.env = KANLayer(1, 2, grid=kan_grid, order=kan_order,
                            x_min=0.0, x_max=1.5)

        # --- envelope-driven memory correction: KAN maps decimated |taps| -> 2 ---
        self.env_mem = KANLayer(self.n_tap, 2, grid=6, order=3,
                                x_min=0.0, x_max=1.5)

        # --- GMP residual on full window (cheap complex coeffs) ---
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=5, cross=2)

        # --- power-conditioned complex gains (one model spans the sweep) ---
        self.pgain = KANLayer(1, 4, grid=6, order=3, x_min=0.0, x_max=1.0)

    def forward(self, xw, p):
        c = self.c
        # decimated tap window (complex)
        xd = xw[:, self.tap_idx]                       # (B, n_tap)

        # linear memory term (shared complex weights)
        lin = self.lin(xd)                             # (B, 1) complex
        lin = lin[:, 0]

        # envelope nonlinearity on centre magnitude -> complex correction
        env = self.env(xw[:, c].abs().unsqueeze(-1))   # (B, 2)
        env_c = torch.complex(1.0 + env[:, 0], env[:, 1])

        # envelope-driven memory correction from decimated magnitudes
        em = self.env_mem(xd.abs())                    # (B, 2)
        em_c = torch.complex(em[:, 0], em[:, 1])

        # power gains: split into (linear/env) gain and gmp gain
        pg = self.pgain(p.unsqueeze(-1))               # (B, 4)
        g_main = torch.complex(pg[:, 0], pg[:, 1])
        g_gmp = torch.complex(pg[:, 2], pg[:, 3])

        # main predistortion: power-scaled envelope-modulated linear + memory corr
        main = g_main * (lin * env_c + xw[:, c] * em_c)

        # GMP residual
        g = self.gmp(xw)
        if g.dim() > 1:
            g = g[:, c] if g.shape[-1] == self.window else \
                g.reshape(g.shape[0], -1)[:, 0]
        gmp_res = g_gmp * g

        return xw[:, c] + main + gmp_res