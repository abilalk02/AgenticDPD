import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """Decimated complex FIR + envelope-only KAN nonlinearity, power-conditioned.

    Prunes the parent's heavy CNN to a single decimated CLinear backbone and a
    compact envelope KAN (bounded splines -> graceful high-drive behaviour).
    Power p modulates the KAN input gains and a small GMP residual so one model
    linearizes the whole sweep. Targets the empty 200-350 RMpS niche.
    """

    def __init__(self, window: int, decim: int = 2, env_taps: int = 5,
                 kan_hid: int = 8):
        super().__init__(window)
        c = window // 2

        # --- decimated tap indices (share taps around the centre) ---
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_lin = len(idx)

        # --- envelope tap indices (coarse, centred) ---
        eidx = torch.linspace(0, window - 1, env_taps).round().long().tolist()
        eidx = sorted(set(eidx))
        self.register_buffer("env_idx", torch.tensor(eidx, dtype=torch.long))
        self.n_env = len(eidx)

        # --- linear complex FIR backbone over decimated taps ---
        self.fir = CLinear(self.n_lin, 1, bias=False)

        # --- envelope KAN: |x| powers at decimated taps -> hidden bases ---
        self.env_kan = KANLayer(self.n_env, kan_hid, grid=8, order=3,
                                x_min=-1.5, x_max=1.5)

        # --- power -> per-basis gain (bounded, splines) ---
        self.pmod = KANLayer(1, kan_hid, grid=6, order=3, x_min=0.0, x_max=1.0)

        # --- KAN head: gated bases -> complex correction (real, imag) ---
        self.head = KANLayer(kan_hid, 2, grid=8, order=3, x_min=-1.5, x_max=1.5)

        # --- cheap GMP residual, power-gated complex scalar ---
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=5, cross=2)
        self.pgain = KANLayer(1, 2, grid=6, order=3, x_min=0.0, x_max=1.0)

    def forward(self, xw, p):
        c = self.window // 2
        pin = p.unsqueeze(-1)  # (B,1)

        # linear complex FIR over decimated taps
        xt = xw[:, self.tap_idx]                        # (B, n_lin) complex
        lin = self.fir(xt)[:, 0]                         # (B,) complex

        # envelope features (parameter-free), squashed into spline support
        env = xw[:, self.env_idx].abs()                 # (B, n_env)
        env = 1.5 * torch.tanh(env / 1.5)
        h = self.env_kan(env)                           # (B, kan_hid)

        # power-conditioned bounded gain on bases
        g = self.pmod(pin)                              # (B, kan_hid)
        h = torch.tanh(h) * (1.0 + 0.5 * torch.tanh(g))

        o = self.head(h)                                # (B, 2)
        nl_res = torch.complex(o[:, 0], o[:, 1])

        # GMP residual gated by learned function of power
        gm = self.gmp(xw)
        if gm.dim() > 1:
            gm = gm[:, c] if gm.shape[-1] == self.window \
                else gm.reshape(gm.shape[0], -1)[:, 0]
        gate = self.pgain(pin)                          # (B,2)
        gmp_res = gm * torch.complex(gate[:, 0], gate[:, 1])

        return lin + nl_res + gmp_res