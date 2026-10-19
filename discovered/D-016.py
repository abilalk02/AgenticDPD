import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RConv1d, RLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """VCSEL/IM-DD physics-biased DPD.

    Three pathways, each an explicit inductive bias:
      * short-memory RO/TIA transfer  -> 8b RConv1d over the window
      * compressive static AM/AM      -> KAN on instantaneous envelope
      * classic memory polynomial     -> GMPBranch odd orders
    A slow thermal EMA of the envelope + power feed a conditioning KAN that
    outputs complex gains modulating the AM/AM and memory contributions,
    letting one model track the whole power sweep and slow thermal drift.

    All learnable weights live inside registered primitives; DPD owns no
    bare parameters (thermal pole is a fixed constant) so the cost
    accountant can enumerate it.
    """

    def __init__(self, window: int, ch: int = 20, k: int = 5,
                 kan_env_grid: int = 10, kan_cond_grid: int = 8,
                 therm_pole: float = 0.85):
        super().__init__(window)
        c = window // 2
        self.c = c
        # fixed thermal EMA pole (non-learnable constant, no bare Parameter)
        self.therm_pole = float(therm_pole)

        # --- short-memory linear-ish RO/TIA transfer (few taps, 8b) ---
        # features: Re, Im, |x|, |x|^2  -> ch channels, small kernel
        self.mem_conv = RConv1d(4, ch, k, padding=k // 2, quant_bits=8)
        self.mem_head = RLinear(ch, 2, quant_bits=8)  # -> complex residual

        # --- compressive static AM/AM curve (learnable 1-D saturation) ---
        self.amam = KANLayer(1, 4, grid=kan_env_grid, order=3,
                             x_min=0.0, x_max=1.5)
        self.amam_head = RLinear(4, 2)

        # --- classic memory polynomial (odd orders, short mem) ---
        self.gmp = GMPBranch(window, orders=(1, 3, 5, 7), mem=5, cross=2)

        # --- thermal + power conditioning ---
        self.cond = KANLayer(2, 6, grid=kan_cond_grid, order=3,
                             x_min=0.0, x_max=1.0)
        self.cond_head = RLinear(6, 6)

    def _thermal_state(self, xw):
        # slow one-pole EMA of |x|^2 along the window (parameter-free,
        # fixed pole). Returns scalar per batch in ~[0,1].
        a = self.therm_pole
        env = xw.abs() ** 2                  # (B, W)
        B, W = env.shape
        s = env[:, 0]
        for t in range(1, W):
            s = a * s + (1.0 - a) * env[:, t]
        return torch.tanh(s)

    def forward(self, xw, p):
        c = self.c
        B = xw.shape[0]
        pin = p.unsqueeze(-1)                    # (B,1)

        xc = xw[:, c]                            # centre phasor
        env_c = xc.abs().unsqueeze(-1)           # (B,1) instantaneous envelope

        # ---- thermal + power conditioning gains ----
        therm = self._thermal_state(xw).unsqueeze(-1)   # (B,1)
        cond_in = torch.cat([therm, pin], dim=-1)       # (B,2)
        g = self.cond_head(self.cond(cond_in))          # (B,6)
        g_mem = torch.complex(g[:, 0], g[:, 1])
        g_am = torch.complex(g[:, 2], g[:, 3])
        g_gmp = torch.complex(g[:, 4], g[:, 5])

        # ---- short-memory RO/TIA pathway ----
        feat = torch.stack([xw.real, xw.imag, xw.abs(), xw.abs() ** 2], dim=1)
        h = torch.tanh(self.mem_conv(feat))[:, :, c]    # (B, ch)
        m = self.mem_head(h)                            # (B,2)
        mem_res = torch.complex(m[:, 0], m[:, 1]) * (1.0 + 0.5 * g_mem)

        # ---- compressive static AM/AM pathway ----
        basis = self.amam(env_c)                        # (B,4)
        a = self.amam_head(basis)                       # (B,2)
        denom = (env_c.squeeze(-1) + 1e-6)
        dirn = xc / denom
        am_res = torch.complex(a[:, 0], a[:, 1]) * dirn * (1.0 + 0.5 * g_am)

        # ---- memory-polynomial residual ----
        gm = self.gmp(xw)
        if gm.dim() > 1:
            gm = gm[:, c] if gm.shape[-1] == self.window \
                else gm.reshape(B, -1)[:, 0]
        gmp_res = gm * g_gmp

        return xc + mem_res + am_res + gmp_res