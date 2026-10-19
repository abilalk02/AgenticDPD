import math
import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, GMPBranch, RLinear


class DPD(DPDModule):
    """VCSEL IM/DD physics-informed DPD.

    Structure:
      - short complex FIR  -> fast linear memory (relaxation osc. / TIA BW)
      - envelope KAN gain  -> compressive static AM/AM (L-I saturation)
      - AM/PM KAN          -> saturation-induced phase rotation
      - slow thermal EMA + power p -> operating-point conditioning
      - small GMPBranch    -> residual short-memory nonlinearity
    """

    def __init__(self, window: int, taps: int = 9):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        c = self.window // 2
        self.c = c
        h = self.taps // 2
        self.h = h

        # --- fast linear FIR memory ---
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, h] = 1.0

        # --- conditioning: [env, slow_thermal_env, p] -> bias features ---
        self.cond = RLinear(3, 4, bias=True)

        # --- compressive AM/AM: 1-D learnable gain over (env, cond) ---
        # inputs squashed into [-1.5,1.5]; env>=0 uses positive half.
        self.amam = KANLayer(5, 1, grid=8, order=3, x_min=-1.5, x_max=1.5)
        # AM/PM phase vs envelope + conditioning
        self.ampm = KANLayer(5, 1, grid=6, order=3, x_min=-1.5, x_max=1.5)

        # --- residual short-memory polynomial nonlinearity ---
        self.gmp = GMPBranch(self.window, orders=(1, 3, 5), mem=3, cross=1)

        # thermal leak coefficient (fixed slow smoothing over window)
        self.register_buffer("leak", torch.tensor(0.85))

    def _squash(self, x, a=1.4):
        return a * torch.tanh(x / a)

    def forward(self, xw, p):
        c, h = self.c, self.h
        centre = xw[:, c]                       # (B,) complex
        seg = xw[:, c - h : c + h + 1]          # (B, taps)

        # fast linear memory
        lin = self.fir(seg).squeeze(-1)         # (B,) complex

        # instantaneous envelope
        env = centre.abs()                      # (B,)

        # slow thermal envelope: causal leaky EMA across the window magnitude
        mag = xw.abs()                          # (B, window)
        B, W = mag.shape
        leak = self.leak
        ema = mag[:, 0]
        # iterate causally (window is small); parameter-free
        for t in range(1, W):
            ema = leak * ema + (1.0 - leak) * mag[:, t]
        slow = ema                              # (B,) thermal proxy

        # conditioning features
        cond_in = torch.stack([env, slow, p], dim=-1)   # (B,3)
        cfeat = torch.tanh(self.cond(cond_in))          # (B,4)

        # AM/AM: learnable compressive gain
        env_s = self._squash(env)
        kin = torch.cat([env_s.unsqueeze(-1), cfeat], dim=-1)   # (B,5)
        gain = 1.0 + self.amam(kin).squeeze(-1)                 # (B,) around unity
        phase = self.ampm(kin).squeeze(-1)                      # (B,)

        # apply AM/AM + AM/PM correction to the fast-linear signal
        rot = torch.polar(torch.ones_like(phase), phase)        # (B,) complex
        shaped = lin * gain.to(lin.dtype) * rot

        # residual short-memory polynomial nonlinearity
        res = self.gmp(xw)
        if res.dim() > 1:
            res = res[:, res.shape[1] // 2] if res.shape[1] > 1 else res.squeeze(-1)

        return shaped + res