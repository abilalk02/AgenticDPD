import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """Scaled envelope-conditioned KAN DPD for the 10^5.5-10^6.0 BOP niche.
       - complex FIR mix -> bounded envelope feature basis
       - smooth learned 1-D p-modulation branch (KAN on p) -> gain/phase bias
       - main KAN static AM/AM + AM/PM, p-biased, hard-squashed into grid
       - power-gated GMP complex memory-NL residual (bounded gate)
       Every spline input is squashed to [-1.5,1.5] for high-drive stability."""

    def __init__(self, window: int, mix: int = 8, kan_grid: int = 10,
                 kan_order: int = 3):
        super().__init__(window)
        self.window = window
        self.mixw = mix

        # ---- 1) complex FIR mix over the whole window -> feature basis ----
        self.mix = CLinear(window, mix, bias=False, quant_bits=8)
        with torch.no_grad():
            self.mix.wr.zero_(); self.mix.wi.zero_()
            # initialise first channel as centre-tap passthrough
            self.mix.wr[0, window // 2] = 1.0

        # ---- 2) smooth 1-D power modulation branch (KAN on p) ----
        # outputs per-channel gain-bias and a global phase-bias seed
        self.pmod = KANLayer(1, mix + 1, grid=6, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 3) main KAN: env-basis + p feature -> gain & phase corrections ----
        self.kan = KANLayer(mix + 1, mix + 1, grid=kan_grid, order=kan_order,
                           x_min=-1.5, x_max=1.5, quant_bits=8)

        # complex head: combine gated feature channels into centre output
        self.head = CLinear(mix, 1, bias=False, quant_bits=8)
        with torch.no_grad():
            self.head.wr.zero_(); self.head.wi.zero_()
            self.head.wr[0, 0] = 1.0

        # ---- 4) short GMP complex memory nonlinear residual ----
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=5, cross=2)
        # power gate: smooth quadratic-in-p gate (6-bit)
        self.pgate = RLinear(2, 1, bias=True, quant_bits=6)
        with torch.no_grad():
            self.pgate.weight.zero_(); self.pgate.bias.fill_(0.0)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2

        # ---- 1) complex FIR feature basis ----
        z = self.mix(xw)                       # complex (B, mix)
        env = z.abs()                          # real (B, mix), free

        # ---- power feature (safe at p=0 and p=1) ----
        pc = p.reshape(B, 1)                   # (B,1)
        pin = 1.5 * torch.tanh(2.0 * (pc - 0.5))   # squashed p feature

        # ---- 2) smooth learned 1-D p-modulation ----
        pmod = self.pmod(pin)                  # (B, mix+1)
        gain_bias = pmod[:, :self.mixw]        # (B, mix)
        phase_bias = pmod[:, self.mixw:self.mixw + 1]  # (B,1)

        # ---- 3) main KAN: static AM/AM + AM/PM, p-biased ----
        env_sq = 1.5 * torch.tanh(env / 1.5)   # squash env into grid
        feat = torch.cat([env_sq, pin], dim=1) # (B, mix+1)
        feat = 1.5 * torch.tanh(feat / 1.5)    # safety re-squash
        kout = self.kan(feat)                  # (B, mix+1)

        gain = 1.0 + torch.tanh(kout[:, :self.mixw] + gain_bias)   # (B, mix)
        phase = torch.tanh(kout[:, self.mixw:self.mixw + 1] + phase_bias)  # (B,1)

        cph = torch.cos(phase); sph = torch.sin(phase)
        rot = torch.complex(cph, sph)          # (B,1)

        # apply per-channel gain and shared phase rotation to feature basis
        zg = z * gain.to(z.dtype) * rot        # (B, mix)
        y_main = self.head(zg).squeeze(-1)     # complex (B,)

        # ---- 4) power-gated GMP complex memory-NL residual ----
        res = self.gmp(xw)                     # complex (B,) or (B,1)
        if res.dim() > 1:
            res = res.squeeze(-1)
        pf = torch.cat([pc, pc * pc], dim=1)   # (B,2)
        g = torch.sigmoid(self.pgate(pf).squeeze(-1))  # bounded (B,)
        res = res * g.to(res.dtype)

        y = y_main + res                       # complex (B,)
        return y