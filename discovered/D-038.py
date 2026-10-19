import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Enriched ultra-low-complexity DPD.
       - 8-bit complex FIR backbone (wider bandwidth memory)
       - p-conditioned bounded-spline KAN static AM/AM & AM/PM, whose shape is
         modulated by a small learned 1-D function of drive power p
       - short 3-order GMP complex memory-NL residual, power-gated (quadratic in p).
       One model linearizes the whole sweep; p enters as KAN feature, KAN-shaped
       modulator, and residual gate."""

    def __init__(self, window: int, taps: int = 13, ntaps: int = 5,
                 kan_grid: int = 8, kan_order: int = 3):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1

        # ---- 1) Linear complex FIR: wider bandwidth-limited memory (8-bit) ----
        self.fir = CLinear(self.taps, 1, bias=False, quant_bits=8)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 1b) Learned 1-D function of p that modulates the KAN inputs ----
        # small KAN: p -> (env-scale, p-embed) shaping factors  (6-bit)
        self.pshape = KANLayer(1, 2, grid=6, order=kan_order,
                               x_min=-1.5, x_max=1.5, quant_bits=6)

        # ---- 2) KAN static AM/AM + AM/PM ----
        # inputs: centre |x|, |x|^2, drive p, mean-env thermal descriptor
        # outputs: gain corr, phase corr  (bounded splines -> graceful high-drive)
        self.amam = KANLayer(4, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 3) Short GMP complex memory nonlinear residual (orders 3,5,7) ----
        self.nl = CLinear(3 * self.ntaps, 1, bias=False, quant_bits=6)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()
        # power gate: learned quadratic function of p (8-bit)
        self.pgain = RLinear(2, 1, bias=True, quant_bits=8)
        with torch.no_grad():
            self.pgain.weight.zero_(); self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2

        xc = xw[:, c]                                          # centre sample
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt()   # (B, window)

        # ---- 1) Linear FIR path ----
        lin = self.fir(xw[:, c - h: c + h + 1]).squeeze(-1)

        # ---- free mean-envelope thermal descriptor ----
        tstate = torch.tanh(env.mean(dim=1))                   # (B,)

        # ---- 1b) learned 1-D shaping from p ----
        pin_raw = 1.5 * torch.tanh(2.0 * (p - 0.5))            # (B,)
        ps = self.pshape(pin_raw.unsqueeze(1))                 # (B,2)
        env_scale = 1.0 + 0.5 * torch.tanh(ps[:, 0])           # modulate env sens.
        p_embed = torch.tanh(ps[:, 1])                         # power embedding

        # ---- 2) Static AM/AM + AM/PM via KAN ----
        ec = env[:, c]
        ein = 1.5 * torch.tanh(ec * env_scale)
        e2in = 1.5 * torch.tanh(ec * ec)
        pin = 1.5 * p_embed
        tin = 1.5 * tstate
        kin = torch.stack([ein, e2in, pin, tin], dim=1)        # (B,4)
        kout = self.amam(kin)                                  # (B,2)
        gain = 1.0 + torch.tanh(kout[:, 0])                    # AM/AM gain
        phase = torch.tanh(kout[:, 1])                         # AM/PM correction

        cph = torch.cos(phase); sph = torch.sin(phase)
        rot = torch.complex(cph, sph)
        amam_out = xc * gain.to(xc.dtype) * rot

        # ---- 3) Short GMP memory nonlinear residual, power-gated ----
        xn = xw[:, c - hn: c + hn + 1]
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = xn * a2 * a2
        t7 = xn * a2 * a2 * a2
        feats = torch.cat([t3, t5, t7], dim=1)                 # (B, 3*ntaps)
        nl = self.nl(feats).squeeze(-1)
        pf = torch.stack([p, p * p], dim=1)
        g = self.pgain(pf).squeeze(-1)
        nl = nl * g.to(nl.dtype)

        return lin + amam_out + nl