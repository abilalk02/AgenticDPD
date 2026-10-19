import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """VCSEL IM/DD-aware DPD, rev.2.

    Improvements over parent:
      * Richer static AM/AM: KAN over [r_-1, r_0, r_+1, env_power, thermal, p]
        -> wider hidden -> complex gain, so AM/AM + AM/PM is captured jointly
        with neighbour memory context.
      * Power scheduling: a compact KAN(p) produces a scalar gate that scales
        the nonlinear correction, letting the model sharpen compression at the
        high-drive ACLR bottleneck without hurting low-drive linearity.
      * Dynamic memory: an envelope-power-modulated complex FIR (memory
        polynomial in FIR form) augments the static identity FIR.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 12, grid: int = 7,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2

        # --- static AM/AM + AM/PM branch ---
        # inputs: r_-1, r_0, r_+1, env_power, thermal, p  (6 features)
        self.kan1 = KANLayer(6, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # --- power scheduling: KAN(p) -> scalar gate on nonlinear correction ---
        self.pkan = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # --- static linear complex FIR (carries identity) ---
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # --- dynamic (envelope-modulated) complex FIR ---
        self.fir_dyn = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # --- memory-polynomial cross term (recent |x|^2) ---
        self.mp = RLinear(5, 2, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            # identity linear path through centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0 / self.FEAT_SCALE
            # dynamic FIR starts near zero
            self.fir_dyn.wr.mul_(0.02)
            self.fir_dyn.wi.mul_(0.02)
            # small nonlinear corrections
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            # power gate init -> approx constant 1
            self.pkan.coef.mul_(0.05)
            self.pkan.wb.mul_(0.05)
            self.mp.weight.mul_(0.05)
            self.mp.bias.zero_()

    def forward(self, xw, p):
        c = self.window // 2
        w = self.window

        xs = xw * self.FEAT_SCALE
        xc = xs[:, c]

        # --- free envelope features ---
        r = torch.abs(xc)
        env = torch.abs(xs)                       # (B, window)
        therm = env.mean(dim=-1)

        idx_m = max(c - 1, 0)
        idx_p = min(c + 1, w - 1)
        idx_m2 = max(c - 2, 0)
        idx_p2 = min(c + 2, w - 1)

        rm = env[:, idx_m]
        rp = env[:, idx_p]
        env_pow = env.pow(2).mean(dim=-1)         # window energy proxy

        # squash into spline support
        rf   = torch.tanh(r * 2.0)
        rmf  = torch.tanh(rm * 2.0)
        rpf  = torch.tanh(rp * 2.0)
        epf  = torch.tanh(env_pow * 3.0)
        tf   = torch.tanh(therm * 2.0)
        pf   = (p * 2.0 - 1.0)

        feat = torch.stack([rf, rmf, rpf, epf, tf, pf], dim=-1)  # (B,6)

        gain = self.kan2(self.kan1(feat))                       # (B,2)

        # power-scheduled gate (around 1)
        gate = 1.0 + self.pkan(pf.unsqueeze(-1))[:, 0]          # (B,)

        gI = gain[:, 0] * gate
        gQ = gain[:, 1] * gate
        # gain applied as correction around identity
        corr = torch.complex(gI, gQ)
        am_out = xc * corr                                      # nonlinear residual

        # --- static linear FIR (identity path) ---
        fir_out = self.fir(xs)[:, 0]

        # --- dynamic envelope-modulated FIR ---
        # modulate window by local envelope power (free), then linear complex map
        mod = (env * env).to(xs.dtype)                          # (B, window) real->complex cast
        xs_mod = xs * mod
        fir_dyn_out = self.fir_dyn(xs_mod)[:, 0] * gate.to(torch.complex64)

        # --- memory-polynomial cross term ---
        p2 = torch.stack([
            env[:, idx_m2] ** 2,
            env[:, idx_m] ** 2,
            env[:, c] ** 2,
            env[:, idx_p] ** 2,
            env[:, idx_p2] ** 2,
        ], dim=-1)                                              # (B,5)
        mp = self.mp(p2)                                        # (B,2)
        mp_c = torch.complex(mp[:, 0], mp[:, 1]) * xc

        out = fir_out + am_out + fir_dyn_out + mp_c

        return (out / self.FEAT_SCALE).to(torch.complex64)