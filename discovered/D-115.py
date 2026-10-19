import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear, SelfAttentionPrim, LayerNormPrim


class DPD(DPDModule):
    """Conformer-lite DPD: envelope-conditioned KAN core (proven) + a small
    self-attention-over-taps residual that only earns its cost at high drive.

    The attention branch reweights tap memory as a function of the whole
    window envelope trajectory, folded in as a near-zero-init residual gate
    around the identity FIR so seeds stay tight and it degrades to the KAN
    baseline if attention doesn't help.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 12, grid: int = 7,
                 order: int = 3, d_model: int = 8, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2

        # --- static AM/AM + AM/PM branch (parent core) ---
        self.kan1 = KANLayer(6, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # --- power scheduling: KAN(p) -> scalar gate on nonlinear correction ---
        self.pkan = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # --- static linear complex FIR (carries identity) ---
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # --- attention branch over the tap window ---
        # project real re/im per tap into d_model channels
        self.in_proj = RLinear(2, d_model, bias=True, quant_bits=quant_bits)
        self.ln = LayerNormPrim(d_model)
        self.attn = SelfAttentionPrim(d_model=d_model, window=window, heads=2)
        self.ln2 = LayerNormPrim(d_model)
        # collapse attended centre-tap features -> complex residual gain (I,Q)
        self.attn_head = RLinear(d_model, 2, bias=True, quant_bits=quant_bits)

        # --- memory-polynomial cross term ---
        self.mp = RLinear(5, 2, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            # identity linear path through centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0 / self.FEAT_SCALE
            # small nonlinear corrections
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            # power gate init -> approx constant 1
            self.pkan.coef.mul_(0.05)
            self.pkan.wb.mul_(0.05)
            # attention residual near zero -> falls back to KAN baseline
            self.attn_head.weight.mul_(0.02)
            self.attn_head.bias.zero_()
            self.mp.weight.mul_(0.05)
            self.mp.bias.zero_()

    def forward(self, xw, p):
        c = self.window // 2
        w = self.window
        B = xw.shape[0]

        xs = xw * self.FEAT_SCALE
        xc = xs[:, c]

        # --- free envelope features ---
        env = torch.abs(xs)                       # (B, window)
        therm = env.mean(dim=-1)

        idx_m = max(c - 1, 0)
        idx_p = min(c + 1, w - 1)
        idx_m2 = max(c - 2, 0)
        idx_p2 = min(c + 2, w - 1)

        r  = env[:, c]
        rm = env[:, idx_m]
        rp = env[:, idx_p]
        env_pow = env.pow(2).mean(dim=-1)

        # squash into spline support
        rf   = torch.tanh(r * 2.0)
        rmf  = torch.tanh(rm * 2.0)
        rpf  = torch.tanh(rp * 2.0)
        epf  = torch.tanh(env_pow * 3.0)
        tf   = torch.tanh(therm * 2.0)
        pf   = (p * 2.0 - 1.0)                     # safe at p=0 and p=1

        feat = torch.stack([rf, rmf, rpf, epf, tf, pf], dim=-1)  # (B,6)
        gain = self.kan2(self.kan1(feat))                       # (B,2)

        gate = 1.0 + self.pkan(pf.reshape(B, 1))[:, 0]          # (B,)

        gI = gain[:, 0] * gate
        gQ = gain[:, 1] * gate
        corr = torch.complex(gI, gQ)
        am_out = xc * corr

        # --- static linear FIR (identity path) ---
        fir_out = self.fir(xs)[:, 0]

        # --- attention branch over taps ---
        # per-tap real feature: [re, im], squashed into a bounded range
        re = torch.tanh(xs.real)                  # (B, window)
        im = torch.tanh(xs.imag)                  # (B, window)
        taps = torch.stack([re, im], dim=-1)      # (B, window, 2)
        h = self.in_proj(taps)                    # (B, window, d_model)
        h = self.ln(h)
        h = self.attn(h)                          # (B, d_model) collapsed
        h = self.ln2(h)
        hc = h                                     # (B, d_model) centre feature
        attn_gain = self.attn_head(hc)            # (B,2)
        # power-modulated residual gain around identity (near-zero init)
        ag = torch.complex(attn_gain[:, 0], attn_gain[:, 1])
        attn_out = xc * ag * gate.to(torch.complex64)

        # --- memory-polynomial cross term ---
        p2 = torch.stack([
            env[:, idx_m2] ** 2,
            env[:, idx_m] ** 2,
            env[:, c] ** 2,
            env[:, idx_p] ** 2,
            env[:, idx_p2] ** 2,
        ], dim=-1)                                # (B,5)
        mp = self.mp(p2)                          # (B,2)
        mp_c = torch.complex(mp[:, 0], mp[:, 1]) * xc

        out = fir_out + am_out + attn_out + mp_c

        return (out / self.FEAT_SCALE).to(torch.complex64)