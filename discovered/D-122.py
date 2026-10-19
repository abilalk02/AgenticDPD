import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """6-9k RMpS wide power-conditioned KAN-residual DPD (g005c2/g003c1 lineage).

    A full-tap complex FIR (CLinear) forms the linear/memory estimate and a set
    of complex latent channels. A WIDE high-grid KAN bends a rich, bounded real
    feature vector (per-channel envelopes, envelope powers, memory envelope, and
    power-conditioning terms) into a complex correction gain (1+delta) applied to
    the linear estimate, plus a small additive complex residual. Linear path
    dominates so high-drive is stable across seeds. Power p enters via centered
    and cross terms only (safe at 0 and 1). All spline inputs tanh-squashed into
    the grid support.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int,
                 nlat: int = 8, grid: int = 10, order: int = 3,
                 quant_bits: int = 8):
        super().__init__(window)
        c = window // 2
        self.c = c

        # memory window around centre for a memory-envelope feature
        lo = max(0, c - 3)
        hi = min(window, c + 4)
        self.register_buffer("mem_idx",
                             torch.tensor(list(range(lo, hi)), dtype=torch.long))

        self.nlat = nlat

        # complex FIR: window taps -> nlat complex latent channels
        # channel 0 is the primary linear/memory estimate (init identity-ish)
        self.mix = CLinear(window, nlat, bias=False, quant_bits=quant_bits)

        # feature dim: per-channel env (nlat) + env^2 (nlat) + [pc, ep, mem]
        self.n_extra = 3
        d_in = 2 * nlat + self.n_extra           # e.g. 8+8+3 = 19

        # wide, high-grid KAN backbone -> correction (gain re,im + resid re,im)
        d_out = 2 * nlat
        self.kan = KANLayer(d_in, d_out, grid, order,
                            x_min=-1.2, x_max=1.2, quant_bits=quant_bits)

        # head combines KAN output channels into the complex correction/resid
        # split kan output into gain part and residual part via two CLinears
        self.gain_head = CLinear(nlat, 1, bias=True, quant_bits=quant_bits)
        self.res_head = CLinear(nlat, 1, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            # init mix so channel 0 ~ centre-sample passthrough
            self.mix.wr.zero_()
            self.mix.wi.zero_()
            self.mix.wr[0, c] = 1.0
            # small random spread on other channels for feature diversity
            self.mix.wr[1:].normal_(0.0, 0.05)
            self.mix.wi[1:].normal_(0.0, 0.05)

            # start KAN near-zero so correction begins ~identity
            self.kan.coef.mul_(0.03)
            self.kan.wb.mul_(0.03)

            # gain head ~ zero delta (identity), residual head ~ zero
            self.gain_head.wr.mul_(0.05)
            self.gain_head.wi.mul_(0.05)
            if hasattr(self.gain_head, "br") and self.gain_head.br is not None:
                self.gain_head.br.zero_()
                self.gain_head.bi.zero_()
            self.res_head.wr.mul_(0.05)
            self.res_head.wi.mul_(0.05)
            if hasattr(self.res_head, "br") and self.res_head.br is not None:
                self.res_head.br.zero_()
                self.res_head.bi.zero_()

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)

        z = self.mix(x)                                   # (B, nlat) complex
        lin = z[:, 0]                                     # (B,) complex primary

        # ---- bounded real feature construction (all free ops) ----
        env = z.abs()                                     # (B, nlat)
        e = torch.tanh(env * 2.0)                         # (B, nlat)
        e2 = torch.tanh(env * env * 3.0)                  # (B, nlat)

        mem_env = x[:, self.mem_idx].abs().mean(dim=-1)   # (B,)
        m = torch.tanh(mem_env * 2.0)                     # (B,)

        pc = (p * 2.0 - 1.0)                              # (B,) in [-1,1]
        ep = torch.tanh(env[:, 0] * p * 2.5)             # (B,) env*p cross

        extra = torch.stack([pc, ep, m], dim=-1)         # (B, 3)
        feat = torch.cat([e, e2, extra], dim=-1)         # (B, d_in)
        feat = 1.2 * torch.tanh(feat / 1.2)              # keep inside grid

        g = self.kan(feat)                               # (B, 2*nlat) real

        gain_feat = g[:, :self.nlat]                     # (B, nlat)
        res_feat = g[:, self.nlat:]                      # (B, nlat)

        # complex-ify for the CLinear heads (feed real part; imag=0)
        gc = torch.complex(gain_feat, torch.zeros_like(gain_feat))
        rc = torch.complex(res_feat, torch.zeros_like(res_feat))

        dgain = self.gain_head(gc)[:, 0]                  # (B,) complex
        dres = self.res_head(rc)[:, 0]                    # (B,) complex

        # bounded correction gain = 1 + delta
        dr = 0.5 * torch.tanh(dgain.real)
        di = 0.5 * torch.tanh(dgain.imag)
        gain = torch.complex(1.0 + dr, di)               # (B,) complex

        # bounded small additive residual
        rr = 0.3 * torch.tanh(dres.real)
        ri = 0.3 * torch.tanh(dres.imag)
        resid = torch.complex(rr, ri)                    # (B,) complex

        o = (lin * gain + resid) / self.FEAT_SCALE
        return o.to(torch.complex64)                     # (B,)