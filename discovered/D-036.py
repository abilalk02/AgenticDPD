import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import (
    CLinear, KANLayer, RLinear, LayerNormPrim,
)


class DPD(DPDModule):
    """Score-attention-over-taps DPD (conv-free).

    Complex FIR backbone (near-identity anchor) + per-tap real feature
    embedding to d_model + a parameter-free softmax attention pooling driven
    by a learned scalar score per tap (RLinear), fused with a bounded KAN
    envelope/power gain. Power p is injected both as a tap-feature channel
    and into the KAN, so one model spans the whole drive sweep.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, d_model: int = 8,
                 heads: int = 1, hidden: int = 8,
                 grid: int = 7, order: int = 3,
                 quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.window = window
        self.d_model = d_model

        # ---- complex FIR backbone (decimated taps, near-identity init) ----
        decim = 2
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # ---- per-tap real feature embedding -> d_model ----
        # features per tap: [re, im, |x|, |x|^2, p]
        self.feat_in = 5
        self.embed = RLinear(self.feat_in, d_model, bias=True,
                             quant_bits=quant_bits)

        # ---- attention scoring over tap sequence (parameter-free softmax) ----
        self.ln_a = LayerNormPrim(d_model)
        self.score = RLinear(d_model, 1, bias=True, quant_bits=quant_bits)

        # ---- position-wise FFN (pre-norm residual) ----
        self.ln_f = LayerNormPrim(d_model)
        self.ff1 = RLinear(d_model, 2 * d_model, bias=True,
                           quant_bits=quant_bits)
        self.ff2 = RLinear(2 * d_model, d_model, bias=True,
                           quant_bits=quant_bits)

        # ---- pooling to context ----
        self.ctx_dim = d_model
        self.pool = RLinear(d_model, self.ctx_dim, bias=True,
                            quant_bits=quant_bits)

        # ---- KAN nonlinearity on (env, p) ----
        self.kan1 = KANLayer(2, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # ---- head: context + kan -> complex gain + additive correction ----
        self.head = RLinear(self.ctx_dim + 2, 4, bias=True,
                            quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.head.weight.mul_(0.05)
            self.head.bias.zero_()
            self.pool.weight.mul_(0.5)
            self.pool.bias.zero_()

    def _build_features(self, x, p):
        re = x.real
        im = x.imag
        amp = torch.abs(x)
        amp2 = amp * amp
        pw = p.unsqueeze(-1).expand(-1, self.window)
        feat = torch.stack([re, im, amp, amp2, pw], dim=-1)  # (B, W, 5)
        return feat

    def forward(self, xw, p):
        x = xw * self.FEAT_SCALE
        B = x.shape[0]

        # linear backbone anchor
        taps = x[:, self.tap_idx]
        lin = self.fir(taps)[:, 0]

        # tap feature embedding
        feat = self._build_features(x, p)               # (B, W, 5)
        h = self.embed(feat)                            # (B, W, d_model)

        # attention block (pre-norm residual, softmax over taps)
        hn = self.ln_a(h)                               # (B, W, d_model)
        s = self.score(hn)[:, :, 0]                     # (B, W)
        w = torch.softmax(s, dim=1).unsqueeze(-1)       # (B, W, 1)
        a = h * w                                        # weighted (B, W, d_model)
        h = h + a

        # FFN (pre-norm residual)
        f = self.ff2(torch.nn.functional.silu(self.ff1(self.ln_f(h))))
        h = h + f                                        # (B, W, d_model)

        # attention-pooled context (parameter-free sum over taps)
        pooled = (h * w).sum(dim=1)                      # (B, d_model)
        ctx = self.pool(pooled)                          # (B, ctx_dim)

        # KAN envelope/power gain
        xc = x[:, self.c]
        env_s = torch.tanh(torch.abs(xc) * 2.0)
        kfeat = torch.stack([env_s, p], dim=-1)
        kg = self.kan2(self.kan1(kfeat))                # (B, 2)

        # fuse -> complex gain + additive correction
        hin = torch.cat([ctx, kg], dim=-1)
        out = self.head(hin)
        gr = out[:, 0]
        gi = out[:, 1]
        cr = out[:, 2]
        ci = out[:, 3]

        gain = torch.complex(1.0 + gr, gi)
        corr = torch.complex(cr, ci)

        o = (lin * gain + corr) / self.FEAT_SCALE
        return o.to(torch.complex64)