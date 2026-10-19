import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, SelfAttentionPrim, LayerNormPrim


class DPD(DPDModule):
    """Conformer-lite DPD: depthwise complex FIR + single-head self-attention
    over the tap window (memory mixing) + power-modulated odd-order GMP
    nonlinear correction. One model covers the full power sweep via FiLM
    conditioning on p and a learned nonlinear gain."""

    def __init__(self, window: int, awin: int = 7, ntaps: int = 5, d_model: int = 4):
        super().__init__(window)
        self.awin = min(awin, window) | 1
        self.ntaps = min(ntaps, window) | 1
        self.d_model = d_model

        # ---- Linear complex FIR (dominant response) ----
        self.taps = self.awin
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- Attention embedding: map (re, im, |x|^2) per tap -> d_model ----
        self.embed = RLinear(3, d_model, bias=True)
        self.ln_in = LayerNormPrim(d_model)
        self.attn = SelfAttentionPrim(d_model, self.awin, heads=1)
        self.ln_out = LayerNormPrim(d_model)

        # FiLM conditioning of attention output by p: scale & shift
        self.film = RLinear(2, 2 * d_model, bias=True)
        with torch.no_grad():
            self.film.weight.zero_()
            self.film.bias.zero_()  # -> scale offset handled below (1+gamma)

        # Project attended centre-tap features -> complex correction (2 reals)
        self.attn_out = RLinear(d_model, 2, bias=False)
        with torch.no_grad():
            self.attn_out.weight.zero_()

        # ---- Nonlinear GMP-style branch (odd orders) ----
        self.nl = CLinear(2 * self.ntaps, 1, bias=False)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()

        # Power gain on nonlinear branch
        self.pgain = RLinear(2, 1, bias=True)
        with torch.no_grad():
            self.pgain.weight.zero_()
            self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        c = self.window // 2
        B = xw.shape[0]

        # ----- Linear path -----
        h = self.awin // 2
        xa = xw[:, c - h : c + h + 1]                     # (B, awin) complex
        lin = self.fir(xa).squeeze(-1)

        # ----- Attention path -----
        a2 = xa.real * xa.real + xa.imag * xa.imag         # (B, awin)
        # per-tap feature vector (parameter-free construction)
        feat = torch.stack([xa.real, xa.imag, a2], dim=-1)  # (B, awin, 3)
        emb = self.embed(feat)                              # (B, awin, d)
        emb = self.ln_in(emb)
        att = self.attn(emb)                                # (B, awin, d)
        att = self.ln_out(att)

        # FiLM conditioning from p
        pf = torch.stack([p, p * p], dim=1)                 # (B, 2)
        film = self.film(pf)                                # (B, 2d)
        gamma, beta = film[:, : self.d_model], film[:, self.d_model :]
        gamma = (1.0 + torch.tanh(gamma)).unsqueeze(1)      # (B,1,d)
        beta = beta.unsqueeze(1)
        att = att * gamma + beta

        centre = att[:, self.awin // 2, :]                  # (B, d)
        corr = self.attn_out(centre)                        # (B, 2)
        attn_corr = torch.complex(corr[:, 0], corr[:, 1])   # (B,) complex

        # ----- Nonlinear GMP branch -----
        hn = self.ntaps // 2
        xn = xw[:, c - hn : c + hn + 1]                     # (B, ntaps) complex
        n2 = xn.real * xn.real + xn.imag * xn.imag
        t3 = xn * n2
        t5 = xn * n2 * n2
        feats = torch.cat([t3, t5], dim=1)                  # (B, 2*ntaps) complex
        nl = self.nl(feats).squeeze(-1)

        g = self.pgain(pf).squeeze(-1)                      # (B,) real
        nl = nl * g.to(nl.dtype)

        return lin + attn_corr.to(lin.dtype) + nl