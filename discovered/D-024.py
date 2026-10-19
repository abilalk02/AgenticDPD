import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import (
    CLinear, RLinear, KANLayer, SelfAttentionPrim, LayerNormPrim
)


class DPD(DPDModule):
    """Attention-augmented DPD:
       linear complex FIR backbone (bandwidth memory)
       + KAN static AM/AM-PM (bounded splines, p-biased operating point)
       + self-attention residual over the tap window (data-dependent memory weighting).
       One model linearizes the whole power sweep via p-conditioned gains."""

    def __init__(self, window: int, taps: int = 9, awin: int = 7,
                 d_model: int = 4, heads: int = 2,
                 kan_grid: int = 8, kan_order: int = 3):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.awin = min(awin, window) | 1
        self.d_model = d_model

        # ---- 1) Linear complex FIR backbone ----
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) Static compressive AM/AM + AM/PM via KAN (splines), p-biased ----
        self.amam = KANLayer(2, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5)

        # ---- 3) Attention residual branch ----
        # Feature-embed each tap: [real, imag, env, env^2] -> d_model (8-bit)
        self.embed = RLinear(4, d_model, bias=True, quant_bits=8)
        self.ln = LayerNormPrim(d_model)
        self.attn = SelfAttentionPrim(d_model, self.awin, heads=heads)
        # Collapse attended centre representation -> complex residual (2 reals)
        self.proj = RLinear(d_model, 2, bias=True, quant_bits=8)
        with torch.no_grad():
            self.proj.weight.zero_(); self.proj.bias.zero_()

        # ---- 4) Power-conditioned gains for AM/AM and attention residual ----
        self.pgain = RLinear(2, 2, bias=True)  # -> [gain_amam, gain_attn]
        with torch.no_grad():
            self.pgain.weight.zero_(); self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        c = self.window // 2
        h = self.taps // 2
        ha = self.awin // 2

        xc = xw[:, c]
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt()  # (B, window)

        # ---- 1) Linear FIR ----
        lin = self.fir(xw[:, c - h : c + h + 1]).squeeze(-1)

        # ---- power features ----
        pf = torch.stack([p, 2.0 * (p - 0.5)], dim=1)
        pg = self.pgain(pf)                      # (B,2)
        g_amam = 1.0 + torch.tanh(pg[:, 0])      # bounded
        g_attn = torch.tanh(pg[:, 1])            # small residual scale

        # ---- 2) Static AM/AM-PM via KAN, biased by p ----
        ec = env[:, c]
        ein = 1.5 * torch.tanh(ec)
        pin = 1.5 * torch.tanh(2.0 * (p - 0.5))
        kout = self.amam(torch.stack([ein, pin], dim=1))   # (B,2)
        gain = 1.0 + torch.tanh(kout[:, 0])
        phase = torch.tanh(kout[:, 1])
        rot = torch.complex(torch.cos(phase), torch.sin(phase))
        amam_out = xc * (gain * g_amam).to(xc.dtype) * rot

        # ---- 3) Attention residual over tap window ----
        xa = xw[:, c - ha : c + ha + 1]                      # (B, awin) complex
        ea = env[:, c - ha : c + ha + 1]                     # (B, awin)
        # per-tap features, squashed for stability
        feat = torch.stack([
            torch.tanh(xa.real),
            torch.tanh(xa.imag),
            torch.tanh(ea),
            torch.tanh(ea * ea),
        ], dim=-1)                                           # (B, awin, 4)
        B, W, _ = feat.shape
        emb = self.embed(feat.reshape(B * W, 4)).reshape(B, W, self.d_model)
        emb = self.ln(emb)
        att = self.attn(emb)                                 # (B, d_model)
        if att.dim() == 3:
            att = att[:, ha, :]                              # fallback if sequence returned
        res = self.proj(att)                                 # (B, 2)
        res_c = torch.complex(res[:, 0], res[:, 1])
        res_c = res_c * g_attn.to(res_c.dtype)

        return lin + amam_out + res_c