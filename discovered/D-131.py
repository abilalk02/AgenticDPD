import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear


class DPD(DPDModule):
    """KAN-backbone power-aware DPD sized for the 6-9k RMpS niche.
    Lineage: g005c2/g003c1-style envelope-conditioned KAN gate that
    clears the 3 dBFS ceiling natively. Spline inputs squashed into
    grid support for tight seeds at high drive."""

    def __init__(self, window: int, taps: int = 9, lat: int = 16,
                 grid: int = 10, order: int = 3):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.lat = lat
        self.sq = 1.3  # squash edge, inside [-1.5,1.5] grid

        # Dominant linear complex FIR
        self.fir = CLinear(self.taps, 1, bias=False)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # Complex tap mixer -> latent for envelope features
        self.mix = CLinear(self.taps, lat, bias=True)

        # Smooth learned 1-D function of p (real) -> 2 conditioning feats
        self.pmap = RLinear(2, 2, bias=True)
        with torch.no_grad():
            self.pmap.weight.zero_(); self.pmap.bias.zero_()

        # KAN backbone: [env(lat) + p-feats(2)] -> lat*2  (gate + nl weights)
        self.kan = KANLayer(d_in=lat + 2, d_out=2 * lat, grid=grid,
                            order=order, x_min=-1.5, x_max=1.5)

        # Complex heads: gate applied to latent, then reduce to output
        self.gate_head = CLinear(lat, 1, bias=False)
        self.nl_head = CLinear(lat, 1, bias=False)
        with torch.no_grad():
            self.gate_head.wr.zero_(); self.gate_head.wi.zero_()
            self.nl_head.wr.zero_(); self.nl_head.wi.zero_()

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        h = self.taps // 2

        xc = xw[:, c - h: c + h + 1]              # (B, taps) complex

        # Linear dominant path
        lin = self.fir(xc).squeeze(-1)           # (B,) complex

        # Latent complex mix
        z = self.mix(xc)                         # (B, lat) complex
        env = z.abs()                            # (B, lat) real, free

        # Power conditioning (safe at p=0,1)
        pc = p.reshape(B, 1)
        pf = torch.cat([pc, pc * pc], dim=1)     # (B,2)
        pfeat = self.pmap(pf)                    # (B,2)

        feat = torch.cat([env, pfeat], dim=1)    # (B, lat+2)
        feat = self.sq * torch.tanh(feat / self.sq)  # squash into grid

        k = self.kan(feat)                       # (B, 2*lat) real
        g_gate = k[:, :self.lat]
        g_nl = k[:, self.lat:]

        # Power-modulated complex gate on latent
        gate = torch.sigmoid(g_gate).to(z.dtype)
        gated = self.gate_head(z * gate).squeeze(-1)   # (B,) complex

        # Nonlinear complex correction weighted by KAN output
        nlw = torch.tanh(g_nl).to(z.dtype)
        nl = self.nl_head(z * nlw).squeeze(-1)         # (B,) complex

        y = lin + gated + nl
        return y.to(torch.complex64)