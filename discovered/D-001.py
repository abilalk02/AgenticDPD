import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import GMPBranch, KANLayer, RLinear, CLinear


class DPD(DPDModule):
    """Quantized power-aware GMP + KAN correction.

    Backbone: complex GMPBranch (power-agnostic polynomial).
    Correction: parameter-free envelope features + normalized drive power p
    -> 8-bit KANLayer -> 8-bit RLinear -> complex gain via 16-bit CLinear,
    applied multiplicatively (and additively) to the GMP output.
    """

    def __init__(self, window: int, orders=(1, 3, 5, 7), mem: int = 5, cross: int = 2):
        super().__init__(window)
        self.window = window
        self.c = window // 2

        # complex polynomial backbone
        self.gmp = GMPBranch(window, orders, mem, cross)

        # ---- parameter-free feature vector size ----
        # per-tap magnitude (window) + centre powers (3) + p and p^2 (2)
        self.n_feat = window + 3 + 2

        # KAN nonlinear feature map (8-bit)
        self.kan = KANLayer(self.n_feat, 8, grid=6, order=3,
                            x_min=-1.5, x_max=1.5, quant_bits=8)

        # real mixing MLP (8-bit)
        self.mix = RLinear(8, 8, bias=True, quant_bits=8)

        # complex correction head (16-bit for phase fidelity)
        # input: 2 complex values -> output: gain + additive term
        self.head = CLinear(2, 2, bias=True, quant_bits=16)

    def _features(self, xw, p):
        mag = torch.abs(xw)                          # (B, window)
        c = xw[:, self.c]
        cmag = torch.abs(c)
        pow2 = cmag * cmag
        pow3 = pow2 * cmag
        centre = torch.stack([cmag, pow2, pow3], dim=-1)  # (B,3)
        pw = p.reshape(-1, 1)
        pfeat = torch.cat([pw, pw * pw], dim=-1)      # (B,2)
        f = torch.cat([mag, centre, pfeat], dim=-1)   # (B, n_feat)
        # squash into KAN spline support
        f = 1.4 * torch.tanh(f / 1.4)
        return f

    def forward(self, xw, p):
        y = self.gmp(xw)                              # (B,) complex

        f = self._features(xw, p)
        h = self.kan(f)                               # (B,8)
        h = torch.tanh(self.mix(torch.tanh(h)))       # (B,8)

        # build a 4-dim real vector -> pack into 2 complex inputs for CLinear
        # split h into two halves, use as real/imag of a 2-vector
        hr = h[:, :2]
        hi = h[:, 2:4]
        cin = torch.complex(hr, hi)                   # (B,2) complex64

        out = self.head(cin)                          # (B,2) complex
        gain = out[:, 0]
        offset = out[:, 1]

        # power-weighted correction: gain modulates, offset adds
        g = 1.0 + p.to(y.dtype) * gain
        corr = y * g + p.to(y.dtype) * offset
        return corr