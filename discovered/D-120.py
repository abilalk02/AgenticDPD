import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """BOp-refined KAN DPD (g003c1 lineage).

    Same decimated complex-FIR backbone + envelope-only KAN gain as parent,
    but mixed-precision fake-quant: FIR at 16-bit, KAN layers at 8-bit to cut
    BOp ~3x while holding the 3 dBFS ACLR (the KAN only emits a bounded
    gain-near-1 correction, so 8-bit quant barely perturbs it).
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2, hidden: int = 8,
                 grid: int = 7, order: int = 3,
                 fir_bits: int = 16, kan_bits: int = 8):
        super().__init__(window)
        c = window // 2
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        # complex FIR over decimated taps (16-bit): carries linear fidelity
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=fir_bits)

        # envelope-only nonlinearity (8-bit): (|x|, p) -> hidden -> (gr, gi)
        self.kan1 = KANLayer(2, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=kan_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=kan_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                       # (B, window) complex
        taps = x[:, self.tap_idx]                      # (B, n_taps) complex
        lin = self.fir(taps)[:, 0]                     # (B,) complex

        xc = x[:, self.c]                              # centre sample (complex)
        env = torch.abs(xc)                            # (B,) real
        env_s = torch.tanh(env * 2.0)                  # squashed into grid
        pc = p.reshape(B).to(env_s.dtype)              # (B,) safe at 0 and 1
        feat = torch.stack([env_s, pc], dim=-1)        # (B, 2)

        g = self.kan2(self.kan1(feat))                 # (B, 2) real
        gain = torch.complex(1.0 + g[:, 0], g[:, 1])   # gain near 1

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)                   # (B,)