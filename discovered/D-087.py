import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Ultra-low-BOP power-modulated KAN residual DPD.

    Aims at the EMPTY ultra-low niche (10^4.0-10^4.5 BOP, ~130-160 RMpS).
    An identity-initialised complex FIR supplies the dominant linear/memory
    estimate (no high-drive collapse, tight seeds). A single compact 4-bit
    KAN bends a lean, fully tanh-squashed real feature basis (envelope powers
    incl. deep-drive env^5, mem envelope, centred p and env*p cross terms)
    into a bounded (1+delta) complex gain plus a tiny additive residual.
    p enters only via centred / cross terms -> safe at p=0 and p=1. 4-bit
    quantisation drops BOP into the empty ultra-low bin at unchanged RMpS.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, grid: int = 6, order: int = 3,
                 quant_bits: int = 4):
        super().__init__(window)
        c = window // 2
        self.c = c

        # compact tap set: centre +/- 2 for the linear/memory estimate
        lo = max(0, c - 2)
        hi = min(window, c + 3)
        idx = list(range(lo, hi))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)

        mlo = max(0, c - 2)
        mhi = min(window, c + 3)
        self.register_buffer("mem_idx",
                             torch.tensor(list(range(mlo, mhi)), dtype=torch.long))

        # complex FIR -> centre-sample linear/memory estimate (8-bit, cheap)
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=8)

        # lean feature basis: [e, e2, e3, e5, m, pc, ep, e2p]
        self.n_feat = 8

        # single KAN -> (gain_re, gain_im, bias_re, bias_im), 4-bit
        self.kan = KANLayer(self.n_feat, 4, grid, order,
                            x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # near-identity start: tiny KAN perturbation
            self.kan.coef.mul_(0.05)
            self.kan.wb.mul_(0.05)

    def _features(self, xw, p):
        x = xw * self.FEAT_SCALE                              # (B, window)
        taps = x[:, self.tap_idx]                             # (B, n_taps)
        lin = self.fir(taps)[:, 0]                            # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                                   # (B,)
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e5 = torch.tanh((env ** 5) * 6.0)                     # deep-drive
        m = torch.tanh(mem_env * 2.0)
        pc = p * 2.0 - 1.0                                    # in [-1,1]
        ep = torch.tanh(env * p * 2.5)                        # env*p cross
        e2p = torch.tanh(env * env * p * 3.0)                 # env^2*p cross

        feat = torch.stack([e, e2, e3, e5, m, pc, ep, e2p], dim=-1)
        return lin, feat

    def forward(self, xw, p):
        lin, feat = self._features(xw, p)                     # (B,) cplx, (B,8)

        k = self.kan(feat)                                    # (B, 4)
        dr = 0.5 * torch.tanh(k[:, 0])
        di = 0.5 * torch.tanh(k[:, 1])
        gain = torch.complex(1.0 + dr, di)                    # (B,) complex

        br = 0.15 * torch.tanh(k[:, 2])
        bi = 0.15 * torch.tanh(k[:, 3])
        bias = torch.complex(br, bi)                          # (B,) complex

        o = (lin * gain + bias * self.FEAT_SCALE) / self.FEAT_SCALE
        return o.to(torch.complex64)                          # (B,)