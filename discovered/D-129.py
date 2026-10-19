import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """6-9k RMpS niche fill: grown envelope-conditioned power-modulated KAN.

    Lineage: g003c1/g005c2-style spline core (natively clears 3 dBFS ceiling),
    scaled UP via a wide KAN (the archive-approved way to grow). An identity-
    init complex FIR over a full tap window gives the linear/memory estimate;
    a wide tanh-squashed real feature basis (env powers, memory env, centred p,
    env*p cross terms) drives a large KAN producing a bounded (1+delta) complex
    gain plus a small additive residual. p enters only via centred/cross terms
    -> safe at p==0 and p==1. All spline inputs squashed into [-1,1].

    RMpS budget (~6.9k):
      fir  CLinear(n_taps->1)  : 4*n_taps
      kan1 12->24 grid6 order3 : 12*24*(2*9+1)=5472
      kan2 12->8  grid6 order3 : 12*8*(2*9+1)=1824
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, grid: int = 6, order: int = 3,
                 quant_bits: int = 8):
        super().__init__(window)
        c = window // 2
        self.c = c

        # full-ish tap set for the linear/memory estimate: centre +/- 5
        lo = max(0, c - 5)
        hi = min(window, c + 6)
        idx = sorted(set(list(range(lo, hi)) + [c]))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.cpos = idx.index(c)

        # memory window for envelope averaging
        mlo = max(0, c - 4)
        mhi = min(window, c + 5)
        self.register_buffer("mem_idx",
                             torch.tensor(list(range(mlo, mhi)), dtype=torch.long))

        # complex FIR -> centre-sample linear/memory estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=8)

        # feature basis dim
        self.n_feat = 12

        # wide KAN -> gain field (24 real channels)
        self.kan1 = KANLayer(self.n_feat, 24, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        # smaller KAN -> reduce to (gain_re, gain_im, bias_re, bias_im) x2 groups
        self.kan2 = KANLayer(self.n_feat, 8, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # cheap linear reducers (real) to fold the wide gain field to 4 outputs
        # kan1(24)->4 done via free reshape+mean grouping; keep it primitive-free
        # actually reduce via averaging groups (parameter-free)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, self.cpos] = 1.0
            self.kan1.coef.mul_(0.05)
            self.kan1.wb.mul_(0.05)
            self.kan2.coef.mul_(0.05)
            self.kan2.wb.mul_(0.05)

    def _features(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE
        taps = x[:, self.tap_idx]                             # (B, n_taps) cplx
        lin = self.fir(taps)[:, 0]                            # (B,) cplx

        xc = x[:, self.c]
        env = torch.abs(xc)
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)

        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e4 = torch.tanh((env ** 4) * 5.0)
        e5 = torch.tanh((env ** 5) * 6.0)
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env * mem_env * 3.0)
        pc = torch.clamp(p * 2.0 - 1.0, -1.0, 1.0)
        ep = torch.tanh(env * p * 2.5)
        e2p = torch.tanh(env * env * p * 3.0)
        e3p = torch.tanh(env * env * env * p * 3.5)
        mp = torch.tanh(mem_env * p * 2.5)

        feat = torch.stack([e, e2, e3, e4, e5, m, m2, pc, ep, e2p, e3p, mp],
                           dim=-1)                            # (B, 12)
        return lin, feat

    def forward(self, xw, p):
        B = xw.shape[0]
        lin, feat = self._features(xw, p)                    # (B,) cplx, (B,12)

        k1 = self.kan1(feat)                                 # (B, 24)
        k2 = self.kan2(feat)                                 # (B, 8)

        # fold wide gain field -> 4 modulation channels (parameter-free mean over groups)
        g = k1.reshape(B, 4, 6).mean(dim=-1)                 # (B, 4)
        # combine with kan2 first 4 channels (residual detail)
        g = g + 0.5 * k2[:, :4]

        dr = 0.5 * torch.tanh(g[:, 0])
        di = 0.5 * torch.tanh(g[:, 1])
        gain = torch.complex(1.0 + dr, di)                   # (B,) cplx

        # additive residual bias from remaining kan2 channels
        br = 0.15 * torch.tanh(g[:, 2] + k2[:, 4])
        bi = 0.15 * torch.tanh(g[:, 3] + k2[:, 5])
        bias = torch.complex(br, bi)                         # (B,) cplx

        o = (lin * gain + bias * self.FEAT_SCALE) / self.FEAT_SCALE
        return o.to(torch.complex64)                         # (B,)