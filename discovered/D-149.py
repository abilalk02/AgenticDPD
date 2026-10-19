import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """g005c2-core dual-KAN residual DPD scaled into the 10^6.5-10^7.0 BOP niche.

    Replaces the 3 dBFS-capped g015c3 entry (ACLR 32.6) with a lineage that
    natively clears the high-drive ceiling. Identity-init complex FIR carries
    the linear/memory estimate (no seed diverges); a wide float32 gain-KAN bends
    a rich squashed feature basis into a bounded (1+delta) complex multiplicative
    correction, and a bias-KAN adds a small envelope-shaped complex residual.
    A hidden complex memory-mix increases capacity. All spline inputs are
    tanh-squashed into [-1,1]; p enters only via centred/cross terms.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, grid: int = 12, order: int = 3,
                 quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c

        idx = list(range(window))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)

        lo = max(0, c - 4)
        hi = min(window, c + 5)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # complex FIR -> centre-sample linear/memory estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # hidden complex memory mix (extra capacity, no activation-matmul)
        self.mem_mix = CLinear(len(mem_idx), 4, bias=True, quant_bits=quant_bits)

        # feature basis: [e,e2,e3,e5,m,m2,pc,ep,mp,e2p, hr, hi] -> 12
        self.n_feat = 12

        self.kan_gain = KANLayer(self.n_feat, 2, grid, order,
                                 x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan_bias = KANLayer(self.n_feat, 2, grid, order,
                                 x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            self.kan_gain.coef.mul_(0.05)
            self.kan_gain.wb.mul_(0.05)
            self.kan_bias.coef.mul_(0.02)
            self.kan_bias.wb.mul_(0.02)

    def _features(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE
        taps = x[:, self.tap_idx]
        lin = self.fir(taps)[:, 0]                        # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)
        mem = x[:, self.mem_idx]                          # (B, M) complex
        mem_env = torch.abs(mem).mean(dim=-1)             # (B,)

        h = self.mem_mix(mem)                             # (B,4) complex
        hmag = torch.abs(h).mean(dim=-1)                  # (B,)
        hph = torch.tanh(h.real.mean(dim=-1) * 2.0)       # (B,)

        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e5 = torch.tanh((env ** 5) * 6.0)
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env * mem_env * 3.0)
        pc = p * 2.0 - 1.0
        ep = torch.tanh(env * p * 2.5)
        mp = torch.tanh(mem_env * p * 2.5)
        e2p = torch.tanh(env * env * p * 3.0)
        hr = torch.tanh(hmag * 2.0)
        hi = hph

        feat = torch.stack(
            [e, e2, e3, e5, m, m2, pc, ep, mp, e2p, hr, hi], dim=-1)
        return lin, feat

    def forward(self, xw, p):
        lin, feat = self._features(xw, p)

        g = self.kan_gain(feat)
        dr = 0.5 * torch.tanh(g[:, 0])
        di = 0.5 * torch.tanh(g[:, 1])
        gain = torch.complex(1.0 + dr, di)

        b = self.kan_bias(feat)
        br = 0.15 * torch.tanh(b[:, 0])
        bi = 0.15 * torch.tanh(b[:, 1])
        bias = torch.complex(br, bi)

        o = (lin * gain + bias * self.FEAT_SCALE) / self.FEAT_SCALE
        return o.to(torch.complex64)                      # (B,)