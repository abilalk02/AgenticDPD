import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Ultra-low-BOP power-conditioned KAN residual DPD.

    Targets the empty low-BOP niche (~10^4.5 BOP, ~330 RMpS). Identity-init
    complex FIR gives a collapse-proof linear/memory estimate; a single 6-bit
    KAN on a fully tanh-squashed real basis (envelope powers, memory env,
    centred p, env*p / env^2*p cross terms) produces a bounded complex gain +
    tiny bias. A smooth variance-gate on p sharpens high-drive behaviour
    (the binding 3 dBFS limiter) without adding BOP. Safe at p=0 and p=1.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, grid: int = 8, order: int = 3,
                 quant_bits: int = 6):
        super().__init__(window)
        c = window // 2
        self.c = c

        lo = max(0, c - 3)
        hi = min(window, c + 4)
        idx = sorted(set(list(range(lo, hi)) + [c]))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.cpos = idx.index(c)

        mlo = max(0, c - 3)
        mhi = min(window, c + 4)
        self.register_buffer("mem_idx",
                             torch.tensor(list(range(mlo, mhi)), dtype=torch.long))

        # complex FIR -> centre-sample linear/memory estimate (8-bit)
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=8)

        # feature basis: [e, e2, e3, e5, m, m2, pc, ep, e2p, e3p]
        self.n_feat = 10

        # KAN -> (gain_re, gain_im, bias_re, bias_im), 6-bit
        self.kan = KANLayer(self.n_feat, 4, grid, order,
                            x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, self.cpos] = 1.0
            # near-identity start
            self.kan.coef.mul_(0.05)
            self.kan.wb.mul_(0.05)

    def _features(self, xw, p):
        x = xw * self.FEAT_SCALE
        taps = x[:, self.tap_idx]
        lin = self.fir(taps)[:, 0]                            # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)
        mem = torch.abs(x[:, self.mem_idx])
        mem_env = mem.mean(dim=-1)
        mem_env2 = (mem * mem).mean(dim=-1)

        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e5 = torch.tanh((env ** 5) * 6.0)
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env2 * 3.0)
        pc = p * 2.0 - 1.0                                    # in [-1,1]
        ep = torch.tanh(env * p * 2.5)
        e2p = torch.tanh(env * env * p * 3.0)
        e3p = torch.tanh(env * env * env * p * 4.0)

        feat = torch.stack([e, e2, e3, e5, m, m2, pc, ep, e2p, e3p], dim=-1)
        return lin, feat, env

    def forward(self, xw, p):
        lin, feat, env = self._features(xw, p)

        k = self.kan(feat)                                   # (B, 4)

        # smooth power-conditioning: emphasise correction at high drive,
        # bounded gate in (0,1], safe at p=0 and p=1
        pgate = 0.5 + 0.5 * torch.tanh(2.0 * p - 1.0)        # (B,)

        dr = 0.5 * torch.tanh(k[:, 0]) * pgate
        di = 0.5 * torch.tanh(k[:, 1]) * pgate
        gain = torch.complex(1.0 + dr, di)

        br = 0.15 * torch.tanh(k[:, 2]) * pgate
        bi = 0.15 * torch.tanh(k[:, 3]) * pgate
        bias = torch.complex(br, bi)

        o = (lin * gain + bias * self.FEAT_SCALE) / self.FEAT_SCALE
        return o.to(torch.complex64)