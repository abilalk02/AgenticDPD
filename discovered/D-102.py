import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Scaled power-modulated dual-KAN residual DPD (g009-lineage scale-up).

    Fills the weak 10^6.0-10^6.5 BOP niche (parent ACLR 36.5 dB @404 RMpS)
    by scaling the proven dual-KAN residual core UP: wider complex FIR
    memory estimate, a richer 16-dim squashed feature basis, a two-stage
    gain KAN (feat->hidden->gain) for high-drive expressiveness, and a
    parallel bias KAN. Identity-init keeps every seed stable at 3 dBFS.
    All spline inputs are tanh-squashed into [-1,1]; p enters via centred
    and cross terms only (safe at p=0 and p=1). 8-bit KANs place BOP in band.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 1,
                 grid: int = 10, order: int = 3, hidden: int = 12,
                 quant_bits: int = 8):
        super().__init__(window)
        c = window // 2
        self.c = c

        # dense tap selection for a strong linear/memory estimate
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)

        lo = max(0, c - 4)
        hi = min(window, c + 5)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        lo2 = max(0, c - 2)
        hi2 = min(window, c + 3)
        near_idx = list(range(lo2, hi2))
        self.register_buffer("near_idx", torch.tensor(near_idx, dtype=torch.long))

        # complex FIR -> centre-sample linear/memory estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # feature basis (all squashed into [-1,1])
        # [e,e2,e3,e5,e7, m,m2,m3, n,n2, pc, ep,mp,np_, e2p, e3p]
        self.n_feat = 16

        # two-stage gain KAN for high-drive expressiveness
        self.kan_h = KANLayer(self.n_feat, hidden, grid, order,
                              x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan_gain = KANLayer(hidden, 2, grid, order,
                                 x_min=-1.5, x_max=1.5, quant_bits=quant_bits)
        # parallel bias KAN: small additive envelope-shaped residual
        self.kan_bias = KANLayer(self.n_feat, 2, grid, order,
                                 x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # start near identity: tiny KAN perturbations
            self.kan_h.coef.mul_(0.10)
            self.kan_h.wb.mul_(0.10)
            self.kan_gain.coef.mul_(0.05)
            self.kan_gain.wb.mul_(0.05)
            self.kan_bias.coef.mul_(0.02)
            self.kan_bias.wb.mul_(0.02)

    def _features(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE
        taps = x[:, self.tap_idx]
        lin = self.fir(taps)[:, 0]                         # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)
        near_env = torch.abs(x[:, self.near_idx]).mean(dim=-1)

        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e5 = torch.tanh((env ** 5) * 6.0)
        e7 = torch.tanh((env ** 7) * 8.0)                  # very-deep drive
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env * mem_env * 3.0)
        m3 = torch.tanh(mem_env * mem_env * mem_env * 4.0)
        n = torch.tanh(near_env * 2.0)
        n2 = torch.tanh(near_env * near_env * 3.0)
        pc = p * 2.0 - 1.0
        ep = torch.tanh(env * p * 2.5)
        mp = torch.tanh(mem_env * p * 2.5)
        np_ = torch.tanh(near_env * p * 2.5)
        e2p = torch.tanh(env * env * p * 3.0)
        e3p = torch.tanh(env * env * env * p * 3.5)

        feat = torch.stack(
            [e, e2, e3, e5, e7, m, m2, m3, n, n2, pc, ep, mp, np_, e2p, e3p],
            dim=-1)
        return lin, feat

    def forward(self, xw, p):
        lin, feat = self._features(xw, p)

        h = self.kan_h(feat)                               # (B, hidden)
        h = 1.5 * torch.tanh(h)                            # squash into next grid
        g = self.kan_gain(h)                               # (B, 2)
        dr = 0.5 * torch.tanh(g[:, 0])
        di = 0.5 * torch.tanh(g[:, 1])
        gain = torch.complex(1.0 + dr, di)

        b = self.kan_bias(feat)                            # (B, 2)
        br = 0.15 * torch.tanh(b[:, 0])
        bi = 0.15 * torch.tanh(b[:, 1])
        bias = torch.complex(br, bi)

        o = (lin * gain + bias * self.FEAT_SCALE) / self.FEAT_SCALE
        return o.to(torch.complex64)