import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Scaled power-modulated KAN residual DPD for the weak ultra-low niche.

    Anchors on an identity-initialised complex FIR (linear/memory estimate,
    no high-drive collapse, tight seeds). A wider grid KAN, fed a rich but
    fully tanh-squashed real envelope basis (env..env^7, mem env, centred p,
    env*p / env^2*p / env^3*p cross terms), produces a bounded complex gain,
    a small additive residual, and per-tap memory gains. p enters only via
    centred/cross terms -> safe at p=0 and p=1. Low-bit fake-quant keeps BOP
    in the empty/weak ultra-low bins at ~430 RMpS.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, grid: int = 8, order: int = 3,
                 quant_bits: int = 6):
        super().__init__(window)
        c = window // 2
        self.c = c

        # tap set for FIR: centre +/- 3
        lo = max(0, c - 3)
        hi = min(window, c + 4)
        idx = sorted(set(list(range(lo, hi)) + [c]))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)

        # memory taps for envelope features
        mlo = max(0, c - 3)
        mhi = min(window, c + 4)
        midx = list(range(mlo, mhi))
        self.register_buffer("mem_idx", torch.tensor(midx, dtype=torch.long))
        self.n_mem = len(midx)

        # complex FIR -> linear/memory estimate (8-bit, cheap)
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=8)

        # feature basis (all squashed to [-1,1])
        # [e, e2, e3, e5, e7, m, m2, pc, ep, e2p, e3p, ep_mem]
        self.n_feat = 12

        # KAN outputs: gain_re, gain_im, bias_re, bias_im  -> 4 outputs
        self.kan = KANLayer(self.n_feat, 4, grid, order,
                            x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # second small KAN -> per-mem-tap real gain modulation (memory DPD)
        self.kan_mem = KANLayer(self.n_feat, self.n_mem, grid, order,
                                x_min=-1.0, x_max=1.0, quant_bits=4)

        # complex mixer for the memory-modulated path
        self.mem_mix = CLinear(self.n_mem, 1, bias=False, quant_bits=8)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # near-identity start
            self.kan.coef.mul_(0.05)
            self.kan.wb.mul_(0.05)
            self.kan_mem.coef.mul_(0.03)
            self.kan_mem.wb.mul_(0.03)
            # memory mixer starts near-zero (pure residual)
            self.mem_mix.wr.mul_(0.05)
            self.mem_mix.wi.mul_(0.05)

    def _features(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                              # (B, window)

        xc = x[:, self.c]
        env = torch.abs(xc)                                   # (B,)
        mem_taps = x[:, self.mem_idx]                         # (B, n_mem) cplx
        mem_abs = torch.abs(mem_taps)                         # (B, n_mem)
        mem_env = mem_abs.mean(dim=-1)                        # (B,)
        mem_env2 = (mem_abs * mem_abs).mean(dim=-1)           # (B,)

        pc = (p * 2.0 - 1.0)                                  # [-1,1]

        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env ** 3 * 4.0)
        e5 = torch.tanh(env ** 5 * 6.0)
        e7 = torch.tanh(env ** 7 * 8.0)
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env2 * 3.0)
        ep = torch.tanh(env * p * 2.5)
        e2p = torch.tanh(env * env * p * 3.0)
        e3p = torch.tanh(env ** 3 * p * 4.0)
        ep_mem = torch.tanh(mem_env * p * 2.5)

        feat = torch.stack(
            [e, e2, e3, e5, e7, m, m2, pc, ep, e2p, e3p, ep_mem], dim=-1)
        return env, mem_taps, feat

    def forward(self, xw, p):
        x = xw * self.FEAT_SCALE
        taps = x[:, self.tap_idx]
        lin = self.fir(taps)[:, 0]                            # (B,) cplx

        env, mem_taps, feat = self._features(xw, p)

        # main gain / bias
        k = self.kan(feat)                                    # (B,4)
        dr = 0.5 * torch.tanh(k[:, 0])
        di = 0.5 * torch.tanh(k[:, 1])
        gain = torch.complex(1.0 + dr, di)                    # (B,) cplx

        br = 0.15 * torch.tanh(k[:, 2])
        bi = 0.15 * torch.tanh(k[:, 3])
        bias = torch.complex(br, bi)                          # (B,) cplx

        # memory-modulated residual: per-tap real gains bend the mem taps
        km = self.kan_mem(feat)                               # (B, n_mem)
        mem_g = torch.tanh(km)                                # (B, n_mem) real
        mem_mod = mem_taps * mem_g.to(mem_taps.dtype)         # (B, n_mem) cplx
        mem_res = self.mem_mix(mem_mod)[:, 0]                 # (B,) cplx

        o = (lin * gain + mem_res + bias * self.FEAT_SCALE) / self.FEAT_SCALE
        return o.to(torch.complex64)                          # (B,)