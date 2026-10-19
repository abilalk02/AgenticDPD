import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Scaled power-modulated dual-KAN residual DPD.

    Targets the EMPTY 10^5.5-10^6.0 BOP niche by scaling the proven parent
    ultra-low KAN core UP (wider complex FIR + richer squashed feature basis +
    two parallel bounded KANs) rather than a fragile CNN/MLP. A complex FIR
    provides the dominant linear/memory estimate (identity-initialised, so no
    seed diverges at high drive). KAN-gain bends a rich real feature vector
    [env powers, mem env, p-cross terms] into a bounded (1+delta) complex gain,
    and KAN-bias adds a small envelope-shaped complex residual. All spline
    inputs are tanh-squashed into [-1,1]; p enters only via centred/cross
    terms (safe at p=0 and p=1). 8-bit KANs keep BOP in-band.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 1,
                 grid: int = 8, order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2

        # denser tap selection for a stronger linear/memory estimate
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        lo = max(0, c - 3)
        hi = min(window, c + 4)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # complex FIR -> centre-sample linear/memory estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # feature basis size: [e, e2, e3, e5, m, m2, pc, ep, mp, e2p]
        self.n_feat = 10

        # gain KAN: features -> (re, im) bounded multiplicative correction
        self.kan_gain = KANLayer(self.n_feat, 2, grid, order,
                                 x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        # bias KAN: features -> (re, im) small additive envelope-shaped residual
        self.kan_bias = KANLayer(self.n_feat, 2, grid, order,
                                 x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # start near identity: tiny KAN perturbations
            self.kan_gain.coef.mul_(0.05)
            self.kan_gain.wb.mul_(0.05)
            self.kan_bias.coef.mul_(0.02)
            self.kan_bias.wb.mul_(0.02)

    def _features(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)
        taps = x[:, self.tap_idx]                         # (B, n_taps)
        lin = self.fir(taps)[:, 0]                        # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        # bounded, free feature construction squashed into [-1,1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        e5 = torch.tanh((env ** 5) * 6.0)                 # deep-drive term
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env * mem_env * 3.0)
        pc = p * 2.0 - 1.0                                # (B,) in [-1,1]
        ep = torch.tanh(env * p * 2.5)                    # env*p cross
        mp = torch.tanh(mem_env * p * 2.5)                # mem*p cross
        e2p = torch.tanh(env * env * p * 3.0)             # env^2*p cross

        feat = torch.stack([e, e2, e3, e5, m, m2, pc, ep, mp, e2p], dim=-1)
        return lin, feat

    def forward(self, xw, p):
        lin, feat = self._features(xw, p)                 # (B,) complex, (B,10)

        g = self.kan_gain(feat)                           # (B, 2)
        dr = 0.5 * torch.tanh(g[:, 0])
        di = 0.5 * torch.tanh(g[:, 1])
        gain = torch.complex(1.0 + dr, di)                # (B,) complex

        b = self.kan_bias(feat)                           # (B, 2)
        br = 0.15 * torch.tanh(b[:, 0])
        bi = 0.15 * torch.tanh(b[:, 1])
        bias = torch.complex(br, bi)                      # (B,) complex

        o = (lin * gain + bias * self.FEAT_SCALE) / self.FEAT_SCALE
        return o.to(torch.complex64)                      # (B,)