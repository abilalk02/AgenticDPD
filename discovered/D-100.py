import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Ultra-lean power-scheduled two-expert envelope corrector.

    A decimated complex FIR gives the linear/memory estimate. Two bounded
    envelope-KAN experts (soft/low-drive and sharp/high-drive squash) each
    emit a complex deviation-from-identity gain. A dedicated tiny KAN acting
    ONLY on p produces (a) a smooth mixing weight blending the two experts and
    (b) a global correction-strength scalar. Thus p never enters as a plain
    feature: it schedules the correction. All spline inputs are tanh-squashed
    into the grid, and p is used via p*2-1 / tanh(env*p) so p==0 and p==1 are
    both finite.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2,
                 grid: int = 6, order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2

        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        lo = max(0, c - 2)
        hi = min(window, c + 3)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # decimated complex FIR -> centre-sample linear/memory estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # two envelope experts: [env, env^2, ep, mem] -> (re, im) gain deviation
        self.kan_lo = KANLayer(4, 2, grid, order, x_min=-1.0, x_max=1.0,
                               quant_bits=quant_bits)
        self.kan_hi = KANLayer(4, 2, grid, order, x_min=-1.0, x_max=1.0,
                               quant_bits=quant_bits)

        # p-scheduler: [p_c] -> (mix_logit, strength_pre)
        self.kan_p = KANLayer(1, 2, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # start near identity gain: tiny expert perturbations
            self.kan_lo.coef.mul_(0.05)
            self.kan_lo.wb.mul_(0.05)
            self.kan_hi.coef.mul_(0.05)
            self.kan_hi.wb.mul_(0.05)
            # p-scheduler starts near neutral
            self.kan_p.coef.mul_(0.05)
            self.kan_p.wb.mul_(0.05)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                       # (B, window)
        taps = x[:, self.tap_idx]                      # (B, n_taps)
        lin = self.fir(taps)[:, 0]                     # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                            # (B,)
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        pc = p * 2.0 - 1.0                             # (B,) in [-1,1]

        # ---- p-scheduler (smooth 1-D function of p only) ----
        pf = torch.tanh(pc).reshape(B, 1)              # (B,1) bounded
        ps = self.kan_p(pf)                            # (B,2)
        mix = torch.sigmoid(ps[:, 0])                  # (B,) 0..1 hi weight
        strength = 0.5 + 1.5 * torch.sigmoid(ps[:, 1])  # (B,) 0.5..2.0

        # ---- low-drive expert (soft squash) ----
        e_lo = torch.tanh(env * 2.0)
        e2_lo = torch.tanh(env * env * 3.0)
        ep_lo = torch.tanh(env * p * 2.5)
        m_lo = torch.tanh(mem_env * 2.0)
        feat_lo = torch.stack([e_lo, e2_lo, ep_lo, m_lo], dim=-1)
        g_lo = self.kan_lo(feat_lo)                    # (B,2)

        # ---- high-drive expert (sharper squash, emphasizes compression) ----
        e_hi = torch.tanh(env * 3.5)
        e2_hi = torch.tanh(env * env * 5.0)
        ep_hi = torch.tanh(env * p * 4.0)
        m_hi = torch.tanh(mem_env * 3.0)
        feat_hi = torch.stack([e_hi, e2_hi, ep_hi, m_hi], dim=-1)
        g_hi = self.kan_hi(feat_hi)                    # (B,2)

        # blend experts by p-schedule
        mixv = mix.reshape(B, 1)
        g = (1.0 - mixv) * g_lo + mixv * g_hi          # (B,2)

        # power-modulated correction strength scales deviation from identity
        sv = strength.reshape(B, 1)
        g = g * sv

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])   # (B,) complex

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)                   # (B,)