import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Ultra-lean power-scheduled THREE-expert envelope corrector.

    Decimated complex FIR gives the linear/memory estimate. Three bounded
    envelope-KAN experts (soft low-drive, mid, sharp compression) each emit a
    complex deviation-from-identity gain. A tiny p-only KAN scheduler produces
    a 3-way soft mixing distribution plus a global correction-strength scalar.
    p never enters as a plain feature: it schedules experts. All spline inputs
    are tanh-squashed into the symmetric grid, and p uses p*2-1 / tanh so both
    p==0 and p==1 are finite. 6-bit experts keep BOP in the target band.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 2,
                 grid: int = 6, order: int = 3, quant_bits: int = 6):
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
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=8)

        # three envelope experts: [env, env^2, ep, mem] -> (re, im) gain dev.
        self.kan_lo = KANLayer(4, 2, grid, order, x_min=-1.0, x_max=1.0,
                               quant_bits=quant_bits)
        self.kan_mid = KANLayer(4, 2, grid, order, x_min=-1.0, x_max=1.0,
                                quant_bits=quant_bits)
        self.kan_hi = KANLayer(4, 2, grid, order, x_min=-1.0, x_max=1.0,
                               quant_bits=quant_bits)

        # p-scheduler: [p_c] -> (mix_logit_lo, mix_logit_mid, mix_logit_hi, strength_pre)
        self.kan_p = KANLayer(1, 4, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=8)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # start near identity gain: tiny expert perturbations
            for k in (self.kan_lo, self.kan_mid, self.kan_hi):
                k.coef.mul_(0.05)
                k.wb.mul_(0.05)
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
        ps = self.kan_p(pf)                            # (B,4)
        # 3-way soft mixing distribution over experts
        mix = torch.softmax(ps[:, 0:3], dim=-1)        # (B,3)
        strength = 0.5 + 1.5 * torch.sigmoid(ps[:, 3]) # (B,) 0.5..2.0

        # ---- low-drive expert (soft squash) ----
        e_lo = torch.tanh(env * 2.0)
        e2_lo = torch.tanh(env * env * 3.0)
        ep_lo = torch.tanh(env * p * 2.5)
        m_lo = torch.tanh(mem_env * 2.0)
        feat_lo = torch.stack([e_lo, e2_lo, ep_lo, m_lo], dim=-1)
        g_lo = self.kan_lo(feat_lo)                    # (B,2)

        # ---- mid-drive expert ----
        e_mid = torch.tanh(env * 2.8)
        e2_mid = torch.tanh(env * env * 4.0)
        ep_mid = torch.tanh(env * p * 3.2)
        m_mid = torch.tanh(mem_env * 2.5)
        feat_mid = torch.stack([e_mid, e2_mid, ep_mid, m_mid], dim=-1)
        g_mid = self.kan_mid(feat_mid)                 # (B,2)

        # ---- high-drive compression expert (sharp squash, env*p emphasis) ----
        e_hi = torch.tanh(env * 3.5)
        e2_hi = torch.tanh(env * env * 5.5)
        ep_hi = torch.tanh(env * p * 4.5)
        m_hi = torch.tanh(mem_env * 3.0)
        feat_hi = torch.stack([e_hi, e2_hi, ep_hi, m_hi], dim=-1)
        g_hi = self.kan_hi(feat_hi)                    # (B,2)

        # blend experts by p-schedule (soft 3-way)
        w_lo = mix[:, 0:1]
        w_mid = mix[:, 1:2]
        w_hi = mix[:, 2:3]
        g = w_lo * g_lo + w_mid * g_mid + w_hi * g_hi  # (B,2)

        # power-modulated correction strength scales deviation from identity
        sv = strength.reshape(B, 1)
        g = g * sv

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])   # (B,) complex

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)                   # (B,)