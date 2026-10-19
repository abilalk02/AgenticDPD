import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Scaled envelope-conditioned KAN targeting the empty high-BOP niche (bin 9,
    10^7.5+ BOP). Full-window complex FIR (robust linear memory) + a rich
    parameter-free feature vector (envelope powers, two memory-envelope scales,
    p and several p cross terms) feeds a three-stage fine-grid KAN producing a
    complex residual gain on the centre sample. One model linearizes the whole
    sweep: p enters directly and via cross terms to bend the AM/AM curve at high
    drive. All spline inputs are squashed into the grid; coefficients start small
    (near-identity FIR) for tight seed spread. Full 32-bit precision lifts BOP
    into the target band while KAN keeps high-drive robustness.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 24,
                 grid: int = 14, order: int = 4, quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.n_taps = window

        # short memory neighbourhood
        lo1 = max(0, c - 2)
        hi1 = min(window, c + 3)
        self.register_buffer("mem_idx",
                             torch.tensor(list(range(lo1, hi1)), dtype=torch.long))
        # wider memory neighbourhood
        lo2 = max(0, c - 5)
        hi2 = min(window, c + 6)
        self.register_buffer("mem_idx2",
                             torch.tensor(list(range(lo2, hi2)), dtype=torch.long))

        # complex FIR over all taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # feature layout (all parameter-free), 12-D:
        # [env, env^2, env^3, mem1, mem1^2, mem2, mem2^2, p,
        #  env*p, env2*p, mem1*p, env*p^2]
        self.n_feat = 12
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.06)
                l.wb.mul_(0.06)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)

        lin = self.fir(x)[:, 0]                            # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)

        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)   # (B,)
        mem_env2 = torch.abs(x[:, self.mem_idx2]).mean(dim=-1) # (B,)

        pv = p.reshape(B)                                 # (B,), safe at 0 and 1

        # bounded squashes into spline support [-1, 1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m1 = torch.tanh(mem_env * 2.0)
        m1b = torch.tanh(mem_env * mem_env * 3.0)
        m2 = torch.tanh(mem_env2 * 2.0)
        m2b = torch.tanh(mem_env2 * mem_env2 * 3.0)

        pp = pv * 2.0 - 1.0                               # center p to [-1,1]
        ep = torch.tanh(env * pv * 2.5)                   # env*p
        e2p = torch.tanh(env * env * pv * 3.0)            # env^2*p
        mp = torch.tanh(mem_env * pv * 2.5)               # mem*p
        ep2 = torch.tanh(env * pv * pv * 2.5)             # env*p^2

        feat = torch.stack(
            [e, e2, e3, m1, m1b, m2, m2b, pp, ep, e2p, mp, ep2], dim=-1)  # (B,12)

        h = self.kan1(feat)
        h = self.kan2(h)
        g = self.kan3(h)                                  # (B, 2)

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])      # residual gain

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)