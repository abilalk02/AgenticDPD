import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Scaled envelope-conditioned KAN targeting the empty 10^7.0-10^7.5 BOP niche.

    Wide complex FIR (linear memory) + a two-stage, fine-grid KAN acting on a
    rich parameter-free feature vector [env powers, memory envelope, p and its
    cross terms] -> complex residual gain on the centre sample. Power p enters
    directly and via cross terms so one model linearizes the whole sweep and can
    bend the AM/AM curve at high drive. All spline inputs are squashed into the
    grid; coefficients start small (near identity FIR) for tight seed spread.
    Full 32-bit precision lifts BOP into the target 10^7 band while KAN keeps
    high-drive robustness (unlike scaled CNN/MLP).
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 16,
                 grid: int = 12, order: int = 4, quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c

        # full tap window for complex FIR (linear memory, robust)
        self.n_taps = window

        # short symmetric neighbourhood for a memory-envelope feature (free)
        lo = max(0, c - 3)
        hi = min(window, c + 4)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # complex FIR over all taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # feature layout (all parameter-free):
        # [env, env^2, env^3, mem_env, mem_env^2, p, env*p, env2*p, mem*p]  (9-D)
        self.n_feat = 9
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            # identity FIR at centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            # small-signal init: start near identity gain, tight seed spread
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.08)
                l.wb.mul_(0.08)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                        # (B, window)

        lin = self.fir(x)[:, 0]                          # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                              # (B,)

        # local memory envelope (mean |x| over short neighbourhood), free
        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        # bounded squashes into spline support [-1, 1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env * mem_env * 3.0)

        pp = p * 2.0 - 1.0                               # center p to [-1,1]
        ep = torch.tanh(env * p * 2.5)                   # env*p cross term
        e2p = torch.tanh(env * env * p * 3.0)            # env^2*p cross term
        mp = torch.tanh(mem_env * p * 2.5)               # mem*p cross term

        feat = torch.stack([e, e2, e3, m, m2, pp, ep, e2p, mp], dim=-1)  # (B, 9)

        h = self.kan1(feat)
        h = self.kan2(h)
        g = self.kan3(h)                                 # (B, 2)

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])     # residual gain

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)