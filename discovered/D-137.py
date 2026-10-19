import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Envelope-conditioned KAN (g017c1-style, native 3 dBFS clearer) scaled into
    the 10^6.5-10^7.0 BOP niche to displace the 3 dBFS-capped g015c3 leader.

    Complex FIR linear memory + three-stage fine-grid KAN on a parameter-free
    feature basis (env powers, memory envelope, p cross terms). Power p enters
    both as a squashed feature and via a smooth learned 1-D KAN function that
    modulates the residual gain (variance-gate style conditioning), so a single
    model linearizes the whole sweep and bends AM/AM at high drive. All spline
    inputs are squashed into grid support; near-identity init keeps seeds tight.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 24,
                 grid: int = 14, order: int = 4, quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.n_taps = window

        lo = max(0, c - 3)
        hi = min(window, c + 4)
        mem_idx = list(range(lo, hi))
        self.register_buffer("mem_idx", torch.tensor(mem_idx, dtype=torch.long))

        # complex FIR over all taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # feature layout (parameter-free):
        # [e, e2, e3, m, m2, pp, ep, e2p, mp]  (9-D)
        self.n_feat = 9
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # smooth learned 1-D function of p -> variance-gate power conditioning
        self.pkan = KANLayer(1, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan1, self.kan2, self.kan3, self.pkan):
                l.coef.mul_(0.08)
                l.wb.mul_(0.08)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                        # (B, window) complex

        lin = self.fir(x)[:, 0]                          # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                              # (B,)

        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        # bounded squashes into spline support [-1, 1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m = torch.tanh(mem_env * 2.0)
        m2 = torch.tanh(mem_env * mem_env * 3.0)

        pp = p * 2.0 - 1.0                               # p -> [-1,1], safe @0,1
        ep = torch.tanh(env * p * 2.5)
        e2p = torch.tanh(env * env * p * 3.0)
        mp = torch.tanh(mem_env * p * 2.5)

        feat = torch.stack([e, e2, e3, m, m2, pp, ep, e2p, mp], dim=-1)  # (B,9)

        h = self.kan1(feat)
        h = self.kan2(h)
        g = self.kan3(h)                                 # (B, 2)

        # power conditioning: smooth 1-D function of p modulates gain
        pin = pp.reshape(B, 1)                           # (B,1) in [-1,1]
        pmod = self.pkan(pin)                            # (B, 2)
        pgr = 1.0 + torch.tanh(pmod[:, 0])               # ~1 scaling
        pgi = torch.tanh(pmod[:, 1])                     # small rotation

        gr = 1.0 + g[:, 0]
        gi = g[:, 1]

        # combine residual gain with power-conditioned modulation
        re = gr * pgr - gi * pgi
        im = gr * pgi + gi * pgr
        gain = torch.complex(re, im)

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)