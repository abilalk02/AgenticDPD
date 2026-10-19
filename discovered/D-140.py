import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """g005c2-lineage envelope-conditioned KAN DPD, sized for the empty
    6-9k RMpS niche (~7-8k RMpS target).

    A full-window identity-initialized complex FIR carries the linear/
    memory estimate (dominant path -> no high-drive divergence). A wide
    bounded main KAN bends a rich real envelope basis into a complex
    residual gain (1 + delta). Power p enters as a smooth learned 1-D
    KAN function of centered p (safe at 0 and 1) that modulates the
    correction depth -- power conditioning, not a raw concat feature.
    All spline inputs are tanh-squashed into [-1,1].
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, grid: int = 8, order: int = 3,
                 quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        c = window // 2
        self.c = c

        # memory neighbourhood for envelope features
        lo = max(0, c - 3)
        hi = min(window, c + 4)
        self.register_buffer("mem_idx",
                             torch.arange(lo, hi, dtype=torch.long))

        # full-window complex FIR -> centre-sample linear/memory estimate
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # complex tap mixer -> richer complex feature bank for envelopes
        self.mix = CLinear(window, 8, bias=True, quant_bits=quant_bits)

        # main bounded KAN: rich real basis -> complex residual (re, im)
        # feature vector (14 dims): 8 mixed-env, env, env^2, env^3,
        #                            mem_env, mem_env^2, pc
        self.n_feat = 14
        self.d_hid = 10
        self.kan = KANLayer(self.n_feat, self.d_hid, grid, order,
                            x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # complex head: hidden -> single complex correction
        self.head = CLinear(self.d_hid, 1, bias=True, quant_bits=quant_bits)

        # power-conditioning 1-D KAN: centered p -> scalar depth gate
        self.pkan = KANLayer(1, 1, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            # FIR identity at centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            # start near identity gain: tiny correction
            self.kan.coef.mul_(0.05)
            self.kan.wb.mul_(0.05)
            self.head.wr.mul_(0.1)
            self.head.wi.mul_(0.1)
            # power gate starts near neutral
            self.pkan.coef.mul_(0.1)
            self.pkan.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)

        # dominant linear/memory path
        lin = self.fir(x)[:, 0]                           # (B,) complex

        # complex mixed features -> bounded real envelopes
        z = self.mix(x)                                   # (B, 8) complex
        zenv = torch.tanh(z.abs() * 2.0)                  # (B, 8) in (-1,1)

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)
        mem = torch.abs(x[:, self.mem_idx]).mean(dim=-1)  # (B,)

        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m = torch.tanh(mem * 2.0)
        m2 = torch.tanh(mem * mem * 3.0)
        pc = (p * 2.0 - 1.0)                              # (B,) in [-1,1]

        feat = torch.cat([
            zenv,
            torch.stack([e, e2, e3, m, m2, pc], dim=-1),
        ], dim=-1)                                        # (B, 14)
        feat = torch.tanh(feat)                           # keep in grid

        h = self.kan(feat)                                # (B, d_hid)
        h = 1.5 * torch.tanh(h / 1.5)                     # bounded hidden

        # complex residual correction
        delta = self.head(h.to(torch.complex64))[:, 0]    # (B,) complex

        # smooth power-depth gate (safe at p=0 and p=1)
        pin = pc.reshape(B, 1)
        depth = torch.sigmoid(self.pkan(pin)[:, 0])       # (B,) in (0,1)
        depth = 0.5 * depth                               # bound residual

        gain = 1.0 + depth.to(torch.complex64) * (0.5 * torch.tanh(delta.real)
                                                  + 1j * 0.5 * torch.tanh(delta.imag))

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)                      # (B,)