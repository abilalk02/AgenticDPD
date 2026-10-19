import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """g005c2-core lift for the 10^6.5-10^7.0 BOP niche.

    Backbone: complex FIR passthrough (linear centre estimate) + a
    multi-tap envelope-driven KAN correction, gated by a smooth 1-D
    power-conditioning KAN on p. FIR guarantees a clean linear limit;
    all spline inputs are squashed into [x_min,x_max]; heavy KANs are
    8-bit QAT to keep BOP inside the band while RMpS ~ 7k.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, decim: int = 1, hidden: int = 20,
                 grid: int = 9, order: int = 3, pgrid: int = 8,
                 env_taps: int = 6, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2
        idx = list(range(0, window, decim))
        if c not in idx:
            idx.append(c)
        idx = sorted(set(idx))
        self.register_buffer("tap_idx", torch.tensor(idx, dtype=torch.long))
        self.n_taps = len(idx)
        self.c = c

        # linear passthrough FIR (full precision, cheap)
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=32)

        # complex mix producing multiple memory-aware "sub-signals" whose
        # envelopes form a rich real feature basis for the KAN
        self.envmix = CLinear(self.n_taps, env_taps, bias=False, quant_bits=32)

        # envelope nonlinearity: [env basis (env_taps), drive, raw] -> hidden -> (gr,gi)
        self.kan1 = KANLayer(env_taps + 2, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # smooth 1-D power conditioning: p -> (drive_scale, gate)
        self.kanp = KANLayer(1, 2, pgrid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=32)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            cpos = idx.index(c)
            self.fir.wr[0, cpos] = 1.0
            # envmix: seed each output near a distinct tap so envelopes differ
            self.envmix.wr.zero_()
            self.envmix.wi.zero_()
            for k in range(env_taps):
                t = min(k, self.n_taps - 1)
                self.envmix.wr[k, t] = 1.0
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.kanp.coef.mul_(0.1)
            self.kanp.wb.mul_(0.1)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                      # (B, window)
        taps = x[:, self.tap_idx]                     # (B, n_taps)
        lin = self.fir(taps)[:, 0]                    # (B,) complex

        # ---- smooth power conditioning (safe at p=0 and p=1) ----
        pc = p.reshape(B, 1)                          # (B,1) in [0,1]
        pin = torch.tanh(2.0 * pc - 1.0)              # monotone into grid
        pout = self.kanp(pin)                         # (B,2)
        drive = 1.0 + 0.5 * torch.tanh(pout[:, 0])    # ~[0.5,1.5]
        gate = torch.sigmoid(pout[:, 1])              # (B,)

        # ---- multi-tap envelope basis ----
        sub = self.envmix(taps)                       # (B, env_taps) complex
        env_basis = torch.tanh(sub.abs() * 2.0)       # (B, env_taps) in (-1,1)

        xc = x[:, self.c]
        env = torch.abs(xc)                           # (B,)
        env_d = torch.tanh(env * 2.0 * drive)         # drive-modulated
        env_r = torch.tanh(env * 2.0)                 # raw
        feat = torch.cat([env_basis,
                          env_d.unsqueeze(-1),
                          env_r.unsqueeze(-1)], dim=-1)  # (B, env_taps+2)
        feat = torch.tanh(feat)                       # stay inside grid

        g = self.kan2(self.kan1(feat))                # (B,2) additive delta
        gr = g[:, 0] * gate
        gi = g[:, 1] * gate
        gain = torch.complex(1.0 + gr, gi)            # gain near 1

        o = (lin * gain) / self.FEAT_SCALE
        return o.to(torch.complex64)