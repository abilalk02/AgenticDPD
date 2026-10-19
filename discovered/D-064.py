import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer


class DPD(DPDModule):
    """Envelope-conditioned residual-gain KAN with a dedicated high-drive
    spline branch, targeting >38 dB worst-seed ACLR at 3 dBFS.

    Core (parent-proven):
      * full-window complex FIR -> centre-sample linear estimate (identity init)
      * parameter-free 12-D feature basis (envelope powers, two memory scales,
        centred p, p cross terms), all squashed into [-1,1]
      * three fine-grid KAN stages producing a complex residual gain

    New high-drive focus:
      * an extra small KAN branch driven by a smooth "high-drive activation"
        scalar (env^2 gated by p) whose residual gain vanishes at low drive,
        so seed-to-seed variance at low/mid drive stays tight while extra
        spline capacity linearizes the 3 dBFS point.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 28,
                 grid: int = 16, order: int = 4, quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        self.c = c
        self.n_taps = window

        lo1 = max(0, c - 2)
        hi1 = min(window, c + 3)
        self.register_buffer("mem_idx",
                             torch.tensor(list(range(lo1, hi1)), dtype=torch.long))
        lo2 = max(0, c - 5)
        hi2 = min(window, c + 6)
        self.register_buffer("mem_idx2",
                             torch.tensor(list(range(lo2, hi2)), dtype=torch.long))

        # complex FIR over all taps -> centre-sample linear estimate
        self.fir = CLinear(self.n_taps, 1, bias=False, quant_bits=quant_bits)

        # main feature KAN (12-D basis)
        self.n_feat = 12
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # dedicated high-drive branch: fine grid, small width, 3-D input
        self.hd_grid = 20
        self.kan_hd1 = KANLayer(3, 12, self.hd_grid, order,
                                x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan_hd2 = KANLayer(12, 2, self.hd_grid, order,
                                x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        with torch.no_grad():
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            for l in (self.kan1, self.kan2, self.kan3):
                l.coef.mul_(0.06)
                l.wb.mul_(0.06)
            # high-drive branch starts near-zero so it perturbs nothing initially
            for l in (self.kan_hd1, self.kan_hd2):
                l.coef.mul_(0.03)
                l.wb.mul_(0.03)

    def forward(self, xw, p):
        B = xw.shape[0]
        x = xw * self.FEAT_SCALE                          # (B, window)

        lin = self.fir(x)[:, 0]                           # (B,) complex

        xc = x[:, self.c]
        env = torch.abs(xc)                               # (B,)

        mem_env = torch.abs(x[:, self.mem_idx]).mean(dim=-1)
        mem_env2 = torch.abs(x[:, self.mem_idx2]).mean(dim=-1)

        pv = p.reshape(B)                                 # (B,), safe at 0 and 1

        # bounded squashes into [-1, 1]
        e = torch.tanh(env * 2.0)
        e2 = torch.tanh(env * env * 3.0)
        e3 = torch.tanh(env * env * env * 4.0)
        m1 = torch.tanh(mem_env * 2.0)
        m1b = torch.tanh(mem_env * mem_env * 3.0)
        m2 = torch.tanh(mem_env2 * 2.0)
        m2b = torch.tanh(mem_env2 * mem_env2 * 3.0)

        pp = pv * 2.0 - 1.0
        ep = torch.tanh(env * pv * 2.5)
        e2p = torch.tanh(env * env * pv * 3.0)
        mp = torch.tanh(mem_env * pv * 2.5)
        ep2 = torch.tanh(env * pv * pv * 2.5)

        feat = torch.stack(
            [e, e2, e3, m1, m1b, m2, m2b, pp, ep, e2p, mp, ep2], dim=-1)

        h = self.kan1(feat)
        h = self.kan2(h)
        g = self.kan3(h)                                  # (B, 2)

        gain = torch.complex(1.0 + g[:, 0], g[:, 1])

        # ---- high-drive branch ----
        # smooth drive gate: ~0 at low drive/low p, ->1 at high drive+high p.
        drive = env * (0.4 + 0.6 * pv)                    # emphasise p high
        gate = torch.tanh(drive * drive * 4.0)            # in [0,1), free
        hd_a = torch.tanh(env * env * 3.0)                # high-drive activation
        hd_b = torch.tanh(env * env * env * pv * 4.0)     # p-modulated cubic
        hd_c = pp                                         # centred power
        hd_feat = torch.stack([hd_a, hd_b, hd_c], dim=-1)
        hh = self.kan_hd1(hd_feat)
        gh = self.kan_hd2(hh)                             # (B, 2)
        # residual gain gated by drive: negligible at low drive -> tight seeds
        gh_gain = torch.complex(gh[:, 0] * gate, gh[:, 1] * gate)

        total_gain = gain + gh_gain

        o = (lin * total_gain) / self.FEAT_SCALE
        return o.to(torch.complex64)