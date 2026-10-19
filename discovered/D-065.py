import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """Seed-robust envelope+power conditioned KAN DPD.

    Static AM/AM & AM/PM correction from a two-layer KAN driven by squashed
    [I, Q, |x|, |x|^2, thermal] features around an identity complex-linear
    path. Power p is injected as a SMOOTH learned 1-D gate (via a tiny KAN)
    that modulates the correction magnitude in a bounded range [0.5, 1.5],
    avoiding the plain-concat-p robustness failure. A short complex FIR
    captures fast memory. 8-bit quant on KAN cores keeps BOP in the
    10^4.5-10^5.0 band while preserving RMpS/ACLR.
    """

    FEAT_SCALE = 0.21
    GRID_A = 1.3   # squash amplitude -> stays inside [-1.5, 1.5] grid

    def __init__(self, window: int, hidden: int = 8, grid: int = 7,
                 order: int = 3, mem: int = 5, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2
        half = mem // 2
        lo = max(0, c - half)
        hi = min(window, c + half + 1)
        self.mem_idx = list(range(lo, hi))
        self.M = len(self.mem_idx)

        # fast complex FIR memory branch (short)
        self.fir = CLinear(self.M, 1, bias=False, quant_bits=quant_bits)

        # static AM/AM-AM/PM KAN: features [I, Q, |x|, |x|^2, thermal]
        self.n_feat = 5
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)

        # smooth learned 1-D power gate: p -> scalar modulation
        self.pgate = KANLayer(1, 1, grid, order,
                              x_min=-1.5, x_max=1.5, quant_bits=quant_bits)

        # identity-initialised complex linear path (2x2 real)
        self.linear_path = RLinear(2, 2, bias=False, quant_bits=quant_bits)

        with torch.no_grad():
            self.linear_path.weight.copy_(torch.eye(2))
            self.fir.wr.mul_(0.05)
            self.fir.wi.mul_(0.05)
            for l in (self.kan1, self.kan2, self.pgate):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

        self.register_buffer("leak", torch.tensor(0.7))

    def _squash(self, t):
        a = self.GRID_A
        return a * torch.tanh(t / a)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        xs = xw * self.FEAT_SCALE
        x = xs[:, c]

        # --- static features (all squashed into grid support) ---
        env = x.abs()
        env2 = env * env
        env_s = self._squash(env)
        env2_s = self._squash(env2)

        # slow thermal: centre-weighted decaying envelope power
        pw = xs.abs() ** 2
        offs = (torch.arange(self.window, device=xw.device) - c).abs().to(torch.float32)
        w = self.leak ** offs
        w = w / w.sum()
        thermal = (pw * w).sum(dim=-1)
        thermal_s = self._squash(thermal * 4.0)

        iq = torch.stack([x.real, x.imag], dim=-1)   # (B,2)
        i_s = self._squash(x.real)
        q_s = self._squash(x.imag)

        f = torch.stack([i_s, q_s, env_s, env2_s, thermal_s], dim=-1)  # (B,5)

        corr = self.kan2(self.kan1(f))               # (B,2)

        # --- smooth bounded power gate (safe at p=0 and p=1) ---
        pc = self._squash(p.reshape(B, 1) * 2.0 - 1.0)   # centre p in [-1,1]
        graw = self.pgate(pc)                              # (B,1)
        gate = 1.0 + 0.5 * torch.tanh(graw)               # in [0.5, 1.5]
        corr = corr * gate

        lin = self.linear_path(iq)                        # (B,2)

        # fast complex FIR memory branch
        xmem = xs[:, self.mem_idx]
        fir_out = self.fir(xmem).squeeze(-1)              # (B,) complex
        fir_iq = torch.stack([fir_out.real, fir_out.imag], dim=-1)

        o = (lin + corr + fir_iq) / self.FEAT_SCALE       # (B,2)
        return torch.complex(o[:, 0], o[:, 1])            # complex64 (B,)