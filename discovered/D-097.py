import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """Envelope+power-conditioned KAN DPD sized for the 10^4.5-10^5.0 BOP niche.

    Identity-initialised complex-linear backbone + short complex FIR memory +
    a KAN AM/AM corrector driven by [I, Q, |x|, |x|^2, thermal, p, p^2].
    Power is used two ways: as a squashed spline feature AND to modulate a
    smooth learned gain on the nonlinear correction (variance-gate style,
    safe at p=0 and p=1). All KAN inputs squashed into [-1,1] grid support so
    they never leave the spline at high drive. 8-bit quant lowers BOP.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 6, grid: int = 7,
                 order: int = 3, mem: int = 7, quant_bits: int = 8):
        super().__init__(window)
        c = window // 2
        half = mem // 2
        lo = max(0, c - half)
        hi = min(window, c + half + 1)
        self.mem_idx = list(range(lo, hi))
        self.M = len(self.mem_idx)

        # fast complex FIR memory branch (short) -- 8-bit
        self.fir = CLinear(self.M, 1, bias=False, quant_bits=quant_bits)

        # static AM/AM KAN features: [I, Q, |x|, |x|^2, thermal, p, p^2]
        self.n_feat = 7
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # smooth learned 1-D gain from p (variance-gate on the correction)
        self.pgain = KANLayer(1, 1, grid, order, x_min=-1.0, x_max=1.0,
                              quant_bits=quant_bits)

        # identity-initialised complex linear path (2x2 real)
        self.linear_path = RLinear(2, 2, bias=False, quant_bits=quant_bits)

        with torch.no_grad():
            self.linear_path.weight.copy_(torch.eye(2))
            self.fir.wr.mul_(0.05)
            self.fir.wi.mul_(0.05)
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            # pgain starts near a small positive bias
            self.pgain.coef.mul_(0.1)
            self.pgain.wb.mul_(0.1)

        self.register_buffer("leak", torch.tensor(0.7))

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        xs = xw * self.FEAT_SCALE
        x = xs[:, c]

        # --- static AM/AM features ---
        env = x.abs()
        env2 = env * env
        env_s = torch.tanh(env)
        env2_s = torch.tanh(env2)

        # slow thermal: centre-weighted decaying envelope-power average
        pw = xs.abs() ** 2
        offs = (torch.arange(self.window, device=xw.device) - c).to(torch.float32)
        w = self.leak ** offs.abs()
        w = w / w.sum()
        thermal = (pw * w).sum(dim=-1)
        thermal_s = torch.tanh(thermal * 4.0)

        pc = p.reshape(B, 1)                     # safe at 0 and 1
        p_s = 2.0 * pc - 1.0                     # centre into [-1,1]
        p2_s = torch.tanh((pc * pc) * 2.0)

        iq = torch.stack([x.real, x.imag], dim=-1)
        f = torch.cat([
            iq,
            env_s.unsqueeze(-1),
            env2_s.unsqueeze(-1),
            thermal_s.unsqueeze(-1),
            p_s,
            p2_s,
        ], dim=-1)
        # extra safety squash so nothing leaves grid support at high drive
        f = torch.tanh(f)

        corr = self.kan2(self.kan1(f))           # (B, 2)

        # power-modulated gain on the nonlinear correction (smooth in p)
        gmod = self.pgain(p_s)                   # (B,1)
        gate = 0.5 + torch.sigmoid(gmod)         # in (0.5, 1.5), safe
        corr = corr * gate

        lin = self.linear_path(iq)               # (B, 2)

        # fast complex FIR memory branch
        xmem = xs[:, self.mem_idx]
        fir_out = self.fir(xmem).squeeze(-1)
        fir_iq = torch.stack([fir_out.real, fir_out.imag], dim=-1)

        o = (lin + corr + fir_iq) / self.FEAT_SCALE
        return torch.complex(o[:, 0], o[:, 1])