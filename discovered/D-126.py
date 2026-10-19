import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear, GMPBranch


class DPD(DPDModule):
    """Lean g005c2-core power-modulated KAN DPD (~300 RMpS band).

    Identity complex-linear backbone + short complex-FIR envelope basis +
    single wide-grid AM/AM KAN correction + GMP memory-polynomial skip.
    A smooth learned 1-D gain g(p) modulates only the nonlinear residual so
    one model linearizes the whole power sweep. All spline inputs are squashed
    into the grid support to guarantee seed-tight high-drive behaviour.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8, grid: int = 6,
                 order: int = 3, mem: int = 5, quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        half = mem // 2
        lo = max(0, c - half)
        hi = min(window, c + half + 1)
        self.mem_idx = list(range(lo, hi))
        self.M = len(self.mem_idx)

        # short complex FIR memory branch (cheap, complex coeffs)
        self.fir = CLinear(self.M, 1, bias=False, quant_bits=quant_bits)

        # GMP memory-polynomial skip (complex, robust, 4 RMpS/term)
        gmp_mem = min(5, window)
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=gmp_mem, cross=1)

        # static AM/AM KAN: [I, Q, |x|, |x|^2, thermal, p]
        self.n_feat = 6
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # smooth learned 1-D gain of p: [p, p^2] -> scalar gate
        self.pgain = KANLayer(2, 1, grid, order,
                              x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # identity-initialised complex linear backbone (2x2 real)
        self.linear_path = RLinear(2, 2, bias=False, quant_bits=quant_bits)

        with torch.no_grad():
            self.linear_path.weight.copy_(torch.eye(2))
            self.fir.wr.mul_(0.05)
            self.fir.wi.mul_(0.05)
            for l in (self.kan1, self.kan2, self.pgain):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

        self.register_buffer("leak", torch.tensor(0.7))

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        xs = xw * self.FEAT_SCALE
        x = xs[:, c]

        # --- static AM/AM features, all squashed into grid support ---
        env = x.abs()
        env_s = torch.tanh(env)
        env2_s = torch.tanh(env * env)

        # slow thermal: centre-weighted decaying envelope power
        pw = xs.abs() ** 2
        offs = torch.arange(self.window, device=xw.device) - c
        w = self.leak ** offs.abs().to(torch.float32)
        w = w / w.sum()
        thermal = (pw * w).sum(dim=-1)
        thermal_s = torch.tanh(thermal * 4.0)

        pc = p.reshape(B, 1).to(torch.float32)         # safe at 0 and 1
        p_s = 2.0 * pc - 1.0                            # [0,1] -> [-1,1]
        p2_s = torch.tanh(2.0 * (pc * pc))

        iq = torch.stack([x.real, x.imag], dim=-1)     # (B,2)
        iq_s = torch.tanh(iq)

        f = torch.cat([
            iq_s,
            env_s.unsqueeze(-1),
            env2_s.unsqueeze(-1),
            thermal_s.unsqueeze(-1),
            p_s,
        ], dim=-1)                                      # (B, 6)

        h = self.kan1(f)
        h = torch.tanh(h)                               # keep stage-2 in grid
        corr = self.kan2(h)                             # (B, 2)

        # smooth per-drive gain modulation of the nonlinear correction
        pgin = torch.tanh(torch.cat([p_s, p2_s], dim=1))   # (B,2)
        gate = torch.sigmoid(self.pgain(pgin))             # (B,1) in (0,1)
        corr = corr * gate

        lin = self.linear_path(iq)                         # (B,2)

        xmem = xs[:, self.mem_idx]
        fir_out = self.fir(xmem).squeeze(-1)               # (B,) complex
        fir_iq = torch.stack([fir_out.real, fir_out.imag], dim=-1)

        # GMP memory-polynomial skip, modulated by the same power gate
        gmp_out = self.gmp(xs)
        if gmp_out.dim() > 1:
            gmp_out = gmp_out.reshape(B, -1)[:, 0]
        gmp_out = gmp_out.reshape(B)
        gmp_out = gmp_out * gate.squeeze(-1).to(gmp_out.dtype)
        gmp_iq = torch.stack([gmp_out.real, gmp_out.imag], dim=-1)

        o = (lin + corr + fir_iq + gmp_iq) / self.FEAT_SCALE
        return torch.complex(o[:, 0], o[:, 1]).to(torch.complex64)