import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """Scaled KAN AM/AM DPD for the 6-9k RMpS niche (10^6.0-10^6.5 BOP).

    Lineage: KAN-backboned g027/g005c2-style core that natively clears the
    3 dBFS ACLR ceiling. Widened + deepened to land in the empty 6-9k band.
    Identity complex-linear path + complex FIR memory + squashed-input KAN
    AM/AM correction, smoothly gated by a learned 1-D function of drive p so
    a single model linearizes the whole power sweep.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 20, grid: int = 5,
                 order: int = 3, mem: int = 9, quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        half = mem // 2
        lo = max(0, c - half)
        hi = min(window, c + half + 1)
        self.mem_idx = list(range(lo, hi))
        self.M = len(self.mem_idx)

        # complex FIR memory branch
        self.fir = CLinear(self.M, 1, bias=False, quant_bits=quant_bits)

        # static AM/AM KAN: features [I, Q, |x|, |x|^2, thermal, p, p^2]
        self.n_feat = 7
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan3 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # smooth learned 1-D gain function of p: [p, p^2] -> scalar gate
        self.pgain = KANLayer(2, 1, grid, order,
                              x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # identity-initialised complex linear path (2x2 real)
        self.linear_path = RLinear(2, 2, bias=False, quant_bits=quant_bits)

        with torch.no_grad():
            self.linear_path.weight.copy_(torch.eye(2))
            self.fir.wr.mul_(0.05)
            self.fir.wi.mul_(0.05)
            for l in (self.kan1, self.kan2, self.kan3, self.pgain):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

        self.register_buffer("leak", torch.tensor(0.7))

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        xs = xw * self.FEAT_SCALE
        x = xs[:, c]

        # --- static AM/AM features (all squashed into grid support) ---
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

        pc = p.reshape(B, 1)               # safe at 0 and 1
        p_s = 2.0 * pc - 1.0               # map [0,1] -> [-1,1] into grid
        p2_s = torch.tanh(2.0 * (pc * pc))

        iq = torch.stack([x.real, x.imag], dim=-1)
        iq_s = torch.tanh(iq)

        f = torch.cat([
            iq_s,
            env_s.unsqueeze(-1),
            env2_s.unsqueeze(-1),
            thermal_s.unsqueeze(-1),
            p_s,
            p2_s,
        ], dim=-1)                          # (B, 7)

        h = self.kan1(f)
        h = torch.tanh(h)                   # keep stage-2 input in grid
        h = self.kan2(h)
        h = torch.tanh(h)
        corr = self.kan3(h)                 # (B, 2)

        # smooth per-drive gain modulation of the correction
        pgin = torch.cat([p_s, p2_s], dim=1)          # (B,2)
        pgin = torch.tanh(pgin)
        gate = torch.sigmoid(self.pgain(pgin))         # (B,1) in (0,1)
        corr = corr * gate                             # broadcast (B,2)

        lin = self.linear_path(iq)                     # (B,2)

        xmem = xs[:, self.mem_idx]
        fir_out = self.fir(xmem).squeeze(-1)           # (B,) complex
        fir_iq = torch.stack([fir_out.real, fir_out.imag], dim=-1)

        o = (lin + corr + fir_iq) / self.FEAT_SCALE
        return torch.complex(o[:, 0], o[:, 1]).to(torch.complex64)