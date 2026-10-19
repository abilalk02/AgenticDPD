import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """VCSEL IM/DD-aware DPD.

    Static compressive AM/AM: KAN driven by [I,Q,|x|,|x|^2,p,thermal] around an
    identity complex-linear path (operating point set by p). A short complex FIR
    (CLinear over a small delay window) captures fast relaxation-osc/TIA memory.
    A leaky-integrated envelope power feeds a slow-thermal feature into the KAN.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8, grid: int = 7,
                 order: int = 3, mem: int = 5, quant_bits: int = 32):
        super().__init__(window)
        c = window // 2
        # short symmetric memory taps around centre
        half = mem // 2
        lo = max(0, c - half)
        hi = min(window, c + half + 1)
        self.mem_idx = list(range(lo, hi))
        self.M = len(self.mem_idx)

        # fast complex FIR memory branch (short)
        self.fir = CLinear(self.M, 1, bias=False, quant_bits=quant_bits)

        # static AM/AM KAN: features [I, Q, |x|, |x|^2, p, thermal]
        self.n_feat = 6
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=quant_bits)

        # identity-initialised complex linear path (2x2 real)
        self.linear_path = RLinear(2, 2, bias=False, quant_bits=quant_bits)

        with torch.no_grad():
            self.linear_path.weight.copy_(torch.eye(2))
            # FIR starts near zero (memory is a small correction)
            self.fir.wr.mul_(0.05)
            self.fir.wi.mul_(0.05)
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)

        # thermal leak factor (fixed, physically slow)
        self.register_buffer("leak", torch.tensor(0.7))

    def forward(self, xw, p):
        c = self.window // 2
        xs = xw * self.FEAT_SCALE          # (B, window) scaled to analyzer RMS
        x = xs[:, c]                        # centre sample

        # --- static AM/AM features ---
        env = x.abs()
        env2 = env * env
        # squash unbounded features into spline support [-1,1]
        env_s = torch.tanh(env)
        env2_s = torch.tanh(env2)

        # slow thermal: leaky-integrated envelope power across window
        pw = (xs.abs() ** 2)               # (B, window)
        w = self.leak ** torch.arange(
            self.window, device=xw.device, dtype=torch.float32).abs()
        # centre-weighted decaying average (parameter-free)
        offs = torch.arange(self.window, device=xw.device) - c
        w = self.leak ** offs.abs().to(torch.float32)
        w = w / w.sum()
        thermal = (pw * w).sum(dim=-1)
        thermal_s = torch.tanh(thermal * 4.0)

        iq = torch.stack([x.real, x.imag], dim=-1)
        f = torch.cat([
            iq,
            env_s.unsqueeze(-1),
            env2_s.unsqueeze(-1),
            p.unsqueeze(-1),
            thermal_s.unsqueeze(-1),
        ], dim=-1)

        corr = self.kan2(self.kan1(f))                      # (B, 2)
        lin = self.linear_path(iq)                          # (B, 2)

        # fast complex FIR memory branch
        xmem = xs[:, self.mem_idx]                           # (B, M)
        fir_out = self.fir(xmem).squeeze(-1)                 # (B,) complex
        fir_iq = torch.stack([fir_out.real, fir_out.imag], dim=-1)

        o = (lin + corr + fir_iq) / self.FEAT_SCALE
        return torch.complex(o[:, 0], o[:, 1])