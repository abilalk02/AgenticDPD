import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """VCSEL IM/DD-aware DPD with mixed-precision quantization and power-gated KAN.

    Static compressive AM/AM: KAN driven by [I,Q,|x|,|x|^2,p,thermal] around an
    identity complex-linear path. A short complex FIR captures fast memory. A tiny
    power gate (function of p and thermal state) scales the KAN correction so the
    high-drive regime — the ACLR bottleneck — is tracked without a full expert path.

    Precision: FIR + linear path @ 8-bit (small corrections, precision-tolerant),
    KAN core @ 16-bit (quality-critical bounded bases), gate @ 8-bit.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8, grid: int = 7,
                 order: int = 3, mem: int = 5):
        super().__init__(window)
        c = window // 2
        half = mem // 2
        lo = max(0, c - half)
        hi = min(window, c + half + 1)
        self.mem_idx = list(range(lo, hi))
        self.M = len(self.mem_idx)

        # fast complex FIR memory branch (8-bit)
        self.fir = CLinear(self.M, 1, bias=False, quant_bits=8)

        # static AM/AM KAN core (16-bit — quality critical)
        self.n_feat = 6
        self.kan1 = KANLayer(self.n_feat, hidden, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=16)
        self.kan2 = KANLayer(hidden, 2, grid, order,
                             x_min=-1.0, x_max=1.0, quant_bits=16)

        # identity-initialised complex linear path (8-bit)
        self.linear_path = RLinear(2, 2, bias=False, quant_bits=8)

        # power gate: [p, p^2, thermal] -> 2 scale terms (I,Q correction gain)
        self.gate = RLinear(3, 2, bias=True, quant_bits=8)

        with torch.no_grad():
            self.linear_path.weight.copy_(torch.eye(2))
            self.fir.wr.mul_(0.05)
            self.fir.wi.mul_(0.05)
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            # gate starts near unity gain (weights ~0, bias ~0 -> tanh->0 -> gain 1)
            self.gate.weight.mul_(0.05)
            if self.gate.bias is not None:
                self.gate.bias.zero_()

        self.register_buffer("leak", torch.tensor(0.7))

    def forward(self, xw, p):
        c = self.window // 2
        xs = xw * self.FEAT_SCALE
        x = xs[:, c]

        env = x.abs()
        env2 = env * env
        env_s = torch.tanh(env)
        env2_s = torch.tanh(env2)

        pw = (xs.abs() ** 2)
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

        # power-dependent gate on the KAN correction (unity at init)
        pf = torch.stack([p, p * p, thermal_s], dim=-1)     # (B, 3)
        gain = 1.0 + torch.tanh(self.gate(pf))              # (B, 2), ~1 at init
        corr = corr * gain

        lin = self.linear_path(iq)                          # (B, 2)

        xmem = xs[:, self.mem_idx]                           # (B, M)
        fir_out = self.fir(xmem).squeeze(-1)                 # (B,) complex
        fir_iq = torch.stack([fir_out.real, fir_out.imag], dim=-1)

        o = (lin + corr + fir_iq) / self.FEAT_SCALE
        return torch.complex(o[:, 0], o[:, 1])