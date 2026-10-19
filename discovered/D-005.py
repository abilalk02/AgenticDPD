import torch
import torch.nn as nn
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import RConv1d, RLinear, KANLayer, GMPBranch


class DPD(DPDModule):
    """Power-conditioned mixture-of-experts DPD.

    Shared 8-bit CNN trunk -> K linear correction experts, blended by a
    smooth softmax over p (from a KAN). A KAN also produces a complex power
    gain. A GMP branch provides a robust classical backbone, gated by p.
    """

    def __init__(self, window: int, ch: int = 24, k: int = 7, depth: int = 3,
                 experts: int = 4, expert_hidden: int = 24):
        super().__init__(window)
        self.K = experts

        # ---- shared conv trunk (real, imag, |x|, |x|^2 -> 4 channels) ----
        convs, c_in = [], 4
        for _ in range(depth):
            convs += [RConv1d(c_in, ch, k, padding=k // 2, quant_bits=8),
                      nn.Tanh()]
            c_in = ch
        self.convs = nn.Sequential(*convs)

        # ---- K correction experts (each a small 2-layer head) ----
        self.expert_in = RLinear(ch, expert_hidden, quant_bits=16)
        self.expert_out = RLinear(expert_hidden, 2 * experts, quant_bits=16)

        # ---- power conditioner: KAN(p) -> mix logits (K) + complex gain(2)
        self.pcond = KANLayer(1, experts + 2, grid=8, order=3,
                              x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- classical GMP backbone, gated by a power gain ----
        self.gmp = GMPBranch(window, orders=(1, 3, 5), mem=5, cross=2)
        self.pgmp = KANLayer(1, 2, grid=6, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        env = xw.abs()

        # shared feature extraction
        f = torch.stack([xw.real, xw.imag, env, env * env], dim=1)
        h = self.convs(f)[:, :, c]                       # (B, ch)

        # experts
        e = torch.tanh(self.expert_in(h))
        e = self.expert_out(e)                           # (B, 2K)
        e = e.view(B, self.K, 2)                         # (B, K, 2)
        expert_c = torch.complex(e[:, :, 0], e[:, :, 1]) # (B, K)

        # power conditioning: mix weights + complex gain
        ps = 3.0 * (p.unsqueeze(-1) - 0.5)               # (B,1) in support
        cond = self.pcond(ps)                            # (B, K+2)
        logits = cond[:, :self.K]                        # (B, K)
        w = torch.softmax(logits, dim=-1)                # (B, K) free
        gr = 1.0 + torch.tanh(cond[:, self.K])
        gi = torch.tanh(cond[:, self.K + 1])
        gain = torch.complex(gr, gi)                     # (B,)

        # blended expert correction (elementwise, no matmul)
        wc = torch.complex(w, torch.zeros_like(w))
        mixed = (expert_c * wc).sum(dim=1)               # (B,)
        corr_nn = mixed * gain

        # classical GMP backbone, power-gated
        gmp_out = self.gmp(xw)                           # (B,) complex
        gp = self.pgmp(ps)                               # (B,2)
        ggr = 1.0 + torch.tanh(gp[:, 0])
        ggi = torch.tanh(gp[:, 1])
        gmp_gain = torch.complex(ggr, ggi)
        corr_gmp = gmp_out * gmp_gain

        return xw[:, c] + corr_nn + corr_gmp