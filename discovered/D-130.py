import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, KANLayer, RLinear, GMPBranch


class DPD(DPDModule):
    """Envelope-conditioned KAN DPD with explicit smooth power gating and a
    high-order GMP high-drive term. Built on the KAN lineage that natively
    clears the 3 dBFS ceiling (NOT g015c3/g021c2).

    Key high-drive moves vs parent:
      - tighter spline squash (1.4 into +/-1.5 grid) so peaks never leave grid
      - dedicated power-gate KAN of [p, p^2] modulating residual gain/corr
      - GMPBranch high-order memory-poly correction (strong at peak envelope)
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int,
                 n_mix: int = 18,
                 hidden: int = 24,
                 grid: int = 12,
                 order: int = 3,
                 quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        self.c = window // 2
        self.SQ = 1.4  # squash bound, safely inside +/-1.5 grid support

        # ---- complex FIR backbone (near-identity init) ----
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # ---- complex tap mix -> envelope basis ----
        self.mix = CLinear(window, n_mix, bias=True, quant_bits=quant_bits)
        self.n_mix = n_mix

        # main KAN input: env basis (n_mix) + |centre| + env^3 centre + p + p^2
        self.kan_in = n_mix + 4
        self.kan1 = KANLayer(self.kan_in, hidden, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)
        self.kan2 = KANLayer(hidden, hidden, grid=grid, order=order,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)

        # ---- smooth power-gate: KAN of [p, p^2] -> gate features ----
        self.pkan = KANLayer(2, hidden, grid=8, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=quant_bits)

        # head: hidden -> [gr, gi, cr, ci]
        self.head = RLinear(hidden, 4, bias=True, quant_bits=quant_bits)

        # ---- high-order GMP high-drive correction ----
        self.gmp = GMPBranch(window, orders=(1, 3, 5, 7), mem=3, cross=2)

        with torch.no_grad():
            # near-identity FIR anchor at centre tap
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, self.c] = 1.0
            # damp KANs so model starts near-linear
            for l in (self.kan1, self.kan2, self.pkan):
                if hasattr(l, "coef"):
                    l.coef.mul_(0.1)
                if hasattr(l, "wb"):
                    l.wb.mul_(0.1)
            self.head.weight.mul_(0.05)
            self.head.bias.zero_()

    def _squash(self, t):
        return self.SQ * torch.tanh(t / self.SQ)

    def forward(self, xw, p):
        x = xw * self.FEAT_SCALE
        B = x.shape[0]

        # linear anchor
        lin = self.fir(x)[:, 0]                           # complex (B,)

        # complex envelope basis
        z = self.mix(x)                                   # complex (B, n_mix)
        env = z.abs()                                     # real (B, n_mix)

        # centre-tap magnitude + higher power feature
        xc = x[:, self.c]
        ec = xc.abs().reshape(B, 1)                       # (B,1)
        ec3 = (ec * ec * ec)                              # (B,1) high-drive emphasis
        pc = p.reshape(B, 1)                              # (B,1), safe at 0/1
        pc2 = pc * pc

        feat = torch.cat([env, ec, ec3, pc, pc2], dim=1)  # (B, kan_in)
        feat = self._squash(feat)

        h = self.kan1(feat)
        h = self._squash(h)
        h = self.kan2(h)                                  # (B, hidden)

        # smooth power gate from [p, p^2]
        pin = self._squash(torch.cat([pc, pc2], dim=1))   # (B,2)
        gate = torch.sigmoid(self.pkan(pin))              # (B, hidden), free

        h = h * gate                                      # power-modulated

        out = self.head(h)                                # (B,4)
        gr = out[:, 0]
        gi = out[:, 1]
        cr = out[:, 2]
        ci = out[:, 3]

        gain = torch.complex(1.0 + gr, gi)
        corr = torch.complex(cr, ci)

        # high-order GMP high-drive correction (complex)
        gmp_out = self.gmp(x)
        if gmp_out.dim() > 1:
            gmp_out = gmp_out.reshape(B, -1)[:, 0]
        gmp_out = gmp_out.to(torch.complex64)

        o = (lin * gain + corr + gmp_out) / self.FEAT_SCALE
        return o.to(torch.complex64)