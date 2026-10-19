import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Mid-scale scaled-KAN DPD for the 1-6k RMpS / empty high-BOP niche.
       - complex FIR linear memory
       - smooth learned 1-D power modulation (KAN on p) -> gain/phase warp seeds
       - wide two-stage envelope-conditioned KAN AM/AM + AM/PM
       - power-gated short GMP complex residual
       KAN backbone scales gracefully & holds high-drive ACLR across seeds."""

    def __init__(self, window: int, taps: int = 11, ntaps: int = 7,
                 kan_grid: int = 10, kan_order: int = 3,
                 nfeat: int = 16, nhid: int = 12):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.ntaps = min(ntaps, window) | 1
        self.nfeat = nfeat
        self.nhid = nhid

        # ---- 1) Linear complex FIR memory ----
        self.fir = CLinear(self.taps, 1, bias=False, quant_bits=8)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # ---- 2) Complex mix of taps -> feature basis (envelope descriptors) ----
        self.mix = CLinear(window, nfeat, bias=True, quant_bits=8)

        # ---- 3) Smooth learned 1-D power modulation via small KAN ----
        # maps squashed p -> nfeat gain-warp coefficients
        self.pmod = KANLayer(1, nfeat, grid=6, order=3,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 4) Wide feature KAN: envelope powers + p-warp -> hidden ----
        # input dim: nfeat envelopes + centre|x| + mean-env + p-scalar = nfeat+3
        self.kan1 = KANLayer(nfeat + 3, nhid, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 5) Output KAN: hidden -> [gain, phase] static NL ----
        self.kan2 = KANLayer(nhid, 2, grid=kan_grid, order=kan_order,
                             x_min=-1.5, x_max=1.5, quant_bits=8)

        # ---- 6) Short GMP complex memory nonlinear residual ----
        self.nl = CLinear(3 * self.ntaps, 1, bias=False, quant_bits=6)
        with torch.no_grad():
            self.nl.wr.zero_(); self.nl.wi.zero_()

        # power gate: learned quadratic in p
        self.pgain = RLinear(2, 1, bias=True, quant_bits=6)
        with torch.no_grad():
            self.pgain.weight.zero_(); self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        h = self.taps // 2
        hn = self.ntaps // 2

        xc = xw[:, c]                                          # centre (complex)
        env = (xw.real * xw.real + xw.imag * xw.imag).sqrt()   # (B, window)

        # ---- 1) linear FIR ----
        lin = self.fir(xw[:, c - h: c + h + 1]).squeeze(-1)

        # ---- 2) complex feature mix -> envelope basis (free abs) ----
        z = self.mix(xw)                                       # complex (B,nfeat)
        zenv = z.abs()                                         # (B,nfeat) real

        # ---- 3) smooth 1-D power modulation ----
        pc = p.reshape(B, 1)
        pin = 1.5 * torch.tanh(2.0 * (pc - 0.5))               # (B,1) in grid
        pwarp = self.pmod(pin)                                 # (B,nfeat)
        pgate = torch.sigmoid(pwarp)                           # (B,nfeat) free

        # modulate envelope features by smooth power warp
        zmod = zenv * pgate                                    # (B,nfeat)

        # ---- free descriptors ----
        ec = env[:, c]                                         # (B,)
        tstate = torch.tanh(env.mean(dim=1))                   # (B,)
        pscalar = torch.tanh(2.0 * (p - 0.5))                  # (B,)

        # ---- 4) wide feature KAN ----
        feat = torch.cat([zmod,
                          ec.reshape(B, 1),
                          tstate.reshape(B, 1),
                          pscalar.reshape(B, 1)], dim=1)        # (B,nfeat+3)
        feat = 1.5 * torch.tanh(feat / 1.5)                    # squash into grid
        hid = self.kan1(feat)                                  # (B,nhid)
        hid = 1.5 * torch.tanh(hid / 1.5)                      # keep in grid for kan2

        # ---- 5) output KAN: static AM/AM + AM/PM ----
        kout = self.kan2(hid)                                  # (B,2)
        gain = 1.0 + torch.tanh(kout[:, 0])                    # multiplicative gain
        phase = torch.tanh(kout[:, 1])                         # phase correction
        cph = torch.cos(phase); sph = torch.sin(phase)
        rot = torch.complex(cph, sph)
        amam_out = xc * gain.to(xc.dtype) * rot

        # ---- 6) short GMP complex residual, power-gated ----
        xn = xw[:, c - hn: c + hn + 1]                         # complex (B,ntaps)
        a2 = (xn.real * xn.real + xn.imag * xn.imag)
        t3 = xn * a2
        t5 = xn * a2 * a2
        t7 = xn * a2 * a2 * a2
        feats = torch.cat([t3, t5, t7], dim=1)                 # (B,3*ntaps)
        nl = self.nl(feats).squeeze(-1)
        pf = torch.stack([p, p * p], dim=1)
        g = self.pgain(pf).squeeze(-1)
        nl = nl * g.to(nl.dtype)

        y = lin + amam_out + nl                                # complex (B,)
        return y