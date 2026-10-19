import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import CLinear, RLinear, KANLayer


class DPD(DPDModule):
    """Ultra-low-BOP envelope-conditioned KAN predistorter.

    Targets the empty low-BOP niche (~10^4.0-10^4.5 BOP, ~200 RMpS) with an
    8-bit quantized KAN backbone. Complex FIR carries the linear response; a
    grid-squashed KAN produces a complex gain modulation driven by envelope
    powers and a smooth power feature. Built for tight high-drive (3 dBFS)
    worst-seed ACLR.
    """

    def __init__(self, window: int, taps: int = 9, nfeat: int = 6,
                 grid: int = 7, qbits: int = 8):
        super().__init__(window)
        self.taps = min(taps, window) | 1
        self.nfeat = nfeat

        # --- Linear complex FIR (dominant response) ---
        self.fir = CLinear(self.taps, 1, bias=False, quant_bits=qbits)
        with torch.no_grad():
            self.fir.wr.zero_(); self.fir.wi.zero_()
            self.fir.wr[0, self.taps // 2] = 1.0

        # --- Tap mixer -> small complex basis for nonlinear branch ---
        self.mix = CLinear(self.taps, nfeat, bias=False, quant_bits=qbits)
        with torch.no_grad():
            self.mix.wr.normal_(0.0, 0.15)
            self.mix.wi.normal_(0.0, 0.15)

        # --- Smooth learned 1-D function of power p (robust conditioning) ---
        self.pmap = RLinear(1, 4, bias=True, quant_bits=qbits)

        # --- KAN backbone on [envelope powers (nfeat), p-features (4)] ---
        # d_in = nfeat + 4 ; squashed into grid support before entry.
        self.kan = KANLayer(d_in=nfeat + 4, d_out=2 * nfeat, grid=grid,
                            order=3, x_min=-1.5, x_max=1.5, quant_bits=qbits)

        # --- Head: combine gated complex basis into correction sample ---
        self.head = CLinear(nfeat, 1, bias=False, quant_bits=qbits)
        with torch.no_grad():
            self.head.wr.zero_(); self.head.wi.zero_()

        # --- Power-dependent scalar gain on the nonlinear correction ---
        self.pgain = RLinear(4, 1, bias=True, quant_bits=qbits)
        with torch.no_grad():
            self.pgain.weight.zero_()
            self.pgain.bias.fill_(1.0)

    def forward(self, xw, p):
        B = xw.shape[0]
        c = self.window // 2
        h = self.taps // 2
        xc = xw[:, c - h : c + h + 1]              # (B, taps) complex

        # Linear path
        lin = self.fir(xc).squeeze(-1)            # (B,) complex

        # Nonlinear complex basis from tap mixer
        z = self.mix(xc)                          # (B, nfeat) complex

        # Free envelope features (real), normalized power terms
        env = z.abs()                             # (B, nfeat) real
        # squash into grid support
        env_s = 1.5 * torch.tanh(env / 1.5)

        # Smooth learned power features (safe at p=0 and p=1, no divides)
        pc = p.reshape(B, 1)                      # (B,1) in ~[0,1]
        pf = torch.tanh(self.pmap(pc))            # (B,4) real, bounded
        pf_s = 1.5 * pf                           # already within [-1.5,1.5]

        feat = torch.cat([env_s, pf_s], dim=1)    # (B, nfeat+4)
        feat = 1.5 * torch.tanh(feat / 1.5)       # final safety squash

        g = self.kan(feat)                        # (B, 2*nfeat) real
        gr = g[:, : self.nfeat]
        gi = g[:, self.nfeat :]
        gate = torch.complex(torch.tanh(gr), torch.tanh(gi))  # (B, nfeat) cplx

        # Gated complex basis -> correction sample
        nl = self.head(z * gate).squeeze(-1)      # (B,) complex

        # Power-dependent real gain on nonlinear correction
        pg = self.pgain(pf).squeeze(-1)           # (B,) real
        nl = nl * pg.to(nl.dtype)

        y = lin + nl                              # (B,) complex64
        return y