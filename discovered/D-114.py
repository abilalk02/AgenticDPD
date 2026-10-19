import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear


class DPD(DPDModule):
    """Ultra-lean power-conditioned single-wide-KAN DPD.

    Targets the 10^4.5-10^5.0 BOP niche (~324 RMpS band), attacking the weak
    high-drive (3 dBFS) ACLR ceiling of the lean archive entries (g021c3 @30.9,
    g027c1 @35.7).  Design:
      - complex FIR memory over the whole window, identity-init pass-through
      - ONE wide KAN over an envelope-dominant, grid-squashed feature basis
        (amp, amp^2 per tap + a couple of real/imag anchors); envelope features
        never leave [-1.5,1.5] so high drive cannot blow up the splines
      - a smooth 1-D KAN(p) -> per-channel gain vector modulating the KAN
        correction channels (drive-dependent basis selection, tight seeds)
      - 8-bit quant on the correction KAN + head to sink BOP into the target
        band while keeping RMpS low
    """

    FEAT_SCALE = 0.21
    GRID_LIM = 1.5

    def __init__(self, window: int, hidden: int = 10, grid: int = 8,
                 order: int = 3, quant_bits: int = 8):
        super().__init__(window)
        self.window = window
        c = window // 2
        offs = [o for o in (-2, -1, 0, 1, 2) if 0 <= c + o < window]
        self.tap_idx = [c + o for o in offs]
        nt = len(self.tap_idx)
        # per tap: [amp, amp^2]  + centre tap re/im anchors
        d_feat = nt * 2 + 2

        # linear memory: complex FIR over whole window (high precision, cheap)
        self.fir = CLinear(window, 1, bias=False, quant_bits=16)

        # wide dense-grid KAN correction core (single layer), quantized
        self.kan = KANLayer(d_feat, hidden, grid, order,
                            x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                            quant_bits=quant_bits)

        # complex head: hidden channels -> complex correction
        self.head = CLinear(hidden, 1, bias=False, quant_bits=quant_bits)

        # smooth 1-D power conditioning -> per-channel gain vector
        self.pg = KANLayer(1, hidden, grid, order,
                           x_min=-self.GRID_LIM, x_max=self.GRID_LIM,
                           quant_bits=quant_bits)

        with torch.no_grad():
            # FIR: identity on centre tap (pass-through linear start)
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0
            # gentle nonlinear correction at init for stability
            for l in (self.kan, self.pg):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.head.wr.mul_(0.1)
            self.head.wi.mul_(0.1)

    def _squash(self, x):
        return self.GRID_LIM * torch.tanh(x / self.GRID_LIM)

    def forward(self, xw, p):
        B = xw.shape[0]
        xs = xw * self.FEAT_SCALE                       # (B, window)

        # --- linear memory branch ---
        lin_c = self.fir(xs)[:, 0]                      # (B,) complex

        # --- envelope-dominant feature stack ---
        cols = []
        for i in self.tap_idx:
            xc = xs[:, i]
            amp = torch.abs(xc)
            cols.append(amp)
            cols.append(amp * amp)
        c = self.window // 2
        xc0 = xs[:, c]
        cols.append(xc0.real)
        cols.append(xc0.imag)
        feats = torch.stack(cols, dim=-1)               # (B, d_feat)
        feats = self._squash(feats)

        # single wide KAN -> hidden channels (real)
        h = self.kan(feats)                             # (B, hidden)

        # --- smooth power-conditioning: per-channel gain vector ---
        pin = (p.reshape(B, 1) * 2.0 - 1.0)             # [0,1] -> [-1,1]
        pin = self._squash(pin)
        g = self.pg(pin)                                # (B, hidden)
        gain = 1.0 + torch.tanh(g)                      # (0, 2) per channel

        h = h * gain                                    # drive-modulated basis
        h = self._squash(h)                             # bound head input

        # complex head: hidden -> complex correction
        corr_c = self.head(h.to(torch.complex64))[:, 0]  # (B,) complex

        out_c = (lin_c + corr_c) / self.FEAT_SCALE
        return out_c.to(torch.complex64)               # (B,) complex64