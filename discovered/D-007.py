import torch
import math
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import KANLayer, CLinear, RLinear


class DPD(DPDModule):
    """VCSEL IM/DD-aware DPD.

    Inductive biases:
      * Compressive static AM/AM (L-I saturation): a KAN on the envelope
        magnitude r=|x| (conditioned on drive power p and slow thermal
        envelope) yields a complex gain applied to the centre sample,
        preserving phase structure — the natural basis for AM/AM + AM/PM.
      * Short memory (relaxation oscillation + TIA bandwidth): a compact
        complex FIR over the tap window (CLinear -> 1 tap) plus an
        envelope-power memory-polynomial cross term.
      * Slow thermal dynamics: a smoothed window envelope drives a gentle
        operating-point shift into the static AM/AM KAN.
    """

    FEAT_SCALE = 0.21

    def __init__(self, window: int, hidden: int = 8, grid: int = 7,
                 order: int = 3, quant_bits: int = 32):
        super().__init__(window)
        self.window = window
        c = window // 2

        # --- static AM/AM branch: 1-D nonlinearity of [r, p, thermal] ---
        # r = |x_centre|, p = drive power, therm = smoothed window envelope.
        self.kan1 = KANLayer(3, hidden, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)
        # outputs a complex gain (gI, gQ) around identity
        self.kan2 = KANLayer(hidden, 2, grid, order, x_min=-1.0, x_max=1.0,
                             quant_bits=quant_bits)

        # --- short-memory complex FIR (relaxation osc + TIA) ---
        self.fir = CLinear(window, 1, bias=False, quant_bits=quant_bits)

        # --- memory-polynomial cross term: recent |x|^2 * centre-neighborhood ---
        # feed [r_{-1}^2, r_0^2, r_{+1}^2] scaled, produce complex correction
        self.mp = RLinear(3, 2, bias=True, quant_bits=quant_bits)

        with torch.no_grad():
            # FIR init: pass the centre tap through (identity linear path)
            self.fir.wr.zero_()
            self.fir.wi.zero_()
            self.fir.wr[0, c] = 1.0 / self.FEAT_SCALE  # undo scale below
            # small nonlinear corrections
            for l in (self.kan1, self.kan2):
                l.coef.mul_(0.1)
                l.wb.mul_(0.1)
            self.mp.weight.mul_(0.05)
            self.mp.bias.zero_()

    def forward(self, xw, p):
        c = self.window // 2
        B = xw.shape[0]

        # scaled window (analyzer scale, inside spline support)
        xs = xw * self.FEAT_SCALE
        xc = xs[:, c]

        # --- envelope features (free) ---
        r = torch.abs(xc)                                   # (B,) centre magnitude
        env = torch.abs(xs)                                 # (B, window)
        therm = env.mean(dim=-1)                            # slow thermal proxy

        # AM/AM static branch input: squash into [-1,1]
        rf = torch.tanh(r * 2.0)                            # magnitude feature
        tf = torch.tanh(therm * 2.0)
        pf = (p * 2.0 - 1.0)                                # p in [0,1] -> [-1,1]
        feat = torch.stack([rf, pf, tf], dim=-1)            # (B,3)

        gain = self.kan2(self.kan1(feat))                   # (B,2) complex gain corr
        g = torch.complex(1.0 + gain[:, 0], gain[:, 1])     # gain around identity
        am_out = xc * g                                     # phase-preserving AM/AM+PM

        # --- short-memory FIR (complex) ---
        fir_out = self.fir(xs)[:, 0]                        # (B,) complex

        # --- memory-polynomial cross term ---
        # recent envelope powers around centre
        idx_m = max(c - 1, 0)
        idx_p = min(c + 1, self.window - 1)
        p2 = torch.stack([
            env[:, idx_m] ** 2,
            env[:, c] ** 2,
            env[:, idx_p] ** 2,
        ], dim=-1)                                          # (B,3)
        mp = self.mp(p2)                                    # (B,2)
        mp_c = torch.complex(mp[:, 0], mp[:, 1]) * xc       # modulate centre

        # combine: FIR (linear memory, carries identity) + AM/AM residual + MP
        # subtract linear centre once so am_out contributes only its correction
        out = fir_out + (am_out - xc) + mp_c

        return (out / self.FEAT_SCALE).to(torch.complex64)