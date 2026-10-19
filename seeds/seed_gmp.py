
import torch
from eqdgm.dpd.interface import DPDModule
from eqdgm.dpd.primitives import GMPBranch


class DPD(DPDModule):
    """Generalized memory polynomial (power-agnostic classic)."""

    def __init__(self, window: int, orders=(1, 3, 5, 7), mem: int = 5, cross: int = 2):
        super().__init__(window)
        self.gmp = GMPBranch(window, orders, mem, cross)

    def forward(self, xw, p):
        return self.gmp(xw)
