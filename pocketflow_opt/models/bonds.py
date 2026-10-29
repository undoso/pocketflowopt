"""BondFlow: conditional flow over bond types to existing atoms.

Bond variable per candidate pair in {none, single, double, triple}; handled
with the same dequantized-flow construction as the element variable
(proposal Eq. 2-3), with the 'none' level absorbing no-bond proposals.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .flows import CondFlow

BOND_TYPES = ("none", "single", "double", "triple")


class BondFlow(nn.Module):
    def __init__(self, d_ctx: int = 256, depth: int = 4, hidden: int = 128):
        super().__init__()
        self.flow = CondFlow(dim=len(BOND_TYPES), d_ctx=d_ctx, depth=depth,
                             hidden=hidden)

    def decode(self, z: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        u = self.flow.decode(z, ctx)
        return self.flow.quantize(u)

    def prior_sample(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.flow.prior_sample(ctx)

    def log_prob(self, bond_onehot: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        u = self.flow.dequantize(bond_onehot)
        return self.flow.log_prob(u, ctx)
