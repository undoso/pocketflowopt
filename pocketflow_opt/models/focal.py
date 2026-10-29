"""FocalNet: frontier-restricted focal-atom selection (WP1).

Scores existing atoms (plus a STOP symbol) as the next focal atom. In
optimization mode the legal focal set is the frontier of the fixed fragment
union the already-grown region -- everything else is masked to -inf.
"""
from __future__ import annotations

import torch
import torch.nn as nn

STOP = -1  # sentinel action meaning "terminate growth"


class FocalNet(nn.Module):
    def __init__(self, h_s: int = 128, d_ctx: int = 256):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(h_s + d_ctx, 128), nn.SiLU(), nn.Linear(128, 1))
        self.stop_logit = nn.Parameter(torch.zeros(1))

    def logits(self, g, ctx: torch.Tensor, frontier: torch.Tensor) -> torch.Tensor:
        """Per-node logits with a STOP logit appended; non-frontier -> -inf.

        g: GeomGraph of the current context; ctx: [d_ctx] context vector;
        frontier: LongTensor of legal focal node indices.
        """
        n = g.coords.shape[0]
        node_in = torch.cat([g.scalars, ctx.expand(n, -1)], dim=-1)
        lg = self.score(node_in).squeeze(-1)
        legal = torch.zeros(n, dtype=torch.bool, device=lg.device)
        legal[frontier] = True
        lg = lg.masked_fill(~legal, float("-inf"))
        return torch.cat([lg, self.stop_logit.to(lg.device)])

    def sample(self, g, ctx: torch.Tensor, frontier: torch.Tensor,
               temperature: float = 1.0) -> int:
        lg = self.logits(g, ctx, frontier) / max(temperature, 1e-6)
        idx = torch.distributions.Categorical(logits=lg).sample().item()
        return STOP if idx == g.coords.shape[0] else idx
