"""PositionPredictor: new-atom coordinates in the focal-local frame.

Predicts a Gaussian over the displacement from the focal atom. Because the
density is built from invariant scalars, the resulting position likelihood
is E(3)-invariant up to the focal frame (proposal Sec. 2.2, 2.4).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionPredictor(nn.Module):
    def __init__(self, h_s: int = 128, d_ctx: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * h_s + d_ctx, 256), nn.SiLU(),
            nn.Linear(256, 6),  # mean (3), raw log-var (3)
        )

    def _params(self, g, focal: int, ctx: torch.Tensor):
        n = g.coords.shape[0]
        agg = g.scalars.mean(dim=0, keepdim=True)  # cheap context summary
        x = torch.cat([g.scalars[focal:focal + 1], agg, ctx.unsqueeze(0)], dim=-1)
        mean, raw_lv = self.net(x).split(3, dim=-1)
        return mean.squeeze(0), raw_lv.squeeze(0).clamp(-6.0, 4.0)

    def predict(self, g, focal: int, ctx: torch.Tensor) -> torch.Tensor:
        """Absolute coordinate sample: focal position + displacement."""
        mean, log_var = self._params(g, focal, ctx)
        disp = mean + torch.randn_like(mean) * (0.5 * log_var).exp()
        return g.coords[focal] + disp

    def log_prob(self, g, focal: int, ctx: torch.Tensor, x_abs: torch.Tensor) -> torch.Tensor:
        mean, log_var = self._params(g, focal, ctx)
        d = x_abs - g.coords[focal]
        return (-0.5 * ((d - mean).pow(2) / log_var.exp()
                        + log_var + math.log(2.0 * math.pi))).sum()
