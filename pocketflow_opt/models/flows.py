"""Conditional affine coupling flows (proposal Sec. 2.3, Eq. 2).

An invertible stack mapping dequantized discrete variables u (e.g. relaxed
one-hot element vectors) to a Gaussian latent z, conditioned on the GDBP
context embedding c. Exact log-likelihoods via the change-of-variables
formula; invertibility is what makes guided latent sampling (WP2) cheap.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

LOG_2PI = math.log(2.0 * math.pi)


def log_normal(z: torch.Tensor) -> torch.Tensor:
    """Standard normal log-density, summed over the last dim."""
    return -0.5 * (z.pow(2).sum(-1) + z.shape[-1] * LOG_2PI)


class AffineCoupling(nn.Module):
    """One conditional affine coupling layer.

    Splits the input into (u_a, u_b); u_b is scaled/shifted by a network
    fed with (u_a, context c). log|det J| accumulates the scale trace.
    """

    def __init__(self, dim: int, d_ctx: int, hidden: int = 256, flip: bool = False):
        super().__init__()
        self.flip = flip
        d_a = dim // 2
        d_b = dim - d_a
        self.net = nn.Sequential(
            nn.Linear(d_a + d_ctx, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * d_b),
        )
        self.d_a, self.d_b = d_a, d_b

    def _split(self, u: torch.Tensor):
        if self.flip:
            return u[..., self.d_b:], u[..., :self.d_b]
        return u[..., :self.d_a], u[..., self.d_a:]

    def _combine(self, a: torch.Tensor, b: torch.Tensor):
        if self.flip:
            return torch.cat([b, a], dim=-1)
        return torch.cat([a, b], dim=-1)

    def forward(self, u: torch.Tensor, c: torch.Tensor):
        """u -> z, returns (z, log_det)."""
        a, b = self._split(u)
        log_s, t = self.net(torch.cat([a, c], dim=-1)).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * 3.0                     # bounded scales
        z_b = b * log_s.exp() + t
        return self._combine(a, z_b), log_s.sum(-1)

    def inverse(self, z: torch.Tensor, c: torch.Tensor):
        """z -> u, returns (u, log_det_of_inverse)."""
        a, z_b = self._split(z)
        log_s, t = self.net(torch.cat([a, c], dim=-1)).chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * 3.0
        b = (z_b - t) * (-log_s).exp()
        return self._combine(a, b), -log_s.sum(-1)


class CondFlow(nn.Module):
    """Conditional normalizing flow over a dim-sized variable (Eq. 2)."""

    def __init__(self, dim: int, d_ctx: int, depth: int = 6, hidden: int = 256):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            AffineCoupling(dim, d_ctx, hidden, flip=(i % 2 == 1))
            for i in range(depth))

    def log_prob(self, u: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """log p(u | c) = log N(z; 0, I) + log |det J|  (Eq. 3)."""
        log_det = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
        for blk in self.blocks:
            u, ld = blk(u, c)
            log_det = log_det + ld
        return log_normal(u) + log_det

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Generator map G_theta: latent -> data space (Sec. 4.2.2)."""
        u = z
        for blk in reversed(self.blocks):
            u, _ = blk.inverse(u, c)
        return u

    def prior_sample(self, c: torch.Tensor) -> torch.Tensor:
        return torch.randn(c.shape[0], self.dim, device=c.device, dtype=c.dtype)

    def dequantize(self, onehot: torch.Tensor) -> torch.Tensor:
        """Discrete -> continuous relaxation used during training (Eq. 2)."""
        return onehot + torch.rand_like(onehot)

    def quantize(self, u: torch.Tensor) -> torch.Tensor:
        """Decode back to a hard one-hot/category index."""
        idx = u.argmax(dim=-1)
        out = torch.zeros_like(u)
        out.scatter_(-1, idx.unsqueeze(-1), 1.0)
        return out
