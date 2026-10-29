"""Guided sampling in flow latent space (proposal Sec. 4.2, Eqs. 8-9).

Targets pi_beta(z) prop_to N(z;0,I) * exp(sum_k beta_k r_k(G(z))) with
Metropolis-adjusted Langevin dynamics. Gradients flow through the
invertible decode map; discrete variables are handled via the dequantized
relaxation (Gumbel-softmax variant available for hard relaxation).
"""
from __future__ import annotations

import math

import torch

from ..models.flows import log_normal


def _energy(model, z: torch.Tensor, ctx, beta: dict) -> torch.Tensor:
    """sum_k beta_k r_k(G_theta(z)) evaluated on the decoded candidate."""
    cand = model.atom_flow.decode(z, ctx.ctx)
    total = torch.zeros((), device=z.device, dtype=z.dtype)
    for _, (weight, fn) in beta.items():
        total = total + weight * torch.as_tensor(fn(cand), device=z.device,
                                                 dtype=z.dtype)
    return total


def _log_target(model, z: torch.Tensor, ctx, beta: dict) -> torch.Tensor:
    return log_normal(z) + _energy(model, z, ctx, beta)


def accept(model, z: torch.Tensor, z_prop: torch.Tensor, ctx, beta: dict,
           eta: float, grad_old: torch.Tensor, grad_new: torch.Tensor) -> bool:
    """Metropolis correction for the discretized Langevin proposal (Eq. 9)."""
    with torch.no_grad():
        log_alpha = (_log_target(model, z_prop, ctx, beta)
                     - _log_target(model, z, ctx, beta))
        # proposal asymmetry terms of MALA
        log_alpha = log_alpha - (z - z_prop - 0.5 * eta * grad_new).pow(2).sum() / (2 * eta)
        log_alpha = log_alpha + (z_prop - z - 0.5 * eta * grad_old).pow(2).sum() / (2 * eta)
    return math.log(torch.rand(1).item()) < float(log_alpha.clamp(max=0.0))


def mala_step(model, ctx, beta: dict, eta: float = 0.02,
              n_steps: int = 8) -> torch.Tensor:
    """Run n_steps of MALA in the atom-flow latent and return the sample."""
    z = model.atom_flow.prior_sample(ctx.ctx)
    z = z.detach().requires_grad_(True)
    for _ in range(n_steps):
        target = _log_target(model, z, ctx, beta)
        grad = torch.autograd.grad(target.sum(), z)[0]
        noise = torch.randn_like(z)
        z_prop = (z + 0.5 * eta * grad + math.sqrt(eta) * noise).detach()
        z_prop.requires_grad_(True)
        target_prop = _log_target(model, z_prop, ctx, beta)
        grad_prop = torch.autograd.grad(target_prop.sum(), z_prop)[0]
        if accept(model, z.detach(), z_prop.detach(), ctx, beta, eta,
                  grad.detach(), grad_prop.detach()):
            z = z_prop
        z = z.detach().requires_grad_(True)
    return z.detach()


def gumbel_relax(logits: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Gumbel-softmax relaxation for discrete element/bond variables (Eq. 10)."""
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
    return torch.softmax((logits + g) / temperature, dim=-1)
