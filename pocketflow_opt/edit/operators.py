"""Editing operators: R-group growth, scaffold hopping, linker redesign.

All three are instances of mask-and-regrow (proposal Table 1, Eq. 5): build
the fixed fragment, then sample the variable region with the focal atom
restricted to the frontier. Guidance (beta) is optional here and handled by
pocketflow_opt.guidance.tilted.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import constraints
from .mask import Fragment, GrownRegion, Mask
from ..models.focal import STOP


# ---------------------------------------------------------------------------
# Operator definitions: each returns (Mask, frontier atoms, extra constraints)
# ---------------------------------------------------------------------------

@dataclass
class RGroupGrowth:
    """Delete substituent(s); regrow from user-specified frontier atoms."""

    substituent_atoms: torch.Tensor
    frontier_atoms: torch.Tensor
    max_new_atoms: int = 24

    def mask(self, n_atoms: int) -> Mask:
        m = torch.zeros(n_atoms, dtype=torch.bool)
        m[self.substituent_atoms] = True
        return Mask(m)


@dataclass
class ScaffoldHop:
    """Delete the core; keep pharmacophore anchors with a soft restraint."""

    core_atoms: torch.Tensor
    anchor_atoms: torch.Tensor        # kept: donor/acceptor/hydrophobe centers
    anchor_targets: torch.Tensor      # [m, 3] pocket-side anchor positions
    restraint_w: float = 5.0          # w_k in U_ph (Eq. 7)

    def mask(self, n_atoms: int) -> Mask:
        m = torch.zeros(n_atoms, dtype=torch.bool)
        m[self.core_atoms] = True
        m[self.anchor_atoms] = False
        return Mask(m)

    def restraint(self, coords: torch.Tensor, anchor_map: dict) -> torch.Tensor:
        """U_ph = sum_k w_k (||x_pi(k) - a_k|| - d0_k)^2  (Eq. 7, distance part)."""
        total = coords.new_zeros(())
        for k, atom_idx in anchor_map.items():
            total = total + self.restraint_w * (
                coords[atom_idx] - self.anchor_targets[k]).pow(2).sum()
        return total


@dataclass
class LinkerRedesign:
    """Keep two warheads; regrow a connected path between them."""

    linker_atoms: torch.Tensor
    warhead_a_frontier: torch.Tensor
    warhead_b_frontier: torch.Tensor
    min_len: int = 2
    max_len: int = 14

    def mask(self, n_atoms: int) -> Mask:
        m = torch.zeros(n_atoms, dtype=torch.bool)
        m[self.linker_atoms] = True
        return Mask(m)


# ---------------------------------------------------------------------------
# The regrowth loop (proposal Algorithm 1)
# ---------------------------------------------------------------------------

def regrow(model, fragment: Fragment, pocket, budget: int = 24,
           beta: dict | None = None, eta: float = 0.02,
           temperature: float = 1.0) -> GrownRegion:
    """Sample a variable region conditioned on a frozen fragment (Eq. 5).

    beta: {predictor_name: (weight, callable)} enabling tilted sampling;
    when None, plain prior sampling (WP1 without guidance).
    """
    from ..guidance.tilted import mala_step  # local import: avoid cycle

    merged = fragment.graph  # pocket merged upstream by the caller
    ctx = model.encode(merged)
    grown = GrownRegion()
    base = merged.coords.shape[0]

    for _ in range(budget):
        frontier = torch.cat([
            fragment.frontier.to(merged.coords.device),
            grown.atoms(base, device=merged.coords.device)])
        focal = model.focal.sample(ctx.graph, ctx.ctx.squeeze(0), frontier,
                                   temperature=temperature)
        if focal is STOP:
            break
        if beta:
            z = mala_step(model, ctx, beta, eta)
        else:
            z = model.atom_flow.prior_sample(ctx.ctx)
        elem = model.atom_flow.decode(z, ctx.ctx)
        xyz = model.position.predict(ctx.graph, focal, ctx.ctx.squeeze(0))
        bonds = model.bond_flow.decode(model.bond_flow.prior_sample(ctx.ctx),
                                       ctx.ctx)
        elem_idx = int(elem.argmax(-1).item())
        bond_map = {focal: _bond_attr(bonds, merged)}
        if not constraints.ok(grown, merged.coords, pocket, elem_idx, xyz,
                              bond_map):
            continue  # support restriction (Eq. 6): illegal action masked
        grown.add(elem_idx, xyz, bond_map)
        merged = fragment.fuse(grown)
        ctx = model.encode_update(ctx, merged)
    return grown


def _bond_attr(bond_onehot: torch.Tensor, ref) -> torch.Tensor:
    """Lift a bond one-hot to the edge-attribute width of the context graph."""
    d = ref.edge_attr.shape[1]
    out = torch.zeros(d, device=ref.edge_attr.device, dtype=ref.edge_attr.dtype)
    k = int(bond_onehot.argmax(-1).item())
    out[k % d] = 1.0
    return out
