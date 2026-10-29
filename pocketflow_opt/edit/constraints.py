"""Chemical-knowledge constraints: the support restriction of Eq. (6).

Local checks applied at every autoregressive step: element valence limits,
bond-length ranges, a light ring-closure rule, and the minimum
protein-ligand distance. Because all checks depend only on the partial
state, they transfer unchanged from de novo to fragment-anchored sampling.
"""
from __future__ import annotations

import torch

VALENCE = {0: 1, 1: 4, 2: 3, 3: 2, 4: 1, 5: 3, 6: 2, 7: 1, 8: 1, 9: 1}
# index order follows models.pocketflow.ELEMENTS: H C N O F P S Cl Br I

BOND_RANGE = {1: (1.0, 1.8), 2: (1.1, 1.6), 3: (1.05, 1.45)}  # Angstrom
MIN_POCKET_DIST = 2.2   # Angstrom; hard clash cutoff
MAX_DOUBLE_BONDS_IN_RING6 = 2


def valence_ok(elem_idx: int, existing_bond_order: int, new_order: int) -> bool:
    return existing_bond_order + new_order <= VALENCE.get(elem_idx, 4)


def bond_length_ok(order: int, dist: float) -> bool:
    lo, hi = BOND_RANGE.get(order, (1.0, 1.8))
    return lo <= dist <= hi


def min_distance_ok(xyz: torch.Tensor, pocket) -> bool:
    if pocket is None or pocket.coords.numel() == 0:
        return True
    d = (pocket.coords - xyz).pow(2).sum(-1).sqrt().min().item()
    return d >= MIN_POCKET_DIST


def ring_rule_ok(grown, order: int) -> bool:
    """Heuristic ring rule: cap double bonds inside a closing 6-ring."""
    if order != 2 or grown.size() < 5:
        return True
    recent = grown._records[-5:]
    n_double = sum(1 for _, _, b in recent
                   if any(int(a.argmax().item()) == 2 for a in b.values()))
    return n_double < MAX_DOUBLE_BONDS_IN_RING6


def ok(grown, fragment, pocket, elem_idx: int, xyz: torch.Tensor,
       bond_map: dict) -> bool:
    """Aggregate step-level legality check (Eq. 6 indicator)."""
    if not min_distance_ok(xyz, pocket):
        return False
    for dst, attr in bond_map.items():
        order = int(attr.argmax().item())
        if order == 0:
            continue
        dist = (fragment.graph.coords[dst] - xyz).norm().item()
        if not bond_length_ok(order, dist):
            return False
        if not valence_ok(elem_idx, 0, order):
            return False
        if not ring_rule_ok(grown, order):
            return False
    return True
