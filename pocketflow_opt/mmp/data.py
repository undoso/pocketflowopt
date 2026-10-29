"""Matched molecular pair assembly (proposal Sec. 4.3).

Input is a ChEMBL-like export with columns:
    target_id, smiles_old, smiles_new, delta_activity
where delta_activity = pA(new) - pA(old) for the shared target. Edit
descriptors (fingerprint deltas) are computed for the AL loop's GP.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import torch


@dataclass
class MMPPair:
    target_id: str
    old: object   # featurized molecule (graph or embedding, per featurizer)
    new: object
    delta_a: float

    @property
    def direction(self) -> float:
        return 1.0 if self.delta_a >= 0 else -1.0


def load_pairs_csv(path: str, featurizer=None) -> list[MMPPair]:
    """Load MMP pairs; featurizer(smiles) -> any representation the scorer
    consumes (defaults to raw SMILES strings)."""
    featurizer = featurizer or (lambda s: s)
    pairs: list[MMPPair] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            pairs.append(MMPPair(
                target_id=row["target_id"],
                old=featurizer(row["smiles_old"]),
                new=featurizer(row["smiles_new"]),
                delta_a=float(row["delta_activity"]),
            ))
    return pairs


def fingerprint_delta(old_bits: torch.Tensor, new_bits: torch.Tensor) -> torch.Tensor:
    """Edit descriptor q used by the GP surrogate in WP4 (Eq. 16 input)."""
    return torch.cat([new_bits - old_bits, new_bits * old_bits], dim=-1)
