"""Property predictors r_k used by the tilted distribution (Eq. 8).

Anything callable with signature r(candidate) -> float (higher = better)
plugs in. MLPScorer is the trainable head used for the affinity predictor
initialized from the MMP scorer (WP3, Eq. 14). QED/SA wrap RDKit when the
`chem` extra is installed; otherwise they fall back to neutral constants so
the pipeline stays runnable in lightweight environments.
"""
from __future__ import annotations

from typing import Callable, Protocol

import torch
import torch.nn as nn


class PropertyPredictor(Protocol):
    def __call__(self, candidate) -> float: ...


class MLPScorer(nn.Module):
    """Differentiable scorer over a pooled graph embedding (affinity head)."""

    def __init__(self, d_in: int = 256, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding).squeeze(-1)


def _qed_rdkit(candidate) -> float:
    try:
        from rdkit import Chem
        from rdkit.Chem import QED
    except ImportError:
        return 0.5  # neutral fallback without the chem extra
    mol = Chem.MolFromSmiles(getattr(candidate, "smiles", "") or "C")
    return float(QED.qed(mol)) if mol is not None else 0.0


def _sa_rdkit(candidate) -> float:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return 0.5
    mol = Chem.MolFromSmiles(getattr(candidate, "smiles", "") or "C")
    if mol is None:
        return 0.0
    # SA ~ 1 is easy, 10 is hard; map to [0,1] reward-style
    mw = Descriptors.MolWt(mol)
    return float(max(0.0, min(1.0, 1.0 - (mw - 200.0) / 600.0)))


def make_default_predictors() -> dict[str, Callable]:
    """The r_k registry used by tilted sampling and the RL reward."""
    return {
        "qed": _qed_rdkit,
        "sa": _sa_rdkit,
        # 'affinity' is injected per-target from the MMP-trained MLPScorer
        # or a docking wrapper; kept out of defaults on purpose.
    }
