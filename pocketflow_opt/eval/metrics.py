"""Evaluation metrics (proposal Sec. 6.3)."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr


def core_rmsd(coords_fixed: torch.Tensor, coords_ref: torch.Tensor) -> float:
    """RMSD of the frozen core vs. its reference pose, after optimal
    rigid superposition (Kabsch). Target: < 1.0 Angstrom."""
    a = coords_fixed.detach().cpu().double()
    b = coords_ref.detach().cpu().double()
    a0 = a - a.mean(0)
    b0 = b - b.mean(0)
    h = a0.T @ b0
    u, _, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(u @ vt))
    rot = u @ torch.diag(torch.tensor([1.0, 1.0, d])) @ vt
    diff = a0 @ rot - b0
    return float((diff.pow(2).sum(-1).mean()).sqrt())


def tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Tanimoto similarity between binary fingerprint vectors (Eq. 10)."""
    inter = np.logical_and(fp_a, fp_b).sum()
    union = np.logical_or(fp_a, fp_b).sum()
    return float(inter / union) if union else 1.0


def spearman(pred: np.ndarray, measured: np.ndarray) -> float:
    """Spearman rho between predicted and measured ddG (target >= 0.5)."""
    return float(spearmanr(pred, measured).statistic)


def validity_rate(samples: list, is_valid) -> float:
    """Fraction passing chemical validity; 1.0 by construction with the
    constraint layer active (Eq. 6) -- monitored as a regression alarm."""
    if not samples:
        return 0.0
    return float(np.mean([bool(is_valid(s)) for s in samples]))
