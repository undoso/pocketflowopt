"""MMP fine-tuning objectives (proposal Eqs. 14-15)."""
from __future__ import annotations

import math

import torch


def bradley_terry(scorer, pairs, pkt=None, l2: float = 1e-4) -> torch.Tensor:
    """Pairwise ranking: -log sigma(sign(dA) * (s(new) - s(old))) + L2.

    scorer: callable (features, pkt) -> scalar per pair. The trained scorer
    becomes the affinity predictor r_1 of the tilted distribution (Eq. 8),
    making guidance target-specific instead of generic docking.
    """
    losses, s_all = [], []
    for p in pairs:
        s_new = scorer(p.new, pkt)
        s_old = scorer(p.old, pkt)
        s_all += [s_new, s_old]
        sign = 1.0 if p.delta_a >= 0 else -1.0
        losses.append(-torch.log(torch.sigmoid(sign * (s_new - s_old)) + 1e-9))
    s_cat = torch.stack([s.reshape(()) for s in s_all])
    return torch.stack(losses).mean() + l2 * s_cat.pow(2).mean()


def edit_weight(delta_a: float, kappa: float = 0.5) -> float:
    """omega(dA) = sigmoid(dA / kappa): up-weights activity-improving edits
    in the edit-conditioned likelihood shaping loss (Eq. 15)."""
    return 1.0 / (1.0 + math.exp(-delta_a / kappa))


def edit_ll(model, pairs, kappa: float = 0.5) -> torch.Tensor:
    """-sum omega(dA) * log p_theta(V' | F, P) over MMP pairs (Eq. 15)."""
    terms = []
    for p in pairs:
        weight = edit_weight(p.delta_a, kappa)
        # p.trace carries the edit trajectory; exact LL via flows (Eq. 3)
        terms.append(-weight * model.trace_loglik(p.trace, p.context))
    return torch.stack(terms).mean()
