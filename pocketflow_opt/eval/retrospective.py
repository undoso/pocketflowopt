"""Retrospective harness: replay FEP benchmark optimization trajectories.

For each target series (congeneric ligands with measured ddG), start from
the earliest/weakest compound as the lead, generate edits with WP1(+WP2),
score them, and report the Spearman rho of predicted vs. measured ddG --
the primary endpoint of the retrospective phase (proposal Sec. 6.4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import spearman


@dataclass
class Series:
    target: str
    lead: object                 # earliest compound (start point)
    candidates: list             # generated edits
    pred_ddg: np.ndarray         # model-predicted ddG vs. lead
    measured_ddg: np.ndarray     # measured ddG vs. lead


def replay(series: list[Series]) -> dict:
    rhos = {}
    for s in series:
        rho = spearman(np.asarray(s.pred_ddg), np.asarray(s.measured_ddg))
        rhos[s.target] = rho
    mean_rho = float(np.mean(list(rhos.values()))) if rhos else float("nan")
    return {"per_target": rhos, "mean_spearman": mean_rho,
            "pass": bool(mean_rho >= 0.5)}
