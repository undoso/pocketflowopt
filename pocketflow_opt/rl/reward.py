"""Multi-objective terminal reward (proposal Eq. 11).

R = w1*(-dG_bind) + w2*QED + w3*SA + w4*Sim(lead) - l_strain*E_strain - mu*1[MW>500]

The MW/strain terms guard against docking-score gaming; affinity is a
predictor (or docking) score in kcal/mol, negated so higher reward = better.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConfig:
    w_affinity: float = 1.0
    w_qed: float = 0.3
    w_sa: float = 0.3
    w_sim: float = 0.5
    lambda_strain: float = 0.1
    mw_penalty: float = 2.0
    mw_cutoff: float = 500.0


def compute_reward(props: dict, cfg: RewardConfig | None = None) -> float:
    """props keys: affinity (kcal/mol, negative is better), qed, sa, sim,
    strain (kcal/mol), mw."""
    cfg = cfg or RewardConfig()
    r = (cfg.w_affinity * (-props.get("affinity", 0.0))
         + cfg.w_qed * props.get("qed", 0.0)
         + cfg.w_sa * props.get("sa", 0.0)
         + cfg.w_sim * props.get("sim", 0.0)
         - cfg.lambda_strain * props.get("strain", 0.0))
    if props.get("mw", 0.0) > cfg.mw_cutoff:
        r -= cfg.mw_penalty
    return float(r)
