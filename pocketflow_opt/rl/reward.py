"""Multi-objective reward for lead optimization (proposal Eq. 11).

In addition to the original affinity/QED/SA/similarity terms, the reward can
include normalized assay/QSAR activity, a logP-range preference, and a
synthetic-feasibility term (SA, MW, and optional structural alerts). These
components are also used by the pose-gated candidate ranker.
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
    w_qsar: float = 0.0
    w_logp: float = 0.0
    w_feasibility: float = 0.0
    qsar_min: float = 4.0
    qsar_max: float = 10.0
    logp_min: float = 1.0
    logp_max: float = 4.0
    logp_margin: float = 2.0
    mw_soft_limit: float = 500.0


def qsar_reward(p_activity: float, cfg: RewardConfig) -> float:
    """Normalize pActivity/pIC50 to [0,1]; higher activity is better."""
    span = max(cfg.qsar_max - cfg.qsar_min, 1e-8)
    return max(0.0, min(1.0, (p_activity - cfg.qsar_min) / span))


def logp_reward(logp: float, cfg: RewardConfig) -> float:
    """Reward logP inside the configured range, decaying outside it."""
    if cfg.logp_min <= logp <= cfg.logp_max:
        return 1.0
    distance = (cfg.logp_min - logp if logp < cfg.logp_min
                else logp - cfg.logp_max)
    return max(0.0, 1.0 - distance / max(cfg.logp_margin, 1e-8))


def feasibility_reward(props: dict, cfg: RewardConfig) -> float:
    """Higher for easier synthesis, acceptable MW, and fewer alerts."""
    sa = props.get("sa", 6.0)
    sa_component = max(0.0, min(1.0, (6.0 - sa) / 5.0))
    mw = props.get("mw", 0.0)
    mw_component = 1.0 if mw <= cfg.mw_soft_limit else max(
        0.0, 1.0 - (mw - cfg.mw_soft_limit) / 200.0)
    alert_component = 1.0 / (1.0 + max(0, props.get("alerts", 0)))
    return float(sa_component * mw_component * alert_component)


def compute_reward(props: dict, cfg: RewardConfig | None = None) -> float:
    """props keys: affinity, qed, sa, sim, strain, mw, qsar_activity, logp,
    alerts. QSAR activity is a measured or predicted pActivity/pIC50."""
    cfg = cfg or RewardConfig()
    r = (cfg.w_affinity * (-props.get("affinity", 0.0))
         + cfg.w_qed * props.get("qed", 0.0)
         + cfg.w_sa * props.get("sa", 0.0)
         + cfg.w_sim * props.get("sim", props.get("similarity", 0.0))
         - cfg.lambda_strain * props.get("strain", 0.0))
    if props.get("mw", 0.0) > cfg.mw_cutoff:
        r -= cfg.mw_penalty
    if props.get("qsar_activity") is not None:
        r += cfg.w_qsar * qsar_reward(props["qsar_activity"], cfg)
    if props.get("logp") is not None:
        r += cfg.w_logp * logp_reward(props["logp"], cfg)
    r += cfg.w_feasibility * feasibility_reward(props, cfg)
    return float(r)
