"""Pose-gated multi-objective candidate ranking for masked regrowth."""
from __future__ import annotations


def sa_score(sa: float) -> float:
    """Map Ertl SA (easy ~1, hard ~10) to a bounded higher-is-better score."""
    return max(0.0, min(1.0, (6.0 - sa) / 5.0))


def candidate_score(record: dict, w_similarity: float = 0.5,
                    w_qed: float = 0.3, w_sa: float = 0.2,
                    mw_penalty: float = 0.2, mw_cutoff: float = 500.0) -> float:
    """Default property score; affinity terms can be added by target plugins."""
    score = (w_similarity * record["similarity"]
             + w_qed * record["qed"]
             + w_sa * sa_score(record["sa"]))
    if record["mw"] > mw_cutoff:
        score -= mw_penalty
    return float(score)


def pareto_front(records: list[dict]) -> list[int]:
    """Indices of non-dominated candidates.

    Objectives: maximize similarity and QED, minimize SA and MW. A candidate is
    dominated only when another is at least as good on every objective and
    strictly better on at least one.
    """
    keep = []
    for i, a in enumerate(records):
        dominated = False
        for j, b in enumerate(records):
            if i == j:
                continue
            no_worse = (b["similarity"] >= a["similarity"]
                        and b["qed"] >= a["qed"]
                        and b["sa"] <= a["sa"]
                        and b["mw"] <= a["mw"])
            strictly_better = (b["similarity"] > a["similarity"]
                               or b["qed"] > a["qed"]
                               or b["sa"] < a["sa"]
                               or b["mw"] < a["mw"])
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def rank_records(records: list[dict], top_n: int = 10,
                 reward_cfg=None) -> list[dict]:
    """Attach score/Pareto membership and return descending-score records.

    When reward_cfg is supplied, the RL reward (including optional QSAR/logP/
    feasibility terms) replaces the default property score.
    """
    if not records:
        raise ValueError("no pose-gated candidates to rank")
    if reward_cfg is not None:
        from ..rl.reward import compute_reward
    front = set(pareto_front(records))
    ranked = []
    for i, record in enumerate(records):
        item = dict(record)
        item["score"] = (compute_reward(record, reward_cfg)
                         if reward_cfg is not None
                         else candidate_score(record))
        item["pareto"] = i in front
        ranked.append(item)
    ranked.sort(key=lambda r: (-r["score"], -r["similarity"], r["sa"]))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked[:top_n]
