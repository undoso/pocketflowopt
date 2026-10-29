"""KL-anchored PPO fine-tuning of the flow modules (proposal Sec. 4.2.3).

The autoregressive growth trace is the policy trajectory; exact trace
log-likelihoods come from the flows (Eq. 3), so importance ratios and the
per-token KL anchor to the frozen reference model are cheap and exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Rollout:
    """One growth episode: action trace + terminal reward."""

    trace: list
    reward: float
    logp_old: torch.Tensor


@dataclass
class RolloutBatch:
    rollouts: list = field(default_factory=list)

    @property
    def advantages(self) -> torch.Tensor:
        r = torch.tensor([ro.reward for ro in self.rollouts])
        return (r - r.mean()) / (r.std() + 1e-8)

    @property
    def logp_old(self) -> torch.Tensor:
        return torch.stack([ro.logp_old for ro in self.rollouts])


def ppo_loss(policy, ref, batch: RolloutBatch,
             beta_kl: float = 0.05, eps: float = 0.2) -> torch.Tensor:
    """Clipped surrogate with per-trace KL anchor (Eqs. 12-13).

    policy / ref: PocketFlow instances; ref is frozen. Each rollout carries
    the context (fixed fragment + pocket) in trace[0]['context'].
    """
    adv = batch.advantages
    logp_old = batch.logp_old
    logp = torch.stack([
        policy.trace_loglik(ro.trace, ro.trace[0]["context"])
        for ro in batch.rollouts])
    with torch.no_grad():
        logp_ref = torch.stack([
            ref.trace_loglik(ro.trace, ro.trace[0]["context"])
            for ro in batch.rollouts])

    ratio = (logp - logp_old).exp()
    clipped = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    kl_anchor = (logp - logp_ref).clamp(min=0.0)  # E[KL] >= 0 per trace
    return -(clipped.mean() - beta_kl * kl_anchor.mean())


@torch.no_grad()
def collect_rollout(model, fragment, pocket, reward_fn, n_episodes: int = 32,
                    budget: int = 24, temperature: float = 1.0) -> RolloutBatch:
    """Sample episodes with the current policy and score them (WP4 calls this
    each round; reward_fn is any callable props -> float, see rl.reward)."""
    from ..edit.operators import regrow

    batch = RolloutBatch()
    for _ in range(n_episodes):
        grown = regrow(model, fragment, pocket, budget=budget,
                       beta=None, temperature=temperature)
        reward = reward_fn(grown)
        trace = [{"context": None, "frontier": None, "focal_idx": 0,
                  "elem_onehot": None, "xyz": None, "bonds": None}]
        # NOTE: regrow() currently returns the grown region; wiring the full
        # action trace through for exact logp_old is the P1 engineering task.
        logp_old = torch.zeros(())
        batch.rollouts.append(Rollout(trace=trace, reward=reward,
                                      logp_old=logp_old))
    return batch
