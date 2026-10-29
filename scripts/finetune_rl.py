#!/usr/bin/env python3
"""WP2 CLI: KL-anchored PPO fine-tuning against the multi-objective reward.

    python scripts/finetune_rl.py --ckpt pocketflow.pt --steps 200
"""
from __future__ import annotations

import argparse
import copy

import torch

from pocketflow_opt.models.pocketflow import PocketFlow
from pocketflow_opt.rl.ppo import collect_rollout, ppo_loss
from pocketflow_opt.rl.reward import RewardConfig, compute_reward


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--beta-kl", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default="pocketflow_opt_rl.pt")
    args = ap.parse_args()

    policy = PocketFlow()
    if args.ckpt:
        policy.load_pretrained(args.ckpt)
    for p in policy.atom_flow.parameters():  # fine-tune flows only
        p.requires_grad_(True)
    for p in policy.bond_flow.parameters():
        p.requires_grad_(True)

    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    cfg = RewardConfig()
    opt = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    def reward_fn(grown) -> float:
        props = {"affinity": -8.0, "qed": 0.6, "sa": 0.7,
                 "sim": 0.8, "strain": 5.0, "mw": 380.0}  # plug real scorers
        return compute_reward(props, cfg)

    for step in range(args.steps):
        batch = collect_rollout(policy, fragment=None, pocket=None,
                                reward_fn=reward_fn, n_episodes=args.episodes)
        loss = ppo_loss(policy, ref, batch, beta_kl=args.beta_kl)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"step {step:4d}  loss {float(loss):+.4f}")

    torch.save({"model": policy.state_dict()}, args.out)
    print(f"saved fine-tuned checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
