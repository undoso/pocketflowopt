#!/usr/bin/env python3
"""Verify recorded upstream action traces by replaying their probabilities.

    /mnt/d/opencode/venv-legacy/bin/python scripts/verify_traces.py \
        --run /path/to/regrowth/output --device cuda:1

Recomputes each selected action from its saved pre-action state and compares
its log-probability with the recorded logp_old. Fails if the maximum absolute
error exceeds --tolerance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocketflow_opt.edit.upstream import (_fragment_generator, _imports,
                                          replay_action_logp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parents[1]
                                          / "vendor/pocketflow/ckpt"
                                          / "ZINC-pretrained-255000.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-episode", type=int, default=2)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()

    Generate, PocketFlow, *_ = _imports()
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = PocketFlow(ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    generate = _fragment_generator(Generate)(model, None, device=args.device)
    traces = torch.load(Path(args.run) / "action_traces.pt",
                        map_location="cpu")

    errors = []
    for episode in traces:
        for step in episode[:args.per_episode]:
            replayed = replay_action_logp(generate, step, args.device)
            errors.append(abs(step["logp_old"] - replayed["logp_old"]))
    max_error = max(errors) if errors else 0.0
    print(f"verified {len(errors)} steps across {len(traces)} episodes; "
          f"max |logp_old - replay| = {max_error:.3e}")
    if max_error > args.tolerance:
        raise SystemExit(f"trace verification failed (>{args.tolerance})")


if __name__ == "__main__":
    main()
