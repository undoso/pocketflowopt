#!/usr/bin/env python3
"""WP4 CLI: closed design-make-test loop with GP + EI acquisition.

    python scripts/al_loop.py --lead lead.sdf --pocket pocket.pdb --rounds 4

Generator / scorer / assay are stubbed callables; replace them with the
WP1 sampler, docking+FEP cascade, and the assay backend respectively.
"""
from __future__ import annotations

import argparse

import numpy as np

from pocketflow_opt.al.loop import ActiveLearningLoop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", required=True)
    ap.add_argument("--pocket", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--n-generate", type=int, default=256)
    ap.add_argument("--n-assay", type=int, default=8)
    args = ap.parse_args()

    rng = np.random.default_rng(0)

    def generate(round_idx: int, n: int):
        return [f"edit_r{round_idx}_{i}" for i in range(n)]

    def describe(cand: str) -> np.ndarray:
        g = np.random.default_rng(abs(hash(cand)) % (2**31))
        return g.normal(size=64)

    def score(cand: str) -> float:
        return float(rng.normal())          # docking ensemble score

    def assay(cand: str) -> float:
        return float(rng.normal(1.0, 0.5))  # measured activity

    loop = ActiveLearningLoop(generate=generate, describe=describe,
                              score=score, assay=assay,
                              n_generate=args.n_generate, n_assay=args.n_assay)
    ranked = loop.run(rounds=args.rounds)
    print(f"assayed {len(ranked)} candidates; best activity {ranked[0][1]:.3f}")


if __name__ == "__main__":
    main()
