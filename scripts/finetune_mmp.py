#!/usr/bin/env python3
"""WP3 CLI: SAR-aware fine-tuning on matched molecular pairs.

    python scripts/finetune_mmp.py --pairs mmp_pairs.csv --epochs 20

CSV columns: target_id, smiles_old, smiles_new, delta_activity
"""
from __future__ import annotations

import argparse

import torch

from pocketflow_opt.guidance.predictors import MLPScorer
from pocketflow_opt.mmp.data import load_pairs_csv
from pocketflow_opt.mmp.losses import bradley_terry


def smiles_featurizer(smiles: str) -> torch.Tensor:
    """Placeholder: hash-based embedding; swap for the GDBP encoder + RDKit
    descriptors in the full pipeline."""
    g = torch.Generator().manual_seed(abs(hash(smiles)) % (2**31))
    return torch.randn(256, generator=g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--out", default="mmp_scorer.pt")
    args = ap.parse_args()

    pairs = load_pairs_csv(args.pairs, featurizer=smiles_featurizer)
    scorer = MLPScorer(d_in=256)
    opt = torch.optim.Adam(scorer.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        loss = bradley_terry(lambda f, _p: scorer(f), pairs, l2=args.l2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 5 == 0:
            print(f"epoch {epoch:3d}  BT loss {float(loss):.4f}")

    torch.save({"model": scorer.state_dict()}, args.out)
    print(f"saved MMP scorer -> {args.out}  (wire it in as affinity predictor r_1)")


if __name__ == "__main__":
    main()
