#!/usr/bin/env python3
"""Train the EGFR QSAR model on the cleaned ChEMBL IC50 dataset.

    python scripts/train_qsar.py --data /mnt/d/opencode/egfr_opt/egfr_ic50_clean.csv \
        --out /mnt/d/opencode/egfr_opt/qsar --device cuda:1

Uses a scaffold split (80/10/10), reports RMSE/MAE/Spearman per split, and
saves the model with its training fingerprints for the applicability domain.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocketflow_opt.qsar.features import featurize, scaffold_split
from pocketflow_opt.qsar.model import (QSARPredictor, evaluate, save_metrics,
                                       train_fps_from_smiles, train_qsar)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ad-threshold", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    smiles = df["smiles"].tolist()
    y = np.clip(df["pIC50"].to_numpy(dtype=np.float32), 2.0, 12.0)
    print(f"dataset: {len(df)} compounds, pIC50 "
          f"[{y.min():.2f}, {y.max():.2f}] median {np.median(y):.2f}")

    X, keep = featurize(smiles)
    smiles = [smiles[i] for i in keep]
    y = y[keep]
    tr, va, te = scaffold_split(smiles, seed=args.seed)
    print(f"split: train {len(tr)} val {len(va)} test {len(te)}")

    bundle = train_qsar(X, y, tr, va, device=args.device,
                        epochs=args.epochs, seed=args.seed)
    train_fps = X[tr, :2048].astype(np.uint8)
    predictor = QSARPredictor(bundle, train_fps, args.ad_threshold)

    metrics = [
        evaluate(predictor, [smiles[i] for i in tr], y[tr], "train", args.device),
        evaluate(predictor, [smiles[i] for i in va], y[va], "val", args.device),
        evaluate(predictor, [smiles[i] for i in te], y[te], "test", args.device),
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predictor.save(out)
    save_metrics(metrics, out / "qsar_metrics.json")
    for m in metrics:
        print(f"{m['split']:5s} rmse={m['rmse']:.3f} mae={m['mae']:.3f} "
              f"spearman={m['spearman']:.3f} n={m['n']}")
    test_sp = next(m["spearman"] for m in metrics if m["split"] == "test")
    print(f"acceptance gate: test spearman {test_sp:.3f} "
          f"{'PASS' if test_sp >= 0.5 else 'FAIL'} (>= 0.5)")


if __name__ == "__main__":
    main()
