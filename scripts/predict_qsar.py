#!/usr/bin/env python3
"""Score molecules with a trained QSAR model and its applicability domain.

    python scripts/predict_qsar.py --model /mnt/d/opencode/egfr_opt/qsar \
        --input generated.sdf --out egfr_predictions.csv

Writes smiles,pIC50,ad_max_tanimoto,in_ad so rank_opt.py --qsar can consume it.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocketflow_opt.qsar.model import QSARPredictor


def read_smiles(path: str) -> list[str]:
    if path.endswith(".sdf"):
        return [Chem.MolToSmiles(m) for m in Chem.SDMolSupplier(path)
                if m is not None]
    df = pd.read_csv(path)
    col = next(c for c in ("smiles", "SMILES") if c in df.columns)
    return df[col].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    smiles = read_smiles(args.input)
    predictor = QSARPredictor.load(args.model)
    preds = predictor.predict(smiles, device=args.device)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["smiles", "pIC50",
                                                "ad_max_tanimoto", "in_ad"])
        writer.writeheader()
        writer.writerows(preds)
    n_in = sum(p["in_ad"] for p in preds)
    print(f"{len(preds)} predictions -> {args.out} "
          f"({n_in} inside AD at tanimoto>=0.4)")


if __name__ == "__main__":
    main()
