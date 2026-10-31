#!/usr/bin/env python3
"""Evaluate a masked-regrowth run from scripts/optimize_upstream.py.

Reports chemical validity, uniqueness, lead similarity, retained-core RMSD,
QED, SA, logP, and molecular weight. Writes optimization_metrics.json in the
run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, QED

from pocketflow_opt.eval.metrics import core_rmsd

EVAL_DIR = Path(__file__).resolve().parents[1] / "vendor" / "eval"
sys.path.insert(0, str(EVAL_DIR))
import sascorer  # noqa: E402


def positions(mol, indices):
    conf = mol.GetConformer()
    return torch.tensor([[conf.GetAtomPosition(i).x,
                          conf.GetAtomPosition(i).y,
                          conf.GetAtomPosition(i).z] for i in indices],
                        dtype=torch.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="regrowth output directory")
    args = ap.parse_args()
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    kept = manifest["retained_atom_indices"]

    lead = Chem.SDMolSupplier(str(run / "lead.sdf"), removeHs=True)[0]
    if lead is None:
        raise SystemExit("could not parse lead.sdf")
    lead_pos = positions(lead, kept)
    lead_fp = AllChem.GetMorganFingerprintAsBitVect(lead, 2, 2048)

    records = []
    for mol in Chem.SDMolSupplier(str(run / "generated.sdf"), removeHs=True):
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        records.append({
            "smiles": Chem.MolToSmiles(mol),
            "similarity": float(DataStructs.TanimotoSimilarity(lead_fp, fp)),
            "core_rmsd": core_rmsd(positions(mol, kept), lead_pos),
            "qed": float(QED.qed(mol)),
            "sa": float(sascorer.calculateScore(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "mw": float(Descriptors.MolWt(mol)),
            "atoms": mol.GetNumAtoms(),
        })
    if not records:
        raise SystemExit("no valid generated molecules")

    summary = {
        "n_generated": len(records),
        "n_unique": len({r["smiles"] for r in records}),
        "similarity_mean": float(np.mean([r["similarity"] for r in records])),
        "similarity_max": float(np.max([r["similarity"] for r in records])),
        "core_rmsd_max": float(np.max([r["core_rmsd"] for r in records])),
        "qed_mean": float(np.mean([r["qed"] for r in records])),
        "sa_mean": float(np.mean([r["sa"] for r in records])),
        "logp_mean": float(np.mean([r["logp"] for r in records])),
        "mw_mean": float(np.mean([r["mw"] for r in records])),
    }
    payload = {"summary": summary, "candidates": records}
    (run / "optimization_metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(summary, indent=2))
    best = max(records, key=lambda r: r["similarity"])
    print(f"most similar: similarity={best['similarity']:.3f}, "
          f"RMSD={best['core_rmsd']:.4f} A, QED={best['qed']:.3f}, "
          f"SMILES={best['smiles']}")


if __name__ == "__main__":
    main()
