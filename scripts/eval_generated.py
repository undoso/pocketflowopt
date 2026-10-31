#!/usr/bin/env python3
"""Evaluate generated molecules against the paper's CrossDocked2020 metric
suite (chemistry side; runs in the main venv, rdkit only).

    python scripts/eval_generated.py --root gen_results [--ref-smiles REF.txt]

Metrics per track: validity, uniqueness, QED, SA, logP, Lipinski violations,
and pairwise Tanimoto diversity. Prints a summary table row per track.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent.parent / "vendor" / "eval"
sys.path.insert(0, str(EVAL_DIR))

import sascorer  # noqa: E402  (vendored rdkit contrib)


def load_mols(sdf: Path):
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(str(sdf), removeHs=True)
    return [m for m in suppl if m is not None]


def props(mols):
    from rdkit import Chem
    from rdkit.Chem import Crippen, Lipinski, QED

    out = []
    for m in mols:
        out.append({
            "qed": QED.qed(m),
            "sa": sascorer.calculateScore(m),
            "logp": Crippen.MolLogP(m),
            "lipinski": sum([
                Lipinski.NumHDonors(m) <= 5,
                Lipinski.NumHAcceptors(m) <= 10,
                Lipinski.NumRotatableBonds(m) <= 10,
                Crippen.MolLogP(m) <= 5]),
            "smiles": Chem.MolToSmiles(m),
        })
    return out


def diversity(mols):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - float(np.mean(sims)) if sims else float("nan")


def eval_track(sdf: Path) -> dict | None:
    all_entries = sdf.read_text(errors="replace").split("$$$$")
    n_raw = len([b for b in all_entries if b.strip()])
    mols = load_mols(sdf)
    if not mols:
        return None
    p = props(mols)
    smiles = [x["smiles"] for x in p]
    return {
        "track": sdf.relative_to(sdf.parents[1]).as_posix(),
        "n_raw": n_raw,
        "validity": len(mols) / max(n_raw, 1),
        "uniqueness": len(set(smiles)) / len(smiles),
        "qed": float(np.mean([x["qed"] for x in p])),
        "sa": float(np.mean([x["sa"] for x in p])),
        "logp": float(np.mean([x["logp"] for x in p])),
        "lipinski": float(np.mean([x["lipinski"] for x in p])),
        "diversity": diversity(mols),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="gen_results")
    args = ap.parse_args()

    root = Path(args.root)
    rows = [r for sdf in sorted(root.rglob("*.sdf"))
            if sdf.name != "lead.sdf" and (r := eval_track(sdf)) is not None]
    hdr = ("track", "n", "valid", "uniq", "QED", "SA", "logP",
           "Lipinski", "diversity")
    print(("%-46s" + "%9s" * 8) % hdr)
    for r in rows:
        print(("%-46s" + "%9s" + "%9.3f" * 7) % (
            r["track"][-46:], r["n_raw"], r["validity"], r["uniqueness"],
            r["qed"], r["sa"], r["logp"], r["lipinski"], r["diversity"]))
    if rows:
        print("-" * 119)
        print(("%-46s" + "%9s" + "%9.3f" * 7) % (
            "MEAN", int(np.mean([r["n_raw"] for r in rows])),
            float(np.mean([r["validity"] for r in rows])),
            float(np.mean([r["uniqueness"] for r in rows])),
            float(np.mean([r["qed"] for r in rows])),
            float(np.mean([r["sa"] for r in rows])),
            float(np.mean([r["logp"] for r in rows])),
            float(np.mean([r["lipinski"] for r in rows])),
            float(np.mean([r["diversity"] for r in rows]))))
        print("\nAcross-target mean ± SD (paper Table 1 convention):")
        for label, key in (("Validity", "validity"), ("Uniqueness", "uniqueness"),
                           ("QED", "qed"), ("SA", "sa"), ("LogP", "logp"),
                           ("Diversity", "diversity")):
            vals = np.asarray([r[key] for r in rows], dtype=float)
            print(f"  {label:12s} {vals.mean():.3f} ± {vals.std(ddof=0):.3f}")


if __name__ == "__main__":
    main()
