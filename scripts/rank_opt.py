#!/usr/bin/env python3
"""Rank pose-gated regrowth candidates and export the selected subset.

    python scripts/rank_opt.py --run /path/to/regrowth/output --top 10

Reads optimization_metrics.json and pose_checks.json, removes failed poses,
scores candidates on similarity/QED/SA/MW, marks the Pareto front, and writes
ranked_candidates.json plus top_candidates.sdf for the viewer.

Use --reward-mode rl to include logP-range and feasibility rewards, and
--qsar predictions.csv (smiles,activity or smiles,pIC50) to add an assay/QSAR
activity reward. RL outputs use an rl_ filename prefix.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem

from pocketflow_opt.eval.rank import rank_records
from pocketflow_opt.rl.reward import RewardConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--w-similarity", type=float, default=0.5)
    ap.add_argument("--w-qed", type=float, default=0.3)
    ap.add_argument("--w-sa", type=float, default=0.2)
    ap.add_argument("--reward-mode", choices=["default", "rl"],
                    default="default")
    ap.add_argument("--qsar", default=None,
                    help="CSV with smiles and activity/pIC50 columns")
    args = ap.parse_args()
    run = Path(args.run)
    metrics = json.loads((run / "optimization_metrics.json").read_text())
    poses = json.loads((run / "pose_checks.json").read_text())
    pose_by_smiles = {r["smiles"]: r for r in poses["candidates"]}
    gated = []
    for record in metrics["candidates"]:
        pose = pose_by_smiles.get(record["smiles"])
        if pose is not None and pose["pose_pass"]:
            item = dict(record)
            item["ifp_conservation"] = pose["ifp_conservation"]
            item["core_rmsd"] = pose["core_rmsd"]
            gated.append(item)
    if not gated:
        raise SystemExit("no candidates passed the pose gate")

    qsar_by_smiles = {}
    if args.qsar:
        with open(args.qsar) as fh:
            for row in csv.DictReader(fh):
                value = next((row[k] for k in ("activity", "pIC50", "pActivity")
                              if row.get(k)), None)
                if value is not None:
                    qsar_by_smiles[row["smiles"]] = float(value)
        for record in gated:
            if record["smiles"] in qsar_by_smiles:
                record["qsar_activity"] = qsar_by_smiles[record["smiles"]]

    reward_cfg = None
    if args.reward_mode == "rl":
        reward_cfg = RewardConfig(
            w_affinity=0.0, w_qed=0.2, w_sa=0.0, w_sim=0.5,
            lambda_strain=0.0, mw_penalty=0.2,
            w_qsar=1.0 if args.qsar else 0.0,
            w_logp=0.2, w_feasibility=0.2)
    ranked = rank_records(gated, top_n=args.top, reward_cfg=reward_cfg)
    prefix = "rl_" if args.reward_mode == "rl" else ""
    payload = {"n_pose_gated": len(gated),
               "n_pareto": sum(r["pareto"] for r in ranked),
               "reward_mode": args.reward_mode,
               "qsar_inputs": len(qsar_by_smiles),
               "weights": {"similarity": args.w_similarity,
                           "qed": args.w_qed, "sa": args.w_sa},
               "candidates": ranked}
    (run / f"{prefix}ranked_candidates.json").write_text(
        json.dumps(payload, indent=2))

    mols = {Chem.MolToSmiles(m): m for m in
            Chem.SDMolSupplier(str(run / "generated.sdf"), removeHs=False)
            if m is not None}
    with open(run / f"{prefix}top_candidates.sdf", "w") as fh:
        for item in ranked:
            mol = mols[item["smiles"]]
            mol.SetProp("_Name", f"rank_{item['rank']}_score_{item['score']:.3f}")
            fh.write(Chem.MolToMolBlock(mol) + "\n$$$$\n")

    print(f"pose-gated {len(gated)} candidates; exported {len(ranked)}")
    for item in ranked[:5]:
        print(f"#{item['rank']} score={item['score']:.3f} "
              f"sim={item['similarity']:.3f} QED={item['qed']:.3f} "
              f"SA={item['sa']:.3f} MW={item['mw']:.1f} "
              f"IFP={item['ifp_conservation']:.3f} pareto={item['pareto']}")


if __name__ == "__main__":
    main()
