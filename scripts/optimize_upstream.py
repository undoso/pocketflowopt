#!/usr/bin/env python3
"""PocketFlow-Opt: masked lead regrowth with the real upstream model.

Requires the legacy environment (see vendor/pocketflow). Example:

    /mnt/d/opencode/venv-legacy/bin/python scripts/optimize_upstream.py \
        --lead lead.sdf --pocket pocket.pdb --variable 12,13 \
        --n 64 --device cuda:1 --out /mnt/d/opencode/opt_results

The deleted atoms are replaced by PocketFlow-generated growth; retained lead
atoms stay in place. Outputs include generated.sdf/.smi, lead.sdf, pocket.pdb,
and manifest.json, ready for the viewer and eval_generated.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pocketflow_opt.edit.upstream import regrow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", required=True, help="lead SDF/MOL with 3D pose")
    ap.add_argument("--pocket", required=True, help="pocket PDB")
    ap.add_argument("--variable", required=True,
                    help="comma-separated atom indices to delete and regrow")
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parents[1]
                                          / "vendor/pocketflow/ckpt"
                                          / "ZINC-pretrained-255000.pt"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="opt_results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-atom-num", type=int, default=40)
    ap.add_argument("--atom-temperature", type=float, default=1.0)
    ap.add_argument("--bond-temperature", type=float, default=1.0)
    args = ap.parse_args()

    variable = [int(x.strip()) for x in args.variable.split(",") if x.strip()]
    name = args.name or Path(args.lead).stem
    out_dir = regrow(args.lead, args.pocket, variable, args.ckpt, args.out,
                     name=name, n=args.n, device=args.device, seed=args.seed,
                     max_atom_num=args.max_atom_num,
                     atom_temperature=args.atom_temperature,
                     bond_temperature=args.bond_temperature)
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
