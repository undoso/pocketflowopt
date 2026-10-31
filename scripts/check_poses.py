#!/usr/bin/env python3
"""Apply core-RMSD and interaction-conservation gates to a regrowth run.

    python scripts/check_poses.py --run /path/to/regrowth/output \
        --min-ifp 0.5 --max-rmsd 1.0

Writes pose_checks.json and prints the number of candidates passing the hard
pose gate. ProLIF is used for interaction fingerprints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pocketflow_opt.eval.pose_checks import check_regrowth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-ifp", type=float, default=0.5)
    ap.add_argument("--max-rmsd", type=float, default=1.0)
    args = ap.parse_args()
    result = check_regrowth(args.run, min_conservation=args.min_ifp,
                            max_core_rmsd=args.max_rmsd)
    out = Path(args.run) / "pose_checks.json"
    out.write_text(json.dumps(result, indent=2))
    summary = {k: v for k, v in result.items() if k != "candidates"}
    print(json.dumps(summary, indent=2))
    print(f"pose gate: {result['n_pass']}/{result['n_candidates']} passed "
          f"(wrote {out})")


if __name__ == "__main__":
    main()
