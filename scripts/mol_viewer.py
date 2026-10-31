#!/usr/bin/env python3
"""Molecule viewer for generation evolution (3Dmol.js; stdlib only).

    python scripts/mol_viewer.py --root gen_results --port 8765

Scans --root recursively for *.sdf tracks (generation runs, AL rounds),
oldest first, and serves an interactive 3D browser at the printed URL.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pocketflow_opt.viewer import serve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="gen_results",
                    help="results root to scan for *.sdf tracks")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--pocket-root", default=None,
                    help="base dir for resolving relative pocket paths from "
                         "run readme.txt (e.g. vendor/pocketflow)")
    args = ap.parse_args()
    serve(Path(args.root), host=args.host, port=args.port,
          pocket_root=args.pocket_root)


if __name__ == "__main__":
    main()
