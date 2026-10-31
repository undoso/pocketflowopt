#!/usr/bin/env python3
"""WP1 CLI: masked regrowth of a lead inside a pocket -> edited candidates.

Example:
    python scripts/generate_edits.py --lead lead.sdf --pocket pocket.pdb \
        --operator rgroup --frontier 12 --budget 24 --n 64 --out edits.sdf
"""
from __future__ import annotations

import argparse

import torch

from pocketflow_opt.chem import featurize
from pocketflow_opt.edit.mask import Fragment
from pocketflow_opt.edit.operators import RGroupGrowth, ScaffoldHop, LinkerRedesign, regrow
from pocketflow_opt.models.pocketflow import PocketFlow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", required=True)
    ap.add_argument("--pocket", required=True)
    ap.add_argument("--operator", choices=["rgroup", "scaffold", "linker"],
                    default="rgroup")
    ap.add_argument("--frontier", type=int, nargs="+", required=True)
    ap.add_argument("--variable", type=int, nargs="*", default=[])
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="edits.sdf")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    lead_graph = featurize.load_ligand(args.lead)
    pocket = featurize.load_pocket(args.pocket, ref_mol=lead_graph)
    # frontier indices refer to lead atoms; lead stays the first block of the
    # merged context graph, so they remain valid after the pocket is appended
    context = featurize.merge_graphs(lead_graph, pocket)
    model = PocketFlow()
    if args.ckpt:
        model.load_pretrained(args.ckpt)

    n_atoms = lead_graph.coords.shape[0]  # operator defaults index the lead
    if args.operator == "rgroup":
        op = RGroupGrowth(
            substituent_atoms=torch.tensor(args.variable or [n_atoms - 1]),
            frontier_atoms=torch.tensor(args.frontier),
            max_new_atoms=args.budget)
    elif args.operator == "scaffold":
        op = ScaffoldHop(
            core_atoms=torch.tensor(args.variable or [n_atoms // 2]),
            anchor_atoms=torch.tensor(args.frontier),
            anchor_targets=pocket.coords[: len(args.frontier)])
    else:
        op = LinkerRedesign(
            linker_atoms=torch.tensor(args.variable),
            warhead_a_frontier=torch.tensor(args.frontier[:1]),
            warhead_b_frontier=torch.tensor(args.frontier[-1:]))

    fragment = Fragment(graph=context, frontier=op.frontier_atoms
                        if hasattr(op, "frontier_atoms")
                        else torch.tensor(args.frontier))

    kept = 0
    with open(args.out, "w") as fh:  # placeholder SDF sink
        for i in range(args.n):
            grown = regrow(model, fragment, pocket, budget=args.budget)
            if grown.size() > 0:
                fh.write(f"# candidate {i}: {grown.size()} new atoms\n")
                kept += 1
    print(f"wrote {kept}/{args.n} edited candidates -> {args.out}")


if __name__ == "__main__":
    main()
