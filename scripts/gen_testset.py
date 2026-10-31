#!/usr/bin/env python3
"""Generate molecules for the CrossDocked2020 test pockets (split_by_name.pt).

    /tmp/opencode/venv-legacy/bin/python scripts/gen_testset.py \
        --data /home/llamosa/pfopt_data/crossdocked_pocket10 \
        --split /mnt/d/opencode/crossdocked_data/split_by_name.pt \
        --ckpt <checkpoint.pt> --n 100 --pockets 0-99 --out gen_results/testset
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "pocketflow"
sys.path.insert(0, str(VENDOR))

import torch  # noqa: E402

from pocket_flow import PocketFlow, Generate  # noqa: E402
from pocket_flow.utils import (  # noqa: E402
    AtomComposer, ComplexData, FeaturizeLigandAtom, FeaturizeProteinAtom,
    Ligand, Protein, RefineData, LigandCountNeighbors, mask_node,
    torchify_dict,
)

ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53]


def build_data(pocket_pdb: Path):
    pro_dict = Protein(str(pocket_pdb)).get_atom_dict(removeHs=True,
                                                      get_surf=True)
    data = ComplexData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pro_dict),
        ligand_dict=torchify_dict(Ligand.empty_dict()))
    data = RefineData()(data)
    data = LigandCountNeighbors()(data)
    data = FeaturizeProteinAtom()(data)
    data = FeaturizeLigandAtom(atomic_numbers=ATOM_TYPES)(data)
    node4mask = torch.arange(data.ligand_pos.size(0))
    data = mask_node(data, torch.empty([0], dtype=torch.long), node4mask,
                     num_atom_type=9, y_pos_std=0.0)
    composer = AtomComposer(knn=16, num_workers=16, for_gen=True,
                            use_protein_bond=True)
    return composer.run(data), composer.run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--pockets", default="0-99", help="inclusive range, e.g. 0-9")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="gen_results/testset")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.pockets.split("-"))
    test = torch.load(args.split)["test"]
    entries = [test[k] for k in range(lo, min(hi + 1, len(test)))]

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = PocketFlow(ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt}; {len(entries)} pockets", flush=True)

    for k, (pdb_rel, sdf_rel) in enumerate(entries, start=lo):
        name = Path(sdf_rel).parent.name + "/" + Path(sdf_rel).stem
        out = Path(args.out) / f"{k:03d}"
        if (list(out.rglob("*.sdf"))):
            print(f"[{k}] skip (exists): {name}", flush=True)
            continue
        print(f"[{k}] {name}", flush=True)
        t0 = time.time()
        try:
            data, compose = build_data(Path(args.data) / pdb_rel)
            generate = Generate(model, compose, temperature=[1.0, 1.0],
                                atom_type_map=ATOM_TYPES, num_bond_type=4,
                                max_atom_num=40, focus_threshold=0.5,
                                max_double_in_6ring=0, min_dist_inter_mol=3.0,
                                bond_length_range=(1.0, 2.0), choose_max=True,
                                device=args.device)
            generate.generate(data, num_gen=args.n, rec_name=name,
                              with_print=False, root_path=str(out))
        except Exception as e:
            print(f"[{k}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        # drop a pocket.pdb + readme so the viewer shows the protein
        ts_dir = Path(generate.out_dir)
        (ts_dir / "pocket.pdb").write_text(
            (Path(args.data) / pdb_rel).read_text(errors="replace"))
        with open(ts_dir / "readme.txt", "a") as fh:
            fh.write(f"\nckpt: {args.ckpt}\npocket: {pdb_rel}\n")
        print(f"    done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
