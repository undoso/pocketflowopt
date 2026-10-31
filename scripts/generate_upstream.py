#!/usr/bin/env python3
"""Generate molecules with the vendored upstream PocketFlow checkpoint.

Requires the legacy env (py3.8, torch 1.13+cu117 -- see vendor/pocketflow):

    /path/to/venv-legacy/bin/python scripts/generate_upstream.py \
        --pocket vendor/pocketflow/test_samples/test_pocket10/1bvr_C_rec_pocket10-surf.pdb \
        --n 100 --out gen_results
"""
from __future__ import annotations

import argparse
import os
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
    torchify_dict, timewait,
)

ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pocket", required=True, help="pocket PDB file")
    ap.add_argument("--ckpt", default=str(VENDOR / "ckpt"
                                          / "ZINC-pretrained-255000.pt"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="gen_results")
    ap.add_argument("--max-atom-num", type=int, default=40)
    args = ap.parse_args()

    name = args.name or Path(args.pocket).name.split("-")[0]
    pro_dict = Protein(args.pocket).get_atom_dict(removeHs=True, get_surf=True)
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
    atom_composer = AtomComposer(knn=16, num_workers=16, for_gen=True,
                                 use_protein_bond=True)
    data = atom_composer.run(data)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = PocketFlow(ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.ckpt}")

    generate = Generate(model, atom_composer.run, temperature=[1.0, 1.0],
                        atom_type_map=ATOM_TYPES, num_bond_type=4,
                        max_atom_num=args.max_atom_num, focus_threshold=0.5,
                        max_double_in_6ring=0, min_dist_inter_mol=3.0,
                        bond_length_range=(1.0, 2.0), choose_max=True,
                        device=args.device)
    start = time.time()
    generate.generate(data, num_gen=args.n, rec_name=name, with_print=False,
                      root_path=args.out)
    readme = os.path.join(generate.out_dir, "readme.txt")
    with open(readme, "a") as fh:
        fh.write(f"\npocket: {os.path.abspath(args.pocket)}\n")
    print(f"results -> {generate.out_dir}")
    print("Time:", timewait(time.time() - start))


if __name__ == "__main__":
    main()
