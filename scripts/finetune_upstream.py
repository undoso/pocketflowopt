#!/usr/bin/env python3
"""Fine-tune the vendored PocketFlow checkpoint on CrossDocked2020
(upstream finetuning.py protocol; IMPLEMENTATION.md P0 item 2).

Requires the legacy env and the processed LMDB:

    /tmp/opencode/venv-legacy/bin/python scripts/finetune_upstream.py \
        --lmdb /home/llamosa/pfopt_data/crossdocked_pocket10_35A.lmdb \
        --split /mnt/d/opencode/crossdocked_data/split_by_name.pt \
        --steps 1000000 --valid-per 2500 --logdir /mnt/d/opencode/ft_log

Checkpoints are saved every --valid-per steps under <logdir>/<date>/ckpt/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "pocketflow"
import sys
sys.path.insert(0, str(VENDOR))

from pocket_flow.gdbp_model import PocketFlow, reset_parameters
from pocket_flow.utils import Experiment, LoadDataset
from pocket_flow.utils.transform import (
    AtomComposer, Combine, FocalMaker, FeaturizeLigandAtom, FeaturizeProteinAtom,
    LigandCountNeighbors, LigandTrajectory, RefineData, TrajCompose, collate_fn,
)

ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53]
RESET_KEYS = ["edge_flow.flow_layers.5", "atom_flow.flow_layers.5",
              "pos_predictor.mu_net", "pos_predictor.logsigma_net",
              "pos_predictor.pi_net", "focal_net.net.1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--split", required=True, help="split_by_name.pt")
    ap.add_argument("--ckpt", default=str(VENDOR / "ckpt"
                                          / "ZINC-pretrained-255000.pt"))
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--valid-per", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--logdir", default="finetuning_log")
    args = ap.parse_args()

    transform = TrajCompose([
        RefineData(),
        LigandCountNeighbors(),
        FeaturizeProteinAtom(),
        FeaturizeLigandAtom(atomic_numbers=ATOM_TYPES),
        Combine(LigandTrajectory(perm_type="mix", num_atom_type=9),
                FocalMaker(r=4, num_work=16, atomic_numbers=ATOM_TYPES),
                AtomComposer(knn=16, num_workers=16, graph_type="knn",
                             radius=10, use_protein_bond=True)),
        collate_fn,
    ])

    # split without transforms (fast name2id pass), then attach
    dataset = LoadDataset(args.lmdb, transform=None)
    test_keys = set(map(tuple, torch.load(args.split)["test"]))
    train_set, valid_set = LoadDataset.split_by_name(dataset, test_keys)
    print(f"train {len(train_set)} | valid(test pockets) {len(valid_set)}",
          flush=True)
    dataset.transform = transform

    ckpt = torch.load(args.ckpt, map_location=args.device)
    model = PocketFlow(ckpt["config"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model = reset_parameters(model, RESET_KEYS)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0, betas=(0.99, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.6, patience=10, min_lr=1e-5)

    exp = Experiment(model, train_set, optimizer, valid_set=valid_set,
                     scheduler=scheduler, device=args.device, use_amp=False)
    exp.fit_step(args.steps, valid_per_step=args.valid_per,
                 train_batch_size=args.batch_size, valid_batch_size=16,
                 print_log=True, with_tb=True, logdir=args.logdir,
                 schedule_key="loss", num_workers=8, pin_memory=False,
                 follow_batch=[], exclude_keys=[], collate_fn=None,
                 max_edge_num_in_batch=400000)


if __name__ == "__main__":
    main()
