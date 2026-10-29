# pocketflow-opt

Reference implementation of **PocketFlow-Opt**: pocket-conditioned **lead optimization**
built on top of the PocketFlow autoregressive flow (Jiang et al., *Nat. Mach. Intell.* 6,
326-337, 2024). De novo generation is treated as the empty-fragment special case of
mask-and-regrow constrained generation.

## Work packages -> packages

| WP | Package | Paper equation(s) |
|----|---------|-------------------|
| WP1 fragment-anchored regrowth | `pocketflow_opt.edit` | masked conditional (Eq. 5), restraint potential (Eq. 7) |
| WP2 property guidance | `pocketflow_opt.guidance`, `pocketflow_opt.rl` | tilted distribution (Eq. 8), MALA (Eq. 9), reward (Eq. 11), PPO + KL (Eqs. 12-13) |
| WP3 SAR-aware fine-tuning | `pocketflow_opt.mmp` | Bradley-Terry (Eq. 14), edit LL shaping (Eq. 15) |
| WP4 active-learning loop | `pocketflow_opt.al` | GP surrogate (Eq. 16), expected improvement (Eq. 17) |

The geometric backbone (`geometry/gdbp.py`) and the flow modules (`models/flows.py`)
mirror the PocketFlow architecture (GDBP equivariance, dequantized conditional flows)
and load pretrained checkpoints unchanged - the generation core is never modified.

## Install

```bash
pip install -e .            # core (torch, numpy, scipy)
pip install -e .[chem,dev]  # + RDKit-backed property scorers and tests
```

## Quickstart

```bash
# WP1: regrow the R-group attached to atom 12 of a lead, in a pocket
python scripts/generate_edits.py --lead lead.sdf --pocket pocket.pdb \
    --operator rgroup --frontier 12 --budget 24 --out edits.sdf

# WP2: KL-anchored PPO fine-tuning against a multi-objective reward
python scripts/finetune_rl.py --ckpt pocketflow.pt --reward configs/reward.yaml

# WP3: MMP pairwise fine-tuning on a ChEMBL-like export
python scripts/finetune_mmp.py --pairs mmp_pairs.csv --ckpt pocketflow.pt

# WP4: closed design-make-test loop (GP + EI acquisition)
python scripts/al_loop.py --lead lead.sdf --pocket pocket.pdb --rounds 4
```

## Smoke tests

```bash
pytest tests/ -q   # equivariance, flow invertibility, validity invariants
```

## Scope

Data pipelines (CrossDocked2020 parsing, co-crystal alignment), FEP+ orchestration and
assay logistics are integration points behind interfaces (`guidance/predictors.py`
accepts any callable scorer; `al/loop.py` accepts any assay backend). This repository
implements the novel logic - masked regrowth, guidance, KL-anchored RL, MMP ranking,
and the AL loop - not the heavy data engineering.

## Next steps

See `IMPLEMENTATION.md` for the ordered roadmap (P0-P3).
