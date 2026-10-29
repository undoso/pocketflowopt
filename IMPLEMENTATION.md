# Instructions for further implementation

This repo is the reference scaffold for **PocketFlow-Opt** (pocket-conditioned lead
optimization on the PocketFlow autoregressive flow). The architecture and math are
specified in the companion proposal (`PocketFlow-Opt_Proposal.pdf`, 15 pp., eqs. 1-17).
All smoke tests pass (`pytest tests/ -q`). Everything below is ordered by dependency.

## P0 - Foundations (do first, nothing else works without these)

1. **Real featurization.** Replace the `load_geom` stub in `scripts/generate_edits.py`
   with RDKit-based SDF/PDB parsing: element typing onto `models.pocketflow.ELEMENTS`,
   bond typing to the 4-class scheme in `models/bonds.py`, pocket extraction (protein
   heavy atoms within ~6 A of the reference ligand), and atom-feature tables for the
   GDBP scalar channels.
2. **Checkpoint parity.** Load the upstream pretrained PocketFlow weights
   (Saoge123/PocketFlow), map parameter names onto `models/*`, and reproduce the
   paper's CrossDocked2020 test metrics *before* any fine-tuning. If parity fails,
   nothing downstream is interpretable.
3. **Incremental context cache.** `PocketFlow.encode_update` currently re-encodes the
   full graph each step; make it incremental or generation will be too slow for the
   AL loop.

## P1 - The optimization core (WP1 + pose verification)

4. **Wire action traces.** `edit/operators.regrow()` must record per-step
   `(frontier, focal_idx, elem_onehot, xyz, bonds, context)`. This unblocks exact
   `logp_old` in `rl/ppo.collect_rollout` (currently zeroed - marked TODO), which
   PPO importance ratios and the KL anchor depend on.
5. **Pose verification layer** (`eval/`, new module `pose_checks.py`):
   - IFP conservation (ProLIF): fraction of must-keep contacts (hinge H-bonds, salt
     bridges, conserved waters) retained vs. the reference co-crystal mode. This is a
     **hard gate**, not a reward term.
   - Core RMSD gate (< 1.0 A; `eval/metrics.core_rmsd` already implements Kabsch).
   - Torsional strain of the generated conformer vs. relaxed geometry.
   - Rationale: generated poses are samples from a prior trained on CrossDocked2020
     (docked, partly cross-docked poses - not crystal structures). Treat every pose as
     a hypothesis until this gate passes.
6. **Diversity = monitoring, never an objective.** In seeded runs, interaction
   conservation is the target. Use mode entropy (RMSD/IFP clustering at 2 A) solely as
   an RL-collapse detector: entropy -> 0 means tighten `beta_kl`. Only scaffold-hop
   candidates may leave the reference interaction cluster, and those are escalated to
   MD-stability/FEP pose checks, never discarded silently.

## P2 - Guidance and fine-tuning (WP2 + WP3)

7. **Docking discipline.** Docking scores are a cheap shortlist filter inside the AL
   loop only. They must NOT appear as a sampling-time guidance signal (gameable
   gradient) or as a ranking endpoint. Sampling guidance comes from the MMP-trained
   scorer (`mmp/`) + QED/SA + IFP-conservation term.
8. **MMP pipeline.** ChEMBL extraction -> `mmp/data.load_pairs_csv` -> Bradley-Terry
   fine-tune (`mmp/losses.bradley_terry`) -> register the scorer as affinity predictor
   `r_1` in `guidance/predictors.make_default_predictors`.
9. **PPO fine-tune.** Only after 4+8 land. Tune `beta_kl` (start 0.05); keep MW/strain
   penalties in `rl/reward.RewardConfig` - they are the anti-reward-hacking terms.
   Watch mode entropy (item 6) every epoch.

## P3 - Closed loop (WP4)

10. **FEP+ orchestration** behind the `assay` interface in `al/loop.py`; keep the
    GP+EI acquisition as-is.
11. **Retrospective evaluation**: replay the public FEP ligand series via
    `eval/retrospective.replay`; primary endpoint Spearman >= 0.5 vs. measured ddG.
    Baselines: Pocket2Mol, TargetDiff, DiffSBDD, Delete, DiffDec, TamGen-refine,
    and the naive linear-reward RL wrapper (the ablation - expected to show reward
    hacking: reward up, validity/diversity/true potency down).

## Testing discipline

Extend `tests/test_smoke.py` whenever a layer lands: full-model equivariance under
random rotation, constraint-layer ablation (validity must stay 100%), trace
round-trip (recorded logp == recomputed logp), IFP-gate regression on a known
co-crystal series.
