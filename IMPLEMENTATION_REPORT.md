# PocketFlow-Opt implementation and evaluation report

**Status date:** 2026-10-31  
**Workspace:** `/home/llamosa/projects/pocketflowopt`  
**Large datasets, checkpoints, and generated results:** `/mnt/d/opencode`

## Executive summary

- Implemented RDKit-based ligand and pocket featurization and connected it to
  `scripts/generate_edits.py`.
- Fixed three runtime bugs affecting multi-step regrowth and GPU PPO/regrowth.
- Vendored the MIT-licensed upstream PocketFlow implementation and its released
  ZINC-pretrained checkpoint. The checkpoint cannot be loaded into this
  scaffold's simplified model because the architectures differ; upstream
  generation is run through a separate legacy-environment CLI.
- Added a local molecule/protein viewer and evaluation/training utilities.
- Ran the paper's ten named target pockets with the released ZINC-pretrained
  checkpoint at 1,000 molecules per target. The property averages are close to
  Table 1, but this is a pre-fine-tuning baseline check, not reproduction of
  the final fine-tuned model. The paper generated 10,000 molecules per target.
- CrossDocked2020 data preparation is complete. A separate fine-tuning run is
  active at approximately 176,000 of 1,000,000 steps; its current setup is
  diagnostic and does not exactly match the paper's validation protocol.
- The incremental context cache remains unimplemented.

## Implementation completed

### Regrowth and PPO fixes

Fixed three issues uncovered by GPU and multi-step execution:

1. `GrownRegion.atoms()` returned CPU indices, causing GPU regrowth to fail
   when concatenating frontier indices. It now supports a device, and the
   regrowth loop places both index tensors on the graph device.
2. Constraint checking indexed fragment-only coordinates with merged-graph
   indices. It now receives merged coordinates, so a previously grown atom can
   be selected as a focal atom without an out-of-range error.
3. `ppo_loss` combined CUDA log probabilities with CPU advantages and
   `logp_old`. Both are now moved to the log-probability device.

### RDKit featurization

Added `pocketflow_opt/chem/featurize.py` with:

- `load_ligand`: reads SDF, MOL, or PDB; maps elements to the model's element
  index order; kekulizes aromatic bonds into the four bond classes while
  retaining aromatic/ring/conjugation flags; and creates the GDBP scalar,
  vector, and edge tensors.
- `load_pocket`: selects standard-residue protein heavy atoms within 6 Å of
  the reference ligand and creates spatial edges.
- `merge_graphs`: combines ligand and pocket graphs and adds spatial contact
  edges.

The ligand scalar table uses 19 defined channels in the 128-channel encoder
input; vector channels start at zero. Protein scalar features currently encode
element and protein-atom identity, rather than a full residue/interaction
feature set. `generate_edits.py` now uses these structures instead of random
placeholder graphs.

### Upstream PocketFlow and generation

The public upstream PocketFlow implementation, its MIT license, the
`ZINC-pretrained-255000.pt` checkpoint, and the ten bundled pocket structures
are vendored under `vendor/pocketflow/`. Inspection showed that upstream and the
scaffold have different architectures and state dictionaries (652 versus 105
tensors), so a parameter-name mapping is not possible. The upstream model is
therefore run separately through `scripts/generate_upstream.py` in a legacy
Python 3.8 / PyTorch 1.13.1+cu117 environment. GPU execution was verified on
the Pascal cards.

### Viewer and evaluation tools

- `scripts/mol_viewer.py` serves SDF generation tracks through a lightweight
  stdlib HTTP server. The page displays molecules in 3D with 3Dmol.js and shows
  the associated pocket as a protein cartoon. The viewer is running at
  `http://127.0.0.1:8765` and scans `/mnt/d/opencode/gen_results`.
- `scripts/eval_generated.py` reports validity, uniqueness, QED, Ertl SA,
  logP, Lipinski-rule count, and Morgan-fingerprint diversity, including
  across-target mean ± standard deviation.
- `scripts/gen_testset.py` generates molecules for selected CrossDocked test
  pockets; `scripts/finetune_upstream.py` runs upstream-style fine-tuning.

### Opt layer: masked lead regrowth

Added the first working PocketFlow-Opt integration in
`pocketflow_opt/edit/upstream.py` and `scripts/optimize_upstream.py`:

- A retained lead fragment is loaded from a real 3D SDF/MOL structure, selected
  atom indices are deleted, and the upstream model regrows the variable region
  inside the pocket.
- The upstream first-step behavior was adapted from de novo generation to
  fragment-anchored growth. New atoms attach to retained-fragment frontier
  atoms, including the first new atom.
- Bond validation now includes the full retained fragment, and the atom limit
  is enforced as a total-molecule cap rather than adding a full de novo budget
  on top of the lead.
- Outputs include generated SDF/SMILES, the reference lead, pocket PDB, and a
  manifest recording deleted/retained indices and run settings.
- `scripts/eval_opt.py` computes lead similarity, retained-core RMSD, QED, SA,
  logP, and molecular weight.

A real 31-atom co-crystal ligand was used with atoms 27–30 deleted. The
32-candidate run produced 32 valid molecules (30 unique), mean/max lead
similarity 0.381/0.494, and retained-core RMSD below 1e-15 Å. These are
successfully generated and evaluated analogues, not yet affinity-guided or
PPO-optimized candidates. The viewer now serves the optimization results from
`/mnt/d/opencode/opt_results`.

### Pose verification

Added `pocketflow_opt/eval/pose_checks.py` and `scripts/check_poses.py` as a
hard candidate gate before property or affinity ranking:

- Retained-core RMSD against the reference pose (Kabsch).
- Conservation of the reference co-crystal interaction fingerprint using
  ProLIF. The fingerprint uses residue/interaction keys; conservation is the
  fraction of reference keys retained by the candidate.

On the 32-candidate R-group run, all 32 passed the gate (core RMSD <= 1.0 Å,
IFP conservation >= 0.5). Minimum/mean conservation was 0.889/0.941 and core
RMSD was effectively zero. ProLIF is now included in the `chem` optional
dependency group.

### Candidate ranking

Added `pocketflow_opt/eval/rank.py` and `scripts/rank_opt.py`. Candidates that
fail the pose gate are removed before scoring. The default score combines lead
similarity, QED, and normalized SA with an MW>500 penalty; a separate Pareto
front uses similarity/QED (maximize) and SA/MW (minimize). The CLI writes
`ranked_candidates.json` and `top_candidates.sdf`.

For the 32-candidate run, all 32 were ranked, 10 were exported, and 9 are on
the Pareto front. The top default-score candidate had similarity 0.494, QED
0.085, SA 5.048, and IFP conservation 0.889. This is property-based selection,
not target-affinity optimization; an affinity predictor or assay backend still
needs to be supplied for target-specific optimization.

### RL-ready reward

Extended `pocketflow_opt/rl/reward.py` with configurable terms for:

- Assay/QSAR activity: normalized pActivity/pIC50 (default range 4–10).
- LogP: full reward inside a configurable range (default 1–4), with a smooth
  penalty outside it.
- Feasibility: Ertl SA, soft molecular-weight limit, and optional structural
  alerts.

`rank_opt.py --reward-mode rl` applies this reward after the pose gate;
`--qsar predictions.csv` accepts `smiles,activity` or `smiles,pIC50` input.
Without a QSAR file, logP and feasibility are active while the activity term
is disabled. On the existing 32 candidates the RL-ranked top candidate scored
0.302 with similarity 0.494 and IFP conservation 0.889. Outputs are written as
`rl_ranked_candidates.json` and `rl_top_candidates.sdf`.

This is reward-guided selection, not yet PPO fine-tuning. The next dependency
was upstream action traces and exact `logp_old`.

### Upstream action traces for PPO

The upstream masked-regrowth adapter now records accepted actions and their
probabilities. Each step stores focal, atom-flow, position, and bond-flow log
probabilities, the total `logp_old`, and the pre-action state needed for later
PPO re-evaluation. Outputs are `action_traces.pt` (full states) and
`action_traces.json` (per-episode summaries). `scripts/verify_traces.py`
replays saved states through the model to check consistency.

A traced two-episode run recorded 13 accepted growth steps per episode with
finite trajectory log probabilities of -21.21 and -32.36. Replaying six steps
across both episodes reproduced `logp_old` with a maximum absolute error of
5.01e-6. The remaining PPO work is to collect reward-labeled rollout batches
and run the KL-anchored update loop; the traces now provide the required
probability bookkeeping.

## CrossDocked2020 data and storage

- Downloaded the processed `crossdocked_pocket10.tar.gz` (1.6 GB) and
  `split_by_name.pt` from the Pocket2Mol data archive.
- Built an LMDB with **140,174** usable complexes after element and
  heavy-atom-count filtering; the LMDB is approximately 5.5 GB.
- The tarball, LMDB, legacy environment, test pockets, checkpoints, logs, and
  generated structures are stored on D: under `/mnt/d/opencode`.
- The active WSL root filesystem is a VHDX on C:. Large files were moved off
  the Linux root filesystem; deleting files inside WSL does not shrink that
  VHDX automatically. C: had about 9.5 GB free at the last check.

## Pre-fine-tuning benchmark check

The released `ZINC-pretrained-255000.pt` checkpoint was evaluated on the
paper's ten targets (2ah9, 5lvq, 5g3n, 1u0f, 4bnw, 2ati, 2hw1, 1bvr, 1zyu,
4i91), with 1,000 generated molecules per target. The table reports the mean
and standard deviation of target-level metric means, following the paper's
presentation. These are baseline results before CrossDocked fine-tuning.

| Metric | PocketFlow paper | Released checkpoint, this run |
|---|---:|---:|
| Validity | 1.000 ± 0.000 | **1.000 ± 0.000** |
| QED | 0.507 ± 0.051 | **0.503 ± 0.054** |
| SA (lower is easier to synthesize) | 2.927 ± 0.372 | **3.011 ± 0.419** |
| LogP | 3.719 ± 1.074 | **3.656 ± 1.164** |
| Diversity | 0.871 ± 0.014 | **0.883 ± 0.016** |

The results are close to the published averages, but that numerical agreement
does not establish parity with the final model: the checkpoint used here is
ZINC-pretrained. The paper describes pretraining on ZINC molecules followed by
fine-tuning on filtered CrossDocked2020 complexes. This evaluation also uses
one-tenth of the paper's 10,000 molecules per target.

## Fine-tuning status and caveat

The CrossDocked fine-tuning process is still running on cuda:0. At the last
check it had reached about **184,000 / 1,000,000 steps**, with checkpoints saved
every 2,500 steps under `/mnt/d/opencode/ft_log/2026-10-29-21-09/ckpt/`.

An 85,000-step checkpoint was sampled at 100 molecules per paper target. It
gave validity 1.000 ± 0.000, QED 0.489 ± 0.028, SA 2.741 ± 0.513, LogP
3.074 ± 0.752, and diversity 0.857 ± 0.036. A 90,000-step checkpoint gave
validity 1.000 ± 0.000, QED 0.497 ± 0.034, SA 2.794 ± 0.524, LogP
2.748 ± 0.675, and diversity 0.846 ± 0.049. Treat these as diagnostic only:
training is incomplete and evaluation samples are smaller than the paper's.
The active fine-tune uses the Pocket2Mol split's surviving test entries for
validation, whereas the paper describes a random 100-complex validation set.
A strict training-protocol reproduction would need that validation split
corrected.

## Verification

- `.venv/bin/python -m pytest tests/ -q` — **14 passed**.
- The tests cover equivariance, flow invertibility, constraint behavior,
  grown-atom focal indexing, multi-step regrowth, RDKit ligand/pocket
  featurization, graph merging, viewer HTTP endpoints, and CUDA regrowth/PPO.
- Verified upstream GPU generation, real-structure featurization through the
  generation CLI, and viewer routes for SDF molecules and pocket PDBs.
- The scaffold's own randomly initialized `PocketFlow` is not a meaningful
  generator yet: real-feature CLI runs may terminate without adding atoms
  because its untrained focal network selects STOP. The upstream pretrained
  generation path is separate from that scaffold path.

## Remaining work

1. Correct the fine-tuning validation split and complete the 1M-step run.
2. Evaluate the converged CrossDocked-fine-tuned checkpoint on all ten paper
   targets at 10,000 molecules per target; the current ZINC-checkpoint run is
   only the pre-fine-tuning baseline.
3. Implement the incremental context cache (`PocketFlow.encode_update`
   currently re-encodes the full graph).
4. Connect the working upstream masked-regrowth adapter to target-specific
   scoring and guidance, and eventually PPO; the scaffold's own `regrow` path
   remains untrained and is not the path used for the successful opt run.
