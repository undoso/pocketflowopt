"""Pose-verification gates for masked-regrowth candidates.

Checks retained-core RMSD and conservation of the reference co-crystal
interaction fingerprint. The IFP gate is a hard filter, not a reward term:
candidates must preserve enough of the reference contacts before any affinity
or property ranking is considered.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from rdkit import Chem

from .metrics import core_rmsd


def atom_positions(mol: Chem.Mol, indices: list[int]) -> torch.Tensor:
    conf = mol.GetConformer()
    return torch.tensor([[conf.GetAtomPosition(i).x,
                          conf.GetAtomPosition(i).y,
                          conf.GetAtomPosition(i).z] for i in indices],
                        dtype=torch.float64)


def prepare_protein(pocket_pdb: str | Path):
    """Load a heavy-atom pocket PDB for ProLIF.

    CrossDocked/Pocket2Mol pockets lack explicit hydrogens and bond orders, so
    the MDAnalysis converter must allow implicit hydrogens.
    """
    import MDAnalysis as mda
    from prolif import Molecule

    universe = mda.Universe(str(pocket_pdb))
    return Molecule.from_mda(universe, "protein", NoImplicit=False, force=True)


def ifp_keys(mol: Chem.Mol, protein) -> set[tuple[str, str]]:
    """Return (protein_residue, interaction) keys detected by ProLIF."""
    from prolif import Fingerprint, Molecule

    fp = Fingerprint()
    fp.run_from_iterable([Molecule(mol)], protein, progress=False)
    return {(column[1], column[2])
            for column in fp.to_dataframe().columns}


def conservation(reference: set, candidate: set) -> float:
    """Fraction of reference interaction keys retained by a candidate."""
    if not reference:
        return 1.0
    return len(reference & candidate) / len(reference)


def check_regrowth(run_dir: str | Path, min_conservation: float = 0.5,
                   max_core_rmsd: float = 1.0) -> dict:
    """Evaluate all generated molecules in an upstream regrowth run."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    kept = manifest["retained_atom_indices"]
    lead = Chem.SDMolSupplier(str(run_dir / "lead.sdf"), removeHs=False)[0]
    if lead is None:
        raise ValueError("could not parse reference lead")
    lead_pos = atom_positions(lead, kept)
    protein = prepare_protein(run_dir / "pocket.pdb")
    reference_keys = ifp_keys(lead, protein)

    records = []
    for mol in Chem.SDMolSupplier(str(run_dir / "generated.sdf"),
                                  removeHs=False):
        if mol is None:
            continue
        keys = ifp_keys(mol, protein)
        rmsd = core_rmsd(atom_positions(mol, kept), lead_pos)
        retained = conservation(reference_keys, keys)
        records.append({
            "smiles": Chem.MolToSmiles(mol),
            "core_rmsd": rmsd,
            "ifp_conservation": retained,
            "n_interactions": len(keys),
            "pose_pass": bool(rmsd <= max_core_rmsd
                              and retained >= min_conservation),
        })
    if not records:
        raise ValueError("no valid generated molecules")
    return {
        "n_candidates": len(records),
        "n_pass": sum(r["pose_pass"] for r in records),
        "n_reference_interactions": len(reference_keys),
        "min_conservation": min(r["ifp_conservation"] for r in records),
        "mean_conservation": sum(r["ifp_conservation"]
                                 for r in records) / len(records),
        "max_core_rmsd": max(r["core_rmsd"] for r in records),
        "gate": {"min_conservation": min_conservation,
                 "max_core_rmsd": max_core_rmsd},
        "candidates": records,
    }
