"""Morgan-fingerprint + RDKit-descriptor features and scaffold splitting."""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

FP_BITS = 2048


def morgan_fp(mol: Chem.Mol, n_bits: int = FP_BITS) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_fps(mols: list[Chem.Mol], n_bits: int = FP_BITS) -> np.ndarray:
    return np.stack([morgan_fp(m, n_bits) for m in mols])


def descriptors(mol: Chem.Mol) -> np.ndarray:
    return np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcTPSA(mol),
        mol.GetNumHeavyAtoms(),
        Lipinski.RingCount(mol),
        Lipinski.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
    ], dtype=np.float32)


N_DESCRIPTORS = 10


def featurize(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """Return (X [n_valid, FP_BITS+N_DESCRIPTORS], valid_indices)."""
    feats, keep = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        feats.append(np.concatenate([morgan_fp(mol), descriptors(mol)]))
        keep.append(i)
    return np.stack(feats), keep


def scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core) if core is not None else ""


def scaffold_split(smiles_list: list[str], frac=(0.8, 0.1, 0.1),
                   seed: int = 0) -> tuple[list[int], list[int], list[int]]:
    """Split indices by Bemis-Murcko scaffold so no scaffold crosses splits."""
    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        groups.setdefault(scaffold_key(smi), []).append(i)
    order = sorted(groups.values(), key=lambda idxs: -len(idxs))
    rng = np.random.default_rng(seed)
    same_size: dict[int, list[list[int]]] = {}
    for idxs in order:
        same_size.setdefault(len(idxs), []).append(idxs)
    order = []
    for size in sorted(same_size, reverse=True):
        chunk = same_size[size]
        rng.shuffle(chunk)
        order.extend(chunk)
    n = len(smiles_list)
    n_train = int(frac[0] * n)
    n_val = int(frac[1] * n)
    train, val, test = [], [], []
    for idxs in order:
        if len(train) + len(idxs) <= n_train:
            train.extend(idxs)
        elif len(val) + len(idxs) <= n_val:
            val.extend(idxs)
        else:
            test.extend(idxs)
    return sorted(train), sorted(val), sorted(test)


def max_tanimoto(fps: np.ndarray, ref_fps: np.ndarray,
                 chunk: int = 1024) -> np.ndarray:
    """Max Tanimoto similarity of each fp to a reference fp set (uint8/01)."""
    ref = ref_fps.astype(np.float32)
    ref_sum = ref.sum(axis=1)
    out = np.zeros(len(fps), dtype=np.float32)
    for start in range(0, len(fps), chunk):
        block = fps[start:start + chunk].astype(np.float32)
        inter = block @ ref.T
        union = block.sum(axis=1)[:, None] + ref_sum[None, :] - inter
        out[start:start + chunk] = (inter / np.maximum(union, 1e-8)).max(axis=1)
    return out
