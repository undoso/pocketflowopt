"""RDKit featurization: SDF/PDB structures -> GeomGraph (IMPLEMENTATION.md P0).

- element typing onto models.pocketflow.ELEMENTS (the same index order the
  valence table in edit.constraints uses)
- bond typing onto the 4-class models.bonds.BOND_TYPES (aromatic rings are
  kekulized; aromaticity survives as a flag channel)
- pocket extraction: protein heavy atoms within POCKET_CUTOFF of the
  reference ligand; pocket edges are spatial, since PDB bond perception is
  unreliable
- scalar atom features occupy the first ATOM_FEAT_DIM of the encoder's 128
  scalar channels; the rest stay zero
"""
from __future__ import annotations

import torch

from ..geometry.gdbp import GeomGraph
from ..models.bonds import BOND_TYPES
from ..models.pocketflow import ELEMENTS

H_S, H_V, D_E = 128, 32, 8
ATOM_FEAT_DIM = 19
POCKET_CUTOFF = 6.0    # Angstrom around the reference ligand
SPATIAL_CUTOFF = 4.5   # Angstrom; non-bonded pocket/contact edges

STANDARD_RESIDUES = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})

_BOND_ORDER = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}
_DONOR_EL = {"N", "O", "S"}
_ACCEPTOR_EL = {"N", "O", "S", "F"}


def _elem_index(symbol: str) -> int:
    try:
        return ELEMENTS.index(symbol)
    except ValueError:
        raise ValueError(
            f"element {symbol!r} not in models.pocketflow.ELEMENTS") from None


def _ligand_atom_feats(atom) -> list[float]:
    """19 scalar channels for a sanitized ligand atom."""
    from rdkit import Chem

    onehot = [0.0] * len(ELEMENTS)
    onehot[_elem_index(atom.GetSymbol())] = 1.0
    sym = atom.GetSymbol()
    n_h = atom.GetTotalNumHs()
    return onehot + [
        atom.GetTotalDegree() / 6.0,
        atom.GetValence(Chem.ValenceType.EXPLICIT) / 8.0,
        atom.GetFormalCharge() / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        int(atom.GetHybridization()) / 8.0,
        float(sym in _DONOR_EL and n_h > 0),        # crude H-bond donor
        float(sym in _ACCEPTOR_EL and n_h == 0),    # crude H-bond acceptor
        0.0,                                        # is_protein
    ]


def _protein_atom_feats(symbol: str) -> list[float]:
    """Element one-hot + is_protein; residue-level channels come later."""
    onehot = [0.0] * len(ELEMENTS)
    onehot[_elem_index(symbol)] = 1.0
    return onehot + [0.0] * 8 + [1.0]


def _bond_attr(order_idx: int, aromatic: bool, in_ring: bool,
               conjugated: bool, spatial: bool) -> list[float]:
    v = [0.0] * D_E
    v[order_idx] = 1.0                      # BOND_TYPES order
    v[4], v[5] = float(aromatic), float(in_ring)
    v[6], v[7] = float(conjugated), float(spatial)
    return v


def _scalars_table(feats: list[list[float]]) -> torch.Tensor:
    s = torch.zeros(len(feats), H_S)
    if feats:
        s[:, :ATOM_FEAT_DIM] = torch.tensor(feats)
    return s


def _spatial_edges(coords: torch.Tensor, cutoff: float) -> torch.Tensor:
    d = torch.cdist(coords, coords)
    dst, src = ((0.0 < d) & (d <= cutoff)).nonzero(as_tuple=True)
    return torch.stack([dst, src])          # symmetric -> both directions


def _read_mol(path: str):
    from rdkit import Chem

    if path.endswith(".sdf"):
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mol = suppl[0] if len(suppl) else None
    elif path.endswith(".mol"):
        mol = Chem.MolFromMolFile(path, removeHs=False)
    elif path.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(path, sanitize=True, removeHs=False)
    else:
        raise ValueError(f"unsupported ligand format: {path}")
    if mol is None:
        raise ValueError(f"RDKit could not parse {path}")
    if mol.GetNumConformers() == 0:
        raise ValueError(f"{path} has no 3D conformer")
    return mol


def load_ligand(path: str, remove_hs: bool = True) -> GeomGraph:
    """SDF/MOL/PDB ligand -> typed graph (covalent edges, both directions)."""
    from rdkit import Chem

    mol = _read_mol(path)
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=False)
    except Chem.rdchem.KekulizeException:
        kek = mol

    coords = torch.tensor(mol.GetConformer().GetPositions(),
                          dtype=torch.float32)
    scalars = _scalars_table([_ligand_atom_feats(a) for a in mol.GetAtoms()])

    dst, src, attr = [], [], []
    for bond in mol.GetBonds():
        kb = kek.GetBondBetweenAtoms(bond.GetBeginAtomIdx(),
                                     bond.GetEndAtomIdx())
        order = _BOND_ORDER.get(str(kb.GetBondType()), 1)
        v = _bond_attr(order, bool(bond.GetIsAromatic()),
                       bool(bond.IsInRing()), bool(bond.GetIsConjugated()),
                       False)
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dst += [a, b]
        src += [b, a]
        attr += [v, v]
    edge_index = (torch.tensor([dst, src], dtype=torch.long)
                  if dst else torch.zeros(2, 0, dtype=torch.long))
    edge_attr = torch.tensor(attr) if attr else torch.zeros(0, D_E)
    n = coords.shape[0]
    return GeomGraph(coords, scalars, torch.zeros(n, H_V, 3),
                     edge_index, edge_attr, torch.zeros(n, dtype=torch.long))


def load_pocket(path: str, ref_mol: GeomGraph | None = None,
                cutoff: float = POCKET_CUTOFF) -> GeomGraph:
    """PDB protein -> pocket graph: heavy atoms of standard residues within
    `cutoff` of the reference ligand; edges are spatial contacts."""
    from rdkit import Chem

    mol = Chem.MolFromPDBFile(path, sanitize=False, removeHs=True)
    if mol is None:
        raise ValueError(f"RDKit could not parse {path}")
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)

    idx, feats = [], []
    for atom in mol.GetAtoms():
        res = atom.GetPDBResidueInfo()
        if (res is None
                or res.GetResidueName().strip() not in STANDARD_RESIDUES
                or atom.GetSymbol() not in ELEMENTS):
            continue
        idx.append(atom.GetIdx())
        feats.append(_protein_atom_feats(atom.GetSymbol()))
    coords = pos[idx] if idx else torch.zeros(0, 3)
    if ref_mol is not None and idx:
        dmin = torch.cdist(coords, ref_mol.coords).min(dim=1).values
        keep = dmin <= cutoff
        coords = coords[keep]
        feats = [f for f, k in zip(feats, keep.tolist()) if k]

    edge_index = (_spatial_edges(coords, SPATIAL_CUTOFF) if feats
                  else torch.zeros(2, 0, dtype=torch.long))
    edge_attr = (torch.tensor([_bond_attr(0, False, False, False, True)]
                              ).repeat(edge_index.shape[1], 1)
                 if edge_index.shape[1] else torch.zeros(0, D_E))
    n = coords.shape[0]
    return GeomGraph(coords, _scalars_table(feats), torch.zeros(n, H_V, 3),
                     edge_index, edge_attr, torch.zeros(n, dtype=torch.long))


def merge_graphs(a: GeomGraph, b: GeomGraph,
                 contact_cutoff: float = SPATIAL_CUTOFF) -> GeomGraph:
    """Concatenate two single-batch graphs and add contact edges (bond type
    'none', spatial flag) between nodes of a and b within contact_cutoff."""
    na, nb = a.coords.shape[0], b.coords.shape[0]
    coords = torch.cat([a.coords, b.coords])
    scalars = torch.cat([a.scalars, b.scalars])
    vectors = torch.cat([a.vectors, b.vectors])
    edge_index = torch.cat([a.edge_index, b.edge_index + na], dim=1)
    edge_attr = torch.cat([a.edge_attr, b.edge_attr])
    if na and nb:
        ii, jj = (torch.cdist(a.coords, b.coords)
                  <= contact_cutoff).nonzero(as_tuple=True)
        cross = torch.stack([torch.cat([ii, jj + na]),
                             torch.cat([jj + na, ii])])
        edge_index = torch.cat([edge_index, cross], dim=1)
        edge_attr = torch.cat([
            edge_attr,
            torch.tensor([_bond_attr(0, False, False, False, True)]
                         ).repeat(cross.shape[1], 1)])
    return GeomGraph(coords, scalars, vectors, edge_index, edge_attr,
                     torch.zeros(coords.shape[0], dtype=torch.long))
