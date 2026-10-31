"""Fixed/Variable partition objects (proposal Sec. 3.1).

A Mask selects the variable atoms to delete; everything left is the fixed
fragment, clamped to its co-crystal pose and used as generation context
only. Frontier bookkeeping lives here so samplers never touch masked atoms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..geometry.gdbp import GeomGraph


@dataclass
class Mask:
    """Boolean partition of lead atoms into fixed (False) / variable (True)."""

    variable: torch.Tensor  # [N] bool

    @property
    def fixed_idx(self) -> torch.Tensor:
        return (~self.variable).nonzero(as_tuple=True)[0]

    @property
    def variable_idx(self) -> torch.Tensor:
        return self.variable.nonzero(as_tuple=True)[0]


@dataclass
class Fragment:
    """Fixed substructure clamped at the co-crystal pose."""

    graph: GeomGraph
    frontier: torch.Tensor  # LongTensor: atoms allowed as growth anchors

    def fuse(self, grown: "GrownRegion") -> GeomGraph:
        """Merge the regrown region into the fragment graph."""
        base = self.graph.coords.shape[0]
        coords = torch.cat(
            [self.graph.coords, grown.coords().to(self.graph.coords.device)],
            dim=0)
        scalars = torch.cat([self.graph.scalars, grown.scalars(self.graph)], dim=0)
        vectors = torch.cat(
            [self.graph.vectors,
             torch.zeros(grown.size(), self.graph.vectors.shape[1], 3,
                         device=coords.device, dtype=coords.dtype)], dim=0)
        edges = [self.graph.edge_index]
        for dst_rel, src_rel, attr in grown.edges(base):
            edges.append(torch.tensor([[dst_rel], [src_rel]],
                                      device=coords.device))
        edge_index = torch.cat(edges, dim=1)
        edge_attr = torch.cat(
            [self.graph.edge_attr, grown.edge_attrs(self.graph, base)], dim=0)
        batch = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        return GeomGraph(coords, scalars, vectors, edge_index, edge_attr, batch)


@dataclass
class GrownRegion:
    """Atoms placed by masked regrowth (the variable set V)."""

    _records: list = field(default_factory=list)  # (elem_idx, xyz, bonds dict)

    def size(self) -> int:
        return len(self._records)

    def atoms(self, base: int = 0, device=None) -> torch.Tensor:
        """Node indices of grown atoms (offset by the fixed-graph size)."""
        if not self._records:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.arange(base, base + len(self._records), device=device)

    def add(self, elem_idx: int, xyz: torch.Tensor, bonds: dict) -> None:
        self._records.append((elem_idx, xyz.detach(), bonds))

    def coords(self) -> torch.Tensor:
        if not self._records:
            return torch.zeros(0, 3)
        return torch.stack([r[1].reshape(3) for r in self._records])

    def scalars(self, ref: GeomGraph) -> torch.Tensor:
        """Element one-hot lifted to the scalar channel width of ref."""
        n, h = self.size(), ref.scalars.shape[1]
        out = torch.zeros(n, h, device=ref.scalars.device, dtype=ref.scalars.dtype)
        for i, (elem_idx, _, _) in enumerate(self._records):
            out[i, elem_idx % h] = 1.0
        return out

    def edges(self, base: int):
        for i, (_, _, bonds) in enumerate(self._records):
            for dst, attr in bonds.items():
                yield base + i, dst, attr

    def edge_attrs(self, ref: GeomGraph, base: int) -> torch.Tensor:
        rows = [attr for _, _, bonds in self._records for attr in bonds.values()]
        if not rows:
            return torch.zeros(0, ref.edge_attr.shape[1],
                               device=ref.edge_attr.device)
        return torch.stack(rows)
