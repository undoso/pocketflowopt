"""E(3)-equivariant geometric double-bottleneck perceptron (GDBP) backbone.

Implements the scalar/vector feature scheme of Eq. (3)-(4) in the proposal:
nodes carry invariant scalars ``s`` and equivariant vector channels ``V``;
messages are built from distances (invariant) and displacement vectors
(equivariant); vector channels pass through a low-rank bottleneck.

Under x -> Qx + t, scalars are invariant and vectors transform as V Q^T.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GeomGraph:
    """Batched geometric graph: ligand fragment, grown region, or pocket."""

    coords: torch.Tensor   # [N, 3]
    scalars: torch.Tensor  # [N, h_s]
    vectors: torch.Tensor  # [N, h_v, 3]
    edge_index: torch.Tensor  # [2, E], directed edges j -> i
    edge_attr: torch.Tensor   # [E, d_e] invariant edge features (bond type one-hot, ...)
    batch: torch.Tensor       # [N] graph id per node

    def to(self, device) -> "GeomGraph":
        return GeomGraph(
            self.coords.to(device), self.scalars.to(device),
            self.vectors.to(device), self.edge_index.to(device),
            self.edge_attr.to(device), self.batch.to(device),
        )


def _scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    return out


class GDBPLayer(nn.Module):
    """One equivariant message-passing layer with a vector bottleneck.

    scalars: s_i' = s_i + sum_j phi_s(s_i, s_j, ||x_j - x_i||^2, a_ij)
    vectors: V_i' = W2 sigma(W1 V_i) + sum_j phi_v(m_ij) * (x_j - x_i)
    """

    def __init__(self, h_s: int, h_v: int, d_e: int, bottleneck: int = 16):
        super().__init__()
        self.phi_s = nn.Sequential(
            nn.Linear(2 * h_s + d_e + 1, h_s), nn.SiLU(),
            nn.Linear(h_s, h_s), nn.SiLU(),
        )
        self.phi_v = nn.Sequential(
            nn.Linear(h_s, h_v), nn.Sigmoid(),  # gate in (0,1)
        )
        # vector bottleneck W1: h_v -> c, W2: c -> h_v applied along channels
        self.w1 = nn.Parameter(torch.randn(bottleneck, h_v) * 0.02)
        self.w2 = nn.Parameter(torch.randn(h_v, bottleneck) * 0.02)
        self.norm_act = nn.SiLU()

    def forward(self, g: GeomGraph) -> GeomGraph:
        dst, src = g.edge_index[0], g.edge_index[1]
        diff = g.coords[src] - g.coords[dst]                    # [E, 3] equivariant
        dist2 = diff.pow(2).sum(-1, keepdim=True)               # [E, 1] invariant

        m_in = torch.cat(
            [g.scalars[dst], g.scalars[src], dist2, g.edge_attr], dim=-1)
        m_s = self.phi_s(m_in)                                  # [E, h_s]
        gate = self.phi_v(m_s)                                  # [E, h_v]
        m_v = gate.unsqueeze(-1) * diff.unsqueeze(1)            # [E, h_v, 3]

        n = g.coords.shape[0]
        s_agg = _scatter_add(m_s, dst, n)
        v_agg = _scatter_add(m_v, dst, n)

        # bottleneck on vector channels: sigma acts on channel norms (invariant)
        low = torch.einsum("cv,nvd->ncd", self.w1, g.vectors)   # [N, c, 3]
        scale = self.norm_act(low.norm(dim=-1))                 # [N, c] invariant
        low = low * scale.unsqueeze(-1)
        v_bottle = torch.einsum("vc,ncd->nvd", self.w2, low)    # [N, h_v, 3]

        return GeomGraph(
            coords=g.coords,
            scalars=g.scalars + s_agg,
            vectors=g.vectors + v_bottle + v_agg,
            edge_index=g.edge_index,
            edge_attr=g.edge_attr,
            batch=g.batch,
        )


class GDBPEncoder(nn.Module):
    """Stack of GDBP layers producing a per-graph context embedding c."""

    def __init__(self, h_s: int = 128, h_v: int = 32, d_e: int = 8,
                 layers: int = 4, bottleneck: int = 16, d_ctx: int = 256):
        super().__init__()
        self.layers = nn.ModuleList(
            GDBPLayer(h_s, h_v, d_e, bottleneck) for _ in range(layers))
        self.head = nn.Sequential(
            nn.Linear(h_s, d_ctx), nn.SiLU(), nn.Linear(d_ctx, d_ctx))

    def forward(self, g: GeomGraph) -> tuple[torch.Tensor, GeomGraph]:
        for layer in self.layers:
            g = layer(g)
        n_graphs = int(g.batch.max().item()) + 1 if g.batch.numel() else 1
        pooled = _scatter_add(g.scalars, g.batch, n_graphs)
        ctx = self.head(pooled)                                  # [B, d_ctx]
        return ctx, g
