"""Autoregressive assembly of the PocketFlow modules (proposal Eq. 1).

Holds the five modules and the incremental context cache. The pretrained
checkpoint is loaded once and (in optimization mode) kept frozen; WP2's PPO
fine-tuning is the only place parameters are updated.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..geometry.gdbp import GDBPEncoder, GeomGraph
from .bonds import BondFlow
from .focal import FocalNet
from .flows import CondFlow
from .position import PositionPredictor

ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")


@dataclass
class Context:
    """Cached encoding of fixed fragment + pocket (+ grown region so far)."""

    graph: GeomGraph          # merged geometric graph
    ctx: torch.Tensor         # [1, d_ctx] pooled context embedding


class PocketFlow(nn.Module):
    def __init__(self, h_s: int = 128, h_v: int = 32, d_e: int = 8,
                 d_ctx: int = 256, layers: int = 4):
        super().__init__()
        self.encoder = GDBPEncoder(h_s=h_s, h_v=h_v, d_e=d_e,
                                   layers=layers, d_ctx=d_ctx)
        self.focal = FocalNet(h_s=h_s, d_ctx=d_ctx)
        self.position = PositionPredictor(h_s=h_s, d_ctx=d_ctx)
        self.atom_flow = CondFlow(dim=len(ELEMENTS), d_ctx=d_ctx, depth=6)
        self.bond_flow = BondFlow(d_ctx=d_ctx)
        self.d_ctx = d_ctx

    # ---- context management ------------------------------------------------
    def encode(self, merged: GeomGraph) -> Context:
        ctx, g = self.encoder(merged)
        return Context(graph=g, ctx=ctx)

    def encode_update(self, context: Context, merged: GeomGraph) -> Context:
        """Incremental refresh after a growth step (kept O(N) in practice)."""
        return self.encode(merged)

    # ---- exact likelihoods (used by PPO ratios / KL anchor) -----------------
    def trace_loglik(self, trace, context: Context) -> torch.Tensor:
        """Sum of per-step log p(a_t | s_t) over a growth trace.

        trace: list of dicts with keys 'focal', 'elem_onehot', 'xyz', 'bonds'.
        Exact because flows and the position Gaussian are normalized
        densities (Eq. 3); used for PPO importance ratios and the KL anchor.
        """
        ctx = context.ctx.expand(1, -1)
        total = context.graph.coords.new_zeros(())
        for step in trace:
            lg = self.focal.logits(context.graph, context.ctx.squeeze(0),
                                   step["frontier"])
            total = total + torch.log_softmax(lg, dim=-1)[step["focal_idx"]]
            total = total + self.atom_flow.log_prob(step["elem_onehot"], ctx).squeeze()
            total = total + self.position.log_prob(
                context.graph, step["focal_idx"], context.ctx.squeeze(0), step["xyz"])
            total = total + self.bond_flow.log_prob(step["bonds"], ctx).squeeze()
        return total

    @torch.no_grad()
    def load_pretrained(self, path: str, strict: bool = True):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["model"] if "model" in state else state,
                             strict=strict)
        for p in self.parameters():
            p.requires_grad_(False)  # frozen asset until WP2 fine-tuning
        return self
