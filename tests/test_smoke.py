"""Smoke tests for the three invariants the WPs rely on:

1. E(3)-equivariance of the GDBP backbone (Eq. 4)
2. Invertibility of the conditional flows (Sec. 2.3)
3. Constraint layer admits only legal steps (Eq. 6)
"""
from __future__ import annotations

import torch

from pocketflow_opt.edit import constraints
from pocketflow_opt.geometry.gdbp import GDBPEncoder, GeomGraph
from pocketflow_opt.models.flows import CondFlow


def _graph(n: int = 6) -> GeomGraph:
    torch.manual_seed(0)
    coords = torch.randn(n, 3)
    scalars = torch.randn(n, 128)
    vectors = torch.zeros(n, 32, 3)
    idx = torch.arange(n)
    edge_index = torch.stack([idx, idx.roll(1)])
    edge_attr = torch.zeros(n, 8)
    batch = torch.zeros(n, dtype=torch.long)
    return GeomGraph(coords, scalars, vectors, edge_index, edge_attr, batch)


def test_equivariance():
    enc = GDBPEncoder(h_s=128, h_v=32, d_e=8, layers=2)
    g = _graph()
    rot = torch.linalg.qr(torch.randn(3, 3))[0]
    ctx_a, ga = enc(g)
    g_rot = GeomGraph(g.coords @ rot, g.scalars, g.vectors @ rot,
                      g.edge_index, g.edge_attr, g.batch)
    ctx_b, gb = enc(g_rot)
    assert torch.allclose(ctx_a, ctx_b, atol=1e-4)          # scalars invariant
    va = ga.vectors
    vb = gb.vectors @ rot.T                                   # V -> V Q^T
    assert torch.allclose(va, vb, atol=1e-4)


def test_flow_invertible():
    flow = CondFlow(dim=10, d_ctx=32, depth=4)
    c = torch.randn(4, 32)
    z = flow.prior_sample(c)
    u = flow.decode(z, c)
    z2 = u
    for blk in flow.blocks:        # forward pass must recover the same latent
        z2, _ = blk(z2, c)
    assert torch.allclose(z, z2, atol=1e-4)
    onehot = torch.eye(10)[0].unsqueeze(0).expand(4, 10).contiguous()
    lp = flow.log_prob(flow.dequantize(onehot), c)
    assert lp.shape == (4,) and torch.isfinite(lp).all()


def test_constraints_legality():
    frag_coords = torch.tensor([[0.0, 0.0, 0.0]])
    pocket = type("P", (), {"coords": torch.tensor([[0.0, 0.0, 0.0]])})()
    xyz_close = torch.tensor([1.0, 0.0, 0.0])     # 1.0 A from pocket -> clash
    xyz_far = torch.tensor([5.0, 0.0, 0.0])
    order1 = torch.tensor([0.0, 1.0, 0.0, 0.0])   # single bond

    frag = type("F", (), {})()
    frag.graph = type("G", (), {"coords": frag_coords})()
    grown = type("R", (), {"size": lambda self: 0, "_records": []})()

    assert not constraints.ok(grown, frag, pocket, 1, xyz_close,
                              {0: order1})                       # clash rejected
    assert constraints.ok(grown, frag, pocket, 1, xyz_far, {})   # no bond -> ok
    xyz_badlen = torch.tensor([5.0, 0.0, 0.0])
    frag2 = type("F", (), {})()
    frag2.graph = type("G", (), {"coords": torch.tensor([[4.0, 0.0, 0.0]])})()
    assert constraints.ok(grown, frag2, pocket, 1, xyz_badlen,
                          {0: order1})                           # 1.0 A bond ok
    xyz_toolong = torch.tensor([8.0, 0.0, 0.0])
    assert not constraints.ok(grown, frag2, pocket, 1, xyz_toolong,
                              {0: order1})                       # 4.0 A rejected


def test_valence():
    assert constraints.valence_ok(1, 3, 1)      # carbon: 3+1 <= 4
    assert not constraints.valence_ok(1, 4, 1)  # carbon: 4+1 > 4
    assert constraints.valence_ok(3, 1, 1)      # oxygen: 1+1 <= 2
