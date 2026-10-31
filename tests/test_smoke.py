"""Smoke tests for the three invariants the WPs rely on:

1. E(3)-equivariance of the GDBP backbone (Eq. 4)
2. Invertibility of the conditional flows (Sec. 2.3)
3. Constraint layer admits only legal steps (Eq. 6)
"""
from __future__ import annotations

import pytest
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

    grown = type("R", (), {"size": lambda self: 0, "_records": []})()

    assert not constraints.ok(grown, frag_coords, pocket, 1, xyz_close,
                              {0: order1})                       # clash rejected
    assert constraints.ok(grown, frag_coords, pocket, 1, xyz_far, {})
    xyz_badlen = torch.tensor([5.0, 0.0, 0.0])
    frag2_coords = torch.tensor([[4.0, 0.0, 0.0]])
    assert constraints.ok(grown, frag2_coords, pocket, 1, xyz_badlen,
                          {0: order1})                           # 1.0 A bond ok
    xyz_toolong = torch.tensor([8.0, 0.0, 0.0])
    assert not constraints.ok(grown, frag2_coords, pocket, 1, xyz_toolong,
                              {0: order1})                       # 4.0 A rejected


def test_constraints_grown_focal():
    """bond_map keys index the merged graph: a grown atom (index >= fragment
    size) as bond target must not run off the fragment coordinate table."""
    from pocketflow_opt.edit.mask import GrownRegion

    pocket = type("P", (), {"coords": torch.zeros(0, 3)})()
    order1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    grown = GrownRegion()
    grown.add(1, torch.tensor([2.4, 0.0, 0.0]), {1: order1})
    merged_coords = torch.tensor([[0.0, 0.0, 0.0],
                                  [1.2, 0.0, 0.0],
                                  [2.4, 0.0, 0.0]])             # grown = idx 2
    assert constraints.ok(grown, merged_coords, pocket, 1,
                          torch.tensor([3.6, 0.0, 0.0]), {2: order1})
    assert not constraints.ok(grown, merged_coords, pocket, 1,
                              torch.tensor([9.0, 0.0, 0.0]), {2: order1})


def test_valence():
    assert constraints.valence_ok(1, 3, 1)      # carbon: 3+1 <= 4
    assert not constraints.valence_ok(1, 4, 1)  # carbon: 4+1 > 4
    assert constraints.valence_ok(3, 1, 1)      # oxygen: 1+1 <= 2


def test_regrow_runs():
    """Multi-step regrowth must not crash (regression: grown-atom focal)."""
    from pocketflow_opt.edit.mask import Fragment
    from pocketflow_opt.edit.operators import regrow
    from pocketflow_opt.models.pocketflow import PocketFlow

    torch.manual_seed(0)
    model = PocketFlow().eval()
    frag = Fragment(graph=_graph(), frontier=torch.arange(6))
    pocket = type("P", (), {"coords": torch.randn(40, 3) * 8})()
    grown = regrow(model, frag, pocket, budget=12)
    assert grown.size() >= 0                    # any outcome, but no crash


def test_featurize_ligand(tmp_path):
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from pocketflow_opt.chem import featurize

    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    Chem.MolToMolFile(mol, str(tmp_path / "bz.mol"))

    g = featurize.load_ligand(str(tmp_path / "bz.mol"))
    assert g.coords.shape == (6, 3)                # heavy atoms only
    assert g.scalars.shape == (6, 128)
    assert g.scalars[:, 1].all()                   # all carbon (ELEMENTS idx 1)
    assert g.vectors.shape == (6, 32, 3)
    assert g.edge_index.shape == (2, 12)           # 6 bonds, both directions
    orders = g.edge_attr[:, :4].argmax(-1)         # kekulized benzene
    assert (orders == 1).sum() == 6 and (orders == 2).sum() == 6
    assert g.edge_attr[:, 4].all()                 # aromatic flag survives


def _pdb_line(i: int, name: str, res: str, chain: str, resnum: int,
              x: float, y: float, z: float, elem: str) -> str:
    return (f"ATOM  {i:5d} {name:<4s} {res:>3s} {chain}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}")


def test_featurize_pocket(tmp_path):
    pytest.importorskip("rdkit")
    from pocketflow_opt.chem import featurize

    lines, i = [], 1
    for resnum, base in ((1, 0.0), (2, 50.0)):     # residue 2 far from ligand
        for name, elem, dx in (("N", "N", 0.0), ("CA", "C", 1.4),
                               ("C", "C", 2.8), ("O", "O", 3.5)):
            lines.append(_pdb_line(i, name, "ALA", "A", resnum,
                                   base + dx, 0.0, 0.0, elem))
            i += 1
    pdb = tmp_path / "prot.pdb"
    pdb.write_text("\n".join(lines) + "\nEND\n")

    ref = GeomGraph(torch.zeros(1, 3), torch.zeros(1, 128),
                    torch.zeros(1, 32, 3), torch.zeros(2, 0, dtype=torch.long),
                    torch.zeros(0, 8), torch.zeros(1, dtype=torch.long))
    g = featurize.load_pocket(str(pdb), ref_mol=ref, cutoff=6.0)
    assert g.coords.shape[0] == 4                  # only residue 1 within 6 A
    assert g.scalars[:, 18].all()                  # is_protein channel
    assert g.edge_index.shape == (2, 12)           # 4 atoms, all within 4.5 A
    assert g.edge_attr[:, 0].all()                 # spatial edges: bond "none"
    assert g.edge_attr[:, 7].all()                 # spatial flag


def test_ifp_conservation_math():
    from pocketflow_opt.eval.pose_checks import conservation

    reference = {("ASP26.A", "VdWContact"), ("ARG82.A", "Hydrophobic")}
    assert conservation(reference, reference) == 1.0
    assert conservation(reference, {("ASP26.A", "VdWContact")}) == 0.5
    assert conservation(reference, set()) == 0.0
    assert conservation(set(), set()) == 1.0


def test_candidate_ranking():
    from pocketflow_opt.eval.rank import pareto_front, rank_records, sa_score

    assert sa_score(1.0) == 1.0
    assert sa_score(6.0) == 0.0
    records = [
        {"similarity": 0.9, "qed": 0.8, "sa": 2.0, "mw": 350.0},
        {"similarity": 0.8, "qed": 0.7, "sa": 3.0, "mw": 400.0},
        {"similarity": 0.7, "qed": 0.6, "sa": 4.0, "mw": 450.0},
    ]
    assert pareto_front(records) == [0]
    ranked = rank_records(records, top_n=2)
    assert len(ranked) == 2 and ranked[0]["rank"] == 1
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_rl_reward_terms():
    from pocketflow_opt.rl.reward import (RewardConfig, compute_reward,
                                          feasibility_reward, logp_reward,
                                          qsar_reward)

    cfg = RewardConfig(w_qsar=1.0, w_logp=0.5, w_feasibility=0.5)
    assert qsar_reward(4.0, cfg) == 0.0
    assert qsar_reward(10.0, cfg) == 1.0
    assert logp_reward(2.5, cfg) == 1.0
    assert logp_reward(-2.0, cfg) == 0.0
    easy = feasibility_reward({"sa": 2.0, "mw": 300.0}, cfg)
    hard = feasibility_reward({"sa": 8.0, "mw": 700.0}, cfg)
    assert easy > hard
    base = compute_reward({"qed": 0.5, "sim": 0.5}, RewardConfig())
    with_qsar = compute_reward({"qed": 0.5, "sim": 0.5,
                                "qsar_activity": 10.0},
                               RewardConfig(w_qsar=1.0))
    assert with_qsar > base


def test_merge_graphs():
    from pocketflow_opt.chem import featurize

    def gg(coords: torch.Tensor) -> GeomGraph:
        n = coords.shape[0]
        return GeomGraph(coords, torch.zeros(n, 128), torch.zeros(n, 32, 3),
                         torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 8),
                         torch.zeros(n, dtype=torch.long))

    a = gg(torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    b = gg(torch.tensor([[1.0, 0.0, 0.0], [50.0, 0.0, 0.0]]))
    m = featurize.merge_graphs(a, b)
    assert m.coords.shape[0] == 4
    # contacts a0-b0 and a1-b0 (both 1 A), each both directions
    assert m.edge_index.shape == (2, 4)
    assert m.edge_attr[:, 7].all()                 # spatial flag
    assert m.batch.shape == (4,)


def test_viewer(tmp_path):
    import threading
    import urllib.request

    from pocketflow_opt.viewer import server

    run = tmp_path / "lead_x" / "round0"
    run.mkdir(parents=True)
    (run / "generated.sdf").write_text("m1\n\n$$$$\nm2\n\n$$$$\n")
    (run / "generated.smi").write_text("CCO\nCCN\n")
    (run / "pocket.pdb").write_text("ATOM      1  CA  ALA A   1       "
                                    "0.000   0.000   0.000  1.00  0.00"
                                    "           C\nEND\n")

    tracks = server.discover(tmp_path)
    assert len(tracks) == 1 and tracks[0]["n_mols"] == 2

    srv = server.make_server(tmp_path, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read()
        assert b"3Dmol" in page
        mol = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/mol?track="
            f"lead_x/round0/generated.sdf&i=1").read()
        assert b"CCN" in mol
        pkt = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/pocket?track="
            f"lead_x/round0/generated.sdf").read()
        assert b"ALA" in pkt
    finally:
        srv.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_regrow_gpu():
    """regrow and ppo_loss must run device-consistently on CUDA."""
    from pocketflow_opt.edit.mask import Fragment
    from pocketflow_opt.edit.operators import regrow
    from pocketflow_opt.models.pocketflow import PocketFlow
    from pocketflow_opt.rl.ppo import Rollout, RolloutBatch, ppo_loss

    dev = "cuda:0"
    torch.manual_seed(0)
    model = PocketFlow().to(dev).eval()
    g = _graph()
    g = type(g)(*[t.to(dev) if torch.is_tensor(t) else t
                  for t in (g.coords, g.scalars, g.vectors,
                            g.edge_index, g.edge_attr, g.batch)])
    frag = Fragment(graph=g, frontier=torch.arange(6, device=dev))
    pocket = type("P", (), {"coords": torch.randn(40, 3, device=dev) * 8})()
    regrow(model, frag, pocket, budget=12)

    policy = PocketFlow().to(dev)
    ctx = policy.encode(g)
    step = {"frontier": torch.arange(6, device=dev), "focal_idx": 2,
            "elem_onehot": torch.randn(1, 10, device=dev),
            "xyz": torch.randn(3, device=dev),
            "bonds": torch.randn(1, 4, device=dev), "context": ctx}
    batch = RolloutBatch()
    for i in range(4):
        batch.rollouts.append(Rollout(trace=[step], reward=float(i),
                                      logp_old=torch.zeros(())))
    loss = ppo_loss(policy, model, batch)
    assert torch.isfinite(loss)
