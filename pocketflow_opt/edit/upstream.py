"""Masked lead regrowth using the vendored upstream PocketFlow generator.

The upstream Generate.run starts from an empty ligand. FragmentGenerate changes
that behavior: when a retained fragment is present, new atoms are attached to
its frontier, including the first new atom. This module is used with the legacy
Python 3.8 / torch 1.13 environment required by the vendored model.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "pocketflow"
ATOM_TYPES = [6, 7, 8, 9, 15, 16, 17, 35, 53]
MAX_VALENCE = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1,
               35: 1, 53: 1}


def _imports():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from pocket_flow import Generate, PocketFlow
    from pocket_flow.utils import (
        AtomComposer, ComplexData, FeaturizeLigandAtom, FeaturizeProteinAtom,
        Ligand, Protein, RefineData, LigandCountNeighbors, mask_node,
        torchify_dict,
    )
    return Generate, PocketFlow, AtomComposer, ComplexData, FeaturizeLigandAtom, \
        FeaturizeProteinAtom, Ligand, Protein, RefineData, LigandCountNeighbors, \
        mask_node, torchify_dict


def _fragment_generator(Generate):
    class FragmentGenerate(Generate):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.max_total_atoms = self.max_atom_num
            self.traces = []

        def _action_logp(self, data, h_cpx, focal_idx, new_atom_type,
                         new_pos_to_add, new_edge_idx, new_bond_type_to_add):
            """Log-probability of an accepted action under the current model."""
            import torch.nn.functional as F
            from pocket_flow.utils import get_tri_edges
            from torch_geometric.nn import knn, radius

            with torch.no_grad():
                focal_pred = self.model.focal_net(
                    h_cpx, data.idx_ligand_ctx_in_cpx).view(-1)
                match = (data.idx_ligand_ctx_in_cpx == focal_idx).nonzero()
                focal_logp = F.logsigmoid(focal_pred[match[0, 0]])

                atom_onehot = F.one_hot(
                    new_atom_type,
                    num_classes=len(self.atom_type_map)).float()
                z_atom, atom_log_jacob = self.model.atom_flow(
                    atom_onehot, h_cpx, focal_idx)
                atom_logp = -(0.5 * z_atom.pow(2) - atom_log_jacob).sum()

                atom_type_emb = self.model.atom_type_embedding(
                    new_atom_type).view(-1, self.hidden_channels)
                _, abs_mu, sigma, pi = self.model.pos_predictor(
                    h_cpx, focal_idx, data.cpx_pos,
                    atom_type_emb=atom_type_emb)
                pos_prob = self.model.pos_predictor.get_mdn_probability(
                    abs_mu, sigma, pi, new_pos_to_add)
                pos_logp = torch.log(pos_prob + 1e-30).sum()

                bond_logp = torch.zeros((), device=h_cpx[0].device)
                if new_edge_idx.numel() > 0:
                    edge_index_query = radius(
                        data.ligand_context_pos, new_pos_to_add, r=4.0,
                        num_workers=16)
                    pos_query_knn_edge_idx = knn(
                        x=data.cpx_pos, y=new_pos_to_add,
                        k=self.model.config.encoder.knn, num_workers=16)
                    (index_real, tri_edge_index,
                     tri_edge_feat) = get_tri_edges(
                        edge_index_query, new_pos_to_add, data.context_idx,
                        data.ligand_context_bond_index,
                        data.ligand_context_bond_type)
                    bond_labels = torch.zeros(
                        edge_index_query.size(1), dtype=torch.long,
                        device=new_pos_to_add.device)
                    for k, (i, j) in enumerate(new_edge_idx.t().tolist()):
                        matches = ((edge_index_query[0] == i)
                                   & (edge_index_query[1] == j)).nonzero()
                        bond_labels[matches[0, 0]] = new_bond_type_to_add[k]
                    edge_onehot = F.one_hot(
                        bond_labels, num_classes=4).float()
                    z_edge, edge_log_jacob = self.model.edge_flow(
                        z_edge=edge_onehot, pos_query=new_pos_to_add,
                        edge_index_query=edge_index_query,
                        cpx_pos=data.cpx_pos, node_attr_compose=h_cpx,
                        edge_index_q_cps_knn=pos_query_knn_edge_idx,
                        index_real_cps_edge_for_atten=index_real,
                        tri_edge_index=tri_edge_index,
                        tri_edge_feat=tri_edge_feat,
                        atom_type_emb=atom_type_emb)
                    bond_logp = -(0.5 * z_edge.pow(2) - edge_log_jacob).sum()

                total = focal_logp + atom_logp + pos_logp + bond_logp
                return {
                    "logp_old": float(total),
                    "focal_logp": float(focal_logp),
                    "atom_logp": float(atom_logp),
                    "pos_logp": float(pos_logp),
                    "bond_logp": float(bond_logp),
                    "focal_idx": int(focal_idx),
                    "new_atom_type": int(new_atom_type),
                    "new_pos": new_pos_to_add.detach().cpu(),
                    "new_edge_idx": new_edge_idx.detach().cpu(),
                    "new_bond_type": new_bond_type_to_add.detach().cpu(),
                    "state": data.clone().detach(),
                }

        def run(self, data):
            from rdkit import Chem
            from pocket_flow.gdbp_model import embed_compose
            from pocket_flow.utils import (add_ligand_atom_to_data, data2mol,
                                           modify, check_alert_structures)

            if data.ligand_context_pos.size(0) > 0:
                self.max_atom_num = max(
                    0, self.max_total_atoms - data.ligand_context_pos.size(0))
            data.max_atom_valence = torch.empty(0, dtype=torch.long)
            data = data.to(self.device)
            trace = []
            with torch.no_grad():
                self.prior_node = torch.distributions.normal.Normal(
                    torch.zeros([len(self.atom_type_map)]).cuda(
                        data.cpx_pos.device),
                    self.temperature[0] * torch.ones(
                        [len(self.atom_type_map)]).cuda(data.cpx_pos.device))
                self.prior_edge = torch.distributions.normal.Normal(
                    torch.zeros([self.num_bond_type]).cuda(
                        data.cpx_pos.device),
                    self.temperature[1] * torch.ones(
                        [self.num_bond_type]).cuda(data.cpx_pos.device))

                rw_mol = Chem.RWMol()
                self.counter = 0
                for atom_idx in range(self.max_atom_num):
                    data = data.to(self.device)
                    h_cpx = embed_compose(
                        data.cpx_feature.float(), data.cpx_pos,
                        data.idx_ligand_ctx_in_cpx, data.idx_protein_in_cpx,
                        self.model.ligand_atom_emb,
                        self.model.protein_atom_emb, self.model.emb_dim)
                    h_cpx = self.model.encoder(
                        node_attr=h_cpx, pos=data.cpx_pos,
                        edge_index=data.cpx_edge_index,
                        edge_feature=data.cpx_edge_feature)
                    self.resample_edge_faild = False
                    self.check_node = True
                    self.resample_node = 0
                    while self.check_node:
                        if self.resample_node > 50:
                            break
                        focal_out = self.choose_focal(
                            h_cpx, data.idx_protein_in_cpx,
                            data.idx_ligand_ctx_in_cpx, data, atom_idx)
                        if focal_out is False:
                            break
                        focal_idx, focal_prob = focal_out
                        new_atom_type, focal_idx = self.atom_generate(
                            h_cpx, focal_idx, focal_prob, atom_idx)
                        atom_type_emb = self.model.atom_type_embedding(
                            new_atom_type).view(-1, self.hidden_channels)
                        new_pos_to_add = self.pos_generate(
                            h_cpx, atom_type_emb, focal_idx, data.cpx_pos,
                            atom_idx)
                        if new_pos_to_add is False:
                            self.resample_node += 1
                            continue
                        rw_mol.AddAtom(Chem.Atom(
                            self.atom_type_map[new_atom_type]))
                        bond_out = self.bond_generate(
                            h_cpx, data, new_pos_to_add, atom_type_emb,
                            atom_idx, rw_mol)
                        if bond_out is not False:
                            rw_mol, new_edge_idx, new_bond_type_to_add = bond_out
                            has_alert = check_alert_structures(
                                rw_mol, ['[O]-[O]', '[N]-[O,Br,Cl,I,F,P]',
                                         '[S,P]-[Br,Cl,I,F]', '[P]-[O]-[P]',
                                         '[Br,Cl,I,F]-[Br,Cl,I,F]'])
                            if has_alert:
                                for ix in range(new_edge_idx.size(1)):
                                    _, j = new_edge_idx[:, ix].tolist()
                                    rw_mol.RemoveBond(atom_idx, j)
                                rw_mol.RemoveAtom(atom_idx)
                                self.resample_node += 1
                                continue
                            break
                        if self.resample_edge_faild:
                            break
                    if (self.resample_edge_faild or self.resample_node > 50
                            or focal_out is False):
                        break
                    trace.append(self._action_logp(
                        data, h_cpx, focal_idx, new_atom_type, new_pos_to_add,
                        new_edge_idx, new_bond_type_to_add))
                    data = data.to('cpu')
                    data = add_ligand_atom_to_data(
                        data, new_pos_to_add.to('cpu'), new_atom_type.to('cpu'),
                        new_edge_idx.to('cpu'), new_bond_type_to_add.to('cpu'),
                        type_map=self.atom_type_map)
                    data = self.transform(data)
            try:
                mol = data2mol(data)
                modified_mol = modify(
                    mol, max_double_in_6ring=self.max_double_in_6ring)
                if trace:
                    self.traces.append(trace)
                return modified_mol, mol
            except Exception:
                mol_ = rw_mol.GetMol()
                print('Invalid mol: ', Chem.MolToSmiles(mol_))
                data2mol(data)
                return None

        def choose_focal(self, h_cpx, cpx_index, idx_ligand_ctx_in_cpx,
                         data, atom_idx):
            if data.ligand_context_pos.size(0) == 0:
                return super().choose_focal(h_cpx, cpx_index,
                                            idx_ligand_ctx_in_cpx, data,
                                            atom_idx)
            focal_idx, focal_prob = Generate._Generate__choose_focal(
                self.model.focal_net, h_cpx, idx_ligand_ctx_in_cpx,
                self.focus_threshold, 1)
            if focal_idx is False:
                return False
            max_valence = torch.tensor(
                [MAX_VALENCE[int(e)] for e in data.ligand_context_element],
                device=focal_idx.device)
            allowed = (max_valence[focal_idx]
                       > data.ligand_context_valence[focal_idx])
            if allowed.sum() == 0:
                return False
            return focal_idx[allowed], focal_prob[allowed]

        def pos_generate(self, h_cpx, atom_type_emb, focal_idx, cpx_pos,
                         atom_idx):
            effective_idx = 1 if cpx_pos.size(0) else atom_idx
            return super().pos_generate(h_cpx, atom_type_emb, focal_idx,
                                        cpx_pos, effective_idx)

        def bond_generate(self, h_cpx, data, new_pos_to_add, atom_type_emb,
                          atom_idx, rw_mol):
            if data.ligand_context_pos.size(0) == 0:
                return super().bond_generate(h_cpx, data, new_pos_to_add,
                                             atom_type_emb, atom_idx, rw_mol)
            # Upstream's scratch RWMol contains only generated atoms. Fragment
            # bonds must instead be validated against the full current context.
            from rdkit import Chem

            full = Chem.RWMol()
            for element in data.ligand_context_element.tolist():
                full.AddAtom(Chem.Atom(int(element)))
            bond_index = data.ligand_context_bond_index.tolist()
            bond_types = data.ligand_context_bond_type.tolist()
            for (i, j), order in zip(zip(*bond_index), bond_types):
                if i < j and order > 0:
                    full.AddBond(i, j, self.bond_type_map[order])
            new_element = rw_mol.GetAtomWithIdx(
                rw_mol.GetNumAtoms() - 1).GetAtomicNum()
            full.AddAtom(Chem.Atom(new_element))
            return super().bond_generate(h_cpx, data, new_pos_to_add,
                                         atom_type_emb, full.GetNumAtoms() - 1,
                                         full)

    return FragmentGenerate


def replay_action_logp(generate, step: dict, device: str) -> dict:
    """Recompute one stored action's log-probability from its saved state.

    This is the trace round-trip check required before PPO importance ratios
    can be trusted. It reruns only the encoder and probability heads, not the
    sampling procedure.
    """
    from pocket_flow.gdbp_model import embed_compose

    data = step["state"].to(device)
    h_cpx = embed_compose(
        data.cpx_feature.float(), data.cpx_pos, data.idx_ligand_ctx_in_cpx,
        data.idx_protein_in_cpx, generate.model.ligand_atom_emb,
        generate.model.protein_atom_emb, generate.model.emb_dim)
    h_cpx = generate.model.encoder(
        node_attr=h_cpx, pos=data.cpx_pos, edge_index=data.cpx_edge_index,
        edge_feature=data.cpx_edge_feature)
    focal_idx = torch.tensor([step["focal_idx"]], device=device)
    new_atom_type = torch.tensor([step["new_atom_type"]], device=device)
    return generate._action_logp(
        data, h_cpx, focal_idx, new_atom_type,
        step["new_pos"].to(device), step["new_edge_idx"].to(device),
        step["new_bond_type"].to(device))


def regrow(lead: str, pocket: str, variable: list[int], ckpt: str,
           out: str, name: str = "lead_opt", n: int = 32,
           device: str = "cuda:0", seed: int = 0, max_atom_num: int = 40,
           atom_temperature: float = 1.0, bond_temperature: float = 1.0) -> Path:
    """Regrow a lead after deleting variable atom indices; return output dir."""
    (Generate, PocketFlow, AtomComposer, ComplexData, FeaturizeLigandAtom,
     FeaturizeProteinAtom, Ligand, Protein, RefineData, LigandCountNeighbors,
     mask_node, torchify_dict) = _imports()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    lead_mol = Ligand(lead)
    n_atoms = lead_mol.num_atoms
    variable = sorted(set(int(i) for i in variable))
    if not variable or any(i < 0 or i >= n_atoms for i in variable):
        raise ValueError(f"variable indices must be within [0, {n_atoms - 1}]")
    if len(variable) == n_atoms:
        raise ValueError("refusing to delete the entire lead fragment")
    kept = [i for i in range(n_atoms) if i not in set(variable)]

    pro_dict = Protein(pocket).get_atom_dict(removeHs=True, get_surf=True)
    data = ComplexData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pro_dict),
        ligand_dict=torchify_dict(lead_mol.to_dict()))
    data = RefineData()(data)
    data = LigandCountNeighbors()(data)
    data = FeaturizeProteinAtom()(data)
    data = FeaturizeLigandAtom(atomic_numbers=ATOM_TYPES)(data)
    data = mask_node(data, torch.tensor(kept, dtype=torch.long),
                     torch.tensor(variable, dtype=torch.long),
                     num_atom_type=9, y_pos_std=0.0)
    composer = AtomComposer(knn=16, num_workers=16, for_gen=True,
                            use_protein_bond=True)
    data = composer.run(data)

    checkpoint = torch.load(ckpt, map_location=device)
    model = PocketFlow(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])

    generate = _fragment_generator(Generate)(
        model, composer.run, temperature=[atom_temperature, bond_temperature],
        atom_type_map=ATOM_TYPES, num_bond_type=4, max_atom_num=max_atom_num,
        focus_threshold=0.5, max_double_in_6ring=0, min_dist_inter_mol=3.0,
        bond_length_range=(1.0, 2.0), choose_max=True, device=device)
    run_name = f"{name}__del_{'-'.join(map(str, variable))}"
    generate.generate(data, num_gen=n, rec_name=run_name, with_print=False,
                      root_path=out)
    out_dir = Path(generate.out_dir)
    shutil.copy2(lead, out_dir / "lead.sdf")
    shutil.copy2(pocket, out_dir / "pocket.pdb")
    manifest = {
        "lead": str(Path(lead).resolve()),
        "pocket": str(Path(pocket).resolve()),
        "checkpoint": str(Path(ckpt).resolve()),
        "deleted_atom_indices": variable,
        "retained_atom_indices": kept,
        "requested": n,
        "generated": (out_dir / "generated.sdf").read_text().count("$$$$")
        if (out_dir / "generated.sdf").exists() else 0,
        "seed": seed,
        "device": device,
        "max_atom_num": max_atom_num,
        "atom_temperature": atom_temperature,
        "bond_temperature": bond_temperature,
    }
    traces = getattr(generate, "traces", [])
    torch.save(traces, out_dir / "action_traces.pt")
    trace_summary = [{
        "episode": i,
        "n_steps": len(trace),
        "logp_old": float(sum(step["logp_old"] for step in trace)),
        "components": {
            key: float(sum(step[key] for step in trace))
            for key in ("focal_logp", "atom_logp", "pos_logp", "bond_logp")
        },
    } for i, trace in enumerate(traces)]
    (out_dir / "action_traces.json").write_text(
        json.dumps(trace_summary, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "readme.txt").write_text(
        f"lead: {lead}\npocket: {pocket}\ndeleted: {variable}\n"
        f"checkpoint: {ckpt}\n")
    return out_dir
