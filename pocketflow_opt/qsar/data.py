"""Download and clean ChEMBL IC50 data for a single target."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
MUTANT_TOKENS = ("mutant", "mutation", "T790M", "L858R", "G719S", "C797S",
                 "L861Q", "exon", "del19")


def fetch_activities(target_id: str, limit: int = 1000,
                     max_pages: int = 100, pause: float = 0.2) -> list[dict]:
    """Fetch exact-relation IC50 (nM) activity records for a ChEMBL target."""
    records: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        params = {
            "target_chembl_id": target_id,
            "standard_type": "IC50",
            "standard_units": "nM",
            "standard_relation": "=",
            "limit": limit,
            "offset": offset,
        }
        url = CHEMBL_API + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=120) as fh:
            payload = json.loads(fh.read().decode())
        page = payload.get("activities", [])
        records.extend(page)
        if not payload.get("page_meta", {}).get("next") or not page:
            break
        offset += limit
        time.sleep(pause)
    return records


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    cols = ["molecule_chembl_id", "canonical_smiles", "standard_value",
            "pchembl_value", "assay_chembl_id", "assay_type",
            "assay_organism", "target_organism", "assay_description",
            "document_chembl_id"]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in records])


def canonical_smiles(smiles: str) -> str | None:
    """Largest-fragment canonical SMILES (salt stripped), or None if invalid."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)


def clean_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to human single-target assays and median pIC50 per compound."""
    df = df.copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    df = df[df["standard_value"] > 0]
    df["target_organism"] = df["target_organism"].fillna("")
    df["assay_organism"] = df["assay_organism"].fillna("")
    human = (df["target_organism"].str.contains("Homo sapiens",
                                                case=False) |
             df["assay_organism"].str.contains("Homo sapiens", case=False))
    df = df[human]
    desc = df["assay_description"].fillna("").str.lower()
    mask = np.ones(len(df), dtype=bool)
    for token in MUTANT_TOKENS:
        mask &= ~desc.str.contains(token.lower(), regex=False)
    df = df[mask]
    pchembl = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["pIC50"] = pchembl.fillna(9.0 - np.log10(df["standard_value"]))
    df["smiles"] = df["canonical_smiles"].map(canonical_smiles)
    df = df.dropna(subset=["smiles"])
    grouped = (df.groupby("smiles")
               .agg(pIC50=("pIC50", "median"),
                    n_measurements=("pIC50", "size"),
                    molecule_chembl_id=("molecule_chembl_id", "first"))
               .reset_index()
               .sort_values("pIC50", ascending=False))
    return grouped


def build_dataset(target_id: str, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = fetch_activities(target_id)
    raw = records_to_frame(records)
    raw.to_csv(out / "activities_raw.csv", index=False)
    clean = clean_activities(raw)
    path = out / "egfr_ic50_clean.csv"
    clean.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="CHEMBL203")
    ap.add_argument("--out", default="/mnt/d/opencode/egfr_opt")
    args = ap.parse_args()
    path = build_dataset(args.target, args.out)
    df = pd.read_csv(path)
    print(f"clean dataset -> {path} ({len(df)} compounds)")
