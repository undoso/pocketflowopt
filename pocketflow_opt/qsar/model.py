"""PyTorch MLP QSAR regressor with early stopping and an applicability domain."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from rdkit import Chem

from .features import featurize, max_tanimoto, morgan_fps, N_DESCRIPTORS, FP_BITS


class QSARNet(nn.Module):
    def __init__(self, in_dim: int = FP_BITS + N_DESCRIPTORS,
                 hidden: tuple[int, ...] = (512, 256), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_qsar(X: np.ndarray, y: np.ndarray, train_idx, val_idx,
               device: str = "cpu", epochs: int = 200, batch: int = 256,
               lr: float = 1e-3, patience: int = 20, seed: int = 0,
               verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_t = torch.device(device)
    desc_mean = X[train_idx, FP_BITS:].mean(axis=0)
    desc_std = X[train_idx, FP_BITS:].std(axis=0) + 1e-8
    Xn = X.copy()
    Xn[:, FP_BITS:] = (Xn[:, FP_BITS:] - desc_mean) / desc_std
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std() + 1e-8)

    Xt = torch.tensor(Xn, dtype=torch.float32)
    yt = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    model = QSARNet(X.shape[1]).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    tr = torch.tensor(train_idx, dtype=torch.long)
    va = torch.tensor(val_idx, dtype=torch.long)

    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            loss = loss_fn(model(Xt[idx].to(device_t)),
                           yt[idx].to(device_t))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xt[va].to(device_t)),
                               yt[va].to(device_t)).item()
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d} val_mse {val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val, best_state, wait = val_loss, \
                {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return {"model": model, "desc_mean": desc_mean, "desc_std": desc_std,
            "y_mean": y_mean, "y_std": y_std, "best_val_mse": best_val}


class QSARPredictor:
    """Trained QSAR model plus nearest-neighbor applicability domain."""

    def __init__(self, bundle: dict, train_fps: np.ndarray,
                 ad_threshold: float = 0.4):
        self.model: QSARNet = bundle["model"]
        self.model.eval()
        self.desc_mean = bundle["desc_mean"]
        self.desc_std = bundle["desc_std"]
        self.y_mean = bundle["y_mean"]
        self.y_std = bundle["y_std"]
        self.train_fps = train_fps.astype(np.uint8)
        self.ad_threshold = ad_threshold

    def predict(self, smiles_list: list[str],
                device: str = "cpu") -> list[dict]:
        X, keep = featurize(smiles_list)
        Xn = X.copy()
        Xn[:, FP_BITS:] = (Xn[:, FP_BITS:] - self.desc_mean) / self.desc_std
        self.model.to(device)
        with torch.no_grad():
            x = torch.tensor(Xn, dtype=torch.float32, device=device)
            pred = self.model(x).cpu().numpy() * self.y_std + self.y_mean
        sims = max_tanimoto(X[:, :FP_BITS], self.train_fps)
        out = []
        for row, i in enumerate(keep):
            out.append({"smiles": smiles_list[i], "pIC50": float(pred[row]),
                        "ad_max_tanimoto": float(sims[row]),
                        "in_ad": bool(sims[row] >= self.ad_threshold)})
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(),
                    "desc_mean": self.desc_mean, "desc_std": self.desc_std,
                    "y_mean": self.y_mean, "y_std": self.y_std,
                    "ad_threshold": self.ad_threshold,
                    "train_fps": self.train_fps}, path / "qsar_model.pt")

    @classmethod
    def load(cls, path: str | Path) -> "QSARPredictor":
        ckpt = torch.load(Path(path) / "qsar_model.pt",
                          map_location="cpu", weights_only=False)
        model = QSARNet()
        model.load_state_dict(ckpt["state_dict"])
        bundle = {"model": model, "desc_mean": ckpt["desc_mean"],
                  "desc_std": ckpt["desc_std"], "y_mean": ckpt["y_mean"],
                  "y_std": ckpt["y_std"]}
        return cls(bundle, ckpt["train_fps"], ckpt.get("ad_threshold", 0.4))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    err = y_pred - y_true
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "spearman": float(spearmanr(y_true, y_pred).statistic),
            "n": int(len(y_true))}


def evaluate(predictor: QSARPredictor, smiles_list, y_true, split_name,
             device="cpu") -> dict:
    preds = predictor.predict(list(smiles_list), device=device)
    smi_to_pred = {p["smiles"]: p["pIC50"] for p in preds}
    y_pred = np.array([smi_to_pred[s] for s in smiles_list])
    m = regression_metrics(np.asarray(y_true, dtype=float), y_pred)
    m["split"] = split_name
    return m


def save_metrics(metrics: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps({"metrics": metrics}, indent=2))


def train_fps_from_smiles(smiles_list: list[str]) -> np.ndarray:
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return morgan_fps([m for m in mols if m is not None]).astype(np.uint8)
