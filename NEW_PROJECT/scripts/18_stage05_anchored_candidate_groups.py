from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_project_path
from src.data_utils import ensure_dir, normalize_column_list, prepare_supervised_dataset, read_numeric_csv, safe_name, save_json
from src.models import make_mlp


@dataclass
class Stage05Data:
    xi: str
    center: str
    ci_features: list[str]
    active_features: list[str]
    X_ci_raw: np.ndarray
    X_a_raw: np.ndarray
    residual_std: np.ndarray
    y_true: np.ndarray
    y_pred_c_oof: np.ndarray
    residual_mean: float
    residual_std_value: float
    train_idx: np.ndarray
    val_idx: np.ndarray
    source_stage03_xi_interface: Path


class AnchoredMultiHeadDGatedResidualNet(nn.Module):
    def __init__(
        self,
        ci_dim: int,
        active_dim: int,
        hidden_dims: Sequence[int],
        dgate_depth: int,
        anchor_indices: Sequence[int],
    ) -> None:
        super().__init__()
        dims = [int(v) for v in hidden_dims]
        if not dims:
            dims = [96, 48]
        if dgate_depth < 2:
            raise ValueError("dgate_depth must be >= 2.")
        self.ci_dim = int(ci_dim)
        self.active_dim = int(active_dim)
        self.head_count = len(anchor_indices)
        self.first_hidden_dim = dims[0]
        self.register_buffer("anchor_indices", torch.tensor(anchor_indices, dtype=torch.long))
        non_anchor = torch.ones(self.head_count, self.active_dim, dtype=torch.float32)
        for head, anchor_idx in enumerate(anchor_indices):
            non_anchor[head, int(anchor_idx)] = 0.0
        self.register_buffer("non_anchor_mask", non_anchor)

        self.omega_ci = nn.Parameter(torch.empty(self.ci_dim, self.first_hidden_dim))
        self.omega_a = nn.Parameter(torch.empty(self.head_count, self.active_dim, self.first_hidden_dim))
        self.gamma = nn.Parameter(torch.ones(self.head_count, int(dgate_depth) - 1, self.active_dim))
        self.bias = nn.Parameter(torch.zeros(self.first_hidden_dim))
        nn.init.kaiming_normal_(self.omega_ci, nonlinearity="relu")
        nn.init.kaiming_normal_(self.omega_a, nonlinearity="relu")

        layers: list[nn.Module] = []
        prev = self.first_hidden_dim
        for dim in dims[1:]:
            layers.append(nn.Linear(prev, int(dim)))
            layers.append(nn.ReLU())
            prev = int(dim)
        layers.append(nn.Linear(prev, 1))
        self.tail = nn.Sequential(*layers)

    def head_gates(self) -> torch.Tensor:
        gates = torch.prod(self.gamma, dim=1)
        anchor_one = torch.zeros_like(gates)
        anchor_one.scatter_(1, self.anchor_indices.view(-1, 1), 1.0)
        return gates * self.non_anchor_mask + anchor_one

    def effective_weights(self, keep: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.omega_a * self.head_gates().unsqueeze(-1)
        if keep is not None:
            weights = weights * keep.unsqueeze(-1)
        return weights

    def effective_group_norms(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.effective_weights(), ord=2, dim=2)

    def dgate_regularizer(self) -> torch.Tensor:
        mask = self.non_anchor_mask
        omega_penalty = torch.sum((self.omega_a**2) * mask.unsqueeze(-1))
        gamma_penalty = torch.sum((self.gamma**2) * mask.unsqueeze(1))
        return omega_penalty + gamma_penalty

    def forward(self, x_ci: torch.Tensor, x_a: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        h_ci = torch.matmul(x_ci, self.omega_ci)
        weights = self.effective_weights(keep=keep)
        h_a = torch.einsum("ba,kah->bkh", x_a, weights)
        h = torch.relu(h_ci.unsqueeze(1) + h_a + self.bias.view(1, 1, -1))
        flat = h.reshape(-1, h.shape[-1])
        out = self.tail(flat).reshape(x_ci.shape[0], self.head_count)
        return out


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def _standardize(train: np.ndarray, apply: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return ((apply - mean) / std).astype(np.float32), mean.squeeze().astype(np.float32), std.squeeze().astype(np.float32)


def _load_stage05_data(stage04_interface: Path, cfg: dict[str, Any], args: argparse.Namespace) -> Stage05Data:
    stage04 = _read_json(stage04_interface)
    stage03 = _read_json(Path(stage04["source_stage03_interface"]))
    xi = str(stage04["xi"])
    xi_interface = Path(stage03["xi_interfaces"][xi])
    xi_if = _read_json(xi_interface)
    stage03_config = _read_json(xi_interface.parents[1] / "stage_config.json")
    stage02 = _read_json(Path(stage03_config["stage02_interface"]))
    stage01 = _read_json(Path(stage02["source_stage01_interface"]))
    exclude_columns = []
    if stage01.get("dgate_run_dir"):
        exclude_columns = normalize_column_list(
            _read_json(Path(stage01["dgate_run_dir"]) / "config.json").get("preprocessing", {}).get("exclude_columns")
        )

    center = str(stage04["center"])
    ci_features = list(stage04["ci_features"])
    active_features = list(stage04["active_candidate_features"])
    if int(args.max_anchors) > 0:
        active_features = active_features[: int(args.max_anchors)]
    all_needed = list(dict.fromkeys([*ci_features, *active_features]))
    dataset_cfg = cfg["dataset"]
    preprocessing = cfg.get("preprocessing", {})
    df = read_numeric_csv(
        resolve_project_path(cfg, dataset_cfg["processed_csv"]),
        drop_all_zero_columns=bool(preprocessing.get("drop_all_zero_columns", False)),
        exclude_columns=exclude_columns,
    )
    missing = [col for col in all_needed if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    oof_df = pd.read_csv(xi_if["oof_residuals_csv"])
    source_rows = oof_df["source_row_index"].to_numpy(dtype=int)
    aligned = df.loc[source_rows, all_needed]
    if aligned.isna().any(axis=None):
        raise ValueError("NaN found after aligning Stage03 OOF rows to raw feature table.")

    n = len(oof_df)
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(n)
    val_size = max(1, int(n * float(args.val_ratio)))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return Stage05Data(
        xi=xi,
        center=center,
        ci_features=ci_features,
        active_features=active_features,
        X_ci_raw=aligned[ci_features].to_numpy(dtype=np.float32),
        X_a_raw=aligned[active_features].to_numpy(dtype=np.float32),
        residual_std=oof_df[str(xi_if.get("residual_target_column", "residual_std"))].to_numpy(dtype=np.float32).reshape(-1, 1),
        y_true=oof_df["y_true"].to_numpy(dtype=np.float32).reshape(-1, 1),
        y_pred_c_oof=oof_df["y_pred_C_oof"].to_numpy(dtype=np.float32).reshape(-1, 1),
        residual_mean=float(xi_if["residual_mean"]),
        residual_std_value=float(xi_if["residual_std_value"]),
        train_idx=train_idx,
        val_idx=val_idx,
        source_stage03_xi_interface=xi_interface,
    )


def _make_training_arrays(data: Stage05Data) -> tuple[DataLoader, tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    X_ci_all, ci_mean, ci_std = _standardize(data.X_ci_raw[data.train_idx], data.X_ci_raw)
    X_a_all, a_mean, a_std = _standardize(data.X_a_raw[data.train_idx], data.X_a_raw)
    y_all = data.residual_std.astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_ci_all[data.train_idx]),
            torch.from_numpy(X_a_all[data.train_idx]),
            torch.from_numpy(y_all[data.train_idx]),
        ),
        batch_size=64,
        shuffle=True,
    )
    return loader, (X_ci_all, X_a_all, y_all), {
        "ci_mean": ci_mean,
        "ci_std": ci_std,
        "active_mean": a_mean,
        "active_std": a_std,
    }


def _eval_multihead(
    model: AnchoredMultiHeadDGatedResidualNet,
    X_ci: np.ndarray,
    X_a: np.ndarray,
    residual_std: np.ndarray,
    data: Stage05Data,
    idx: np.ndarray,
    batch_size: int,
    device: torch.device,
    keep: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    keep_t = torch.from_numpy(keep.astype(np.float32)).to(device) if keep is not None else None
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_ci[idx].astype(np.float32)), torch.from_numpy(X_a[idx].astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for xb_ci, xb_a in loader:
            preds.append(model(xb_ci.to(device), xb_a.to(device), keep=keep_t).detach().cpu().numpy())
    pred_std = np.concatenate(preds, axis=0)
    y_std = residual_std[idx]
    mse_by_head = np.mean((pred_std - y_std) ** 2, axis=0)
    r2_by_head = np.array([_r2_np(y_std, pred_std[:, h]) for h in range(pred_std.shape[1])], dtype=np.float32)
    pred_raw_residual = pred_std * data.residual_std_value + data.residual_mean
    y_hat = data.y_pred_c_oof[idx] + pred_raw_residual
    y_r2_by_head = np.array([_r2_np(data.y_true[idx], y_hat[:, h]) for h in range(pred_std.shape[1])], dtype=np.float32)
    y_mse_by_head = np.mean((y_hat - data.y_true[idx]) ** 2, axis=0)
    return {
        "pred_std": pred_std,
        "residual_mse_by_head": mse_by_head,
        "residual_r2_by_head": r2_by_head,
        "y_r2_by_head": y_r2_by_head,
        "y_mse_by_head": y_mse_by_head,
    }


def _train_trial(
    data: Stage05Data,
    trial_dir: Path,
    lambda_dgate: float,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    ensure_dir(trial_dir)
    _set_seed(seed, device)
    train_loader, arrays, scalers = _make_training_arrays(data)
    X_ci_all, X_a_all, y_all = arrays
    model = AnchoredMultiHeadDGatedResidualNet(
        ci_dim=len(data.ci_features),
        active_dim=len(data.active_features),
        hidden_dims=args.hidden_dims,
        dgate_depth=int(args.dgate_depth),
        anchor_indices=list(range(len(data.active_features))),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    best = {"val_mse_mean": math.inf, "val_y_r2_mean": -math.inf, "epoch": 0}
    best_state: dict[str, torch.Tensor] | None = None
    log_rows = []
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for xb_ci, xb_a, yb in train_loader:
            xb_ci = xb_ci.to(device)
            xb_a = xb_a.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb_ci, xb_a)
            mse = torch.mean((pred - yb) ** 2)
            loss = mse + float(lambda_dgate) * model.dgate_regularizer()
            loss.backward()
            optimizer.step()
            losses.append(float(mse.detach().cpu()))
        val_eval = _eval_multihead(model, X_ci_all, X_a_all, y_all, data, data.val_idx, int(args.batch_size), device)
        val_mse_mean = float(np.mean(val_eval["residual_mse_by_head"]))
        val_y_r2_mean = float(np.mean(val_eval["y_r2_by_head"]))
        norms = model.effective_group_norms().detach().cpu().numpy()
        log_rows.append(
            {
                "epoch": epoch,
                "train_residual_mse_mean": float(np.mean(losses)),
                "val_residual_mse_mean": val_mse_mean,
                "val_y_r2_mean": val_y_r2_mean,
                "median_effective_norm": float(np.median(norms)),
                "max_effective_norm": float(np.max(norms)),
            }
        )
        if val_mse_mean < best["val_mse_mean"] - float(args.min_delta):
            best = {"val_mse_mean": val_mse_mean, "val_y_r2_mean": val_y_r2_mean, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if int(args.patience) > 0 and epoch >= 50 and stale >= int(args.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_eval = _eval_multihead(model, X_ci_all, X_a_all, y_all, data, data.val_idx, int(args.batch_size), device)
    train_eval = _eval_multihead(model, X_ci_all, X_a_all, y_all, data, data.train_idx, int(args.batch_size), device)
    norms = model.effective_group_norms().detach().cpu().numpy()

    threshold_rows = []
    for threshold in args.threshold_sweep:
        head_rows = _extract_heads(
            data=data,
            model=model,
            norms=norms,
            val_eval=val_eval,
            threshold_ratio=float(threshold),
            X_ci_all=X_ci_all,
            X_a_all=X_a_all,
            y_all=y_all,
            args=args,
            device=device,
        )
        for row in head_rows:
            row["threshold_ratio"] = float(threshold)
            threshold_rows.append(row)

    pd.DataFrame(log_rows).to_csv(trial_dir / "train_log.csv", index=False, encoding="utf-8-sig")
    head_df = pd.DataFrame(threshold_rows)
    head_df.to_csv(trial_dir / "head_candidate_groups.csv", index=False, encoding="utf-8-sig")
    np.save(trial_dir / "effective_group_norms.npy", norms)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": "AnchoredMultiHeadDGatedResidualNet",
            "ci_features": data.ci_features,
            "active_features": data.active_features,
            "hidden_dims": list(args.hidden_dims),
            "dgate_depth": int(args.dgate_depth),
            "lambda_dgate": float(lambda_dgate),
            **scalers,
        },
        trial_dir / "anchored_multihead_model.pt",
    )
    return {
        "trial": trial_dir.name,
        "trial_dir": str(trial_dir.resolve()),
        "lambda_dgate": float(lambda_dgate),
        "best_epoch": int(best["epoch"]),
        "train_residual_mse_mean": float(np.mean(train_eval["residual_mse_by_head"])),
        "train_y_r2_mean": float(np.mean(train_eval["y_r2_by_head"])),
        "val_residual_mse_mean": float(np.mean(val_eval["residual_mse_by_head"])),
        "val_y_r2_mean": float(np.mean(val_eval["y_r2_by_head"])),
        "min_group_size": int(head_df["group_size"].min()),
        "median_group_size": float(head_df["group_size"].median()),
        "max_group_size": int(head_df["group_size"].max()),
        "head_groups_csv": str((trial_dir / "head_candidate_groups.csv").resolve()),
    }


def _extract_heads(
    *,
    data: Stage05Data,
    model: AnchoredMultiHeadDGatedResidualNet,
    norms: np.ndarray,
    val_eval: dict[str, np.ndarray | float],
    threshold_ratio: float,
    X_ci_all: np.ndarray,
    X_a_all: np.ndarray,
    y_all: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for h, anchor in enumerate(data.active_features):
        non_anchor_norms = np.delete(norms[h], h)
        base = float(np.max(non_anchor_norms)) if len(non_anchor_norms) else 0.0
        selected = [anchor]
        for j, feature in enumerate(data.active_features):
            if j == h:
                continue
            if base > 0 and float(norms[h, j]) >= threshold_ratio * base:
                selected.append(feature)
        keep_full = np.ones((len(data.active_features), len(data.active_features)), dtype=np.float32)
        keep_anchor_off = keep_full.copy()
        keep_anchor_off[h, h] = 0.0
        masked_eval = _eval_multihead(
            model, X_ci_all, X_a_all, y_all, data, data.val_idx, int(args.batch_size), device, keep=keep_anchor_off
        )
        residual_mse = float(val_eval["residual_mse_by_head"][h])
        rows.append(
            {
                "head_id": h,
                "anchor": anchor,
                "group_size": len(selected),
                "candidate_group": ";".join(selected),
                "residual_val_mse": residual_mse,
                "residual_val_r2": float(val_eval["residual_r2_by_head"][h]),
                "reconstructed_y_val_mse": float(val_eval["y_mse_by_head"][h]),
                "reconstructed_y_val_r2": float(val_eval["y_r2_by_head"][h]),
                "anchor_sensitivity": float(masked_eval["residual_mse_by_head"][h] - residual_mse),
                "anchor_status": "active_anchor"
                if float(masked_eval["residual_mse_by_head"][h] - residual_mse) >= float(args.anchor_sensitivity_min)
                else "inactive_or_dummy_anchor",
            }
        )
    return rows


def _eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = []
    ys = []
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            losses.append(float(nn.functional.mse_loss(pred, yb).detach().cpu()) * len(xb))
            ys.append(yb.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
    y = np.concatenate(ys, axis=0)
    pred = np.concatenate(preds, axis=0)
    return float(np.sum(losses) / len(loader.dataset)), _r2_np(y, pred)


def _validate_group_worker(payload: dict[str, Any]) -> dict[str, Any]:
    features = list(payload["features"])
    all_features = list(payload["all_features"])
    idx = [all_features.index(feature) for feature in features]
    device = torch.device(payload["device"])
    seed = int(payload["seed"])
    _set_seed(seed, device)
    X_train = payload["X_train"][:, idx]
    X_test = payload["X_test"][:, idx]
    y_train = payload["y_train"]
    y_test = payload["y_test"]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=int(payload["batch_size"]),
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32))),
        batch_size=int(payload["batch_size"]),
        shuffle=False,
    )
    model = make_mlp(len(features), payload["hidden_dims"], dropout=float(payload["dropout"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(payload["lr"]))
    best = {"r2": -math.inf, "mse": math.inf, "epoch": 0}
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, int(payload["epochs"]) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
        test_mse, test_r2 = _eval_model(model, test_loader, device)
        if test_r2 > best["r2"] + float(payload["min_delta"]):
            best = {"r2": float(test_r2), "mse": float(test_mse), "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if int(payload["patience"]) > 0 and epoch >= 50 and stale >= int(payload["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_mse, train_r2 = _eval_model(model, train_loader, device)
    test_mse, test_r2 = _eval_model(model, test_loader, device)
    return {
        "group_id": payload["group_id"],
        "group_size": len(payload["candidate_group"]),
        "candidate_group": ";".join(payload["candidate_group"]),
        "feature_count_with_ci": len(features),
        "features_with_ci": ";".join(features),
        "source_heads": payload["source_heads"],
        "source_anchors": payload["source_anchors"],
        "best_test_r2": best["r2"],
        "best_test_mse": best["mse"],
        "best_epoch": best["epoch"],
        "final_train_r2": float(train_r2),
        "final_train_mse": float(train_mse),
        "final_test_r2": float(test_r2),
        "final_test_mse": float(test_mse),
    }


def _run_group_validation(
    *,
    data: Stage05Data,
    groups_df: pd.DataFrame,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    all_group_features = sorted(set(feature for group in groups_df["candidate_group"] for feature in str(group).split(";") if feature))
    validation_features = list(dict.fromkeys([*data.ci_features, *all_group_features]))
    stage03_config = _read_json(data.source_stage03_xi_interface.parents[1] / "stage_config.json")
    stage02 = _read_json(Path(stage03_config["stage02_interface"]))
    stage01 = _read_json(Path(stage02["source_stage01_interface"]))
    exclude_columns = []
    if stage01.get("dgate_run_dir"):
        exclude_columns = normalize_column_list(
            _read_json(Path(stage01["dgate_run_dir"]) / "config.json").get("preprocessing", {}).get("exclude_columns")
        )
    training = cfg.get("training", {})
    preprocessing = cfg.get("preprocessing", {})
    bundle = prepare_supervised_dataset(
        data_path=resolve_project_path(cfg, cfg["dataset"]["processed_csv"]),
        center=data.center,
        features=validation_features,
        train_ratio=float(training.get("train_ratio", 0.8)),
        random_state=int(training.get("random_state", 42)),
        drop_all_zero_columns=bool(preprocessing.get("drop_all_zero_columns", False)),
        exclude_columns=exclude_columns,
    )
    devices = [item.strip() for item in str(args.parallel_devices).split(",") if item.strip()]
    if not devices:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
    workers = min(int(args.num_workers) if int(args.num_workers) > 0 else len(devices), len(groups_df))
    jobs = []
    for n, row in groups_df.reset_index(drop=True).iterrows():
        candidate_group = [item for item in str(row["candidate_group"]).split(";") if item]
        features = list(dict.fromkeys([*data.ci_features, *candidate_group]))
        seed_text = f"{row['group_id']}|{row['candidate_group']}"
        jobs.append(
            {
                "group_id": row["group_id"],
                "candidate_group": candidate_group,
                "features": features,
                "all_features": validation_features,
                "X_train": bundle.X_train,
                "y_train": bundle.y_train,
                "X_test": bundle.X_test,
                "y_test": bundle.y_test,
                "hidden_dims": list(args.validation_hidden_dims),
                "epochs": int(args.validation_epochs),
                "batch_size": int(args.validation_batch_size),
                "lr": float(args.validation_lr),
                "dropout": float(args.validation_dropout),
                "patience": int(args.validation_patience),
                "min_delta": float(args.min_delta),
                "seed": int(args.seed) + zlib.adler32(seed_text.encode("utf-8")) % 100000,
                "device": devices[n % len(devices)],
                "source_heads": row["source_heads"],
                "source_anchors": row["source_anchors"],
            }
        )
    rows = []
    path = out_dir / "group_validation.csv"
    ctx = mp.get_context("spawn")
    if workers <= 1:
        for job in jobs:
            rows.append(_validate_group_worker(job))
            pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = {executor.submit(_validate_group_worker, job): job["group_id"] for job in jobs}
            for future in as_completed(futures):
                rows.append(future.result())
                pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows).sort_values(["best_test_r2", "group_size"], ascending=[False, True])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def _dedupe_groups(head_df: pd.DataFrame, trial: str, target_r2: float) -> pd.DataFrame:
    rows = []
    for group, sub in head_df.groupby("candidate_group", dropna=False):
        features = [item for item in str(group).split(";") if item]
        rows.append(
            {
                "group_id": f"group_{len(rows) + 1:03d}",
                "trial": trial,
                "candidate_group": ";".join(features),
                "group_size": len(features),
                "source_heads": ";".join(str(v) for v in sub["head_id"].tolist()),
                "source_anchors": ";".join(str(v) for v in sub["anchor"].tolist()),
                "mean_residual_val_mse": float(sub["residual_val_mse"].mean()),
                "mean_reconstructed_y_val_r2": float(sub["reconstructed_y_val_r2"].mean()),
                "mean_anchor_sensitivity": float(sub["anchor_sensitivity"].mean()),
                "inactive_anchor_count": int((sub["anchor_status"] != "active_anchor").sum()),
                "target_r2": float(target_r2),
            }
        )
    return pd.DataFrame(rows).sort_values(["group_size", "mean_reconstructed_y_val_r2"], ascending=[True, False])


def _plot_validation(df: pd.DataFrame, target_r2: float, out_path: Path) -> None:
    ordered = df.sort_values("group_size")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(ordered["group_size"], ordered["best_test_r2"], s=55, alpha=0.85)
    ax.axhline(target_r2, color="#d62728", linestyle="--", linewidth=1.5, label=f"target={target_r2:g}")
    ax.set_xlabel("Candidate group size |Q|")
    ax.set_ylabel("DNN test R2 on C_i + Q")
    ax.set_title("Stage 05 Candidate Group Validation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_report(run_dir: Path, data: Stage05Data, best: dict[str, Any], validation_df: pd.DataFrame) -> None:
    valid = validation_df[validation_df["best_test_r2"] >= float(best["target_r2"])]
    selected = valid.sort_values(["group_size", "best_test_r2"], ascending=[True, False]).iloc[0] if not valid.empty else validation_df.iloc[0]
    lines = [
        "# 阶段 05：锚定多头残差补偿门控",
        "",
        "## 目标",
        "",
        "本阶段以阶段 04 的活跃候选集 `A` 为候选池。每个活跃字段作为一个锚头，锚字段强制进入该头，其他字段通过 D-gating 压缩，输出候选替代组 `Q_k`。",
        "",
        "候选组随后用普通 DNN 在 `C_i + Q` 上重新训练验证，避免只依赖残差模型内部指标。",
        "",
        "## 本次字段",
        "",
        f"- xi: `{data.xi}`",
        f"- anchor/head count: `{len(data.active_features)}`",
        f"- C_i 字段数: `{len(data.ci_features)}`",
        "",
        "## 选中的压缩 trial",
        "",
        f"- trial: `{best['trial']}`",
        f"- lambda_dgate: `{best['lambda_dgate']}`",
        f"- threshold_ratio: `{best['threshold_ratio']}`",
        f"- unique candidate groups: `{int(best['unique_group_count'])}`",
        "",
        "## 最小达标候选组",
        "",
        f"- group_id: `{selected['group_id']}`",
        f"- |Q|: `{int(selected['group_size'])}`",
        f"- C_i + Q feature count: `{int(selected['feature_count_with_ci'])}`",
        f"- best test R2: `{float(selected['best_test_r2']):.6f}`",
        f"- source anchors: `{selected['source_anchors']}`",
        "",
        "## 产物",
        "",
        "- `stage05_candidate_group_interface.json`: 给下一阶段使用的候选替代组接口。",
        "- `xi=<field>/02_dgating_sweep/sweep_summary.csv`: D-gating 压缩强度 sweep。",
        "- `xi=<field>/03_group_validation/group_validation.csv`: 候选组真实 `C_i + Q` 验证。",
        "- `xi=<field>/04_candidate_groups/candidate_groups.csv`: 去重后的候选替代组。",
    ]
    (run_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage05: anchored multi-head D-gating candidate replacement groups.")
    parser.add_argument(
        "--stage04-interface",
        default=str(
            PROJECT_ROOT
            / "conditional_residual_compensation_outputs"
            / "CenterOn_net_actual_interchange_mw"
            / "stage04_residual_showing"
            / "run_20260706_000122"
            / "stage04_active_candidate_interface.json"
        ),
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "conditional_residual_compensation_outputs"))
    parser.add_argument("--stage-dir", default="stage05_candidate_groups")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-anchors", type=int, default=0, help="Use only the first N active anchors for a small probe.")
    parser.add_argument("--target-r2", type=float, default=0.95)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0007)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[96, 48, 24])
    parser.add_argument("--dgate-depth", type=int, default=3)
    parser.add_argument("--lambda-sweep", nargs="+", type=float, default=[1e-5, 3e-5, 1e-4, 3e-4])
    parser.add_argument("--threshold-sweep", nargs="+", type=float, default=[0.25, 0.35, 0.50])
    parser.add_argument("--anchor-sensitivity-min", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validation-epochs", type=int, default=140)
    parser.add_argument("--validation-batch-size", type=int, default=50)
    parser.add_argument("--validation-lr", type=float, default=0.0008)
    parser.add_argument("--validation-patience", type=int, default=30)
    parser.add_argument("--validation-hidden-dims", nargs="+", type=int, default=[64, 32, 16])
    parser.add_argument("--validation-dropout", type=float, default=0.0)
    parser.add_argument("--parallel-devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = _load_stage05_data(Path(args.stage04_interface), cfg, args)
    stage_root = Path(args.output_root).resolve() / f"CenterOn_{data.center}" / args.stage_dir
    run_dir = stage_root / (args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    xi_dir = ensure_dir(run_dir / f"xi={safe_name(data.xi)}")
    data_dir = ensure_dir(xi_dir / "01_data")
    sweep_dir = ensure_dir(xi_dir / "02_dgating_sweep")
    validation_dir = ensure_dir(xi_dir / "03_group_validation")
    groups_dir = ensure_dir(xi_dir / "04_candidate_groups")

    save_json(
        data_dir / "data_summary.json",
        {
            "xi": data.xi,
            "center": data.center,
            "ci_features": data.ci_features,
            "active_features": data.active_features,
            "anchor_count": len(data.active_features),
            "train_rows": int(len(data.train_idx)),
            "val_rows": int(len(data.val_idx)),
            "source_stage04_interface": str(Path(args.stage04_interface).resolve()),
        },
    )
    save_json(
        run_dir / "stage_config.json",
        {
            "stage": "stage05_candidate_groups",
            "stage04_interface": str(Path(args.stage04_interface).resolve()),
            "xi": data.xi,
            "target_r2": args.target_r2,
            "lambda_sweep": args.lambda_sweep,
            "threshold_sweep": args.threshold_sweep,
            "anchor_count": len(data.active_features),
            "validation_devices": args.parallel_devices,
        },
    )

    device = torch.device(args.device)
    trial_rows = []
    all_group_frames = []
    for trial_id, lambda_dgate in enumerate(args.lambda_sweep, start=1):
        trial_name = f"trial_{trial_id:02d}_lam{str(lambda_dgate).replace('.', 'p')}"
        trial_dir = sweep_dir / trial_name
        row = _train_trial(data, trial_dir, float(lambda_dgate), args, device, int(args.seed) + trial_id * 1009)
        trial_rows.append(row)
        head_df = pd.read_csv(row["head_groups_csv"])
        for threshold, sub in head_df.groupby("threshold_ratio"):
            deduped = _dedupe_groups(sub, trial_name, float(args.target_r2))
            deduped["lambda_dgate"] = float(lambda_dgate)
            deduped["threshold_ratio"] = float(threshold)
            all_group_frames.append(deduped)
        pd.DataFrame(trial_rows).to_csv(sweep_dir / "sweep_summary.csv", index=False, encoding="utf-8-sig")

    all_groups = pd.concat(all_group_frames, ignore_index=True)
    all_groups["global_group_key"] = all_groups["candidate_group"]
    all_groups = all_groups.sort_values(["group_size", "mean_reconstructed_y_val_r2"], ascending=[True, False])
    all_groups.to_csv(groups_dir / "all_trial_candidate_groups.csv", index=False, encoding="utf-8-sig")

    probe_candidates = all_groups.drop_duplicates("global_group_key").head(24).copy()
    probe_candidates["group_id"] = [f"group_{i + 1:03d}" for i in range(len(probe_candidates))]
    probe_candidates.to_csv(groups_dir / "candidate_groups.csv", index=False, encoding="utf-8-sig")

    validation_df = _run_group_validation(data=data, groups_df=probe_candidates, cfg=cfg, args=args, out_dir=validation_dir)
    merged = probe_candidates.merge(validation_df, on=["group_id", "candidate_group", "group_size"], how="left", suffixes=("", "_validation"))
    merged = merged.sort_values(["best_test_r2", "group_size"], ascending=[False, True])
    merged.to_csv(groups_dir / "candidate_groups_with_validation.csv", index=False, encoding="utf-8-sig")

    valid = merged[merged["best_test_r2"] >= float(args.target_r2)]
    if valid.empty:
        chosen = merged.iloc[0]
    else:
        chosen = valid.sort_values(["group_size", "best_test_r2"], ascending=[True, False]).iloc[0]
    best_trial = {
        "trial": chosen["trial"],
        "lambda_dgate": float(chosen["lambda_dgate"]),
        "threshold_ratio": float(chosen["threshold_ratio"]),
        "unique_group_count": int(len(probe_candidates)),
        "target_r2": float(args.target_r2),
    }

    _plot_validation(merged, float(args.target_r2), groups_dir / "candidate_group_validation.png")
    save_json(
        groups_dir / "selected_candidate_group.json",
        {
            "group_id": chosen["group_id"],
            "candidate_group": [item for item in str(chosen["candidate_group"]).split(";") if item],
            "group_size": int(chosen["group_size"]),
            "source_anchors": str(chosen["source_anchors"]),
            "best_test_r2": float(chosen["best_test_r2"]),
            "best_test_mse": float(chosen["best_test_mse"]),
            "target_r2": float(args.target_r2),
            "meets_target": bool(float(chosen["best_test_r2"]) >= float(args.target_r2)),
        },
    )
    save_json(
        run_dir / "stage05_candidate_group_interface.json",
        {
            "schema_version": 1,
            "stage": "stage05_candidate_groups",
            "center": data.center,
            "target": data.center,
            "xi": data.xi,
            "source_stage04_interface": str(Path(args.stage04_interface).resolve()),
            "run_dir": str(run_dir.resolve()),
            "ci_features": data.ci_features,
            "active_candidate_features": data.active_features,
            "selected_candidate_group_json": str((groups_dir / "selected_candidate_group.json").resolve()),
            "candidate_groups_csv": str((groups_dir / "candidate_groups_with_validation.csv").resolve()),
            "group_validation_csv": str((validation_dir / "group_validation.csv").resolve()),
            "sweep_summary_csv": str((sweep_dir / "sweep_summary.csv").resolve()),
            "selected_candidate_group": [item for item in str(chosen["candidate_group"]).split(";") if item],
            "selected_group_test_r2": float(chosen["best_test_r2"]),
            "selected_group_size": int(chosen["group_size"]),
            "target_r2": float(args.target_r2),
            "meets_target": bool(float(chosen["best_test_r2"]) >= float(args.target_r2)),
        },
    )
    _write_report(run_dir, data, best_trial, merged)
    print(f"Stage 05 run saved to {run_dir}")


if __name__ == "__main__":
    main()
