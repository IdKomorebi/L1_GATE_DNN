from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_center_spec, resolve_project_path
from src.data_utils import (
    center_output_dir,
    ensure_dir,
    normalize_column_list,
    prepare_supervised_dataset,
    read_numeric_csv,
    safe_name,
    save_json,
    write_name_mapping,
)
from src.models import make_mlp


MODEL_NAME = "MinimalSubstitutionHardGateDNN"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_device(name: str) -> torch.device:
    text = str(name or "auto").lower()
    if text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Requested CUDA, but torch.cuda.is_available() is false.")
    return torch.device(text)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if float(ss_tot.detach().cpu()) <= 0:
        return 0.0
    return float((1.0 - ss_res / ss_tot).detach().cpu())


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.net = make_mlp(in_dim, hidden_dims, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HardGateRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        candidate_idx: Sequence[int],
        fixed_idx: Sequence[int] | None = None,
        dropout: float = 0.0,
        gate_init: float = 0.0,
        gate_init_noise: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.candidate_idx = torch.as_tensor(list(candidate_idx), dtype=torch.long)
        self.fixed_idx = torch.as_tensor(list(fixed_idx or []), dtype=torch.long)
        init = torch.full((len(self.candidate_idx),), float(gate_init), dtype=torch.float32)
        if gate_init_noise > 0 and len(init) > 0:
            init = init + torch.randn_like(init) * float(gate_init_noise)
        self.logits = nn.Parameter(init)
        self.net = make_mlp(in_dim, hidden_dims, dropout=dropout)

    def soft_candidate_gates(self, temperature: float = 1.0) -> torch.Tensor:
        temperature = max(float(temperature), 1e-6)
        return torch.sigmoid(self.logits / temperature)

    def mask(self, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
        device = self.logits.device
        mask = torch.zeros(self.in_dim, device=device)
        if len(self.fixed_idx) > 0:
            mask[self.fixed_idx.to(device)] = 1.0
        soft = self.soft_candidate_gates(temperature)
        if hard:
            hard_gate = (soft >= 0.5).to(soft.dtype)
            candidate_gate = hard_gate.detach() - soft.detach() + soft
        else:
            candidate_gate = soft
        if len(self.candidate_idx) > 0:
            mask[self.candidate_idx.to(device)] = candidate_gate
        return mask

    def forward(self, x: torch.Tensor, temperature: float = 1.0, hard: bool = True) -> torch.Tensor:
        return self.net(x * self.mask(temperature=temperature, hard=hard))


@dataclass
class FitResult:
    mse: float
    r2: float
    best_epoch: int
    log: pd.DataFrame
    predictions: pd.DataFrame


def _subset_fit_worker(payload: Dict[str, Any]) -> tuple[tuple[str, ...], FitResult]:
    key = tuple(payload["features"])
    bundle = payload["bundle"]
    idx = [bundle.features.index(f) for f in key]
    device = torch.device(payload["device"])
    tag_text = str(payload["tag"]) + "|" + "|".join(key)
    tag_hash = zlib.adler32(tag_text.encode("utf-8")) % 100000
    seed = int(payload["seed_base"]) + tag_hash
    fit = train_plain_mlp(
        X_train=bundle.X_train[:, idx] if idx else bundle.X_train[:, :0],
        y_train=bundle.y_train,
        X_test=bundle.X_test[:, idx] if idx else bundle.X_test[:, :0],
        y_test=bundle.y_test,
        hidden_dims=payload["hidden_dims"],
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        lr=float(payload["lr"]),
        device=device,
        seed=seed,
        dropout=float(payload["dropout"]),
        feature_count=len(idx),
        patience=int(payload["patience"]),
        min_delta=float(payload["min_delta"]),
    )
    return key, fit


class SubsetEvaluator:
    def __init__(
        self,
        bundle: Any,
        hidden_dims: Sequence[int],
        epochs: int,
        batch_size: int,
        lr: float,
        device: torch.device,
        dropout: float,
        seed_base: int,
        patience: int,
        min_delta: float,
        parallel_devices: Sequence[str] | None = None,
        max_workers: int = 1,
    ) -> None:
        self.bundle = bundle
        self.hidden_dims = list(hidden_dims)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.device = device
        self.dropout = float(dropout)
        self.seed_base = int(seed_base)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.parallel_devices = [str(v) for v in (parallel_devices or [str(device)])]
        self.max_workers = max(1, int(max_workers))
        self.cache: Dict[tuple[str, ...], FitResult] = {}

    def fit(self, features: Sequence[str], tag: str = "") -> FitResult:
        return self.fit_many([(features, tag)])[tuple(features)]

    def _fit_uncached(self, features: Sequence[str], tag: str, device: torch.device) -> FitResult:
        key = tuple(features)
        tag_text = str(tag) + "|" + "|".join(key)
        tag_hash = zlib.adler32(tag_text.encode("utf-8")) % 100000
        seed = self.seed_base + tag_hash
        idx = [self.bundle.features.index(f) for f in features]
        return train_plain_mlp(
            X_train=self.bundle.X_train[:, idx] if idx else self.bundle.X_train[:, :0],
            y_train=self.bundle.y_train,
            X_test=self.bundle.X_test[:, idx] if idx else self.bundle.X_test[:, :0],
            y_test=self.bundle.y_test,
            hidden_dims=self.hidden_dims,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=device,
            seed=seed,
            dropout=self.dropout,
            feature_count=len(idx),
            patience=self.patience,
            min_delta=self.min_delta,
        )

    def fit_many(self, jobs: Sequence[tuple[Sequence[str], str]]) -> Dict[tuple[str, ...], FitResult]:
        results: Dict[tuple[str, ...], FitResult] = {}
        missing: List[tuple[tuple[str, ...], str]] = []
        seen_missing: set[tuple[str, ...]] = set()
        for features, tag in jobs:
            key = tuple(features)
            if key in self.cache:
                results[key] = self.cache[key]
            elif key not in seen_missing:
                seen_missing.add(key)
                missing.append((key, tag))

        if not missing:
            return results

        if self.max_workers <= 1 or len(missing) == 1:
            for job_index, (key, tag) in enumerate(missing):
                key_out, fit = _subset_fit_worker(self._worker_payload(job_index, key, tag))
                self.cache[key_out] = fit
                results[key_out] = fit
            return results

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(self.max_workers, len(missing)), mp_context=ctx) as executor:
            futures = [
                executor.submit(_subset_fit_worker, self._worker_payload(i, key, tag))
                for i, (key, tag) in enumerate(missing)
            ]
            for fut in as_completed(futures):
                key_out, fit = fut.result()
                self.cache[key_out] = fit
                results[key_out] = fit
        return results

    def _worker_payload(self, job_index: int, key: tuple[str, ...], tag: str) -> Dict[str, Any]:
        return {
            "bundle": self.bundle,
            "features": list(key),
            "tag": tag,
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "device": self.parallel_devices[job_index % len(self.parallel_devices)],
            "dropout": self.dropout,
            "seed_base": self.seed_base,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }

    def fit_many_ordered(self, jobs: Sequence[tuple[Sequence[str], str]]) -> List[FitResult]:
        result_map = self.fit_many(jobs)
        return [result_map[tuple(features)] for features, _ in jobs]

    def fit_legacy_serial(self, features: Sequence[str], tag: str = "") -> FitResult:
        key = tuple(features)
        if key not in self.cache:
            tag_hash = zlib.adler32(str(tag).encode("utf-8")) % 1000
            seed = self.seed_base + len(self.cache) * 17 + tag_hash
            idx = [self.bundle.features.index(f) for f in features]
            self.cache[key] = train_plain_mlp(
                X_train=self.bundle.X_train[:, idx] if idx else self.bundle.X_train[:, :0],
                y_train=self.bundle.y_train,
                X_test=self.bundle.X_test[:, idx] if idx else self.bundle.X_test[:, :0],
                y_test=self.bundle.y_test,
                hidden_dims=self.hidden_dims,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
                seed=seed,
                dropout=self.dropout,
                feature_count=len(idx),
                patience=self.patience,
                min_delta=self.min_delta,
            )
        return self.cache[key]


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_rows: List[torch.Tensor] = []
    p_rows: List[torch.Tensor] = []
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss_sum += float(nn.functional.mse_loss(pred, yb).detach().cpu()) * xb.size(0)
            y_rows.append(yb.detach().cpu())
            p_rows.append(pred.detach().cpu())
    y_cat = torch.cat(y_rows, dim=0)
    p_cat = torch.cat(p_rows, dim=0)
    return {"mse": loss_sum / len(loader.dataset), "r2": r2_score_torch(y_cat, p_cat)}


def eval_hard_gate_model(
    model: HardGateRegressor,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    hard: bool = True,
) -> Dict[str, float]:
    model.eval()
    y_rows: List[torch.Tensor] = []
    p_rows: List[torch.Tensor] = []
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb, temperature=temperature, hard=hard)
            loss_sum += float(nn.functional.mse_loss(pred, yb).detach().cpu()) * xb.size(0)
            y_rows.append(yb.detach().cpu())
            p_rows.append(pred.detach().cpu())
    y_cat = torch.cat(y_rows, dim=0)
    p_cat = torch.cat(p_rows, dim=0)
    return {"mse": loss_sum / len(loader.dataset), "r2": r2_score_torch(y_cat, p_cat)}


def predict_model(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int) -> pd.DataFrame:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    preds: List[np.ndarray] = []
    actuals: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
            actuals.append(yb.numpy().reshape(-1))
    return pd.DataFrame({"y_true_norm": np.concatenate(actuals), "y_pred_norm": np.concatenate(preds)})


def train_plain_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dims: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    dropout: float = 0.0,
    feature_count: int | None = None,
    patience: int = 0,
    min_delta: float = 0.0,
) -> FitResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    feature_count = int(X_train.shape[1] if feature_count is None else feature_count)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )
    model = MLPRegressor(feature_count, hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows: List[Dict[str, float]] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best = {"mse": math.inf, "r2": -math.inf, "epoch": 0}
    stale_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

        train_metrics = eval_model(model, train_loader, device)
        test_metrics = eval_model(model, test_loader, device)
        rows.append(
            {
                "epoch": epoch,
                "train_mse": train_metrics["mse"],
                "train_r2": train_metrics["r2"],
                "test_mse": test_metrics["mse"],
                "test_r2": test_metrics["r2"],
            }
        )
        if test_metrics["r2"] > best["r2"] + float(min_delta):
            best = {"mse": float(test_metrics["mse"]), "r2": float(test_metrics["r2"]), "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if int(patience) > 0 and stale_epochs >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    predictions = predict_model(model, X_test, y_test, device, batch_size)
    return FitResult(
        mse=float(best["mse"]),
        r2=float(best["r2"]),
        best_epoch=int(best["epoch"]),
        log=pd.DataFrame(rows),
        predictions=predictions,
    )


def train_hard_gate(
    bundle: Any,
    candidate_features: Sequence[str],
    fixed_features: Sequence[str],
    hidden_dims: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    gate_lr: float,
    lambda_sparse: float,
    lambda_budget: float,
    max_features: int,
    device: torch.device,
    seed: int,
    dropout: float,
    gate_init: float,
    gate_init_noise: float,
    temp_start: float,
    temp_end: float,
    binary_beta: float,
    experiment: str,
) -> tuple[HardGateRegressor, pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    fixed_idx = [bundle.features.index(f) for f in fixed_features]
    candidate_idx = [bundle.features.index(f) for f in candidate_features]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.X_train), torch.from_numpy(bundle.y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.X_test), torch.from_numpy(bundle.y_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    model = HardGateRegressor(
        in_dim=len(bundle.features),
        hidden_dims=hidden_dims,
        fixed_idx=fixed_idx,
        candidate_idx=candidate_idx,
        dropout=dropout,
        gate_init=gate_init,
        gate_init_noise=gate_init_noise,
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.net.parameters(), "lr": lr},
            {"params": [model.logits], "lr": gate_lr},
        ]
    )

    rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_score = -math.inf

    for epoch in range(1, int(epochs) + 1):
        if epochs <= 1:
            temperature = temp_end
        else:
            progress = (epoch - 1) / max(1, epochs - 1)
            temperature = temp_start * (temp_end / temp_start) ** progress

        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb, temperature=temperature, hard=True)
            mse = nn.functional.mse_loss(pred, yb)
            soft_gate = model.soft_candidate_gates(temperature=temperature)
            sparse = torch.sum(soft_gate)
            if int(max_features) > 0:
                budget = torch.relu(sparse - float(max_features)) ** 2
            else:
                budget = torch.zeros((), device=device)
            binary = torch.mean(soft_gate * (1.0 - soft_gate)) if len(soft_gate) else torch.zeros((), device=device)
            loss = mse + float(lambda_sparse) * sparse + float(lambda_budget) * budget + float(binary_beta) * binary
            loss.backward()
            optimizer.step()

        train_metrics = eval_hard_gate_model(model, train_loader, device, temperature=temperature, hard=True)
        test_metrics = eval_hard_gate_model(model, test_loader, device, temperature=temperature, hard=True)
        soft_np = model.soft_candidate_gates(temperature=temperature).detach().cpu().numpy()
        active_count = int(np.sum(soft_np >= 0.5))
        rows.append(
            {
                "epoch": epoch,
                "temperature": temperature,
                "train_mse": train_metrics["mse"],
                "train_r2": train_metrics["r2"],
                "test_mse": test_metrics["mse"],
                "test_r2": test_metrics["r2"],
                "candidate_active_0p5": active_count,
                "candidate_gate_sum": float(np.sum(soft_np)),
                "budget_max_features": int(max_features),
            }
        )
        for feature, value in zip(candidate_features, soft_np):
            gate_rows.append(
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "feature": feature,
                    "role": "candidate",
                    "gate_value": float(value),
                    "selected_0p5": bool(value >= 0.5),
                }
            )
        for feature in fixed_features:
            gate_rows.append(
                {
                    "experiment": experiment,
                    "epoch": epoch,
                    "feature": feature,
                    "role": "fixed_context",
                    "gate_value": 1.0,
                    "selected_0p5": True,
                }
            )

        score = float(test_metrics["r2"] - 0.0005 * active_count)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), pd.DataFrame(gate_rows)


def selected_by_gate(
    gate_values: pd.DataFrame,
    candidate_features: Sequence[str],
    threshold: float,
    min_count: int = 1,
) -> List[str]:
    final_epoch = int(gate_values["epoch"].max())
    final = gate_values[(gate_values["epoch"] == final_epoch) & (gate_values["role"] == "candidate")].copy()
    value_by_feature = dict(zip(final["feature"], final["gate_value"]))
    selected = [f for f in candidate_features if float(value_by_feature.get(f, 0.0)) >= threshold]
    if len(selected) < min_count:
        ranked = sorted(candidate_features, key=lambda f: float(value_by_feature.get(f, 0.0)), reverse=True)
        selected = ranked[:min(min_count, len(ranked))]
    return selected


def ranked_by_final_gate(gate_values: pd.DataFrame, candidate_features: Sequence[str]) -> List[str]:
    final_epoch = int(gate_values["epoch"].max())
    final = gate_values[(gate_values["epoch"] == final_epoch) & (gate_values["role"] == "candidate")].copy()
    value_by_feature = dict(zip(final["feature"], final["gate_value"]))
    return sorted(candidate_features, key=lambda f: float(value_by_feature.get(f, 0.0)), reverse=True)


def choose_compact_path_seed(
    gate_values: pd.DataFrame,
    candidate_features: Sequence[str],
    evaluator: SubsetEvaluator,
    tau: float,
    sizes: Sequence[int],
) -> tuple[List[str], FitResult, pd.DataFrame, str]:
    ranked = ranked_by_final_gate(gate_values, candidate_features)
    clean_sizes = sorted({max(1, min(int(size), len(ranked))) for size in sizes if int(size) > 0})
    jobs = [(ranked[:size], f"compact_main_top_{size}") for size in clean_sizes]
    fits = evaluator.fit_many_ordered(jobs)
    rows: List[Dict[str, Any]] = []
    for size, (features_for_size, _), fit in zip(clean_sizes, jobs, fits):
        rows.append(
            {
                "candidate_size": size,
                "r2": fit.r2,
                "mse": fit.mse,
                "tau": tau,
                "meets_tau": bool(fit.r2 >= tau),
                "features": feature_list_text(features_for_size),
            }
        )
    df = pd.DataFrame(rows)
    meets = df[df["meets_tau"] == True]
    if not meets.empty:
        chosen_size = int(meets.sort_values(["candidate_size", "r2"], ascending=[True, False]).iloc[0]["candidate_size"])
        status = "compact_meets_tau"
    else:
        chosen_size = int(df.sort_values(["r2", "candidate_size"], ascending=[False, True]).iloc[0]["candidate_size"])
        status = "below_tau_compact_choice"
    chosen_features = ranked[:chosen_size]
    chosen_fit = fits[clean_sizes.index(chosen_size)]
    return chosen_features, chosen_fit, df, status


def expand_until_tau(
    current: Sequence[str],
    candidate_pool: Sequence[str],
    gate_values: pd.DataFrame,
    evaluator: SubsetEvaluator,
    tau: float,
    protected: Sequence[str] | None = None,
    expand_step: int = 1,
) -> tuple[List[str], FitResult, List[Dict[str, Any]]]:
    selected = list(dict.fromkeys(current))
    protected_set = set(protected or [])
    fit = evaluator.fit(selected, tag="expand_start")
    rows: List[Dict[str, Any]] = []
    if fit.r2 >= tau:
        return selected, fit, rows

    final_epoch = int(gate_values["epoch"].max())
    final = gate_values[(gate_values["epoch"] == final_epoch) & (gate_values["role"] == "candidate")].copy()
    value_by_feature = dict(zip(final["feature"], final["gate_value"]))
    ranked = sorted(candidate_pool, key=lambda f: float(value_by_feature.get(f, 0.0)), reverse=True)
    expand_step = max(1, int(expand_step))
    pending: List[str] = []
    for feature in ranked:
        if feature not in selected:
            pending.append(feature)
    prefix_jobs: List[tuple[List[str], str]] = []
    prefix_added: List[List[str]] = []
    for start in range(0, len(pending), expand_step):
        added = pending[start : start + expand_step]
        prefix_added.append(added)
        prefix_jobs.append((list(dict.fromkeys([*selected, *pending[: start + expand_step]])), f"expand_add_{'_'.join(added)}"))

    prefix_fits = evaluator.fit_many_ordered(prefix_jobs)
    for (prefix_features, _), added, fit in zip(prefix_jobs, prefix_added, prefix_fits):
        selected = list(prefix_features)
        print(
            f"  expand_until_tau: add={feature_list_text(added)}, count={len(selected)}, R2={fit.r2:.6f}, tau={tau:.6f}",
            flush=True,
        )
        rows.append(
            {
                "action": "add_until_tau",
                "feature": feature_list_text(added),
                "feature_count": len(selected),
                "r2": fit.r2,
                "mse": fit.mse,
                "reached_tau": bool(fit.r2 >= tau),
            }
        )
        if fit.r2 >= tau:
            break
    if fit.r2 < tau:
        missing = [f for f in candidate_pool if f not in selected and f not in protected_set]
        for feature in missing:
            selected.append(feature)
        fit = evaluator.fit(selected, tag="expand_all")
    return selected, fit, rows


def prune_minimal_path(
    path: Sequence[str],
    evaluator: SubsetEvaluator,
    tau: float,
    tag: str,
    locked_keep: Sequence[str] | None = None,
) -> tuple[List[str], FitResult, pd.DataFrame]:
    current = list(dict.fromkeys(path))
    locked = set(locked_keep or [])
    rows: List[Dict[str, Any]] = []
    changed = True
    pass_id = 0

    while changed:
        changed = False
        pass_id += 1
        features_to_try = [feature for feature in current if feature not in locked]
        jobs = [([f for f in current if f != feature], f"{tag}_drop_{pass_id}_{feature}") for feature in features_to_try]
        fits = evaluator.fit_many_ordered(jobs) if jobs else []
        candidates_to_drop: List[tuple[str, FitResult]] = []
        for feature, fit in zip(features_to_try, fits):
            trial = [f for f in current if f != feature]
            can_drop = bool(fit.r2 >= tau)
            print(
                f"  prune[{tag}]: try_drop={feature}, trial_count={len(trial)}, R2={fit.r2:.6f}, drop={can_drop}",
                flush=True,
            )
            rows.append(
                {
                    "stage": tag,
                    "pass": pass_id,
                    "feature": feature,
                    "trial_feature_count": len(trial),
                    "trial_r2": fit.r2,
                    "trial_mse": fit.mse,
                    "tau": tau,
                    "dropped": can_drop,
                    "reason": "redundant_under_tau_rule" if can_drop else "required_under_tau_rule",
                }
            )
            if can_drop:
                candidates_to_drop.append((feature, fit))
        if candidates_to_drop:
            drop_feature, drop_fit = max(candidates_to_drop, key=lambda item: item[1].r2)
            current = [f for f in current if f != drop_feature]
            changed = True
            print(
                f"  prune[{tag}]: dropped_best={drop_feature}, retained_count={len(current)}, trial_R2={drop_fit.r2:.6f}",
                flush=True,
            )

    final_fit = evaluator.fit(current, tag=f"{tag}_final")
    final_rows: List[Dict[str, Any]] = []
    final_jobs = [([f for f in current if f != feature], f"{tag}_final_without_{feature}") for feature in current]
    final_fits = evaluator.fit_many_ordered(final_jobs) if final_jobs else []
    for feature, fit in zip(current, final_fits):
        trial = [f for f in current if f != feature]
        final_rows.append(
            {
                "stage": f"{tag}_final_required_check",
                "pass": pass_id + 1,
                "feature": feature,
                "trial_feature_count": len(trial),
                "trial_r2": fit.r2,
                "trial_mse": fit.mse,
                "tau": tau,
                "dropped": False,
                "reason": "required_final_path",
                "final_path_r2": final_fit.r2,
                "final_path_mse": final_fit.mse,
                "delta_r2_if_removed": final_fit.r2 - fit.r2,
                "delta_mse_if_removed": fit.mse - final_fit.mse,
            }
        )
    return current, final_fit, pd.DataFrame([*rows, *final_rows])


def feature_list_text(values: Sequence[str]) -> str:
    return ";".join(values)


def plot_training_curve(log_df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(log_df["epoch"], log_df["test_mse"], label="test")
    if "train_mse" in log_df:
        axes[0].plot(log_df["epoch"], log_df["train_mse"], label="train", alpha=0.75)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(log_df["epoch"], log_df["test_r2"], label="test")
    if "train_r2" in log_df:
        axes[1].plot(log_df["epoch"], log_df["train_r2"], label="train", alpha=0.75)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R2")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gate_bar(gate_df: pd.DataFrame, output_path: Path, title: str, top_n: int = 30) -> None:
    final = gate_df[(gate_df["epoch"] == gate_df["epoch"].max()) & (gate_df["role"] == "candidate")].copy()
    final = final.sort_values("gate_value", ascending=False).head(top_n)
    y = np.arange(len(final))
    fig, ax = plt.subplots(figsize=(11, max(5.0, 0.28 * len(final) + 1.5)))
    ax.barh(y, final["gate_value"], color="#2f5597")
    ax.set_yticks(y)
    ax.set_yticklabels(final["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#b03030", linestyle="--", linewidth=1.2, label="0.5")
    ax.set_xlabel("Final soft gate")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_main_ablation(main_df: pd.DataFrame, output_path: Path) -> None:
    final = main_df[main_df["stage"] == "main_final_required_check"].copy()
    if final.empty:
        return
    final = final.sort_values("delta_r2_if_removed", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.34 * len(final) + 1.2)))
    ax.barh(final["feature"], final["delta_r2_if_removed"], color="#8064a2")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Final path R2 - R2 after removing feature")
    ax.set_title("Main path required-field impact")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_residual_summary(residual_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = residual_df["case"].tolist()
    values = residual_df["r2"].tolist()
    colors = ["#70ad47" if bool(v) else "#c0504d" for v in residual_df["meets_tau"].tolist()]
    ax.bar(labels, values, color=colors)
    tau = float(residual_df["tau"].iloc[0])
    ax.axhline(tau, color="black", linestyle="--", linewidth=1.2, label=f"tau={tau:.4f}")
    ax.set_ylabel("R2")
    ax.set_title("Residual / independent path tests")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_substitution_summary(sub_df: pd.DataFrame, output_dir: Path, tau: float) -> None:
    if sub_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5.0, 0.32 * len(sub_df) + 1.2)))
    order = sub_df.sort_values("context_plus_R_r2", ascending=True)
    ax.barh(order["removed_feature"], order["context_r2"], label="context", alpha=0.75)
    ax.barh(order["removed_feature"], order["context_plus_R_r2"], label="context + R", alpha=0.65)
    ax.axvline(tau, color="black", linestyle="--", linewidth=1.2, label=f"tau={tau:.4f}")
    ax.set_xlabel("R2")
    ax.set_title("Substitution feasibility by removed main-path field")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "substitution_r2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    repl = sub_df.copy()
    repl["replacement_count"] = repl["replacement_fields"].fillna("").map(lambda x: 0 if not x else len(str(x).split(";")))
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.30 * len(repl) + 1.2)))
    ax.barh(repl["removed_feature"], repl["replacement_count"], color="#4f81bd")
    ax.set_xlabel("Replacement field count")
    ax.set_title("Minimal replacement size")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "substitution_replacement_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gate_heatmap(gate_df: pd.DataFrame, output_path: Path, top_n: int = 35) -> None:
    final = gate_df[gate_df["epoch"] == gate_df["epoch"].max()].copy()
    candidates = final[final["role"] == "candidate"].copy()
    if candidates.empty:
        return
    pivot = candidates.pivot_table(index="feature", columns="experiment", values="gate_value", aggfunc="max", fill_value=0.0)
    pivot["max_gate"] = pivot.max(axis=1)
    pivot = pivot.sort_values("max_gate", ascending=False).drop(columns=["max_gate"]).head(top_n)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.48 * len(pivot.columns) + 5.0), max(6.0, 0.28 * len(pivot) + 1.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Final gate values across hard-gate experiments")
    fig.colorbar(im, ax=ax, label="gate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_dgating_effective_path(source_run_dir: Path, threshold: float) -> List[str]:
    path = source_run_dir / "dgate_effective_group_norms.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing D-Gating effective norm file: {path}")
    df = pd.read_csv(path)
    final_epoch = int(df["epoch"].max())
    final = df[df["epoch"] == final_epoch].copy()
    selected = final[final["effective_group_l2"].astype(float) > float(threshold)].copy()
    selected = selected.sort_values("effective_group_l2", ascending=False)
    return [str(v) for v in selected["feature"].tolist()]


def load_selected_features_json(path: Path) -> List[str]:
    if not path.exists():
        return []
    data = load_json(path)
    return [str(row["name"]) for row in data.get("features", []) if row.get("name")]


def initial_candidate_sets(source_run_dir: Path, effective_threshold: float) -> Dict[str, List[str]]:
    candidates: Dict[str, List[str]] = {}
    effective = load_dgating_effective_path(source_run_dir, effective_threshold)
    if effective:
        candidates[f"dgate_effective_gt_{effective_threshold:g}"] = effective
    best = load_selected_features_json(source_run_dir / "best_epoch_selected_features.json")
    if best:
        candidates["dgate_best_epoch_gate_selected"] = best
    final = load_selected_features_json(source_run_dir / "selected_features.json")
    if final:
        candidates["dgate_final_gate_selected"] = final
    return candidates


def parse_feature_sequence(value: str | None) -> List[str]:
    if not value:
        return []
    path = Path(value).expanduser()
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = str(value)
    raw = text.replace("\n", ";").replace(",", ";").split(";")
    return [item.strip() for item in raw if item.strip()]


def resolve_data_path(cfg: Dict[str, Any], run_cfg: Dict[str, Any] | None, args: argparse.Namespace) -> Path:
    if args.csv:
        return Path(args.csv).expanduser().resolve()
    if run_cfg and run_cfg.get("data_path"):
        candidate = Path(str(run_cfg["data_path"])).expanduser()
        if candidate.exists():
            return candidate.resolve()
    value = cfg.get("dataset", {}).get("processed_csv")
    if not value:
        raise ValueError("No CSV path was provided and config.dataset.processed_csv is empty.")
    return resolve_project_path(cfg, value).expanduser().resolve()


def load_source_run_config(source_run: str | None) -> tuple[Dict[str, Any] | None, Path | None]:
    if not source_run:
        return None, None
    path = Path(source_run).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if path.is_dir():
        cfg_path = path / "config.json"
        return (load_json(cfg_path), path) if cfg_path.exists() else (None, path)
    if path.exists():
        return load_json(path), path.parent
    return None, path


def choose_features(
    data_path: Path,
    target_col: str,
    run_cfg: Dict[str, Any] | None,
    exclude_columns: Sequence[str],
    drop_all_zero_columns: bool,
    use_all_features: bool,
) -> List[str]:
    if not use_all_features and run_cfg and run_cfg.get("features"):
        return [str(v) for v in run_cfg["features"]]
    df = read_numeric_csv(data_path, drop_all_zero_columns=drop_all_zero_columns, exclude_columns=exclude_columns)
    return [str(c) for c in df.columns if str(c) != target_col]


def write_readme(
    output_dir: Path,
    metrics: Dict[str, Any],
    main_path: Sequence[str],
    residual_row: Dict[str, Any],
    substitution_df: pd.DataFrame,
) -> None:
    replaceable_count = int(substitution_df["is_replaceable"].sum()) if not substitution_df.empty else 0
    lines = [
        "# Minimal Substitution Hard Gate DNN",
        "",
        "## Design",
        "",
        "This run first trains a full-field MLP baseline, then sets `tau = 0.95 * R2_full`.",
        "A straight-through binary gate searches for a sparse main inference path. The selected path is then retrained and greedily ablated until every remaining field is required to keep `R2 >= tau`.",
        "The residual feature set is tested as a completely disjoint path. Finally, each main-path field is removed and the residual feature set is tested as a conditional replacement pool.",
        "",
        "## Key Results",
        "",
        f"- Full-field R2: {metrics['r2_full']:.6f}",
        f"- Full-field MSE: {metrics['mse_full']:.6f}",
        f"- Tau: {metrics['tau']:.6f}",
        f"- Main path fields: {len(main_path)}",
        f"- Residual-only R2: {residual_row.get('r2', float('nan')):.6f}",
        f"- Residual reaches tau: {bool(residual_row.get('meets_tau', False))}",
        f"- Replaceable main-path fields: {replaceable_count}/{len(substitution_df)}",
        "",
        "## Main Path",
        "",
    ]
    for feature in main_path:
        lines.append(f"- {feature}")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `full_metrics.csv`: full-field baseline metrics and tau.",
            "- `main_path.csv`: main path selection, pruning, and final removal impact.",
            "- `residual_test.csv`: residual-only and disjoint independent-path checks.",
            "- `substitution_results.csv`: conditional replacement tests for each main-path field.",
            "- `gate_values.csv`: final and historical gate values for the main path and substitution gates.",
            "- `figures/`: training curves, gate bars, ablation impact, residual and substitution summaries.",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_txt(
    output_dir: Path,
    metrics: Dict[str, Any],
    main_path: Sequence[str],
    residual_df: pd.DataFrame,
    substitution_df: pd.DataFrame,
    necessity_df: pd.DataFrame,
) -> None:
    residual_row = residual_df[residual_df["case"] == "R_only"].iloc[0].to_dict()
    substitutable = substitution_df[substitution_df["is_replaceable"] == True] if not substitution_df.empty else pd.DataFrame()
    non_sub = substitution_df[substitution_df["status"] == "non_substitutable"] if not substitution_df.empty else pd.DataFrame()
    redundant = substitution_df[substitution_df["status"] == "redundant_in_path"] if not substitution_df.empty else pd.DataFrame()
    top_nec = necessity_df.sort_values("Nec_j", ascending=False).head(12) if not necessity_df.empty else pd.DataFrame()

    lines = [
        "MinimalSubstitutionHardGateDNN summary",
        "",
        f"Full-field baseline: R2_full={metrics['r2_full']:.6f}, MSE_full={metrics['mse_full']:.6f}, tau={metrics['tau']:.6f}.",
        f"Main path P has {len(main_path)} fields and reaches R2={metrics['main_path_r2']:.6f}, MSE={metrics['main_path_mse']:.6f}, meets_tau={bool(metrics.get('main_path_meets_tau', False))}.",
        f"Main path compact selection status: {metrics.get('main_compact_status', 'unknown')}.",
        "Main path fields:",
        *[f"  - {feature}" for feature in main_path],
        "",
        f"Residual-only test R=U-P: feature_count={int(residual_row['feature_count'])}, R2={float(residual_row['r2']):.6f}, MSE={float(residual_row['mse']):.6f}, reaches_tau={bool(residual_row['meets_tau'])}.",
    ]
    if metrics.get("independent_paths"):
        lines.append("Disjoint paths found inside R:")
        for row in metrics["independent_paths"]:
            lines.append(f"  - P_disjoint_{row['path_id']}: count={row['feature_count']}, R2={row['r2']:.6f}, fields={row['features']}")
    else:
        lines.append("No fully disjoint minimal path was confirmed beyond the residual-only diagnostic.")

    lines.extend(["", "Conditional substitution:"])
    if redundant.empty:
        lines.append("  - No main-path field was redundant under the context-only tau test.")
    else:
        lines.append("  - Redundant in path:")
        for _, row in redundant.iterrows():
            lines.append(f"    * {row['removed_feature']}: context_R2={row['context_r2']:.6f}")

    if non_sub.empty:
        lines.append("  - No field was classified as non_substitutable.")
    else:
        lines.append("  - Non-substitutable fields:")
        for _, row in non_sub.iterrows():
            lines.append(f"    * {row['removed_feature']}: context+R R2={row['context_plus_R_r2']:.6f}")

    if substitutable.empty:
        lines.append("  - No substitutable field was confirmed.")
    else:
        lines.append("  - Substitutable fields and replacement groups:")
        for _, row in substitutable.iterrows():
            repl = row["replacement_fields"] if str(row["replacement_fields"]) else "(empty replacement; context alone reaches tau)"
            lines.append(
                f"    * {row['removed_feature']}: replacement_count={int(row['replacement_count'])}, "
                f"Pi_min_R2={float(row['replacement_path_r2']):.6f}, Qi={repl}"
            )

    lines.extend(["", "Top global necessity scores Nec_j = R2_full - R2(U - {xj}):"])
    for _, row in top_nec.iterrows():
        lines.append(f"  - {row['feature']}: Nec_j={row['Nec_j']:.6f}, R2_minus_j={row['R2_minus_j']:.6f}")
    lines.append("")
    lines.append("Note: Nec_j is a global necessity diagnostic, not a direct feature-importance score.")
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find minimal inference paths and conditional substitutes with straight-through hard gates.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument("--source-run", default=str(PROJECT_ROOT / "outputs" / "data2025_Processed_V2" / "CenterOn_net_actual_interchange_mw" / "L1GateDNN" / "run_20260603_l1_lr0p00065_thr0p10_combo5_L1GateDNN"))
    parser.add_argument("--csv", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--combo", default="5")
    parser.add_argument("--use-all-features", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--baseline-epochs", type=int, default=180)
    parser.add_argument("--gate-epochs", type=int, default=240)
    parser.add_argument("--retrain-epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)
    parser.add_argument("--expand-step", type=int, default=5, help="Number of gate-ranked fields to add per expand-until-tau validation.")
    parser.add_argument("--compact-main-sizes", nargs="+", type=int, default=[8, 10, 12, 14, 16, 18])
    parser.add_argument("--disable-compact-main", action="store_true", help="Use old expand-until-tau path seed instead of compact top-k seeds.")
    parser.add_argument("--run-substitution-when-main-below-tau", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--gate-lr", type=float, default=0.006)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32, 16])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-sparse", type=float, default=0.0025)
    parser.add_argument("--lambda-budget", type=float, default=0.002)
    parser.add_argument("--max-features", type=int, default=0, help="If >0, penalize hard-gate selections above this budget.")
    parser.add_argument("--substitution-lambda-sparse", type=float, default=0.003)
    parser.add_argument("--substitution-lambda-budget", type=float, default=0.002)
    parser.add_argument("--max-sub-features", type=int, default=0, help="If >0, penalize substitution gates above this budget.")
    parser.add_argument("--binary-beta", type=float, default=0.0005)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--gate-init", type=float, default=0.0)
    parser.add_argument("--gate-init-noise", type=float, default=0.02)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.25)
    parser.add_argument("--r2-ratio", type=float, default=0.95)
    parser.add_argument("--main-r2-threshold", type=float, default=None, help="Absolute R2 threshold for main-path pruning. Overrides --r2-ratio * R2_full.")
    parser.add_argument("--substitution-r2-threshold", type=float, default=None, help="Absolute R2 threshold for conditional replacement paths. Defaults to the main threshold.")
    parser.add_argument("--max-independent-paths", type=int, default=3)
    parser.add_argument("--use-dgating-initial-main", action="store_true", help="Skip main hard-gate search and use D-Gating effective-norm threshold as initial main path.")
    parser.add_argument("--dgating-effective-threshold", type=float, default=0.1)
    parser.add_argument("--fixed-main-path", default=None, help="Semicolon/comma/newline separated feature list, or a text file containing that list.")
    parser.add_argument("--skip-main-prune", action="store_true", help="Do not greedily prune the initial/fixed main path; only run final drop-one checks.")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parallel-devices", default=None, help="Comma-separated devices for parallel subset MLP fits. Defaults to all visible CUDA devices.")
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel subset-fit workers. Defaults to number of parallel devices.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_cfg, source_run_dir = load_source_run_config(args.source_run)
    combo_spec = resolve_center_spec(cfg, args.combo) if args.combo else None
    target_col = args.target_col or (run_cfg or {}).get("center") or (combo_spec or {}).get("center")
    if not target_col:
        raise ValueError("Please provide --target-col or --combo.")
    target_col = str(target_col)

    data_path = resolve_data_path(cfg, run_cfg, args)
    source_preprocessing = (run_cfg or {}).get("preprocessing") or cfg.get("preprocessing", {})
    drop_all_zero_columns = bool(source_preprocessing.get("drop_all_zero_columns", False))
    exclude_columns = normalize_column_list(source_preprocessing.get("exclude_columns"))
    if not exclude_columns and combo_spec:
        exclude_columns = normalize_column_list(combo_spec.get("exclude_columns"))
    train_ratio = float(args.train_ratio if args.train_ratio is not None else (run_cfg or {}).get("params", {}).get("train_ratio", cfg.get("training", {}).get("train_ratio", 0.8)))
    features = choose_features(data_path, target_col, run_cfg, exclude_columns, drop_all_zero_columns, args.use_all_features)
    if not features:
        raise ValueError("No candidate input features were found.")

    device = choose_device(args.device)
    if args.parallel_devices:
        parallel_devices = [item.strip() for item in str(args.parallel_devices).split(",") if item.strip()]
    elif torch.cuda.is_available() and str(device).startswith("cuda"):
        parallel_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        parallel_devices = [str(device)]
    num_workers = int(args.num_workers) if int(args.num_workers) > 0 else len(parallel_devices)
    bundle = prepare_supervised_dataset(
        data_path=data_path,
        center=target_col,
        features=features,
        train_ratio=train_ratio,
        random_state=args.random_state,
        drop_all_zero_columns=drop_all_zero_columns,
        exclude_columns=exclude_columns,
    )

    output_root = Path(args.output_root) if args.output_root else center_output_dir(resolve_project_path(cfg, cfg["dataset"]["output_root"]), target_col)
    model_root = ensure_dir(output_root / MODEL_NAME)
    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_combo{combo_spec['id'] if combo_spec else 'custom'}_{MODEL_NAME}"
    run_dir = model_root / safe_name(run_name)
    suffix = 2
    while run_dir.exists():
        run_dir = model_root / f"{safe_name(run_name)}_{suffix}"
        suffix += 1
    ensure_dir(run_dir)
    figure_dir = ensure_dir(run_dir / "figures")
    log_dir = ensure_dir(run_dir / "logs")

    save_json(
        run_dir / "config.json",
        {
            "model": MODEL_NAME,
            "target_col": target_col,
            "data_path": str(data_path),
            "source_run": str(source_run_dir) if source_run_dir else None,
            "combo": combo_spec,
            "features": features,
            "preprocessing": {"drop_all_zero_columns": drop_all_zero_columns, "exclude_columns": exclude_columns},
            "params": vars(args),
        },
    )
    write_name_mapping(run_dir / "name_mapping.csv", target_col, features)

    print(f"[1/5] Full-field MLP baseline: features={len(features)}, device={device}, parallel_subset_devices={parallel_devices}")
    evaluator = SubsetEvaluator(
        bundle=bundle,
        hidden_dims=args.hidden_dims,
        epochs=args.retrain_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        dropout=args.dropout,
        seed_base=args.random_state + 10000,
        patience=args.patience,
        min_delta=args.early_stop_min_delta,
        parallel_devices=parallel_devices,
        max_workers=num_workers,
    )
    full_fit = train_plain_mlp(
        bundle.X_train,
        bundle.y_train,
        bundle.X_test,
        bundle.y_test,
        hidden_dims=args.hidden_dims,
        epochs=args.baseline_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.random_state,
        dropout=args.dropout,
        feature_count=len(features),
        patience=args.patience,
        min_delta=args.early_stop_min_delta,
    )
    full_fit.log.to_csv(run_dir / "full_mlp_log.csv", index=False, encoding="utf-8-sig")
    full_fit.predictions.to_csv(run_dir / "full_predictions.csv", index=False, encoding="utf-8-sig")
    ratio_tau = float(args.r2_ratio * full_fit.r2)
    tau = float(args.main_r2_threshold) if args.main_r2_threshold is not None else ratio_tau
    substitution_tau = float(args.substitution_r2_threshold) if args.substitution_r2_threshold is not None else tau
    full_metrics_df = pd.DataFrame(
        [
            {
                "target_col": target_col,
                "feature_count": len(features),
                "r2_full": full_fit.r2,
                "mse_full": full_fit.mse,
                "best_epoch": full_fit.best_epoch,
                "r2_ratio": args.r2_ratio,
                "ratio_tau": ratio_tau,
                "tau": tau,
                "main_tau": tau,
                "substitution_tau": substitution_tau,
                "train_ratio": train_ratio,
                "random_state": args.random_state,
            }
        ]
    )
    full_metrics_df.to_csv(run_dir / "full_metrics.csv", index=False, encoding="utf-8-sig")
    plot_training_curve(full_fit.log, figure_dir / "full_mlp_training.png", "Full-field MLP baseline")

    main_expand_rows: List[Dict[str, Any]] = []
    substitution_gate_frames: List[pd.DataFrame] = []
    fixed_main_path = parse_feature_sequence(args.fixed_main_path)
    if fixed_main_path:
        missing_fixed = [feature for feature in fixed_main_path if feature not in features]
        if missing_fixed:
            raise ValueError(f"Fixed main path contains unknown features: {missing_fixed}")
        print(f"[2/5] Fixed main path: count={len(fixed_main_path)}, tau={tau:.6f}", flush=True)
        main_selected = list(dict.fromkeys(fixed_main_path))
        main_fit_before_prune = evaluator.fit(main_selected, tag="fixed_main_path")
        compact_status = "fixed_main_path"
        pd.DataFrame(
            [
                {
                    "name": "fixed_main_path",
                    "feature_count": len(main_selected),
                    "r2": main_fit_before_prune.r2,
                    "mse": main_fit_before_prune.mse,
                    "best_epoch": main_fit_before_prune.best_epoch,
                    "tau": tau,
                    "meets_tau": bool(main_fit_before_prune.r2 >= tau),
                    "r2_gap_to_full": full_fit.r2 - main_fit_before_prune.r2,
                    "features": feature_list_text(main_selected),
                }
            ]
        ).to_csv(run_dir / "initial_candidate_comparison.csv", index=False, encoding="utf-8-sig")
        main_gate_history = pd.DataFrame(
            [
                {
                    "experiment": "fixed_main_path",
                    "epoch": 0,
                    "feature": feature,
                    "role": "candidate",
                    "gate_value": 1.0 if feature in set(main_selected) else 0.0,
                    "selected_0p5": bool(feature in set(main_selected)),
                }
                for feature in features
            ]
        )
    elif args.use_dgating_initial_main:
        if source_run_dir is None:
            raise ValueError("--use-dgating-initial-main requires --source-run to point to a D-Gating run directory.")
        print(
            f"[2/5] D-Gating initial main path from effective norm > {args.dgating_effective_threshold:g}: tau={tau:.6f}",
            flush=True,
        )
        candidates = initial_candidate_sets(source_run_dir, args.dgating_effective_threshold)
        if not candidates:
            raise ValueError(f"No initial candidate paths found in {source_run_dir}.")
        candidate_jobs = [(candidate_features, name) for name, candidate_features in candidates.items()]
        candidate_fits = evaluator.fit_many_ordered(candidate_jobs)
        candidate_rows = []
        for (candidate_features, name), fit in zip(candidate_jobs, candidate_fits):
            candidate_rows.append(
                {
                    "name": name,
                    "feature_count": len(candidate_features),
                    "r2": fit.r2,
                    "mse": fit.mse,
                    "best_epoch": fit.best_epoch,
                    "tau": tau,
                    "meets_tau": bool(fit.r2 >= tau),
                    "r2_gap_to_full": full_fit.r2 - fit.r2,
                    "features": feature_list_text(candidate_features),
                }
            )
        initial_df = pd.DataFrame(candidate_rows).sort_values(["feature_count", "r2"], ascending=[True, False])
        initial_df.to_csv(run_dir / "initial_candidate_comparison.csv", index=False, encoding="utf-8-sig")
        initial_key = f"dgate_effective_gt_{args.dgating_effective_threshold:g}"
        if initial_key not in candidates:
            initial_key = initial_df.iloc[0]["name"]
        main_selected = list(candidates[str(initial_key)])
        main_fit_before_prune = evaluator.fit(main_selected, tag=f"initial_{initial_key}")
        compact_status = f"dgating_initial_{initial_key}"
        main_gate_history = pd.DataFrame(
            [
                {
                    "experiment": "dgating_initial_main",
                    "epoch": 0,
                    "feature": feature,
                    "role": "candidate",
                    "gate_value": 1.0 if feature in set(main_selected) else 0.0,
                    "selected_0p5": bool(feature in set(main_selected)),
                }
                for feature in features
            ]
        )
    else:
        print(f"[2/5] Main hard-gate path search: tau={tau:.6f}")
        main_gate_model, main_log, main_gate_history = train_hard_gate(
            bundle=bundle,
            candidate_features=features,
            fixed_features=[],
            hidden_dims=args.hidden_dims,
            epochs=args.gate_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            gate_lr=args.gate_lr,
            lambda_sparse=args.lambda_sparse,
            lambda_budget=args.lambda_budget,
            max_features=args.max_features,
            device=device,
            seed=args.random_state + 1,
            dropout=args.dropout,
            gate_init=args.gate_init,
            gate_init_noise=args.gate_init_noise,
            temp_start=args.temp_start,
            temp_end=args.temp_end,
            binary_beta=args.binary_beta,
            experiment="main_path",
        )
        main_log.to_csv(log_dir / "main_gate_log.csv", index=False, encoding="utf-8-sig")
        torch.save({"model_state": main_gate_model.state_dict(), "features": features, "target_col": target_col}, run_dir / "main_gate_model.pth")
        compact_status = "disabled"
        if args.disable_compact_main:
            main_selected = selected_by_gate(main_gate_history, features, args.gate_threshold, min_count=1)
            main_selected, main_fit_before_prune, main_expand_rows = expand_until_tau(
                current=main_selected,
                candidate_pool=features,
                gate_values=main_gate_history,
                evaluator=evaluator,
                tau=tau,
                expand_step=args.expand_step,
            )
        else:
            main_selected, main_fit_before_prune, compact_df, compact_status = choose_compact_path_seed(
                gate_values=main_gate_history,
                candidate_features=features,
                evaluator=evaluator,
                tau=tau,
                sizes=args.compact_main_sizes,
            )
            compact_df.to_csv(log_dir / "main_compact_candidates.csv", index=False, encoding="utf-8-sig")
            print(
                f"  compact_main: status={compact_status}, count={len(main_selected)}, R2={main_fit_before_prune.r2:.6f}, tau={tau:.6f}",
                flush=True,
            )
    main_path, main_fit, main_prune_df = prune_minimal_path(
        main_selected,
        evaluator=evaluator,
        tau=tau,
        tag="main",
        locked_keep=main_selected if args.skip_main_prune else None,
    )
    main_prune_df.insert(0, "path_feature_count", len(main_path))
    if main_expand_rows:
        pd.DataFrame(main_expand_rows).to_csv(log_dir / "main_expand_until_tau.csv", index=False, encoding="utf-8-sig")
    main_path_df = main_prune_df.copy()
    main_path_df["main_path"] = feature_list_text(main_path)
    main_path_df["main_path_r2"] = main_fit.r2
    main_path_df["main_path_mse"] = main_fit.mse
    main_path_df["compact_status"] = compact_status
    main_path_df.to_csv(run_dir / "main_path.csv", index=False, encoding="utf-8-sig")
    main_fit.log.to_csv(log_dir / "main_path_retrain_log.csv", index=False, encoding="utf-8-sig")
    if not args.use_dgating_initial_main and not fixed_main_path:
        plot_training_curve(main_log, figure_dir / "main_gate_training.png", "Main straight-through gate training")
        plot_gate_bar(main_gate_history, figure_dir / "main_final_gate_values.png", "Main path final gate values")
    plot_main_ablation(main_path_df, figure_dir / "main_path_ablation_impact.png")

    print(f"[3/5] Residual test after main path: main_count={len(main_path)}")
    residual_features = [f for f in features if f not in set(main_path)]
    residual_fit = evaluator.fit(residual_features, tag="residual_only")
    residual_rows: List[Dict[str, Any]] = [
        {
            "case": "R_only",
            "feature_count": len(residual_features),
            "r2": residual_fit.r2,
            "mse": residual_fit.mse,
            "tau": tau,
            "meets_tau": bool(residual_fit.r2 >= tau),
            "features": feature_list_text(residual_features),
        }
    ]
    independent_paths: List[Dict[str, Any]] = []
    remaining = list(residual_features)
    if residual_fit.r2 >= tau:
        for path_id in range(1, int(args.max_independent_paths) + 1):
            if not remaining:
                break
            gate_model, gate_log, gate_history = train_hard_gate(
                bundle=bundle,
                candidate_features=remaining,
                fixed_features=[],
                hidden_dims=args.hidden_dims,
                epochs=args.gate_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                gate_lr=args.gate_lr,
                lambda_sparse=args.lambda_sparse,
                lambda_budget=args.lambda_budget,
                max_features=args.max_features,
                device=device,
                seed=args.random_state + 100 + path_id,
                dropout=args.dropout,
                gate_init=args.gate_init,
                gate_init_noise=args.gate_init_noise,
                temp_start=args.temp_start,
                temp_end=args.temp_end,
                binary_beta=args.binary_beta,
                experiment=f"independent_path_{path_id}",
            )
            gate_log.to_csv(log_dir / f"independent_path_{path_id}_gate_log.csv", index=False, encoding="utf-8-sig")
            candidate = selected_by_gate(gate_history, remaining, args.gate_threshold, min_count=1)
            candidate, _, expand_rows = expand_until_tau(candidate, remaining, gate_history, evaluator, tau, expand_step=args.expand_step)
            path, fit, prune_df = prune_minimal_path(candidate, evaluator=evaluator, tau=tau, tag=f"independent_{path_id}")
            prune_df.to_csv(log_dir / f"independent_path_{path_id}_prune.csv", index=False, encoding="utf-8-sig")
            main_gate_history = pd.concat([main_gate_history, gate_history], ignore_index=True)
            independent_paths.append(
                {
                    "path_id": path_id,
                    "feature_count": len(path),
                    "r2": fit.r2,
                    "mse": fit.mse,
                    "meets_tau": bool(fit.r2 >= tau),
                    "features": feature_list_text(path),
                }
            )
            residual_rows.append(
                {
                    "case": f"independent_path_{path_id}",
                    "feature_count": len(path),
                    "r2": fit.r2,
                    "mse": fit.mse,
                    "tau": tau,
                    "meets_tau": bool(fit.r2 >= tau),
                    "features": feature_list_text(path),
                }
            )
            if expand_rows:
                pd.DataFrame(expand_rows).to_csv(log_dir / f"independent_path_{path_id}_expand_until_tau.csv", index=False, encoding="utf-8-sig")
            if fit.r2 < tau:
                break
            remaining = [f for f in remaining if f not in set(path)]
            remaining_fit = evaluator.fit(remaining, tag=f"remaining_after_independent_{path_id}")
            if remaining_fit.r2 < tau:
                break
    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(run_dir / "residual_test.csv", index=False, encoding="utf-8-sig")
    save_json(run_dir / "independent_paths.json", {"paths": independent_paths})
    plot_residual_summary(residual_df, figure_dir / "residual_test_summary.png")

    print(f"[4/5] Conditional substitution tests for {len(main_path)} main-path fields")
    substitution_rows: List[Dict[str, Any]] = []
    substitution_pre_jobs: List[tuple[List[str], str]] = []
    substitution_contexts: Dict[str, List[str]] = {}
    substitution_context_plus_r: Dict[str, List[str]] = {}
    for removed in main_path:
        context = [f for f in main_path if f != removed]
        context_plus_R = list(dict.fromkeys([*context, *residual_features]))
        substitution_contexts[removed] = context
        substitution_context_plus_r[removed] = context_plus_R
        substitution_pre_jobs.append((context, f"context_without_{removed}"))
        substitution_pre_jobs.append((context_plus_R, f"context_plus_R_without_{removed}"))
    substitution_pre_fits = evaluator.fit_many_ordered(substitution_pre_jobs) if substitution_pre_jobs else []
    substitution_pre_map = {
        tuple(features): fit for (features, _), fit in zip(substitution_pre_jobs, substitution_pre_fits)
    }
    for i, removed in enumerate(main_path, start=1):
        print(f"  substitution {i}/{len(main_path)}: removed={removed}", flush=True)
        context = substitution_contexts[removed]
        context_plus_R = substitution_context_plus_r[removed]
        context_fit = substitution_pre_map[tuple(context)]
        context_plus_R_fit = substitution_pre_map[tuple(context_plus_R)]
        row: Dict[str, Any] = {
            "removed_feature": removed,
            "context_feature_count": len(context),
            "residual_candidate_count": len(residual_features),
            "context_r2": context_fit.r2,
            "context_mse": context_fit.mse,
            "context_plus_R_r2": context_plus_R_fit.r2,
            "context_plus_R_mse": context_plus_R_fit.mse,
            "tau": substitution_tau,
            "main_tau": tau,
            "substitution_tau": substitution_tau,
            "status": "not_evaluated",
            "is_replaceable": False,
            "replacement_count": 0,
            "replacement_fields": "",
            "replacement_path_count": 0,
            "replacement_path": "",
            "replacement_path_r2": np.nan,
            "replacement_path_mse": np.nan,
        }
        if main_fit.r2 < tau and not args.run_substitution_when_main_below_tau:
            row["status"] = "main_path_below_tau_skip_substitution"
        elif context_fit.r2 >= substitution_tau:
            row.update(
                {
                    "status": "redundant_in_path",
                    "is_replaceable": True,
                    "replacement_count": 0,
                    "replacement_fields": "",
                    "replacement_path_count": len(context),
                    "replacement_path": feature_list_text(context),
                    "replacement_path_r2": context_fit.r2,
                    "replacement_path_mse": context_fit.mse,
                }
            )
        elif context_plus_R_fit.r2 < substitution_tau or not residual_features:
            row["status"] = "non_substitutable"
        else:
            row["status"] = "substitution_gate_search"
            sub_gate_model, sub_log, sub_gate_history = train_hard_gate(
                bundle=bundle,
                candidate_features=residual_features,
                fixed_features=context,
                hidden_dims=args.hidden_dims,
                epochs=args.gate_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                gate_lr=args.gate_lr,
                lambda_sparse=args.substitution_lambda_sparse,
                lambda_budget=args.substitution_lambda_budget,
                max_features=args.max_sub_features,
                device=device,
                seed=args.random_state + 1000 + i,
                dropout=args.dropout,
                gate_init=args.gate_init,
                gate_init_noise=args.gate_init_noise,
                temp_start=args.temp_start,
                temp_end=args.temp_end,
                binary_beta=args.binary_beta,
                experiment=f"substitute__{safe_name(removed, 80)}",
            )
            sub_log.to_csv(log_dir / f"substitute__{safe_name(removed, 80)}_gate_log.csv", index=False, encoding="utf-8-sig")
            replacement = selected_by_gate(sub_gate_history, residual_features, args.gate_threshold, min_count=1)
            replacement_path_seed = list(dict.fromkeys([*context, *replacement]))
            replacement_path_seed, _, expand_rows = expand_until_tau(
                current=replacement_path_seed,
                candidate_pool=residual_features,
                gate_values=sub_gate_history,
                evaluator=evaluator,
                tau=substitution_tau,
                protected=context,
                expand_step=args.expand_step,
            )
            replacement_path, replacement_fit, prune_df = prune_minimal_path(
                replacement_path_seed,
                evaluator=evaluator,
                tau=substitution_tau,
                tag=f"substitute_{safe_name(removed, 80)}",
            )
            replacement_fields = [f for f in replacement_path if f not in set(context)]
            prune_df.to_csv(log_dir / f"substitute__{safe_name(removed, 80)}_prune.csv", index=False, encoding="utf-8-sig")
            if expand_rows:
                pd.DataFrame(expand_rows).to_csv(log_dir / f"substitute__{safe_name(removed, 80)}_expand_until_tau.csv", index=False, encoding="utf-8-sig")
            substitution_gate_frames.append(sub_gate_history)
            row.update(
                {
                    "status": "substitutable" if replacement_fit.r2 >= substitution_tau else "substitution_gate_failed_tau",
                    "is_replaceable": bool(replacement_fit.r2 >= substitution_tau),
                    "replacement_count": len(replacement_fields),
                    "replacement_fields": feature_list_text(replacement_fields),
                    "replacement_path_count": len(replacement_path),
                    "replacement_path": feature_list_text(replacement_path),
                    "replacement_path_r2": replacement_fit.r2,
                    "replacement_path_mse": replacement_fit.mse,
                }
            )
        substitution_rows.append(row)

    substitution_df = pd.DataFrame(substitution_rows)
    substitution_df.to_csv(run_dir / "substitution_results.csv", index=False, encoding="utf-8-sig")
    plot_substitution_summary(substitution_df, figure_dir, substitution_tau)

    print(f"[5/6] Global necessity scores for {len(features)} fields")
    necessity_rows: List[Dict[str, Any]] = []
    necessity_jobs: List[tuple[List[str], str]] = []
    for feature in features:
        necessity_jobs.append(([f for f in features if f != feature], f"global_minus_{feature}"))
    necessity_fits = evaluator.fit_many_ordered(necessity_jobs)
    for j, (feature, minus_fit) in enumerate(zip(features, necessity_fits), start=1):
        print(f"  necessity {j}/{len(features)}: minus={feature}", flush=True)
        necessity_rows.append(
            {
                "feature": feature,
                "feature_index": j - 1,
                "R2_minus_j": minus_fit.r2,
                "MSE_minus_j": minus_fit.mse,
                "Nec_j": full_fit.r2 - minus_fit.r2,
                "R2_full": full_fit.r2,
                "MSE_full": full_fit.mse,
                "tau": tau,
                "minus_meets_tau": bool(minus_fit.r2 >= tau),
                "is_in_main_path": bool(feature in set(main_path)),
                "is_in_residual": bool(feature in set(residual_features)),
            }
        )
    necessity_df = pd.DataFrame(necessity_rows).sort_values("Nec_j", ascending=False)
    necessity_df.to_csv(run_dir / "necessity_scores.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 7))
    top_nec = necessity_df.head(25).sort_values("Nec_j", ascending=True)
    ax.barh(top_nec["feature"], top_nec["Nec_j"], color="#9bbb59")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Nec_j = R2_full - R2(U - {xj})")
    ax.set_title("Top global necessity scores")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "necessity_scores_top.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("[6/6] Writing aggregate gate values, figures, and result note")
    all_gate_values = pd.concat([main_gate_history, *substitution_gate_frames], ignore_index=True) if substitution_gate_frames else main_gate_history
    all_gate_values.to_csv(run_dir / "gate_values.csv", index=False, encoding="utf-8-sig")
    plot_gate_heatmap(all_gate_values, figure_dir / "gate_values_heatmap.png")

    metrics = {
        "target_col": target_col,
        "feature_count": len(features),
        "r2_full": full_fit.r2,
        "mse_full": full_fit.mse,
        "tau": tau,
        "ratio_tau": ratio_tau,
        "main_tau": tau,
        "substitution_tau": substitution_tau,
        "main_path_count": len(main_path),
        "main_path": main_path,
        "main_path_r2": main_fit.r2,
        "main_path_mse": main_fit.mse,
        "main_path_meets_tau": bool(main_fit.r2 >= tau),
        "main_compact_status": compact_status,
        "main_gate_selected_before_prune_count": len(main_selected),
        "main_gate_selected_before_prune_r2": main_fit_before_prune.r2,
        "residual_feature_count": len(residual_features),
        "residual_r2": residual_fit.r2,
        "residual_mse": residual_fit.mse,
        "residual_meets_tau": bool(residual_fit.r2 >= tau),
        "independent_paths": independent_paths,
        "replaceable_count": int(substitution_df["is_replaceable"].sum()) if not substitution_df.empty else 0,
        "top_necessity_features": necessity_df.head(12).to_dict(orient="records"),
        "source_run": str(source_run_dir) if source_run_dir else None,
        "device": str(device),
    }
    save_json(run_dir / "metrics.json", metrics)
    write_readme(
        output_dir=run_dir,
        metrics=metrics,
        main_path=main_path,
        residual_row=residual_rows[0],
        substitution_df=substitution_df,
    )
    write_summary_txt(
        output_dir=run_dir,
        metrics=metrics,
        main_path=main_path,
        residual_df=residual_df,
        substitution_df=substitution_df,
        necessity_df=necessity_df,
    )

    print(f"Saved run to {run_dir}")
    print(f"Full R2={full_fit.r2:.6f}, tau={tau:.6f}, main_path_count={len(main_path)}, residual_R2={residual_fit.r2:.6f}")


if __name__ == "__main__":
    main()
