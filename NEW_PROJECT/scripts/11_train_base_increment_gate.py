from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
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


MODEL_NAME = "BaseIncrementDecomposedGateDNN"


def configure_chinese_font() -> None:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, *plt.rcParams.get("font.sans-serif", [])]
            break
    plt.rcParams["axes.unicode_minus"] = False


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


def choose_device(name: str) -> torch.device:
    text = str(name or "auto").lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA, but torch.cuda.is_available() is false.")
    return torch.device(text)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.net = make_mlp(in_dim, hidden_dims, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseIncrementGateRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        num_paths: int,
        dropout: float = 0.0,
        gate_init: float = -2.2,
        gate_init_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_paths = int(num_paths)
        self.base_logits = nn.Parameter(torch.full((in_dim,), float(gate_init)))
        increment_init = torch.full((self.num_paths, in_dim), float(gate_init))
        noise = float(gate_init_noise or 0.0)
        if noise > 0:
            increment_init = increment_init + torch.randn_like(increment_init) * noise
        self.increment_logits = nn.Parameter(increment_init)
        self.net = make_mlp(in_dim, hidden_dims, dropout=dropout)

    def gates(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = torch.sigmoid(self.base_logits)
        u = torch.sigmoid(self.increment_logits)
        g = b.unsqueeze(0) + (1.0 - b).unsqueeze(0) * u
        return b, u, g

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        b, u, g = self.gates()
        base_pred = self.net(x * b)
        path_preds = [self.net(x * g[k]) for k in range(self.num_paths)]
        return base_pred, path_preds, b, u, g


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys: List[torch.Tensor] = []
    preds: List[torch.Tensor] = []
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            total_loss += float(loss.detach().cpu()) * xb.size(0)
            ys.append(yb.detach().cpu())
            preds.append(pred.detach().cpu())
    y_cat = torch.cat(ys, dim=0)
    p_cat = torch.cat(preds, dim=0)
    return {"mse": total_loss / len(loader.dataset), "r2": r2_score_torch(y_cat, p_cat)}


def eval_gate_model(model: BaseIncrementGateRegressor, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    base_y: List[torch.Tensor] = []
    base_p: List[torch.Tensor] = []
    path_y: List[List[torch.Tensor]] = [[] for _ in range(model.num_paths)]
    path_p: List[List[torch.Tensor]] = [[] for _ in range(model.num_paths)]
    base_loss_sum = 0.0
    path_loss_sum = [0.0 for _ in range(model.num_paths)]
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred_base, pred_paths, _, _, _ = model(xb)
            base_loss_sum += float(nn.functional.mse_loss(pred_base, yb).detach().cpu()) * xb.size(0)
            base_y.append(yb.detach().cpu())
            base_p.append(pred_base.detach().cpu())
            for k, pred in enumerate(pred_paths):
                path_loss_sum[k] += float(nn.functional.mse_loss(pred, yb).detach().cpu()) * xb.size(0)
                path_y[k].append(yb.detach().cpu())
                path_p[k].append(pred.detach().cpu())

    out: Dict[str, float] = {
        "base_mse": base_loss_sum / len(loader.dataset),
        "base_r2": r2_score_torch(torch.cat(base_y, dim=0), torch.cat(base_p, dim=0)),
    }
    for k in range(model.num_paths):
        out[f"path{k + 1}_mse"] = path_loss_sum[k] / len(loader.dataset)
        out[f"path{k + 1}_r2"] = r2_score_torch(torch.cat(path_y[k], dim=0), torch.cat(path_p[k], dim=0))
    return out


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
) -> tuple[Dict[str, float], pd.DataFrame]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

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
    model = MLPRegressor(X_train.shape[1], hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best = {"best_test_mse": math.inf, "best_test_r2": -math.inf, "best_epoch": 0}
    rows = []
    for epoch in range(1, epochs + 1):
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
        if test_metrics["r2"] > best["best_test_r2"]:
            best = {
                "best_test_mse": float(test_metrics["mse"]),
                "best_test_r2": float(test_metrics["r2"]),
                "best_epoch": int(epoch),
            }
    return best, pd.DataFrame(rows)


def train_gate_model(
    bundle: Any,
    hidden_dims: Sequence[int],
    num_paths: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    params: Dict[str, Any],
) -> tuple[BaseIncrementGateRegressor, pd.DataFrame, pd.DataFrame]:
    seed = int(params.get("random_state", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

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

    model = BaseIncrementGateRegressor(
        in_dim=len(bundle.features),
        hidden_dims=hidden_dims,
        num_paths=num_paths,
        dropout=float(params.get("dropout", 0.0)),
        gate_init=float(params.get("gate_init", -2.2)),
        gate_init_noise=float(params.get("gate_init_noise", 0.0)),
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": [model.base_logits, model.increment_logits], "lr": float(params.get("gate_lr", lr))},
            {"params": model.net.parameters(), "lr": lr},
        ]
    )

    lambda_b = float(params.get("lambda_b", 0.002))
    lambda_u = float(params.get("lambda_u", 0.002))
    lambda_base_pred = float(params.get("lambda_base_pred", 0.5))
    lambda_overlap = float(params.get("lambda_overlap", 0.01))
    lambda_bu = float(params.get("lambda_bu", 0.01))
    lambda_exist = float(params.get("lambda_exist", 0.1))
    lambda_effect = float(params.get("lambda_effect", 0.1))
    lambda_polar = float(params.get("lambda_polar", 0.001))
    eps = float(params.get("eps", 1.0))
    margin = float(params.get("margin", 0.02))

    log_rows = []
    gate_rows = []
    best_state = None
    best_score = -math.inf
    base_frozen = False

    for epoch in range(1, epochs + 1):
        freeze_after = int(params.get("freeze_base_after_epochs", 0) or 0)
        if freeze_after > 0 and not base_frozen and epoch == freeze_after + 1:
            tau = float(params.get("freeze_base_tau", 0.5))
            hard_logit = float(params.get("frozen_base_logit", 12.0))
            with torch.no_grad():
                b_now = torch.sigmoid(model.base_logits)
                hard_b = torch.where(
                    b_now >= tau,
                    torch.full_like(model.base_logits, hard_logit),
                    torch.full_like(model.base_logits, -hard_logit),
                )
                model.base_logits.copy_(hard_b)
                if bool(params.get("reset_increment_after_freeze", False)):
                    init = torch.full_like(model.increment_logits, float(params.get("gate_init", 0.0)))
                    noise = float(params.get("gate_init_noise", 0.0) or 0.0)
                    if noise > 0:
                        init = init + torch.randn_like(init) * noise
                    model.increment_logits.copy_(init)
            model.base_logits.requires_grad_(False)
            base_frozen = True

        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred_base, pred_paths, b, u, _ = model(xb)

            e_base = nn.functional.mse_loss(pred_base, yb)
            path_mses = torch.stack([nn.functional.mse_loss(pred, yb) for pred in pred_paths])
            pred_loss = torch.sum(path_mses) + lambda_base_pred * e_base

            base_sparse = torch.sum(b)
            inc = (1.0 - b).unsqueeze(0) * u
            inc_sparse = torch.sum(inc)
            bu_separation = torch.sum(b.unsqueeze(0) * u)

            overlap = torch.zeros((), device=device)
            for k in range(num_paths):
                for l in range(k + 1, num_paths):
                    overlap = overlap + torch.sum(((1.0 - b) ** 2) * u[k] * u[l])

            exists = torch.sum(torch.relu(eps - torch.sum(inc, dim=1)) ** 2)
            deltas = e_base.detach() - path_mses
            effective = torch.sum(torch.relu(margin - deltas) ** 2)
            polar = torch.sum(b * (1.0 - b)) + torch.sum(u * (1.0 - u))

            loss = (
                pred_loss
                + lambda_b * base_sparse
                + lambda_u * inc_sparse
                + lambda_overlap * overlap
                + lambda_bu * bu_separation
                + lambda_exist * exists
                + lambda_effect * effective
                + lambda_polar * polar
            )
            loss.backward()
            optimizer.step()

        train_metrics = eval_gate_model(model, train_loader, device)
        test_metrics = eval_gate_model(model, test_loader, device)
        with torch.no_grad():
            b_np, u_np, _ = [t.detach().cpu().numpy() for t in model.gates()]
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"test_{k}": v for k, v in test_metrics.items()}}
        row["base_active_0p5"] = int(np.sum(b_np >= 0.5))
        for k in range(num_paths):
            row[f"inc{k + 1}_active_0p5"] = int(np.sum((u_np[k] >= 0.5) & (b_np < 0.5)))
        log_rows.append(row)

        for i, feature in enumerate(bundle.features):
            gate_rows.append({"epoch": epoch, "feature_index": i, "feature": feature, "gate_type": "base", "path": 0, "value": float(b_np[i])})
            for k in range(num_paths):
                gate_rows.append(
                    {
                        "epoch": epoch,
                        "feature_index": i,
                        "feature": feature,
                        "gate_type": "increment",
                        "path": k + 1,
                        "value": float(u_np[k, i]),
                    }
                )

        score = float(np.mean([test_metrics[f"path{k + 1}_r2"] for k in range(num_paths)]))
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(log_rows), pd.DataFrame(gate_rows)


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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def plot_loss_r2(log_df: pd.DataFrame, output_dir: Path, num_paths: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(log_df["epoch"], log_df["test_base_mse"], label="Base", linewidth=2.0)
    for k in range(num_paths):
        axes[0].plot(log_df["epoch"], log_df[f"test_path{k + 1}_mse"], label=f"Path {k + 1}", linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test MSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(log_df["epoch"], log_df["test_base_r2"], label="Base", linewidth=2.0)
    for k in range(num_paths):
        axes[1].plot(log_df["epoch"], log_df[f"test_path{k + 1}_r2"], label=f"Path {k + 1}", linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"Test $R^2$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_r2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_active_counts(log_df: pd.DataFrame, output_dir: Path, num_paths: int) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(log_df["epoch"], log_df["base_active_0p5"], label="Base B", linewidth=2.2)
    for k in range(num_paths):
        ax.plot(log_df["epoch"], log_df[f"inc{k + 1}_active_0p5"], label=f"Increment I{k + 1}", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active Feature Count @ 0.5")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "active_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gate_history(gate_df: pd.DataFrame, output_dir: Path, num_paths: int, top_n: int = 12) -> None:
    final = gate_df[gate_df["epoch"] == gate_df["epoch"].max()].copy()
    base_final = final[final["gate_type"] == "base"].sort_values("value", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for feature in base_final["feature"].tolist():
        sub = gate_df[(gate_df["gate_type"] == "base") & (gate_df["feature"] == feature)]
        ax.plot(sub["epoch"], sub["value"], linewidth=1.5, label=feature)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Base Gate b")
    ax.set_title(f"Top {top_n} Base Gate Trajectories")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "base_gate_history_top.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(num_paths, 1, figsize=(10.5, max(4.0, 3.0 * num_paths)), sharex=True)
    if num_paths == 1:
        axes = [axes]
    for k, ax in enumerate(axes, start=1):
        path_final = final[(final["gate_type"] == "increment") & (final["path"] == k)].sort_values("value", ascending=False).head(top_n)
        for feature in path_final["feature"].tolist():
            sub = gate_df[(gate_df["gate_type"] == "increment") & (gate_df["path"] == k) & (gate_df["feature"] == feature)]
            ax.plot(sub["epoch"], sub["value"], linewidth=1.3, label=feature)
        ax.set_ylabel(f"u{k}")
        ax.grid(True, alpha=0.22)
        ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"Top {top_n} Increment Gate Trajectories", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "increment_gate_history_top.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_gates(gate_values: pd.DataFrame, output_dir: Path, num_paths: int, top_n: int = 20) -> None:
    ranked = gate_values.sort_values(["base_gate", "max_increment_gate"], ascending=False).head(top_n).copy()
    y = np.arange(len(ranked))
    fig, ax = plt.subplots(figsize=(11, max(5.0, 0.32 * len(ranked) + 2.0)))
    ax.barh(y - 0.18, ranked["base_gate"], height=0.32, label="Base b", color="#1f4e79")
    for k in range(num_paths):
        ax.barh(y + 0.18 + 0.11 * k, ranked[f"u{k + 1}"], height=0.10, label=f"u{k + 1}", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(ranked["feature"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gate Value")
    ax.set_xlim(0, 1.02)
    ax.grid(True, axis="x", alpha=0.22)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "final_gate_values_top.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_path_summary(paths: Dict[str, Any], output_dir: Path) -> None:
    configure_chinese_font()
    lines = ["最终输出："]
    base = paths["base_features"]
    lines.append("B = {" + "，".join(base[:12]) + ("，..." if len(base) > 12 else "") + "}")
    for path in paths["paths"]:
        if not path["is_effective"]:
            continue
        inc = path["increment_features"]
        full = path["full_features"]
        lines.append("")
        lines.append(f"I{path['path']} = " + "{" + "，".join(inc[:10]) + ("，..." if len(inc) > 10 else "") + "}")
        lines.append(f"S{path['path']} = B ∪ I{path['path']}，字段数={len(full)}，R²={path['retrained_r2']:.4f}")
    if all(not path["is_effective"] for path in paths["paths"]):
        lines.append("")
        lines.append("未得到满足判定条件的有效增量路径。")

    fig, ax = plt.subplots(figsize=(11, max(4.8, 0.42 * len(lines) + 1.5)))
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12, linespacing=1.45)
    fig.tight_layout()
    fig.savefig(output_dir / "path_summary_zh.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_gate_values(
    model: BaseIncrementGateRegressor,
    features: Sequence[str],
    tau_b: float,
    tau_u: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, List[str], List[List[str]], List[List[str]]]:
    with torch.no_grad():
        b, u, g = [t.detach().cpu().numpy() for t in model.gates()]
    rows = []
    base_features = [features[i] for i, value in enumerate(b) if value >= tau_b]
    increments: List[List[str]] = []
    full_sets: List[List[str]] = []
    for k in range(u.shape[0]):
        inc = [features[i] for i, value in enumerate(u[k]) if value >= tau_u and b[i] < tau_b]
        full = list(dict.fromkeys([*base_features, *inc]))
        increments.append(inc)
        full_sets.append(full)
    for i, feature in enumerate(features):
        row = {
            "feature_index": i,
            "feature": feature,
            "base_gate": float(b[i]),
            "base_selected": bool(b[i] >= tau_b),
            "max_increment_gate": float(np.max(u[:, i])),
            "max_final_gate": float(np.max(g[:, i])),
        }
        for k in range(u.shape[0]):
            row[f"u{k + 1}"] = float(u[k, i])
            row[f"g{k + 1}"] = float(g[k, i])
            row[f"increment{k + 1}_selected"] = bool(u[k, i] >= tau_u and b[i] < tau_b)
        rows.append(row)
    gate_values = pd.DataFrame(rows).sort_values(["base_selected", "max_final_gate", "base_gate"], ascending=False)
    gate_values.to_csv(output_dir / "gate_values.csv", index=False, encoding="utf-8-sig")
    return gate_values, base_features, increments, full_sets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train base-increment decomposed gate model for tabular time-series inference-source localization.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument("--source-run", default=str(PROJECT_ROOT / "outputs" / "data2025_Processed_V2" / "CenterOn_net_actual_interchange_mw" / "L1GateDNN" / "run_20260603_l1_lr0p00065_thr0p10_combo5_L1GateDNN"))
    parser.add_argument("--csv", default=None, help="CSV path. Defaults to source run data path, then config dataset.processed_csv.")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--combo", default="5")
    parser.add_argument("--use-all-features", action="store_true", help="Use every numeric non-target column after exclusions instead of source-run selected features.")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--baseline-epochs", type=int, default=180)
    parser.add_argument("--retrain-epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--gate-lr", type=float, default=0.004)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32, 16])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gate-init", type=float, default=0.0, help="Initial gate logit. 0 means sigmoid gate starts at 0.5.")
    parser.add_argument("--gate-init-noise", type=float, default=0.02, help="Independent random logit noise for increment gates, used to break path symmetry.")
    parser.add_argument("--lambda-b", type=float, default=0.002)
    parser.add_argument("--lambda-u", type=float, default=0.002)
    parser.add_argument("--lambda-base-pred", type=float, default=0.5, help="Weight of base-only prediction loss, used to keep B as a meaningful common core.")
    parser.add_argument("--lambda-overlap", type=float, default=0.01)
    parser.add_argument("--lambda-bu", type=float, default=0.01, help="Penalty for b_i and u_i^k being high at the same time.")
    parser.add_argument("--lambda-exist", type=float, default=0.1)
    parser.add_argument("--lambda-effect", type=float, default=0.1)
    parser.add_argument("--lambda-polar", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--r2-ratio", type=float, default=0.95, help="A valid S_k must reach this ratio of full-field R2.")
    parser.add_argument("--base-improve-margin", type=float, default=0.005, help="A valid S_k must improve over retrained B by at least this R2 margin.")
    parser.add_argument(
        "--legacy-delta-validity",
        action="store_true",
        help="Use old validity rule: direct E_base-E_k >= margin. Otherwise this value is diagnostic only.",
    )
    parser.add_argument("--freeze-base-after-epochs", type=int, default=0, help="If >0, discover B softly for this many epochs, then freeze b as a hard base for the remaining epochs.")
    parser.add_argument("--freeze-base-tau", type=float, default=0.5)
    parser.add_argument("--frozen-base-logit", type=float, default=12.0)
    parser.add_argument("--reset-increment-after-freeze", action="store_true")
    parser.add_argument("--tau-b", type=float, default=0.5)
    parser.add_argument("--tau-u", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="auto")
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
    params = {
        "random_state": args.random_state,
        "dropout": args.dropout,
        "gate_lr": args.gate_lr,
        "gate_init": args.gate_init,
        "gate_init_noise": args.gate_init_noise,
        "lambda_b": args.lambda_b,
        "lambda_u": args.lambda_u,
        "lambda_base_pred": args.lambda_base_pred,
        "lambda_overlap": args.lambda_overlap,
        "lambda_bu": args.lambda_bu,
        "lambda_exist": args.lambda_exist,
        "lambda_effect": args.lambda_effect,
        "lambda_polar": args.lambda_polar,
        "eps": args.eps,
        "margin": args.margin,
        "freeze_base_after_epochs": args.freeze_base_after_epochs,
        "freeze_base_tau": args.freeze_base_tau,
        "frozen_base_logit": args.frozen_base_logit,
        "reset_increment_after_freeze": args.reset_increment_after_freeze,
    }

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

    print(f"Training full-field baseline MLP with {len(features)} features on {device} ...")
    full_metrics, full_log = train_plain_mlp(
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
    )
    full_log.to_csv(run_dir / "full_mlp_log.csv", index=False, encoding="utf-8-sig")
    r2_full = float(full_metrics["best_test_r2"])

    print(f"Training {MODEL_NAME}: K={args.k}, features={len(features)} ...")
    gate_model, log_df, gate_df = train_gate_model(
        bundle=bundle,
        hidden_dims=args.hidden_dims,
        num_paths=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        params=params,
    )
    log_df.to_csv(run_dir / "log.csv", index=False, encoding="utf-8-sig")
    gate_df.to_csv(run_dir / "gate_history.csv", index=False, encoding="utf-8-sig")
    torch.save({"model_state": gate_model.state_dict(), "features": features, "target_col": target_col}, run_dir / "model.pth")

    gate_values, base_features, increments, full_sets = write_gate_values(gate_model, features, args.tau_b, args.tau_u, run_dir)

    if base_features:
        base_idx = [features.index(f) for f in base_features]
        base_retrained_metrics, base_retrained_log = train_plain_mlp(
            bundle.X_train[:, base_idx],
            bundle.y_train,
            bundle.X_test[:, base_idx],
            bundle.y_test,
            hidden_dims=args.hidden_dims,
            epochs=args.retrain_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.random_state + 1009,
            dropout=args.dropout,
        )
        base_retrained_log.to_csv(run_dir / "base_retrain_log.csv", index=False, encoding="utf-8-sig")
        base_retrained_mse = float(base_retrained_metrics["best_test_mse"])
        base_retrained_r2 = float(base_retrained_metrics["best_test_r2"])
    else:
        base_retrained_mse = None
        base_retrained_r2 = None

    path_rows = []
    for k, full_set in enumerate(full_sets, start=1):
        if full_set:
            idx = [features.index(f) for f in full_set]
            retrained_metrics, retrained_log = train_plain_mlp(
                bundle.X_train[:, idx],
                bundle.y_train,
                bundle.X_test[:, idx],
                bundle.y_test,
                hidden_dims=args.hidden_dims,
                epochs=args.retrain_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                seed=args.random_state + k,
                dropout=args.dropout,
            )
            retrained_log.to_csv(run_dir / f"path{k}_retrain_log.csv", index=False, encoding="utf-8-sig")
            retrained_mse = float(retrained_metrics["best_test_mse"])
            retrained_r2 = float(retrained_metrics["best_test_r2"])
        else:
            retrained_mse = None
            retrained_r2 = None

        direct_base_mse = float(log_df["test_base_mse"].iloc[-1])
        direct_path_mse = float(log_df[f"test_path{k}_mse"].iloc[-1])
        delta = direct_base_mse - direct_path_mse
        r2_threshold = args.r2_ratio * r2_full
        r2_gain_over_base = None if (retrained_r2 is None or base_retrained_r2 is None) else retrained_r2 - base_retrained_r2
        if args.legacy_delta_validity:
            is_effective = bool(retrained_r2 is not None and retrained_r2 >= r2_threshold and delta >= args.margin)
            validity_rule = "legacy_direct_delta"
        else:
            is_effective = bool(
                retrained_r2 is not None
                and len(increments[k - 1]) > 0
                and retrained_r2 >= r2_threshold
                and (base_retrained_r2 is None or retrained_r2 >= base_retrained_r2 + args.base_improve_margin)
            )
            validity_rule = "binary_retrained_r2_and_base_gain"
        path_rows.append(
            {
                "path": k,
                "base_count": len(base_features),
                "increment_count": len(increments[k - 1]),
                "full_count": len(full_set),
                "direct_base_mse": direct_base_mse,
                "direct_path_mse": direct_path_mse,
                "direct_delta_mse": delta,
                "direct_path_r2": float(log_df[f"test_path{k}_r2"].iloc[-1]),
                "retrained_mse": retrained_mse,
                "retrained_r2": retrained_r2,
                "base_retrained_mse": base_retrained_mse,
                "base_retrained_r2": base_retrained_r2,
                "r2_gain_over_base": r2_gain_over_base,
                "r2_full": r2_full,
                "r2_threshold": r2_threshold,
                "validity_rule": validity_rule,
                "is_effective": is_effective,
                "increment_features": increments[k - 1],
                "full_features": full_set,
            }
        )

    paths = {"base_features": base_features, "paths": path_rows}
    save_json(run_dir / "paths.json", paths)
    pd.DataFrame(path_rows).drop(columns=["increment_features", "full_features"]).to_csv(run_dir / "path_metrics.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "target_col": target_col,
        "feature_count": len(features),
        "base_count": len(base_features),
        "num_paths": args.k,
        "r2_full": r2_full,
        "full_mlp_best_test_mse": float(full_metrics["best_test_mse"]),
        "full_mlp_best_epoch": int(full_metrics["best_epoch"]),
        "base_retrained_mse": base_retrained_mse,
        "base_retrained_r2": base_retrained_r2,
        "r2_ratio": float(args.r2_ratio),
        "base_improve_margin": float(args.base_improve_margin),
        "effective_path_count": int(sum(1 for row in path_rows if row["is_effective"])),
        "base_features": base_features,
        "paths": path_rows,
    }
    save_json(run_dir / "metrics.json", metrics)

    plot_loss_r2(log_df, run_dir, args.k)
    plot_active_counts(log_df, run_dir, args.k)
    plot_gate_history(gate_df, run_dir, args.k)
    plot_final_gates(gate_values, run_dir, args.k)
    plot_path_summary(paths, run_dir)

    print(f"Saved run to {run_dir}")
    print(f"Full MLP R2={r2_full:.6f}, base_count={len(base_features)}, effective_paths={metrics['effective_path_count']}")


if __name__ == "__main__":
    main()
