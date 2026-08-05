from __future__ import annotations

import argparse
import json
import math
import sys
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
from src.data_utils import ensure_dir, normalize_column_list, read_numeric_csv, safe_name, save_json


@dataclass
class ResidualDataset:
    xi: str
    center: str
    ci_features: list[str]
    r_features: list[str]
    source_row_index: np.ndarray
    X_ci_raw: np.ndarray
    X_r_raw: np.ndarray
    residual_std: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray


class DGatedResidualShowingNet(nn.Module):
    def __init__(self, ci_dim: int, r_dim: int, hidden_dims: Sequence[int], dgate_depth: int, dropout_r: float) -> None:
        super().__init__()
        dims = [int(v) for v in hidden_dims]
        if not dims:
            dims = [64, 32]
        if dgate_depth < 2:
            raise ValueError("dgate_depth must be >= 2.")
        self.ci_dim = int(ci_dim)
        self.r_dim = int(r_dim)
        self.first_hidden_dim = dims[0]
        self.dropout_r = float(dropout_r)
        self.omega_ci = nn.Parameter(torch.empty(self.ci_dim, self.first_hidden_dim))
        self.omega_r = nn.Parameter(torch.empty(self.r_dim, self.first_hidden_dim))
        self.gamma = nn.Parameter(torch.ones(int(dgate_depth) - 1, self.r_dim))
        self.bias = nn.Parameter(torch.zeros(self.first_hidden_dim))
        nn.init.kaiming_normal_(self.omega_ci, nonlinearity="relu")
        nn.init.kaiming_normal_(self.omega_r, nonlinearity="relu")

        layers: list[nn.Module] = []
        prev = self.first_hidden_dim
        for dim in dims[1:]:
            layers.append(nn.Linear(prev, int(dim)))
            layers.append(nn.ReLU())
            prev = int(dim)
        layers.append(nn.Linear(prev, 1))
        self.tail = nn.Sequential(*layers)

    def get_gates(self) -> torch.Tensor:
        return torch.prod(self.gamma, dim=0)

    def effective_r_weight(self) -> torch.Tensor:
        return self.omega_r * self.get_gates().unsqueeze(1)

    def effective_group_norms(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.effective_r_weight(), ord=2, dim=1)

    def dgate_regularizer(self) -> torch.Tensor:
        return torch.sum(self.omega_r**2) + torch.sum(self.gamma**2)

    def forward(self, x_ci: torch.Tensor, x_r: torch.Tensor, r_keep: torch.Tensor | None = None) -> torch.Tensor:
        if self.training and self.dropout_r > 0:
            keep_prob = 1.0 - self.dropout_r
            mask = torch.bernoulli(torch.full_like(x_r, keep_prob)) / keep_prob
            x_r = x_r * mask
        if r_keep is not None:
            x_r = x_r * r_keep.reshape(1, -1)
        h_ci = torch.matmul(x_ci, self.omega_ci)
        h_r = torch.matmul(x_r, self.effective_r_weight())
        h = torch.relu(h_ci + h_r + self.bias)
        return self.tail(h)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _standardize(train: np.ndarray, apply: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return ((apply - mean) / std).astype(np.float32), mean.squeeze().astype(np.float32), std.squeeze().astype(np.float32)


def _load_dataset(stage03_xi_interface: Path, cfg: dict[str, Any], args: argparse.Namespace) -> ResidualDataset:
    xi_if = _read_json(stage03_xi_interface)
    xi = str(xi_if["xi"])
    center = str(xi_if["target"])
    ci_features = list(xi_if["ci_features"])
    r_features = list(xi_if["residual_candidate_features"])
    all_needed = list(dict.fromkeys([*xi_if["main_path_features"], *r_features]))

    stage03_run = Path(stage03_xi_interface).resolve().parents[1]
    stage03_config = _read_json(stage03_run / "stage_config.json")
    stage02_interface = _read_json(Path(stage03_config["stage02_interface"]))
    stage01_interface = _read_json(Path(stage02_interface["source_stage01_interface"]))
    exclude_columns = []
    dgate_run_dir = stage01_interface.get("dgate_run_dir")
    if dgate_run_dir:
        exclude_columns = normalize_column_list(
            _read_json(Path(dgate_run_dir) / "config.json").get("preprocessing", {}).get("exclude_columns")
        )

    dataset_cfg = cfg["dataset"]
    preprocessing = cfg.get("preprocessing", {})
    df = read_numeric_csv(
        resolve_project_path(cfg, dataset_cfg["processed_csv"]),
        drop_all_zero_columns=bool(preprocessing.get("drop_all_zero_columns", False)),
        exclude_columns=exclude_columns,
    )
    missing = [col for col in [center, *all_needed] if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    oof_df = pd.read_csv(xi_if["oof_residuals_csv"])
    residual_col = str(xi_if.get("residual_target_column", "residual_std"))
    source_rows = oof_df["source_row_index"].to_numpy(dtype=int)
    aligned = df.loc[source_rows, all_needed]
    if aligned.isna().any(axis=None):
        raise ValueError("NaN found after aligning OOF residual rows to raw feature table.")

    X_ci = aligned[ci_features].to_numpy(dtype=np.float32)
    X_r = aligned[r_features].to_numpy(dtype=np.float32)
    y = oof_df[residual_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    n = len(oof_df)
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(n)
    val_size = max(1, int(n * float(args.val_ratio)))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return ResidualDataset(
        xi=xi,
        center=center,
        ci_features=ci_features,
        r_features=r_features,
        source_row_index=source_rows,
        X_ci_raw=X_ci,
        X_r_raw=X_r,
        residual_std=y,
        train_idx=train_idx,
        val_idx=val_idx,
    )


def _make_loaders(
    data: ResidualDataset,
    batch_size: int,
) -> tuple[DataLoader, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train_idx = data.train_idx
    val_idx = data.val_idx
    X_ci_all, ci_mean, ci_std = _standardize(data.X_ci_raw[train_idx], data.X_ci_raw)
    X_r_all, r_mean, r_std = _standardize(data.X_r_raw[train_idx], data.X_r_raw)
    y_all = data.residual_std.astype(np.float32)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_ci_all[train_idx]),
            torch.from_numpy(X_r_all[train_idx]),
            torch.from_numpy(y_all[train_idx]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    arrays = (X_ci_all, X_r_all, y_all)
    scalers = (ci_mean, ci_std, r_mean, r_std)
    return train_loader, arrays, scalers


def _eval_arrays(
    model: DGatedResidualShowingNet,
    X_ci: np.ndarray,
    X_r: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    batch_size: int,
    device: torch.device,
    r_keep: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    keep_tensor = None
    if r_keep is not None:
        keep_tensor = torch.from_numpy(r_keep.astype(np.float32)).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_ci[idx].astype(np.float32)), torch.from_numpy(X_r[idx].astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for xb_ci, xb_r in loader:
            pred = model(xb_ci.to(device), xb_r.to(device), r_keep=keep_tensor).detach().cpu().numpy()
            preds.append(pred)
    pred_all = np.concatenate(preds, axis=0)
    y_eval = y[idx]
    mse = float(np.mean((y_eval - pred_all) ** 2))
    r2 = _r2_score_np(y_eval, pred_all)
    return mse, r2, pred_all


def _train_trial(
    *,
    data: ResidualDataset,
    trial_dir: Path,
    lambda_dgate: float,
    dropout_r: float,
    dgate_depth: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    ensure_dir(trial_dir)
    _set_seed(seed, device)
    train_loader, arrays, scalers = _make_loaders(data, int(args.batch_size))
    X_ci_all, X_r_all, y_all = arrays
    ci_mean, ci_std, r_mean, r_std = scalers
    model = DGatedResidualShowingNet(
        ci_dim=len(data.ci_features),
        r_dim=len(data.r_features),
        hidden_dims=args.hidden_dims,
        dgate_depth=dgate_depth,
        dropout_r=dropout_r,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    best = {"val_mse": math.inf, "val_r2": -math.inf, "epoch": 0}
    best_state: dict[str, torch.Tensor] | None = None
    rows = []
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses = []
        for xb_ci, xb_r, yb in train_loader:
            xb_ci = xb_ci.to(device)
            xb_r = xb_r.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb_ci, xb_r)
            mse = nn.functional.mse_loss(pred, yb)
            loss = mse + float(lambda_dgate) * model.dgate_regularizer()
            loss.backward()
            optimizer.step()
            train_losses.append(float(mse.detach().cpu()))
        val_mse, val_r2, _ = _eval_arrays(
            model, X_ci_all, X_r_all, y_all, data.val_idx, int(args.batch_size), device
        )
        gate_norms = model.effective_group_norms().detach().cpu().numpy()
        rows.append(
            {
                "epoch": epoch,
                "train_mse": float(np.mean(train_losses)),
                "val_mse": val_mse,
                "val_r2": val_r2,
                "max_effective_norm": float(np.max(gate_norms)),
                "median_effective_norm": float(np.median(gate_norms)),
            }
        )
        if val_mse < best["val_mse"] - float(args.min_delta):
            best = {"val_mse": val_mse, "val_r2": val_r2, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if int(args.patience) > 0 and epoch >= 50 and stale >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    train_mse, train_r2, _ = _eval_arrays(
        model, X_ci_all, X_r_all, y_all, data.train_idx, int(args.batch_size), device
    )
    val_mse, val_r2, _ = _eval_arrays(
        model, X_ci_all, X_r_all, y_all, data.val_idx, int(args.batch_size), device
    )
    full_val_mse = val_mse
    gate_norms = model.effective_group_norms().detach().cpu().numpy()
    gate_abs = np.abs(model.get_gates().detach().cpu().numpy())
    gate_score = gate_norms / (float(gate_norms.max()) + 1e-12)

    sensitivity_rows = []
    for j, feature in enumerate(data.r_features):
        keep = np.ones(len(data.r_features), dtype=np.float32)
        keep[j] = 0.0
        masked_mse, masked_r2, _ = _eval_arrays(
            model, X_ci_all, X_r_all, y_all, data.val_idx, int(args.batch_size), device, r_keep=keep
        )
        sensitivity_rows.append(
            {
                "feature": feature,
                "mask_mse": masked_mse,
                "mask_r2": masked_r2,
                "mask_sensitivity": masked_mse - full_val_mse,
            }
        )
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sens_pos = np.maximum(sensitivity_df["mask_sensitivity"].to_numpy(dtype=float), 0.0)
    sens_score = sens_pos / (float(np.max(sens_pos)) + 1e-12)
    candidate_df = pd.DataFrame(
        {
            "feature": data.r_features,
            "gate_abs": gate_abs,
            "effective_norm": gate_norms,
            "gate_score": gate_score,
            "mask_sensitivity": sensitivity_df["mask_sensitivity"],
            "mask_sensitivity_score": sens_score,
        }
    )
    candidate_df["combined_score"] = (
        float(args.gate_weight) * candidate_df["gate_score"]
        + (1.0 - float(args.gate_weight)) * candidate_df["mask_sensitivity_score"]
    )
    candidate_df = candidate_df.sort_values("combined_score", ascending=False)

    topk_rows = []
    max_k = min(int(args.max_topk), len(data.r_features))
    ordered_features = candidate_df["feature"].tolist()
    feature_pos = {feature: idx for idx, feature in enumerate(data.r_features)}
    for k in range(1, max_k + 1):
        keep = np.zeros(len(data.r_features), dtype=np.float32)
        for feature in ordered_features[:k]:
            keep[feature_pos[feature]] = 1.0
        mse, r2, _ = _eval_arrays(model, X_ci_all, X_r_all, y_all, data.val_idx, int(args.batch_size), device, r_keep=keep)
        topk_rows.append(
            {
                "topk": k,
                "val_mse": mse,
                "val_r2": r2,
                "mse_ratio_to_full": mse / (full_val_mse + 1e-12),
                "features": ";".join(ordered_features[:k]),
            }
        )
    topk_df = pd.DataFrame(topk_rows)
    ok = topk_df[
        (topk_df["mse_ratio_to_full"] <= 1.0 + float(args.mse_tolerance))
        & (topk_df["topk"] >= int(args.min_active))
        & (topk_df["topk"] <= int(args.max_active))
    ]
    if ok.empty:
        bounded = topk_df[(topk_df["topk"] >= int(args.min_active)) & (topk_df["topk"] <= int(args.max_active))]
        selected_row = bounded.sort_values(["mse_ratio_to_full", "topk"]).iloc[0] if not bounded.empty else topk_df.iloc[-1]
        selection_reason = "best_within_active_bounds"
    else:
        selected_row = ok.sort_values(["topk", "mse_ratio_to_full"]).iloc[0]
        selection_reason = "smallest_topk_within_mse_tolerance"
    active_features = str(selected_row["features"]).split(";") if str(selected_row["features"]) else []

    pd.DataFrame(rows).to_csv(trial_dir / "train_log.csv", index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(trial_dir / "mask_sensitivity.csv", index=False, encoding="utf-8-sig")
    candidate_df.to_csv(trial_dir / "candidate_scores.csv", index=False, encoding="utf-8-sig")
    topk_df.to_csv(trial_dir / "topk_compression.csv", index=False, encoding="utf-8-sig")
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": "DGatedResidualShowingNet",
            "ci_features": data.ci_features,
            "r_features": data.r_features,
            "hidden_dims": list(args.hidden_dims),
            "dgate_depth": int(dgate_depth),
            "dropout_r": float(dropout_r),
            "lambda_dgate": float(lambda_dgate),
            "ci_mean": ci_mean,
            "ci_std": ci_std,
            "r_mean": r_mean,
            "r_std": r_std,
        },
        trial_dir / "residual_showing_model.pt",
    )
    save_json(
        trial_dir / "active_candidates.json",
        {
            "active_features": active_features,
            "active_count": len(active_features),
            "selection_reason": selection_reason,
            "selected_topk": int(selected_row["topk"]),
            "selected_val_mse": float(selected_row["val_mse"]),
            "selected_val_r2": float(selected_row["val_r2"]),
            "selected_mse_ratio_to_full": float(selected_row["mse_ratio_to_full"]),
        },
    )
    return {
        "trial": trial_dir.name,
        "trial_dir": str(trial_dir.resolve()),
        "lambda_dgate": float(lambda_dgate),
        "dropout_r": float(dropout_r),
        "dgate_depth": int(dgate_depth),
        "best_epoch": int(best["epoch"]),
        "train_mse": train_mse,
        "train_r2": train_r2,
        "val_mse": val_mse,
        "val_r2": val_r2,
        "active_count": len(active_features),
        "active_features": ";".join(active_features),
        "selection_reason": selection_reason,
        "selected_topk": int(selected_row["topk"]),
        "selected_val_mse": float(selected_row["val_mse"]),
        "selected_val_r2": float(selected_row["val_r2"]),
        "selected_mse_ratio_to_full": float(selected_row["mse_ratio_to_full"]),
    }


def _plot_active_scores(candidate_path: Path, out_path: Path, top_n: int = 20) -> None:
    df = pd.read_csv(candidate_path).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4.5, 0.32 * len(df))))
    ax.barh(df["feature"], df["combined_score"], color="#4c78a8")
    ax.set_xlabel("Combined score")
    ax.set_title("Stage 04 Active Candidate Scores")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_gate_sensitivity(candidate_path: Path, out_path: Path) -> None:
    df = pd.read_csv(candidate_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["gate_score"], df["mask_sensitivity_score"], s=36, alpha=0.75)
    ax.set_xlabel("D-gating effective-norm score")
    ax.set_ylabel("Mask-sensitivity score")
    ax.set_title("Gate vs Mask Sensitivity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_report(run_dir: Path, xi: str, best: dict[str, Any], trial_count: int) -> None:
    lines = [
        "# 阶段 04：条件化残差补偿显影",
        "",
        "## 目标",
        "",
        "本阶段只针对阶段 03 已完成的一个字段执行：在固定 `C_i = P \\ {x_i}` 的条件下，让路径外候选字段 `R` 去预测标准化残差 `residual_std`。",
        "",
        "模型使用轻微 D-gating 作用在 `R` 的第一层有效权重上，同时只对 `R` 做随机 dropout。这里的目标不是最终替代认证，而是显影可能补偿 `x_i` 缺口的活跃候选集 `A`。",
        "",
        "## 本次字段",
        "",
        f"- xi: `{xi}`",
        f"- sweep trial 数: `{trial_count}`",
        "",
        "## 选中的显影结果",
        "",
        f"- trial: `{best['trial']}`",
        f"- lambda_dgate: `{best['lambda_dgate']}`",
        f"- R dropout: `{best['dropout_r']}`",
        f"- dgate_depth: `{best['dgate_depth']}`",
        f"- full-R validation MSE: `{best['val_mse']:.6f}`",
        f"- full-R validation R2: `{best['val_r2']:.6f}`",
        f"- active_count: `{best['active_count']}`",
        f"- selected_topk MSE/full MSE: `{best['selected_mse_ratio_to_full']:.6f}`",
        "",
        "## 活跃候选集 A",
        "",
        *(f"- `{feature}`" for feature in str(best["active_features"]).split(";") if feature),
        "",
        "## 产物",
        "",
        "- `stage04_active_candidate_interface.json`: 给下一阶段使用的标准接口。",
        "- `xi=<field>/01_data/data_summary.json`: 数据、字段、划分说明。",
        "- `xi=<field>/02_sweep/sweep_summary.csv`: 多强度 D-gating 显影结果。",
        "- `xi=<field>/02_sweep/trial_*/candidate_scores.csv`: 每个候选 R 字段的 gate、遮蔽敏感度和 combined score。",
        "- `xi=<field>/02_sweep/trial_*/topk_compression.csv`: 按 combined score 逐步压缩 A 的验证。",
        "- `xi=<field>/03_active_set/active_set.json`: 本阶段推荐的活跃候选集 A。",
    ]
    (run_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage04: residual compensation showing with light D-gating.")
    parser.add_argument(
        "--stage03-interface",
        default=str(
            PROJECT_ROOT
            / "conditional_residual_compensation_outputs"
            / "CenterOn_net_actual_interchange_mw"
            / "stage03_oof_residuals"
            / "run_20260705_231558"
            / "stage03_oof_residual_interface.json"
        ),
    )
    parser.add_argument("--xi", default=None)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "conditional_residual_compensation_outputs"))
    parser.add_argument("--stage-dir", default="stage04_residual_showing")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0007)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[96, 48, 24])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda-sweep", nargs="+", type=float, default=[1e-6, 3e-6, 1e-5, 3e-5])
    parser.add_argument("--dropout-sweep", nargs="+", type=float, default=[0.30, 0.45])
    parser.add_argument("--dgate-depth", type=int, default=3)
    parser.add_argument("--gate-weight", type=float, default=0.5)
    parser.add_argument("--mse-tolerance", type=float, default=0.15)
    parser.add_argument("--min-active", type=int, default=3)
    parser.add_argument("--max-active", type=int, default=24)
    parser.add_argument("--max-topk", type=int, default=24)
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage03 = _read_json(Path(args.stage03_interface))
    if args.xi:
        xi = str(args.xi)
        xi_interface = stage03["xi_interfaces"].get(xi)
    else:
        xi = str(stage03["executed_xi_features"][0])
        xi_interface = stage03["xi_interfaces"][xi]
    if not xi_interface:
        raise ValueError(f"No completed stage03 xi_interface for xi={xi!r}.")

    data = _load_dataset(Path(xi_interface), cfg, args)
    center = data.center
    stage_root = Path(args.output_root).resolve() / f"CenterOn_{center}" / args.stage_dir
    run_dir = stage_root / (args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    xi_dir = ensure_dir(run_dir / f"xi={safe_name(data.xi)}")
    data_dir = ensure_dir(xi_dir / "01_data")
    sweep_dir = ensure_dir(xi_dir / "02_sweep")
    active_dir = ensure_dir(xi_dir / "03_active_set")
    save_json(
        data_dir / "data_summary.json",
        {
            "xi": data.xi,
            "center": data.center,
            "ci_features": data.ci_features,
            "residual_candidate_features": data.r_features,
            "train_rows": int(len(data.train_idx)),
            "val_rows": int(len(data.val_idx)),
            "source_stage03_xi_interface": str(Path(xi_interface).resolve()),
        },
    )
    save_json(
        run_dir / "stage_config.json",
        {
            "stage": "stage04_residual_showing",
            "stage03_interface": str(Path(args.stage03_interface).resolve()),
            "xi": data.xi,
            "center": data.center,
            "lambda_sweep": args.lambda_sweep,
            "dropout_sweep": args.dropout_sweep,
            "dgate_depth": args.dgate_depth,
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden_dims": args.hidden_dims,
                "patience": args.patience,
                "device": args.device,
            },
            "active_selection": {
                "gate_weight": args.gate_weight,
                "mse_tolerance": args.mse_tolerance,
                "min_active": args.min_active,
                "max_active": args.max_active,
                "max_topk": args.max_topk,
            },
        },
    )

    device = torch.device(args.device)
    trial_rows = []
    trial_id = 0
    for lambda_dgate in args.lambda_sweep:
        for dropout_r in args.dropout_sweep:
            trial_id += 1
            trial_name = f"trial_{trial_id:02d}_lam{str(lambda_dgate).replace('.', 'p')}_drop{str(dropout_r).replace('.', 'p')}"
            trial_rows.append(
                _train_trial(
                    data=data,
                    trial_dir=sweep_dir / trial_name,
                    lambda_dgate=float(lambda_dgate),
                    dropout_r=float(dropout_r),
                    dgate_depth=int(args.dgate_depth),
                    args=args,
                    device=device,
                    seed=int(args.seed) + trial_id * 1009,
                )
            )
            pd.DataFrame(trial_rows).to_csv(sweep_dir / "sweep_summary.csv", index=False, encoding="utf-8-sig")

    sweep_df = pd.DataFrame(trial_rows)
    bounded = sweep_df[(sweep_df["active_count"] >= int(args.min_active)) & (sweep_df["active_count"] <= int(args.max_active))]
    if bounded.empty:
        best = sweep_df.sort_values(["selected_mse_ratio_to_full", "active_count"]).iloc[0].to_dict()
    else:
        best = bounded.sort_values(["selected_mse_ratio_to_full", "active_count"]).iloc[0].to_dict()
    best_trial_dir = Path(best["trial_dir"])
    best_candidate_path = best_trial_dir / "candidate_scores.csv"
    best_topk_path = best_trial_dir / "topk_compression.csv"
    best_active = [feature for feature in str(best["active_features"]).split(";") if feature]

    candidate_df = pd.read_csv(best_candidate_path)
    active_df = candidate_df[candidate_df["feature"].isin(best_active)].copy()
    active_df.to_csv(active_dir / "active_candidates.csv", index=False, encoding="utf-8-sig")
    pd.read_csv(best_topk_path).to_csv(active_dir / "selected_trial_topk_compression.csv", index=False, encoding="utf-8-sig")
    save_json(
        active_dir / "active_set.json",
        {
            "xi": data.xi,
            "active_features": best_active,
            "active_count": len(best_active),
            "selected_trial": best["trial"],
            "selected_trial_dir": best["trial_dir"],
            "selection_metric": "minimum selected_mse_ratio_to_full within active count bounds",
            "selected_mse_ratio_to_full": float(best["selected_mse_ratio_to_full"]),
            "selected_val_mse": float(best["selected_val_mse"]),
            "selected_val_r2": float(best["selected_val_r2"]),
            "full_r_val_mse": float(best["val_mse"]),
            "full_r_val_r2": float(best["val_r2"]),
        },
    )
    _plot_active_scores(best_candidate_path, active_dir / "active_candidate_scores.png")
    _plot_gate_sensitivity(best_candidate_path, active_dir / "gate_vs_mask_sensitivity.png")

    save_json(
        run_dir / "stage04_active_candidate_interface.json",
        {
            "schema_version": 1,
            "stage": "stage04_residual_showing",
            "center": data.center,
            "target": data.center,
            "source_stage03_interface": str(Path(args.stage03_interface).resolve()),
            "run_dir": str(run_dir.resolve()),
            "xi": data.xi,
            "ci_features": data.ci_features,
            "residual_candidate_features": data.r_features,
            "active_candidate_features": best_active,
            "active_count": len(best_active),
            "selected_trial": best["trial"],
            "selected_trial_dir": best["trial_dir"],
            "active_set_json": str((active_dir / "active_set.json").resolve()),
            "active_candidates_csv": str((active_dir / "active_candidates.csv").resolve()),
            "candidate_scores_csv": str(best_candidate_path.resolve()),
            "topk_compression_csv": str(best_topk_path.resolve()),
            "sweep_summary_csv": str((sweep_dir / "sweep_summary.csv").resolve()),
        },
    )
    _write_report(run_dir, data.xi, best, trial_count=len(trial_rows))
    print(f"Stage 04 run saved to {run_dir}")


if __name__ == "__main__":
    main()
