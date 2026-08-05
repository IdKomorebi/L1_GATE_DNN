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
from src.models import make_mlp


@dataclass
class SplitData:
    selected_index: np.ndarray
    train_pos: np.ndarray
    test_pos: np.ndarray
    all_features: list[str]
    y_raw: np.ndarray
    x_raw: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    return ((apply - mean) / std).astype(np.float32), mean.squeeze(), std.squeeze()


def _make_train_test_split(
    *,
    cfg: dict[str, Any],
    center: str,
    all_features: Sequence[str],
    exclude_columns: Sequence[str],
) -> SplitData:
    dataset_cfg = cfg["dataset"]
    training = cfg.get("training", {})
    preprocessing = cfg.get("preprocessing", {})
    data_path = resolve_project_path(cfg, dataset_cfg["processed_csv"])
    df = read_numeric_csv(
        data_path,
        drop_all_zero_columns=bool(preprocessing.get("drop_all_zero_columns", False)),
        exclude_columns=exclude_columns,
    )
    missing = [col for col in [center, *all_features] if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    selected = df[[center, *all_features]].dropna(axis=0, how="any")
    n = len(selected)
    if n < 5:
        raise ValueError(f"Not enough rows for split: {n}")

    rng = np.random.default_rng(int(training.get("random_state", 42)))
    perm = rng.permutation(n)
    train_ratio = float(training.get("train_ratio", 0.8))
    train_size = max(1, min(n - 1, int(n * train_ratio)))
    train_pos = perm[:train_size]
    test_pos = perm[train_size:]

    return SplitData(
        selected_index=selected.index.to_numpy(),
        train_pos=train_pos,
        test_pos=test_pos,
        all_features=list(all_features),
        y_raw=selected[center].to_numpy(dtype=np.float32).reshape(-1, 1),
        x_raw=selected[list(all_features)].to_numpy(dtype=np.float32),
    )


def _make_kfold_indices(n: int, k: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if k < 2:
        raise ValueError("k_folds must be >= 2.")
    if n < k:
        raise ValueError(f"Not enough train rows {n} for k={k}.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    result = []
    all_idx = np.arange(n)
    for valid_idx in folds:
        mask = np.ones(n, dtype=bool)
        mask[valid_idx] = False
        train_idx = all_idx[mask]
        result.append((train_idx, valid_idx))
    return result


def _set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _eval_model(
    model: nn.Module,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    X_std = ((X_raw - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_std)), batch_size=batch_size, shuffle=False)
    preds_std = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            pred = model(xb.to(device)).detach().cpu().numpy()
            preds_std.append(pred)
    pred_std = np.concatenate(preds_std, axis=0)
    pred_raw = pred_std * y_std + y_mean
    mse = float(np.mean((y_raw.reshape(-1, 1) - pred_raw.reshape(-1, 1)) ** 2))
    r2 = _r2_score_np(y_raw, pred_raw)
    return mse, r2, pred_raw.reshape(-1, 1)


def _train_mlp_raw(
    *,
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    X_eval_raw: np.ndarray,
    y_eval_raw: np.ndarray,
    hidden_dims: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    patience: int,
    min_delta: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    _set_seed(seed, device)
    X_train_std, x_mean, x_std = _standardize(X_train_raw, X_train_raw)
    y_mean = float(y_train_raw.mean())
    y_std = float(y_train_raw.std() + 1e-8)
    y_train_std = ((y_train_raw - y_mean) / y_std).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_std.astype(np.float32)), torch.from_numpy(y_train_std)),
        batch_size=batch_size,
        shuffle=True,
    )
    model = make_mlp(X_train_raw.shape[1], hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best = {"r2": -math.inf, "mse": math.inf, "epoch": 0}
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
        eval_mse, eval_r2, _ = _eval_model(
            model, X_eval_raw, y_eval_raw, x_mean, x_std, y_mean, y_std, batch_size, device
        )
        if eval_r2 > best["r2"] + min_delta:
            best = {"r2": float(eval_r2), "mse": float(eval_mse), "epoch": int(epoch)}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if patience > 0 and epoch >= 50 and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_mse, final_r2, pred_eval_raw = _eval_model(
        model, X_eval_raw, y_eval_raw, x_mean, x_std, y_mean, y_std, batch_size, device
    )
    train_mse, train_r2, pred_train_raw = _eval_model(
        model, X_train_raw, y_train_raw, x_mean, x_std, y_mean, y_std, batch_size, device
    )
    return {
        "model": model,
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": y_mean,
        "y_std": y_std,
        "best_eval_r2": best["r2"],
        "best_eval_mse": best["mse"],
        "best_epoch": best["epoch"],
        "final_train_mse": float(train_mse),
        "final_train_r2": float(train_r2),
        "final_eval_mse": float(final_mse),
        "final_eval_r2": float(final_r2),
        "pred_train_raw": pred_train_raw,
        "pred_eval_raw": pred_eval_raw,
    }


def _plot_oof(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true.reshape(-1), y_pred.reshape(-1), s=12, alpha=0.55)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    ax.plot([low, high], [low, high], color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_xlabel("y true")
    ax.set_ylabel("OOF f_C(C_i) prediction")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_residual(residual: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residual.reshape(-1), bins=40, color="#4c78a8", alpha=0.85)
    ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel("OOF residual: y - f_C(C_i)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_xi_pending(xi_dir: Path, xi: str, reason: str) -> None:
    ensure_dir(xi_dir)
    save_json(
        xi_dir / "status.json",
        {
            "xi": xi,
            "status": "pending",
            "reason": reason,
        },
    )


def _run_one_xi(
    *,
    xi: str,
    xi_dir: Path,
    split: SplitData,
    center: str,
    main_path: list[str],
    residual_features: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    ensure_dir(xi_dir)
    ci_features = [feature for feature in main_path if feature != xi]
    feature_to_idx = {feature: idx for idx, feature in enumerate(split.all_features)}
    ci_idx = [feature_to_idx[feature] for feature in ci_features]
    X_train_raw = split.x_raw[split.train_pos][:, ci_idx]
    y_train_raw = split.y_raw[split.train_pos]
    X_test_raw = split.x_raw[split.test_pos][:, ci_idx]
    y_test_raw = split.y_raw[split.test_pos]

    kfolds = _make_kfold_indices(len(split.train_pos), int(args.k_folds), int(args.seed))
    oof_pred = np.zeros_like(y_train_raw, dtype=np.float32)
    fold_rows = []
    for fold_id, (fold_train_idx, fold_valid_idx) in enumerate(kfolds, start=1):
        fit = _train_mlp_raw(
            X_train_raw=X_train_raw[fold_train_idx],
            y_train_raw=y_train_raw[fold_train_idx],
            X_eval_raw=X_train_raw[fold_valid_idx],
            y_eval_raw=y_train_raw[fold_valid_idx],
            hidden_dims=args.hidden_dims,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            dropout=float(args.dropout),
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            seed=int(args.seed) + fold_id * 1009,
            device=device,
        )
        oof_pred[fold_valid_idx] = fit["pred_eval_raw"].astype(np.float32)
        fold_rows.append(
            {
                "xi": xi,
                "fold": fold_id,
                "train_rows": int(len(fold_train_idx)),
                "valid_rows": int(len(fold_valid_idx)),
                "best_valid_r2": fit["best_eval_r2"],
                "best_valid_mse": fit["best_eval_mse"],
                "best_epoch": fit["best_epoch"],
                "final_train_r2": fit["final_train_r2"],
                "final_train_mse": fit["final_train_mse"],
                "final_valid_r2": fit["final_eval_r2"],
                "final_valid_mse": fit["final_eval_mse"],
            }
        )

    residual = (y_train_raw - oof_pred).astype(np.float32)
    residual_mean = float(residual.mean())
    residual_std_value = float(residual.std() + 1e-8)
    residual_std = ((residual - residual_mean) / residual_std_value).astype(np.float32)
    gap_energy = float(np.var(residual) / (np.var(y_train_raw) + 1e-12))
    oof_r2 = _r2_score_np(y_train_raw, oof_pred)
    oof_mse = float(np.mean((y_train_raw - oof_pred) ** 2))

    full_fit = _train_mlp_raw(
        X_train_raw=X_train_raw,
        y_train_raw=y_train_raw,
        X_eval_raw=X_test_raw,
        y_eval_raw=y_test_raw,
        hidden_dims=args.hidden_dims,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        dropout=float(args.dropout),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        seed=int(args.seed) + 90001,
        device=device,
    )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(xi_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")

    train_source_rows = split.selected_index[split.train_pos]
    oof_df = pd.DataFrame(
        {
            "source_row_index": train_source_rows.astype(int),
            "train_position": np.arange(len(train_source_rows), dtype=int),
            "xi": xi,
            "y_true": y_train_raw.reshape(-1),
            "y_pred_C_oof": oof_pred.reshape(-1),
            "residual": residual.reshape(-1),
            "residual_std": residual_std.reshape(-1),
        }
    )
    oof_df.to_csv(xi_dir / "oof_residuals.csv", index=False, encoding="utf-8-sig")

    residual_stats = {
        "xi": xi,
        "target": center,
        "train_rows": int(len(y_train_raw)),
        "test_rows": int(len(y_test_raw)),
        "k_folds": int(args.k_folds),
        "oof_r2": float(oof_r2),
        "oof_mse": float(oof_mse),
        "residual_mean": residual_mean,
        "residual_std_value": residual_std_value,
        "residual_variance": float(np.var(residual)),
        "y_train_variance": float(np.var(y_train_raw)),
        "gap_energy": gap_energy,
    }
    save_json(xi_dir / "residual_stats.json", residual_stats)

    final_metrics = {
        "xi": xi,
        "model": "f_C_full",
        "input_feature_count": len(ci_features),
        "train_r2": full_fit["final_train_r2"],
        "train_mse": full_fit["final_train_mse"],
        "test_r2": full_fit["final_eval_r2"],
        "test_mse": full_fit["final_eval_mse"],
        "best_test_r2": full_fit["best_eval_r2"],
        "best_test_mse": full_fit["best_eval_mse"],
        "best_epoch": full_fit["best_epoch"],
    }
    pd.DataFrame([final_metrics]).to_csv(xi_dir / "final_model_metrics.csv", index=False, encoding="utf-8-sig")

    torch.save(
        {
            "model_state": full_fit["model"].state_dict(),
            "model_type": "make_mlp",
            "hidden_dims": list(args.hidden_dims),
            "dropout": float(args.dropout),
            "target": center,
            "xi": xi,
            "ci_features": ci_features,
            "x_mean": full_fit["x_mean"],
            "x_std": full_fit["x_std"],
            "y_mean": full_fit["y_mean"],
            "y_std": full_fit["y_std"],
            "residual_mean": residual_mean,
            "residual_std_value": residual_std_value,
        },
        xi_dir / "f_C_full_model.pt",
    )

    save_json(
        xi_dir / "xi_interface.json",
        {
            "schema_version": 1,
            "stage": "stage03_oof_residuals",
            "status": "complete",
            "target": center,
            "xi": xi,
            "main_path_features": main_path,
            "ci_features": ci_features,
            "residual_candidate_features": residual_features,
            "oof_residuals_csv": str((xi_dir / "oof_residuals.csv").resolve()),
            "residual_stats_json": str((xi_dir / "residual_stats.json").resolve()),
            "fold_metrics_csv": str((xi_dir / "fold_metrics.csv").resolve()),
            "f_C_full_model_pt": str((xi_dir / "f_C_full_model.pt").resolve()),
            "final_model_metrics_csv": str((xi_dir / "final_model_metrics.csv").resolve()),
            "residual_target_column": "residual_std",
            "raw_residual_column": "residual",
            "residual_mean": residual_mean,
            "residual_std_value": residual_std_value,
            "gap_energy": gap_energy,
        },
    )
    save_json(xi_dir / "status.json", {"xi": xi, "status": "complete"})

    _plot_oof(y_train_raw, oof_pred, xi_dir / "oof_predictions.png", f"OOF f_C for {xi}")
    _plot_residual(residual, xi_dir / "oof_residual_hist.png", f"OOF residual for {xi}")

    return {
        **residual_stats,
        **{f"final_{k}": v for k, v in final_metrics.items() if k not in {"xi", "model"}},
        "xi_dir": str(xi_dir.resolve()),
        "ci_feature_count": len(ci_features),
    }


def _write_report(run_dir: Path, selected_xi: str, summary: dict[str, Any], executed_count: int, total_xi: int) -> None:
    lines = [
        "# 阶段 03：OOF 残差接口生成",
        "",
        "## 目标",
        "",
        "本阶段为每个可替代主路径字段 `x_i` 生成删除该字段后的 out-of-fold 残差目标。",
        "",
        "固定主路径为剪枝后的 9 字段集合 `P`。对每个 `x_i`，上下文为 `C_i = P \\ {x_i}`，先训练 `f_C(C_i) -> y`，再用未见过该样本的折外预测构造残差 `e_i = y - f_C(C_i)`。",
        "",
        "## 本次执行范围",
        "",
        f"- 可替代字段总数: `{total_xi}`",
        f"- 本次实际执行字段数: `{executed_count}`",
        f"- 本次执行字段: `{selected_xi}`",
        "",
        "第一次只跑一个字段，用来确认目录结构、残差文件、标准化参数和后续接口是否完整。",
        "",
        "## 核心结果",
        "",
        f"- OOF R2: `{summary['oof_r2']:.6f}`",
        f"- OOF MSE: `{summary['oof_mse']:.6f}`",
        f"- residual_mean: `{summary['residual_mean']:.6f}`",
        f"- residual_std_value: `{summary['residual_std_value']:.6f}`",
        f"- gap_energy = Var(residual) / Var(y): `{summary['gap_energy']:.6f}`",
        f"- f_C_full test R2: `{summary['final_test_r2']:.6f}`",
        "",
        "## 产物",
        "",
        "- `stage03_oof_residual_interface.json`: 给下一阶段使用的标准接口。",
        "- `processed_xi_summary.csv`: 本 run 已执行字段的摘要。",
        f"- `xi={safe_name(selected_xi)}/oof_residuals.csv`: 训练集 OOF 残差，后续残差补偿网络应使用 `residual_std` 作为目标。",
        f"- `xi={safe_name(selected_xi)}/residual_stats.json`: 残差标准化参数与 gap energy。",
        f"- `xi={safe_name(selected_xi)}/f_C_full_model.pt`: 使用完整训练集训练的最终 `f_C_full`。",
        f"- `xi={safe_name(selected_xi)}/xi_interface.json`: 单字段接口。",
    ]
    (run_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage03: generate OOF residuals for replaceable main-path fields.")
    parser.add_argument(
        "--stage02-interface",
        default=str(
            PROJECT_ROOT
            / "conditional_residual_compensation_outputs"
            / "CenterOn_net_actual_interchange_mw"
            / "stage02_replaceability"
            / "run_20260705_220511"
            / "stage02_replaceability_interface.json"
        ),
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "conditional_residual_compensation_outputs"),
    )
    parser.add_argument("--stage-dir", default="stage03_oof_residuals")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--xi", default=None, help="Specific replaceable field to run. Defaults to lowest C_i R2.")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32, 16])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage02 = _read_json(Path(args.stage02_interface))
    center = str(stage02["center"])
    main_path = list(stage02.get("repruned_main_path_features") or stage02["main_path_features"])
    replaceable = list(stage02.get("repruned_replaceable_features") or stage02["replaceable_features"])
    all_features = list(stage02["repruned_residual_candidate_features"])
    for feature in main_path:
        if feature not in all_features:
            all_features.append(feature)
    residual_features = [feature for feature in all_features if feature not in set(main_path)]

    summary_csv = stage02.get("repruned_replaceability_summary_csv")
    if args.xi:
        selected_xi = str(args.xi)
    elif summary_csv and Path(summary_csv).exists():
        summary_df = pd.read_csv(summary_csv)
        summary_df = summary_df[summary_df["target_feature"].isin(replaceable)].sort_values("C_i_r2")
        selected_xi = str(summary_df.iloc[0]["target_feature"])
    else:
        selected_xi = replaceable[0]
    if selected_xi not in replaceable:
        raise ValueError(f"Selected xi={selected_xi!r} is not in replaceable features: {replaceable}")

    dgate_config = _read_json(Path(stage02["source_stage01_interface"]))
    exclude_columns = []
    dgate_run_dir = dgate_config.get("dgate_run_dir")
    if dgate_run_dir:
        exclude_columns = normalize_column_list(
            _read_json(Path(dgate_run_dir) / "config.json").get("preprocessing", {}).get("exclude_columns")
        )

    stage_root = Path(args.output_root).resolve() / f"CenterOn_{center}" / args.stage_dir
    run_dir = stage_root / (args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    split = _make_train_test_split(cfg=cfg, center=center, all_features=all_features, exclude_columns=exclude_columns)
    device = torch.device(args.device)

    xi_dirs = {}
    for xi in replaceable:
        xi_dir = run_dir / f"xi={safe_name(xi)}"
        xi_dirs[xi] = xi_dir
        if xi != selected_xi:
            _write_xi_pending(xi_dir, xi, "not_run_in_first_stage03_trial")

    save_json(
        run_dir / "stage_config.json",
        {
            "stage": "stage03_oof_residuals",
            "stage02_interface": str(Path(args.stage02_interface).resolve()),
            "center": center,
            "main_path_features": main_path,
            "replaceable_features": replaceable,
            "selected_xi": selected_xi,
            "executed_xi_count": 1,
            "k_folds": args.k_folds,
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden_dims": args.hidden_dims,
                "dropout": args.dropout,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "seed": args.seed,
                "device": str(device),
            },
            "split": {
                "train_rows": int(len(split.train_pos)),
                "test_rows": int(len(split.test_pos)),
            },
        },
    )

    summary = _run_one_xi(
        xi=selected_xi,
        xi_dir=xi_dirs[selected_xi],
        split=split,
        center=center,
        main_path=main_path,
        residual_features=residual_features,
        args=args,
        device=device,
    )
    pd.DataFrame([summary]).to_csv(run_dir / "processed_xi_summary.csv", index=False, encoding="utf-8-sig")

    xi_interfaces = {
        xi: str((xi_dirs[xi] / "xi_interface.json").resolve()) if xi == selected_xi else None for xi in replaceable
    }
    save_json(
        run_dir / "stage03_oof_residual_interface.json",
        {
            "schema_version": 1,
            "stage": "stage03_oof_residuals",
            "center": center,
            "target": center,
            "source_stage02_interface": str(Path(args.stage02_interface).resolve()),
            "run_dir": str(run_dir.resolve()),
            "main_path_features": main_path,
            "replaceable_features": replaceable,
            "residual_candidate_features": residual_features,
            "executed_xi_features": [selected_xi],
            "pending_xi_features": [xi for xi in replaceable if xi != selected_xi],
            "xi_interfaces": xi_interfaces,
            "processed_xi_summary_csv": str((run_dir / "processed_xi_summary.csv").resolve()),
        },
    )
    _write_report(run_dir, selected_xi, summary, executed_count=1, total_xi=len(replaceable))
    print(f"Stage 03 run saved to {run_dir}")


if __name__ == "__main__":
    main()
