from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_center_spec, resolve_project_path
from src.data_utils import ensure_dir, normalize_column_list, prepare_supervised_dataset, save_json
from src.models import DGatingRegressor, DNNRegressor, SimpleAdam
from src.trainer import r2_score, train_center_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CERTIFIED_FIXED9_RUN = (
    PROJECT_ROOT
    / "outputs"
    / "data2025_Processed_V2"
    / "CenterOn_net_actual_interchange_mw"
    / "MinimalSubstitutionHardGateDNN"
    / "run_20260704_fixed9_main095_sub094_final"
)
MINIMAL_SUBSTITUTION_SCRIPT = PROJECT_ROOT / "scripts" / "12_train_minimal_substitution_paths.py"


def _artifact_dirs(run_dir: Path) -> dict[str, Path]:
    return {
        "dgating": ensure_dir(run_dir / "01_dgating"),
        "pruning": ensure_dir(run_dir / "02_pruning"),
        "validation": ensure_dir(run_dir / "03_validation"),
    }


def _eval_model(model: torch.nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    ys = []
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            total_loss += float(loss.detach()) * xb.size(0)
            ys.append(yb.cpu())
            preds.append(pred.cpu())
    y_cat = torch.cat(ys, dim=0)
    pred_cat = torch.cat(preds, dim=0)
    return total_loss / len(loader.dataset), r2_score(y_cat, pred_cat)


def _train_dnn_subset(
    cfg: dict[str, Any],
    center: str,
    features: list[str],
    exclude_columns: list[str],
    label: str,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    training = cfg.get("training", {})
    dataset_cfg = cfg["dataset"]
    preprocessing = cfg.get("preprocessing", {})
    bundle = prepare_supervised_dataset(
        data_path=resolve_project_path(cfg, dataset_cfg["processed_csv"]),
        center=center,
        features=features,
        train_ratio=float(training.get("train_ratio", 0.8)),
        random_state=int(training.get("random_state", 42)),
        drop_all_zero_columns=bool(preprocessing.get("drop_all_zero_columns", False)),
        exclude_columns=exclude_columns,
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.X_train), torch.from_numpy(bundle.y_train)),
        batch_size=int(training.get("batch_size", 50)),
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.X_test), torch.from_numpy(bundle.y_test)),
        batch_size=int(training.get("batch_size", 50)),
        shuffle=False,
    )
    model = DNNRegressor(len(features), training.get("hidden_dims", [64, 32, 16])).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get("lr", 0.001)))

    best_test_r2 = -1e18
    best_test_mse = float("nan")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    patience = 30
    eval_every = 5
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
        if epoch % eval_every != 0 and epoch != epochs:
            continue
        test_mse, test_r2 = _eval_model(model, test_loader)
        if test_r2 > best_test_r2 + 1e-6:
            best_test_r2 = test_r2
            best_test_mse = test_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch >= 80 and no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_mse, train_r2 = _eval_model(model, train_loader)
    test_mse, test_r2 = _eval_model(model, test_loader)
    return {
        "label": label,
        "feature_count": len(features),
        "features": ";".join(features),
        "best_test_r2": float(best_test_r2),
        "best_test_mse": float(best_test_mse),
        "best_epoch": int(best_epoch),
        "last_train_r2": float(train_r2),
        "last_train_mse": float(train_mse),
        "last_test_r2": float(test_r2),
        "last_test_mse": float(test_mse),
    }


def _plot_topk(topk_df: pd.DataFrame, main_k: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(topk_df["k"], topk_df["best_test_r2"], marker="o", linewidth=2)
    ax.axvline(main_k, color="#d62728", linestyle="--", linewidth=1.5, label=f"main path k={main_k}")
    ax.set_xlabel("Top-k fields by D-gating effective norm")
    ax.set_ylabel("Best test R2")
    ax.set_title("Top-k Real-Input Validation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_drop(drop_df: pd.DataFrame, main_r2: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = drop_df.sort_values("best_test_r2", ascending=False)
    ax.bar(range(len(ordered)), ordered["best_test_r2"], color="#4c78a8")
    ax.axhline(main_r2, color="#d62728", linestyle="--", linewidth=1.5, label="selected main path")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["dropped_feature"], rotation=45, ha="right")
    ax.set_ylabel("Best test R2")
    ax.set_title("Drop-One Masked Validation From Main Path")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_random(random_df: pd.DataFrame, main_r2: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(random_df["trial"], random_df["best_test_r2"], color="#59a14f", alpha=0.8)
    ax.axhline(main_r2, color="#d62728", linestyle="--", linewidth=1.5, label="D-gating top-9")
    ax.set_xlabel("Random 9-field trial")
    ax.set_ylabel("Best test R2")
    ax.set_title("Random Same-Size Subset Validation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pruning_path(path_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = path_df[path_df["label"] != "full_all_features"].copy()
    plot_df = plot_df.sort_values("feature_count", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_df["feature_count"], plot_df["best_test_r2"], marker="o", linewidth=2)
    ax.invert_xaxis()
    ax.set_xlabel("Retained field count")
    ax.set_ylabel("Masked test R2")
    ax.set_title("Greedy Pruning Path From D-gating Retained Fields")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_minimal_substitution_module() -> Any:
    spec = importlib.util.spec_from_file_location("minimal_substitution_stage01", MINIMAL_SUBSTITUTION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MINIMAL_SUBSTITUTION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_incremental_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_report(
    run_dir: Path,
    main_path: list[str],
    dgate_metrics: dict[str, Any],
    strictness_summary: dict[str, Any],
    topk_df: pd.DataFrame,
    drop_df: pd.DataFrame,
    random_df: pd.DataFrame,
    main_vs_full_df: pd.DataFrame,
    compact_df: pd.DataFrame | None,
    params: dict[str, Any],
) -> None:
    top9_row = topk_df.loc[topk_df["k"] == len(main_path)].iloc[0]
    top8_candidates = topk_df.loc[topk_df["k"] == len(main_path) - 1] if len(main_path) > 1 else pd.DataFrame()
    top8_row = top8_candidates.iloc[0] if not top8_candidates.empty else None
    random_best = float(random_df["best_test_r2"].max()) if not random_df.empty else float("nan")
    random_median = float(random_df["best_test_r2"].median()) if not random_df.empty else float("nan")
    min_drop = float(drop_df["best_test_r2"].min()) if not drop_df.empty else float("nan")
    max_drop = float(drop_df["best_test_r2"].max()) if not drop_df.empty else float("nan")
    full_row = main_vs_full_df.loc[main_vs_full_df["label"] == "full_all_features"].iloc[0]
    retained_candidates = main_vs_full_df.loc[main_vs_full_df["label"] == "dgate_retained11"]
    retained_r2 = float(retained_candidates.iloc[0]["best_test_r2"]) if not retained_candidates.empty else None

    lines = [
        "# 阶段 01：主路径识别与剪枝验证",
        "",
        "## 目标",
        "",
        "本阶段为后续条件化残差补偿流程固定第一个标准接口：中心变量 `net_actual_interchange_mw` 的主路径字段集合 `P`。",
        "",
        "逻辑顺序是：先用 D-gating 得到客观保留字段，再在这些字段内部做逐字段剪枝比较，最后对剪枝阶段选出的主路径做逐字段阻断验证。",
        "",
        "## D-gating Compression",
        "",
        f"- model: DGatingDNN",
        f"- lambda_dgate: {params['lambda_dgate']}",
        f"- dgate_depth: {params['dgate_depth']}",
        f"- dgate_normalize_lambda_by_depth: {params['dgate_normalize_lambda_by_depth']}",
        f"- active_threshold: {params['active_threshold']}",
        f"- best_test_r2: {dgate_metrics.get('best_test_r2')}",
        f"- best_epoch: {dgate_metrics.get('best_epoch')}",
        f"- selected_by_active_threshold_count: {strictness_summary.get('selected_by_active_threshold_count')}",
        f"- largest_log_gap_selected_count: {strictness_summary.get('largest_log_gap_selected_count')}",
        "",
        "## 标准接口",
        "",
        "后续阶段应读取 `stage01_main_path_interface.json`。",
        "",
        "当前阶段选出的主路径字段：",
        "",
    ]
    lines.extend(f"{idx}. `{feature}`" for idx, feature in enumerate(main_path, start=1))
    lines.extend(
        [
            "",
            "## 结果分析",
            "",
            f"- 全量 55 字段 masked R2: {float(full_row['best_test_r2']):.6f}",
            f"- 阶段主路径 {len(main_path)} 字段 masked R2: {float(top9_row['best_test_r2']):.6f}",
            f"- 对应 D-gating best epoch: {int(top9_row['best_epoch'])}",
        ]
    )
    if top8_row is not None:
        lines.append(f"- {len(main_path) - 1} 字段 masked R2: {float(top8_row['best_test_r2']):.6f}")
    if retained_r2 is not None:
        lines.append(f"- D-gating 客观保留 11 字段 masked R2: {retained_r2:.6f}")
    if compact_df is not None and not compact_df.empty:
        path_rows = compact_df[compact_df["label"] != "full_all_features"].sort_values("feature_count", ascending=False)
        for _, row in path_rows.iterrows():
            lines.append(f"- 剪枝路径 {int(row['feature_count'])} 字段 masked R2: {float(row['best_test_r2']):.6f}")
    lines.append(f"- 03 验证中逐字段 drop-one R2 范围: {min_drop:.6f} 到 {max_drop:.6f}")
    lines.append("")
    lines.append("结论：如果剪掉任意一个字段后都低于阈值，说明 02 阶段选出的主路径在当前 D-gating 模型下已经不能继续安全压缩。")
    if not random_df.empty:
        lines.extend(
            [
                f"- random 9-field median best_test_r2: {random_median:.6f}",
                f"- random 9-field best best_test_r2: {random_best:.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "## 产物说明",
            "",
            "- `stage01_main_path_interface.json`: 后续阶段读取的标准接口。",
            "- `01_dgating/`: D-gating 模型、日志、门控历史和诊断图。",
            "- `02_pruning/dgate11_compact_candidates.csv`: 从 D-gating 保留 11 字段开始的贪心剪枝路径摘要。",
            "- `02_pruning/dgate11_pruning_trials.csv`: 每一步尝试 drop 每个字段后的完整准确率对比。",
            "- `02_pruning/pruning_path.png`: 11 -> 10 -> 9 等剪枝路径准确率曲线。",
            "- `03_validation/main_vs_full_validation.csv`: 最终主路径与全量字段的 masked 对比。",
            "- `03_validation/drop_one_validation.csv` 和 `03_validation/drop_one_validation.png`: 对最终主路径逐字段阻断验证。",
        ]
    )
    if not random_df.empty:
        lines.append("- `random_9field_validation.csv` and `random_9field_validation.png`: random same-size subset baseline.")
    (run_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rank_features(strictness_path: Path) -> list[str]:
    df = pd.read_csv(strictness_path)
    metric = "final_effective_group_l2" if "final_effective_group_l2" in df.columns else "final_gate_abs"
    ranked = df.sort_values(metric, ascending=False)["feature"].astype(str).tolist()
    return ranked


def _retained_dgate_features(strictness_path: Path) -> list[str]:
    df = pd.read_csv(strictness_path)
    metric = "final_effective_group_l2" if "final_effective_group_l2" in df.columns else "final_gate_abs"
    mask = df["selected_by_active_threshold"].astype(bool)
    if not bool(mask.any()) and "selected_by_largest_log_gap" in df.columns:
        mask = df["selected_by_largest_log_gap"].astype(bool)
    return df.loc[mask].sort_values(metric, ascending=False)["feature"].astype(str).tolist()


def _fitresult_row(label: str, features: list[str], fit: Any, **extra: Any) -> dict[str, Any]:
    return {
        "label": label,
        "feature_count": len(features),
        "features": ";".join(features),
        "best_test_r2": float(fit.r2),
        "best_test_mse": float(fit.mse),
        "best_epoch": int(fit.best_epoch),
        **extra,
    }


def _evaluate_dgate_compact_path(
    cfg: dict[str, Any],
    center: str,
    all_features: list[str],
    retained_features: list[str],
    exclude_columns: list[str],
    run_dir: Path,
    dgate_dir: Path,
    pruning_dir: Path,
    validation_dir: Path,
    seed: int,
    compact_tau: float,
    min_prune_k: int,
    obvious_drop_r2: float,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del seed
    retained_features = list(dict.fromkeys(retained_features))
    state = torch.load(dgate_dir / "model.pth", map_location=DEVICE)
    params = state["params"]
    model_features = list(state["features"])
    if model_features != list(all_features):
        raise ValueError("D-gating model feature order does not match stage feature order.")
    model = DGatingRegressor(
        len(model_features),
        params.get("hidden_dims", [64, 32, 16]),
        dgate_depth=int(params.get("dgate_depth", 4)),
        dropout=float(params.get("dropout", 0.0)),
    ).to(DEVICE)
    model.load_state_dict(state["model_state"])
    model.eval()

    preprocessing = cfg.get("preprocessing", {})
    dataset_cfg = cfg["dataset"]
    training = cfg.get("training", {})
    bundle = prepare_supervised_dataset(
        data_path=resolve_project_path(cfg, dataset_cfg["processed_csv"]),
        center=center,
        features=model_features,
        train_ratio=float(params.get("train_ratio", training.get("train_ratio", 0.8))),
        random_state=int(params.get("random_state", training.get("random_state", 42))),
        drop_all_zero_columns=bool(preprocessing.get("drop_all_zero_columns", False)),
        exclude_columns=exclude_columns,
    )
    x_train = torch.from_numpy(bundle.X_train).to(DEVICE)
    y_train = torch.from_numpy(bundle.y_train).to(DEVICE)
    x_test = torch.from_numpy(bundle.X_test).to(DEVICE)
    y_test = torch.from_numpy(bundle.y_test).to(DEVICE)
    feature_to_idx = {feature: idx for idx, feature in enumerate(model_features)}

    def eval_masked(label: str, features: list[str], **extra: Any) -> dict[str, Any]:
        mask = torch.zeros(len(model_features), dtype=x_test.dtype, device=DEVICE)
        for feature in features:
            mask[feature_to_idx[feature]] = 1.0
        with torch.no_grad():
            train_pred = model(x_train * mask)
            test_pred = model(x_test * mask)
            train_mse = float(nn.functional.mse_loss(train_pred, y_train).detach().cpu())
            test_mse = float(nn.functional.mse_loss(test_pred, y_test).detach().cpu())
            train_r2 = r2_score(y_train.detach().cpu(), train_pred.detach().cpu())
            test_r2 = r2_score(y_test.detach().cpu(), test_pred.detach().cpu())
        return {
            "label": label,
            "feature_count": len(features),
            "features": ";".join(features),
            "best_test_r2": float(test_r2),
            "best_test_mse": float(test_mse),
            "best_epoch": int(_read_json(dgate_dir / "metrics.json").get("best_epoch", 0)),
            "last_train_r2": float(train_r2),
            "last_train_mse": float(train_mse),
            "last_test_r2": float(test_r2),
            "last_test_mse": float(test_mse),
            "evaluation_mode": "dgating_best_model_masked_inputs",
            **extra,
        }

    full_row = eval_masked("full_all_features", model_features)
    full_row["candidate_size"] = len(model_features)
    full_row["compact_tau"] = compact_tau
    full_row["meets_tau"] = bool(full_row["best_test_r2"] >= compact_tau)
    retained_row = eval_masked("dgate_retained11", retained_features, candidate_size=len(retained_features))
    retained_row["compact_tau"] = compact_tau
    retained_row["meets_tau"] = bool(retained_row["best_test_r2"] >= compact_tau)

    compact_rows = [full_row, retained_row]
    trial_rows: list[dict[str, Any]] = []
    current_features = list(retained_features)
    current_r2 = float(retained_row["best_test_r2"])
    retained_r2 = current_r2
    step = 0
    while len(current_features) > 1:
        step += 1
        candidate_rows = []
        for dropped in current_features:
            trial_features = [feature for feature in current_features if feature != dropped]
            row = eval_masked(
                f"try_drop_{dropped}",
                trial_features,
                step=step,
                from_feature_count=len(current_features),
                candidate_size=len(trial_features),
                dropped_feature=dropped,
                compact_tau=compact_tau,
            )
            row["delta_r2_from_previous"] = current_r2 - float(row["best_test_r2"])
            row["delta_r2_from_retained11"] = retained_r2 - float(row["best_test_r2"])
            row["meets_tau"] = bool(row["best_test_r2"] >= compact_tau)
            row["chosen_for_next_step"] = False
            candidate_rows.append(row)

        best_idx = int(np.argmax([float(row["best_test_r2"]) for row in candidate_rows]))
        candidate_rows[best_idx]["chosen_for_next_step"] = True
        best_row = candidate_rows[best_idx]
        trial_rows.extend(candidate_rows)

        current_features = str(best_row["features"]).split(";")
        current_r2 = float(best_row["best_test_r2"])
        compact_rows.append(
            {
                **best_row,
                "label": f"greedy_compact{len(current_features)}",
                "r2_drop_from_retained11": retained_r2 - current_r2,
                "obvious_drop": bool((retained_r2 - current_r2) >= obvious_drop_r2 or current_r2 < compact_tau),
            }
        )

        if len(current_features) <= min_prune_k and (
            (retained_r2 - current_r2) >= obvious_drop_r2 or current_r2 < compact_tau
        ):
            break

    trial_df = pd.DataFrame(trial_rows)
    if not trial_df.empty:
        trial_df.to_csv(pruning_dir / "dgate11_pruning_trials.csv", index=False, encoding="utf-8-sig")

    compact_df = pd.DataFrame(compact_rows)
    compact_df.to_csv(pruning_dir / "dgate11_compact_candidates.csv", index=False, encoding="utf-8-sig")

    meets = compact_df[(compact_df["meets_tau"] == True) & (compact_df["label"] != "full_all_features")].copy()
    if not meets.empty:
        chosen = meets.sort_values(["feature_count", "best_test_r2"], ascending=[True, False]).iloc[0]
    else:
        chosen = compact_df.sort_values(["best_test_r2", "feature_count"], ascending=[False, True]).iloc[0]
    main_path = str(chosen["features"]).split(";")

    drop_rows = []
    for idx, dropped in enumerate(main_path, start=1):
        features = [f for f in main_path if f != dropped]
        row = eval_masked(f"drop_{dropped}", list(features))
        row["dropped_feature"] = dropped
        row["compact_tau"] = compact_tau
        row["still_meets_tau"] = bool(row["best_test_r2"] >= compact_tau)
        drop_rows.append(row)
        _write_incremental_csv(validation_dir / "drop_one_validation.csv", drop_rows)
    drop_df = pd.DataFrame(drop_rows).sort_values("best_test_r2", ascending=False)

    chosen_row = dict(chosen)
    main_vs_full_df = pd.DataFrame(
        [
            chosen_row | {"label": "dgate11_compact_main_path", "compact_source": "dgate_retained11"},
            full_row,
            retained_row,
        ]
    )
    topk_df = pd.DataFrame(
        [
            chosen_row | {"label": "dgate11_compact_main_path", "k": len(main_path), "compact_source": "dgate_retained11"}
        ]
    )
    main_vs_full_df.to_csv(validation_dir / "main_vs_full_validation.csv", index=False, encoding="utf-8-sig")
    topk_df.to_csv(validation_dir / "topk_validation.csv", index=False, encoding="utf-8-sig")
    drop_df.to_csv(validation_dir / "drop_one_validation.csv", index=False, encoding="utf-8-sig")
    _plot_pruning_path(compact_df, pruning_dir / "pruning_path.png")
    return main_path, topk_df, drop_df, main_vs_full_df, compact_df


def _load_certified_fixed9_validation(source_run: Path) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    initial = pd.read_csv(source_run / "initial_candidate_comparison.csv")
    full = pd.read_csv(source_run / "full_metrics.csv")
    drop = pd.read_csv(source_run / "main_path.csv")

    main_row = initial.loc[initial["name"] == "fixed_main_path"].iloc[0]
    full_row = full.iloc[0]
    main_path = str(main_row["features"]).split(";")
    main_vs_full = pd.DataFrame(
        [
            {
                "label": "fixed9_certified_main_path",
                "feature_count": len(main_path),
                "features": ";".join(main_path),
                "best_test_r2": float(main_row["r2"]),
                "best_test_mse": float(main_row["mse"]),
                "best_epoch": int(main_row["best_epoch"]),
                "source": str(source_run / "initial_candidate_comparison.csv"),
            },
            {
                "label": "full_all_features",
                "feature_count": int(full_row["feature_count"]),
                "features": "ALL_AVAILABLE_FEATURES",
                "best_test_r2": float(full_row["r2_full"]),
                "best_test_mse": float(full_row["mse_full"]),
                "best_epoch": int(full_row["best_epoch"]),
                "source": str(source_run / "full_metrics.csv"),
            },
        ]
    )
    topk_df = pd.DataFrame(
        [
            {
                "label": "fixed9_certified_main_path",
                "feature_count": len(main_path),
                "features": ";".join(main_path),
                "best_test_r2": float(main_row["r2"]),
                "best_test_mse": float(main_row["mse"]),
                "best_epoch": int(main_row["best_epoch"]),
                "k": len(main_path),
                "source": str(source_run / "initial_candidate_comparison.csv"),
            }
        ]
    )

    drop_rows = []
    for _, row in drop.iterrows():
        feature = str(row["feature"])
        subset = [value for value in main_path if value != feature]
        drop_rows.append(
            {
                "label": f"drop_{feature}",
                "feature_count": len(subset),
                "features": ";".join(subset),
                "best_test_r2": float(row["trial_r2"]),
                "best_test_mse": float(row["trial_mse"]),
                "best_epoch": None,
                "dropped_feature": feature,
                "source": str(source_run / "main_path.csv"),
            }
        )
    return main_path, topk_df, pd.DataFrame(drop_rows), main_vs_full


def _make_run_dir(root: Path, run_name: str | None) -> Path:
    if run_name:
        if not run_name.startswith("run_"):
            raise ValueError("--run-name must start with run_ and should contain only the timestamp.")
        run_dir = root / run_name
    else:
        run_dir = root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def _parse_k_values(text: str, max_k: int) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            values.extend(range(int(left), int(right) + 1))
        else:
            values.append(int(part))
    return sorted({k for k in values if 1 <= k <= max_k})


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 01 main-path identification and validation.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument("--combo", default="5")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "conditional_residual_compensation_outputs"))
    parser.add_argument("--stage-dir", default="stage01_main_path")
    parser.add_argument("--run-name")
    parser.add_argument("--lambda-dgate", type=float, default=0.03)
    parser.add_argument("--dgate-depth", type=int, default=4)
    parser.add_argument("--main-k", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validation-epochs", type=int, default=200)
    parser.add_argument("--topk", default="9")
    parser.add_argument("--random-trials", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-run-dir", help="Reuse a run directory whose D-gating artifacts already exist.")
    parser.add_argument(
        "--main-path-source",
        choices=["dgate_compact", "fixed9_certified", "dgate_topk"],
        default="dgate_compact",
        help="How to populate the stage interface after D-gating has produced ranking evidence.",
    )
    parser.add_argument("--compact-tau", type=float, default=0.95)
    parser.add_argument("--min-prune-k", type=int, default=9, help="Keep pruning at least until this field count is evaluated.")
    parser.add_argument("--obvious-drop-r2", type=float, default=0.02, help="Stop after min-prune-k once R2 has dropped by at least this much from retained-11.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    combo = resolve_center_spec(cfg, args.combo)
    center = combo["center"]
    exclude_columns = normalize_column_list(combo["exclude_columns"])

    output_root = Path(args.output_root).resolve()
    center_dir = output_root / f"CenterOn_{center}"
    stage_root = center_dir / args.stage_dir
    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
    else:
        run_dir = _make_run_dir(stage_root, args.run_name)
    dirs = _artifact_dirs(run_dir)
    dgate_dir = dirs["dgating"]
    pruning_dir = dirs["pruning"]
    validation_dir = dirs["validation"]

    dgate_params = {
        "epochs": args.epochs,
        "lambda_dgate": args.lambda_dgate,
        "dgate_depth": args.dgate_depth,
        "dgate_normalize_lambda_by_depth": False,
        "active_threshold": 0.001,
        "record_epoch0_gate": True,
        "net_weight_decay": 0.0,
    }
    stage_config = {
        "stage": "stage01_main_path",
        "center": center,
        "combo": args.combo,
        "exclude_columns": exclude_columns,
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "artifact_dirs": {name: str(path) for name, path in dirs.items()},
        "main_k": args.main_k,
        "dgate_params": dgate_params,
        "validation": {
            "validation_epochs": args.validation_epochs,
            "topk": args.topk,
            "random_trials": args.random_trials,
            "include_full_field_validation": True,
            "compact_tau": args.compact_tau,
            "min_prune_k": args.min_prune_k,
            "obvious_drop_r2": args.obvious_drop_r2,
            "seed": args.seed,
            "resume_run_dir": str(run_dir) if args.resume_run_dir else None,
        },
    }
    save_json(run_dir / "stage_config.json", stage_config)

    dgate_done = (dgate_dir / "dgate_strictness_features.csv").exists() and (dgate_dir / "metrics.json").exists()
    if not dgate_done:
        train_center_model(
            cfg=cfg,
            center=center,
            model_name="DGatingDNN",
            overrides=dgate_params,
            run_name=run_dir.name,
            force_relations=False,
            exclude_columns=exclude_columns,
            combo_name=f"combo{combo['id']}",
            output_run_dir=dgate_dir,
        )

    ranked_features = _rank_features(dgate_dir / "dgate_strictness_features.csv")
    all_features = _read_json(dgate_dir / "config.json")["features"]
    compact_df = pd.DataFrame()
    if args.main_path_source == "dgate_compact":
        retained_features = _retained_dgate_features(dgate_dir / "dgate_strictness_features.csv")
        main_path, topk_df, drop_df, main_vs_full_df, compact_df = _evaluate_dgate_compact_path(
            cfg=cfg,
            center=center,
            all_features=list(all_features),
            retained_features=retained_features,
            exclude_columns=exclude_columns,
            run_dir=run_dir,
            dgate_dir=dgate_dir,
            pruning_dir=pruning_dir,
            validation_dir=validation_dir,
            seed=args.seed,
            compact_tau=args.compact_tau,
            min_prune_k=args.min_prune_k,
            obvious_drop_r2=args.obvious_drop_r2,
        )
        random_df = pd.DataFrame()
    elif args.main_path_source == "fixed9_certified":
        main_path, topk_df, drop_df, main_vs_full_df = _load_certified_fixed9_validation(CERTIFIED_FIXED9_RUN)
        random_df = pd.DataFrame()
        topk_df.to_csv(validation_dir / "topk_validation.csv", index=False, encoding="utf-8-sig")
        drop_df.to_csv(validation_dir / "drop_one_validation.csv", index=False, encoding="utf-8-sig")
        main_vs_full_df.to_csv(validation_dir / "main_vs_full_validation.csv", index=False, encoding="utf-8-sig")
    else:
        main_path = ranked_features[: args.main_k]
        rng = np.random.default_rng(args.seed)
        validation_rows = []
        k_values = _parse_k_values(args.topk, len(ranked_features))
        for k in k_values:
            subset = ranked_features[:k]
            validation_rows.append(
                _train_dnn_subset(
                    cfg,
                    center,
                    subset,
                    exclude_columns,
                    label=f"top{k}",
                    epochs=args.validation_epochs,
                    seed=args.seed,
            )
            | {"k": k}
            )
            _write_incremental_csv(validation_dir / "topk_validation.csv", validation_rows)
        topk_df = pd.DataFrame(validation_rows).sort_values("k")
        topk_df.to_csv(validation_dir / "topk_validation.csv", index=False, encoding="utf-8-sig")

        main_result = topk_df.loc[topk_df["k"] == args.main_k]
        if main_result.empty:
            main_eval = _train_dnn_subset(
                cfg,
                center,
                main_path,
                exclude_columns,
                label=f"top{args.main_k}",
                epochs=args.validation_epochs,
                seed=args.seed,
            ) | {"k": args.main_k}
            topk_df = pd.concat([topk_df, pd.DataFrame([main_eval])], ignore_index=True).sort_values("k")
            topk_df.to_csv(validation_dir / "topk_validation.csv", index=False, encoding="utf-8-sig")
            main_result = topk_df.loc[topk_df["k"] == args.main_k]

        full_row = _train_dnn_subset(
            cfg,
            center,
            list(all_features),
            exclude_columns,
            label="full_all_features",
            epochs=args.validation_epochs,
            seed=args.seed,
        )
        main_vs_full_df = pd.DataFrame(
            [
                dict(main_result.iloc[0]) | {"label": f"top{args.main_k}_main_path"},
                full_row,
            ]
        )
        main_vs_full_df.to_csv(validation_dir / "main_vs_full_validation.csv", index=False, encoding="utf-8-sig")

        drop_rows = []
        for dropped in main_path:
            subset = [feature for feature in main_path if feature != dropped]
            row = _train_dnn_subset(
                cfg,
                center,
                subset,
                exclude_columns,
                label=f"drop_{dropped}",
                epochs=args.validation_epochs,
                seed=args.seed,
            )
            row["dropped_feature"] = dropped
            drop_rows.append(row)
            _write_incremental_csv(validation_dir / "drop_one_validation.csv", drop_rows)
        drop_df = pd.DataFrame(drop_rows)
        drop_df.to_csv(validation_dir / "drop_one_validation.csv", index=False, encoding="utf-8-sig")

        random_rows = []
        random_pool = list(all_features)
        for trial in range(1, args.random_trials + 1):
            subset = sorted(rng.choice(random_pool, size=args.main_k, replace=False).tolist())
            row = _train_dnn_subset(
                cfg,
                center,
                subset,
                exclude_columns,
                label=f"random9_trial{trial}",
                epochs=args.validation_epochs,
                seed=args.seed + trial,
            )
            row["trial"] = trial
            random_rows.append(row)
            _write_incremental_csv(validation_dir / "random_9field_validation.csv", random_rows)
        random_df = pd.DataFrame(random_rows)
        if not random_df.empty:
            random_df.to_csv(validation_dir / "random_9field_validation.csv", index=False, encoding="utf-8-sig")

    path_set = set(main_path)
    residual_features = [feature for feature in all_features if feature not in path_set]
    main_r2 = float(main_vs_full_df.loc[main_vs_full_df["label"] != "full_all_features"].iloc[0]["best_test_r2"])

    _plot_topk(topk_df, len(main_path), validation_dir / "topk_validation.png")
    _plot_drop(drop_df, main_r2, validation_dir / "drop_one_validation.png")
    if not random_df.empty:
        _plot_random(random_df, main_r2, validation_dir / "random_9field_validation.png")

    dgate_metrics = _read_json(dgate_dir / "metrics.json")
    strictness_summary = _read_json(dgate_dir / "dgate_strictness_summary.json")
    interface = {
        "schema_version": 1,
        "stage": "stage01_main_path",
        "center": center,
        "target": center,
        "combo": args.combo,
        "source_run_dir": str(run_dir),
        "dgate_run_dir": str(dgate_dir),
        "pruning_dir": str(pruning_dir),
        "validation_dir": str(validation_dir),
        "main_path_count": len(main_path),
        "main_path_features": main_path,
        "residual_candidate_features": residual_features,
        "all_available_features": all_features,
        "feature_ranking_source": str(dgate_dir / "dgate_strictness_features.csv"),
        "main_path_source": args.main_path_source,
        "certified_fixed9_source_run": str(CERTIFIED_FIXED9_RUN) if args.main_path_source == "fixed9_certified" else None,
        "dgate_retained_compact_candidates_csv": str(pruning_dir / "dgate11_compact_candidates.csv")
        if args.main_path_source == "dgate_compact"
        else None,
        "dgate_params": dgate_params,
        "real_input_validation": {
            "main_path_dnn_best_test_r2": main_r2,
            "full_field_dnn_best_test_r2": float(
                main_vs_full_df.loc[main_vs_full_df["label"] == "full_all_features"].iloc[0]["best_test_r2"]
            ),
            "evaluation_mode": "dgating_best_model_masked_inputs",
            "main_path_masked_best_test_r2": main_r2,
            "full_field_masked_best_test_r2": float(
                main_vs_full_df.loc[main_vs_full_df["label"] == "full_all_features"].iloc[0]["best_test_r2"]
            ),
            "main_vs_full_validation_csv": str(validation_dir / "main_vs_full_validation.csv"),
            "topk_validation_csv": str(validation_dir / "topk_validation.csv"),
            "drop_one_validation_csv": str(validation_dir / "drop_one_validation.csv"),
            "random_9field_validation_csv": str(validation_dir / "random_9field_validation.csv") if not random_df.empty else None,
        },
    }
    save_json(run_dir / "stage01_main_path_interface.json", interface)

    _write_report(
        run_dir=run_dir,
        main_path=main_path,
        dgate_metrics=dgate_metrics,
        strictness_summary=strictness_summary,
        topk_df=topk_df,
        drop_df=drop_df,
        random_df=random_df,
        main_vs_full_df=main_vs_full_df,
        compact_df=compact_df,
        params=dgate_params,
    )
    print(f"Stage 01 run saved to {run_dir}")


if __name__ == "__main__":
    main()
