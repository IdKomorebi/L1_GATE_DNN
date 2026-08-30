#!/usr/bin/env python3
"""RTS-GMLC 前置可推断性体检。

目的不是做字段选择，而是回答三个更基础的问题：
1. 27 个候选发布字段能否支撑 130 个细粒度目标（剔除零方差后为 108 个）？
2. 联合预测表现差时，是否只是异质目标互相干扰，按物理对象分组能否改善？
3. 把相对低敏感的节点量逐级移入 X 后，高敏感目标的可推断性如何变化？

主模型是普通多输出 MLP。采用严格时间顺序 70%/10%/20% 划分、仅用训练段拟合
标准化器、验证集早停，并用三个随机种子报告均值与标准差。Ridge 仅作线性参照。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_rts_feasibility_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "01_data" / "raw" / "rts_gmlc_2020"
DEFAULT_DATA = DATASET_DIR / "rts_gmlc_hourly_2020_acpf_base.csv"
DEFAULT_DICT = DATASET_DIR / "field_dictionary.csv"
DEFAULT_OUT = ROOT / "04_outputs" / "rts_gmlc_2020" / "00_feasibility_check"

# 与 PJM 主实验的 12 个目标规模保持接近。选择依据不是单纯挑最高 R²，
# 而是先保证节点/线路/机组三类对象、三个区域和不同机组类型都有代表，
# 再要求目标非恒定、属于 T3 且 public-27 分组模型 R² >= 0.70。
CORE_TARGETS = [
    {"field": "bus_115_va_deg", "core_group": "bus_angle", "selection_reason": "区域 1 高负荷代表节点相角"},
    {"field": "bus_215_va_deg", "core_group": "bus_angle", "selection_reason": "区域 2 高负荷代表节点相角"},
    {"field": "bus_315_va_deg", "core_group": "bus_angle", "selection_reason": "区域 3 高负荷代表节点相角"},
    {"field": "branch_ab1_loading_pct", "core_group": "interarea_loading", "selection_reason": "关键跨区支路 AB1 负载率"},
    {"field": "branch_ab2_loading_pct", "core_group": "interarea_loading", "selection_reason": "关键跨区支路 AB2 负载率"},
    {"field": "branch_ab3_loading_pct", "core_group": "interarea_loading", "selection_reason": "关键跨区支路 AB3 负载率"},
    {"field": "branch_ca_1_loading_pct", "core_group": "interarea_loading", "selection_reason": "关键跨区支路 CA-1 负载率"},
    {"field": "branch_cb_1_loading_pct", "core_group": "interarea_loading", "selection_reason": "关键跨区支路 CB-1 负载率"},
    {"field": "gen_121_nuclear_1_pg_mw", "core_group": "generator", "selection_reason": "区域 1 核电机组有功出力"},
    {"field": "gen_218_cc_1_pg_mw", "core_group": "generator", "selection_reason": "区域 2 联合循环机组有功出力"},
    {"field": "gen_317_wind_1_pg_mw", "core_group": "generator", "selection_reason": "区域 3 风电机组有功出力"},
    {"field": "gen_321_cc_1_status", "core_group": "generator", "selection_reason": "启停较均衡的联合循环机组状态"},
]


@dataclass
class Config:
    seeds: tuple[int, ...] = (42, 43, 44)
    hidden: tuple[int, ...] = (192, 128, 64)
    epochs: int = 240
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 35
    min_epochs: int = 60
    train_ratio: float = 0.70
    validation_ratio: float = 0.10
    ridge_alpha: float = 1.0
    constant_tolerance: float = 1e-12
    strong_r2: float = 0.90
    usable_r2: float = 0.70
    weak_r2: float = 0.30
    device: str = "cpu"


class MultiOutputMLP(nn.Module):
    def __init__(self, n_input: int, n_output: int, hidden: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_input
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU()])
            prev = width
        layers.append(nn.Linear(prev, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize(train: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    return (full - mean) / scale, mean, scale


def r2_columns(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ss_res = np.square(y - pred).sum(axis=0)
    ss_tot = np.square(y - y.mean(axis=0)).sum(axis=0)
    # 时间测试段没有变化时，R² 在数学上无定义。不能用一个极小分母硬算，
    # 否则会产生 -1e12 量级的假失败并污染宏平均。
    return np.divide(
        ss_tot - ss_res,
        ss_tot,
        out=np.full_like(ss_tot, np.nan, dtype=float),
        where=ss_tot > 1e-12,
    )


def pearson_columns(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    yc = y - y.mean(axis=0)
    pc = pred - pred.mean(axis=0)
    denom = np.sqrt(np.square(yc).sum(axis=0) * np.square(pc).sum(axis=0))
    return np.divide((yc * pc).sum(axis=0), denom, out=np.zeros_like(denom), where=denom > 0)


def grade_r2(value: float, cfg: Config) -> str:
    if not np.isfinite(value):
        return "test_segment_has_no_variance"
    if value >= cfg.strong_r2:
        return "strong_R2_ge_0.90"
    if value >= cfg.usable_r2:
        return "usable_R2_0.70_0.90"
    if value >= cfg.weak_r2:
        return "weak_R2_0.30_0.70"
    return "not_inferable_R2_lt_0.30"


def target_metrics(
    fields: list[str], y: np.ndarray, pred: np.ndarray, tier_map: dict[str, str]
) -> pd.DataFrame:
    r2 = r2_columns(y, pred)
    rmse = np.sqrt(np.square(y - pred).mean(axis=0))
    mae = np.abs(y - pred).mean(axis=0)
    std = y.std(axis=0)
    corr = pearson_columns(y, pred)
    rows = []
    for i, field in enumerate(fields):
        is_status = field.endswith("_status")
        row = {
            "target": field,
            "sensitivity_tier": tier_map[field],
            "target_kind": "binary_status" if is_status else "continuous",
            "r2": float(r2[i]),
            "rmse": float(rmse[i]),
            "mae": float(mae[i]),
            "test_std": float(std[i]),
            "nrmse_by_test_std": float(rmse[i] / max(std[i], 1e-12)),
            "pearson_r": float(corr[i]),
            "test_r2_evaluable": int(np.isfinite(r2[i])),
        }
        if is_status:
            true = (y[:, i] >= 0.5).astype(int)
            hard = (pred[:, i] >= 0.5).astype(int)
            row["accuracy"] = float((true == hard).mean())
            row["balanced_accuracy"] = float(balanced_accuracy_score(true, hard))
            row["roc_auc"] = float(roc_auc_score(true, pred[:, i])) if len(np.unique(true)) == 2 else math.nan
        else:
            row["accuracy"] = math.nan
            row["balanced_accuracy"] = math.nan
            row["roc_auc"] = math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def train_mlp(
    X: np.ndarray,
    Y: np.ndarray,
    targets: list[str],
    tier_map: dict[str, str],
    train_end: int,
    val_end: int,
    cfg: Config,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    set_seed(seed)
    device = torch.device(cfg.device)

    Xs, _, _ = standardize(X[:train_end], X)
    Ys, y_mean, y_scale = standardize(Y[:train_end], Y)
    xt = torch.tensor(Xs[:train_end], dtype=torch.float32, device=device)
    yt = torch.tensor(Ys[:train_end], dtype=torch.float32, device=device)
    xv = torch.tensor(Xs[train_end:val_end], dtype=torch.float32, device=device)
    yv = torch.tensor(Ys[train_end:val_end], dtype=torch.float32, device=device)
    xte = torch.tensor(Xs[val_end:], dtype=torch.float32, device=device)

    model = MultiOutputMLP(X.shape[1], Y.shape[1], cfg.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    best_loss = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(train_end, generator=generator)
        total = 0.0
        for start in range(0, train_end, cfg.batch_size):
            idx = order[start : start + cfg.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(idx)

        model.eval()
        with torch.no_grad():
            val_pred_std = model(xv)
            val_loss = float(loss_fn(val_pred_std, yv).cpu())
            val_pred = val_pred_std.cpu().numpy() * y_scale + y_mean
            val_r2 = r2_columns(Y[train_end:val_end], val_pred)
        history.append(
            {
                "epoch": epoch,
                "train_standardized_mse": total / train_end,
                "validation_standardized_mse": val_loss,
                "validation_macro_r2": float(np.nanmean(val_r2)),
                "validation_median_r2": float(np.nanmedian(val_r2)),
            }
        )

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= cfg.min_epochs and stale >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError("MLP training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(xte).cpu().numpy() * y_scale + y_mean
    metrics = target_metrics(targets, Y[val_end:], test_pred, tier_map)
    meta = {
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_standardized_mse": best_loss,
        "elapsed_seconds": time.time() - started,
        "device": cfg.device,
    }
    return metrics, pd.DataFrame(history), meta


def ridge_metrics(
    X: np.ndarray,
    Y: np.ndarray,
    targets: list[str],
    tier_map: dict[str, str],
    train_end: int,
    val_end: int,
    cfg: Config,
) -> pd.DataFrame:
    Xs, _, _ = standardize(X[:train_end], X)
    Ys, y_mean, y_scale = standardize(Y[:train_end], Y)
    model = Ridge(alpha=cfg.ridge_alpha)
    model.fit(Xs[:train_end], Ys[:train_end])
    pred = model.predict(Xs[val_end:]) * y_scale + y_mean
    return target_metrics(targets, Y[val_end:], pred, tier_map)


def sensitivity_tier(row: pd.Series, is_constant: bool) -> tuple[str, str]:
    if is_constant:
        return "T0_constant", "全年不变，没有可评估的时变信息；后续应先剥离"
    quantity = row["physical_quantity"]
    entity = row["entity_type"]
    if quantity == "vm_pu":
        return "T1_relative_low", "节点电压幅值；相对较低敏但仍是运行状态，不等于可无条件公开"
    if entity == "bus" and quantity not in {"va_deg"}:
        return "T2_operational", "节点有功/无功注入或平衡注入；细粒度运行量，中等敏感"
    return "T3_high", "节点相角、线路状态或具名机组状态；作为高敏感主目标保留"


def build_tiers(data: pd.DataFrame, dictionary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    target_def = dictionary[dictionary["role"].eq("sensitive_target")]
    for row in target_def.to_dict("records"):
        field = row["column_name"]
        variance = float(data[field].var(ddof=0))
        n_unique = int(data[field].nunique(dropna=False))
        constant = variance <= cfg.constant_tolerance
        tier, reason = sensitivity_tier(pd.Series(row), constant)
        rows.append(
            {
                "field": field,
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "physical_quantity": row["physical_quantity"],
                "mathematical_status": row["mathematical_status"],
                "variance": variance,
                "n_unique": n_unique,
                "is_constant": int(constant),
                "sensitivity_tier": tier,
                "tier_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def target_group(row: pd.Series) -> str:
    field = row["field"]
    if row["entity_type"] == "branch":
        return "branch"
    if row["entity_type"] == "generator":
        return "generator_status" if field.endswith("_status") else "generator_continuous"
    return "bus_electrical"


def build_partitions(
    public: list[str], tiers: pd.DataFrame
) -> list[dict]:
    variable = tiers[tiers["is_constant"].eq(0)].copy()
    variable["group"] = variable.apply(target_group, axis=1)
    all_targets = variable["field"].tolist()
    partitions = [
        {
            "partition": "01_public27_joint",
            "description": "27 个候选发布字段联合预测全部 108 个时变目标",
            "x": public,
            "y": all_targets,
        }
    ]
    for group in ["bus_electrical", "branch", "generator_continuous", "generator_status"]:
        partitions.append(
            {
                "partition": f"02_public27_grouped/{group}",
                "description": f"仍只用 27 个候选发布字段，单独预测 {group} 目标组",
                "x": public,
                "y": variable.loc[variable["group"].eq(group), "field"].tolist(),
            }
        )

    tier1 = variable.loc[variable["sensitivity_tier"].eq("T1_relative_low"), "field"].tolist()
    tier2 = variable.loc[variable["sensitivity_tier"].eq("T2_operational"), "field"].tolist()
    variable_bus = variable.loc[variable["entity_type"].eq("bus"), "field"].tolist()
    partitions.extend(
        [
            {
                "partition": "03_public_plus_tier1_voltage",
                "description": "候选发布字段加时变节点电压幅值，预测其余目标",
                "x": public + tier1,
                "y": [f for f in all_targets if f not in tier1],
            },
            {
                "partition": "04_public_plus_tier1_tier2_bus_operational",
                "description": "再加入节点 P/Q 与平衡注入，预测 T3 高敏感目标",
                "x": public + tier1 + tier2,
                "y": variable.loc[variable["sensitivity_tier"].eq("T3_high"), "field"].tolist(),
            },
            {
                "partition": "05_public_plus_all_bus_diagnostic",
                "description": "诊断上界：加入全部时变节点量，预测线路和机组目标",
                "x": public + variable_bus,
                "y": variable.loc[variable["entity_type"].isin(["branch", "generator"]), "field"].tolist(),
            },
        ]
    )
    return partitions


def summarize_metrics(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    keys = group_cols + ["target", "sensitivity_tier", "target_kind"]
    numeric = [
        "r2",
        "rmse",
        "mae",
        "test_std",
        "nrmse_by_test_std",
        "pearson_r",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "test_r2_evaluable",
    ]
    out = metrics.groupby(keys, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    out.columns = ["_".join([x for x in col if x]) for col in out.columns.to_flat_index()]
    return out


def partition_summary(all_metrics: pd.DataFrame, ridge: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for part, d in all_metrics.groupby("partition", sort=False):
        # 先按目标跨种子平均，再统计目标分布，避免把随机种子当成额外目标样本。
        by_target = d.groupby("target", as_index=False).agg(
            r2=("r2", "mean"),
            r2_seed_std=("r2", "std"),
            sensitivity_tier=("sensitivity_tier", "first"),
            target_kind=("target_kind", "first"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
        )
        rr = ridge[ridge["partition"].eq(part)]
        rows.append(
            {
                "partition": part,
                "description": d["description"].iloc[0],
                "n_input_declared": int(d["n_input_declared"].iloc[0]),
                "n_input_variable": int(d["n_input_variable"].iloc[0]),
                "n_targets": len(by_target),
                "n_targets_r2_evaluable": int(by_target["r2"].notna().sum()),
                "macro_r2": float(by_target["r2"].mean(skipna=True)),
                "median_r2": float(by_target["r2"].median()),
                "q25_r2": float(by_target["r2"].quantile(0.25)),
                "min_r2": float(by_target["r2"].min()),
                "targets_r2_ge_0_90": int((by_target["r2"] >= cfg.strong_r2).sum()),
                "targets_r2_ge_0_70": int((by_target["r2"] >= cfg.usable_r2).sum()),
                "targets_r2_ge_0_30": int((by_target["r2"] >= cfg.weak_r2).sum()),
                "mean_seed_std_r2": float(by_target["r2_seed_std"].fillna(0).mean()),
                "ridge_macro_r2": float(rr["r2"].mean(skipna=True)),
                "ridge_median_r2": float(rr["r2"].median()),
                "status_mean_balanced_accuracy": float(by_target["balanced_accuracy"].mean()),
                "status_mean_roc_auc": float(by_target["roc_auc"].mean()),
            }
        )
    return pd.DataFrame(rows)


def intrinsic_dimension(data: pd.DataFrame, fields: list[str], train_end: int) -> pd.DataFrame:
    z = data[fields].iloc[:train_end].to_numpy(float)
    z = (z - z.mean(axis=0)) / np.maximum(z.std(axis=0), 1e-12)
    singular = np.linalg.svd(z, compute_uv=False)
    explained = np.square(singular) / np.square(singular).sum()
    cumulative = np.cumsum(explained)
    return pd.DataFrame(
        {
            "component": np.arange(1, len(singular) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )


def create_plots(out: Path, summary: pd.DataFrame, target_summary: pd.DataFrame) -> None:
    labels = [p.replace("02_public27_grouped/", "grouped/") for p in summary["partition"]]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    y = np.arange(len(summary))
    ax.barh(y, summary["macro_r2"], color="#2b6cb0", label="ordinary MLP")
    ax.scatter(summary["ridge_macro_r2"], y, color="#c53030", s=38, zorder=3, label="Ridge reference")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0.7, color="0.45", ls="--", lw=1)
    ax.axvline(0.9, color="0.45", ls=":", lw=1)
    ax.set_xlabel("Macro test R² across targets")
    ax.set_title("RTS-GMLC feasibility check: ordinary MLP by partition")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig_partition_macro_r2.png", dpi=180)
    plt.close(fig)

    base = target_summary[
        target_summary["partition"].eq("01_public27_joint") & target_summary["r2_mean"].notna()
    ].sort_values("r2_mean")
    colors = base["sensitivity_tier"].map(
        {"T1_relative_low": "#68d391", "T2_operational": "#ed8936", "T3_high": "#c53030"}
    )
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.scatter(np.arange(len(base)), base["r2_mean"], c=colors, s=16, alpha=0.85)
    for threshold, style in [(0.3, "--"), (0.7, "--"), (0.9, ":")]:
        ax.axhline(threshold, color="0.45", ls=style, lw=1)
    ax.set_xlabel(f"{len(base)} R²-evaluable variable targets sorted by test R²")
    ax.set_ylabel("Mean test R² across seeds")
    ax.set_title("Public-27 joint MLP: per-target inferability")
    for tier, color, label in [
        ("T1_relative_low", "#68d391", "T1 relative-low"),
        ("T2_operational", "#ed8936", "T2 operational"),
        ("T3_high", "#c53030", "T3 high"),
    ]:
        ax.scatter([], [], c=color, s=24, label=label)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig_public27_target_r2.png", dpi=180)
    plt.close(fig)


def write_summary(
    out: Path,
    cfg: Config,
    data: pd.DataFrame,
    public: list[str],
    tiers: pd.DataFrame,
    part_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
    intrinsic: pd.DataFrame,
    core_targets: pd.DataFrame,
    recommendations: pd.DataFrame,
    train_end: int,
    val_end: int,
) -> None:
    base = target_summary[target_summary["partition"].eq("01_public27_joint")]
    grouped = target_summary[target_summary["partition"].str.startswith("02_public27_grouped/")]
    common = base[["target", "r2_mean"]].merge(
        grouped[["target", "r2_mean"]], on="target", suffixes=("_joint", "_grouped")
    )
    p90 = int(intrinsic.loc[intrinsic["cumulative_explained_variance_ratio"] >= 0.90, "component"].iloc[0])
    p95 = int(intrinsic.loc[intrinsic["cumulative_explained_variance_ratio"] >= 0.95, "component"].iloc[0])
    p99 = int(intrinsic.loc[intrinsic["cumulative_explained_variance_ratio"] >= 0.99, "component"].iloc[0])
    public_constant = [f for f in public if data[f].var(ddof=0) <= cfg.constant_tolerance]
    variable_targets = tiers[tiers["is_constant"].eq(0)]

    lines = [
        "# 00 前置可推断性检测：普通神经网络",
        "",
        "## 一、先给结论",
        "",
        "本目录只回答‘现有 X/Y 定义是否具备继续做剥离、初筛和推断源定位的基本可行性’，",
        "不在这里做门控选择，也不把一个目标字段偷放进另一个目标的输入池。",
        "",
        f"- 名义候选发布字段为 **{len(public)}** 个，其中 {len(public_constant)} 个全年恒定，"
        f"实际携带时变信息的输入为 **{len(public) - len(public_constant)}** 个。",
        f"- 名义敏感类型目标为 **{len(tiers)}** 个，其中 {int(tiers['is_constant'].sum())} 个全年恒定，"
        f"进入神经网络评估的时变目标为 **{len(variable_targets)}** 个。",
        f"- 108 个时变目标并不是 108 个相互独立的自由度：训练段中前 {p90}/{p95}/{p99} 个"
        "主成分分别解释 90%/95%/99% 的标准化目标方差。",
        "- 因此不能仅凭‘X 比 Y 少’判定不可推断；应看每个目标在严格时间测试集上的 R²。",
        "",
        "## 二、统一实验设置",
        "",
        f"- 数据：`rts_gmlc_hourly_2020_acpf_base.csv`，{len(data)} 个小时。",
        f"- 时间划分：前 {train_end} 行训练、中间 {val_end - train_end} 行验证、"
        f"最后 {len(data) - val_end} 行测试；不随机打乱跨越时间边界。",
        f"- 普通 MLP：{'-'.join(map(str, cfg.hidden))} 隐层，最多 {cfg.epochs} 轮，"
        f"验证集早停，随机种子 {list(cfg.seeds)}。",
        "- 所有 X/Y 标准化参数只由训练段估计。主指标逐目标计算测试集 R²，"
        "同时保留 RMSE、MAE、相关系数；二值机组状态另报 balanced accuracy 和 ROC-AUC。",
        "- Ridge 仅作线性参照，不参与最终风险判定。",
        "",
        "## 三、各划分总体结果",
        "",
        "| 划分 | X 名义/时变 | Y/可评估 | MLP 宏平均 R² | 中位 R² | R²≥0.9 | R²≥0.7 | Ridge 宏平均 R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in part_summary.itertuples():
        lines.append(
            f"| `{row.partition}` | {row.n_input_declared}/{row.n_input_variable} | "
            f"{row.n_targets}/{row.n_targets_r2_evaluable} | "
            f"{row.macro_r2:.4f} | {row.median_r2:.4f} | {row.targets_r2_ge_0_90} | "
            f"{row.targets_r2_ge_0_70} | {row.ridge_macro_r2:.4f} |"
        )

    delta = common["r2_mean_grouped"] - common["r2_mean_joint"]
    lines += [
        "",
        "## 四、如何解释分组和增加输入",
        "",
        f"只改变 Y 的分组、不增加任何输入时，逐目标 R² 相比联合模型平均变化 "
        f"**{delta.mean():+.4f}**，中位变化 **{delta.median():+.4f}**。",
        "如果这个变化很小，说明‘输出太多导致网络互相干扰’不是主要问题；"
        "如果明显上升，则后续应按节点、线路、连续机组量和机组状态分开建模。",
        "",
        "逐级加 X 的三组并不是最终发布建议，而是敏感性诊断：",
        "",
        "1. `03` 只加入时变节点电压幅值（T1，相对低敏感）。",
        "2. `04` 再加入节点 P/Q 和平衡注入（T2，中等敏感），只预测 T3 高敏感目标。",
        "3. `05` 加入全部时变节点量，包括相角，只作为‘已知节点状态时能否恢复线路/机组’的诊断上界，"
        "不能据此把相角建议为公开字段。",
        "",
        "## 五、目标分级建议",
        "",
        "分级同时区分两个维度，避免把‘敏感程度’和‘容易预测’混为一谈：",
        "",
        "- T0：全年恒定，无时变信息，先从目标集合剥离。",
        "- T1：节点电压幅值，相对低敏但仍是运行状态；只能作为扩展 X 的敏感性实验。",
        "- T2：节点 P/Q 与平衡注入，中等敏感；不建议直接并入默认公开 X。",
        "- T3：节点相角、线路潮流/负载率、具名机组出力与状态，作为论文主目标池。",
        "",
        "在同一敏感等级内部，再按 public-27 模型的测试 R² 标注 strong/usable/weak/not-inferable。"
        "完整逐字段结果见 `target_recommendations.csv`；论文主实验可优先选择 T3 且"
        "跨种子 R² 稳定的目标，同时保留少量难推目标作为负对照。",
        "",
        "## 六、建议用于正式流程的 12 个核心目标",
        "",
        "为了控制后续逐目标剥离、初筛和门控实验的规模，建议主文先使用下面 12 个代表目标，"
        "其余时变目标放入补充材料。选择时优先保证物理对象、区域和机组类型覆盖，"
        "而不是只挑 R² 最高的字段；所有入选字段均为非恒定 T3 目标，且仅用 public-27 的"
        "分组普通 MLP 测试 R² 不低于 0.70。",
        "",
        "| 类别 | 目标字段 | 选择理由 | 分组 MLP R² | 跨种子标准差 |",
        "|---|---|---|---:|---:|",
    ]
    for row in core_targets.itertuples():
        lines.append(
            f"| {row.core_group} | `{row.field}` | {row.selection_reason} | "
            f"{row.r2_mean_grouped:.4f} | {row.r2_std_grouped:.4f} |"
        )
    hard = recommendations[
        recommendations["recommended_role"].isin(
            ["hard_target_or_negative_control", "test_segment_no_variance_needs_followup"]
        )
    ]
    lines += [
        "",
        "建议另保留以下难推/不可评估字段作为负对照或补充分析：",
        "",
    ]
    for row in hard.itertuples():
        value = "测试段无方差" if pd.isna(row.r2_mean_grouped) else f"分组 R²={row.r2_mean_grouped:.4f}"
        lines.append(f"- `{row.field}`：{value}。")
    lines += [
        "",
        "这 12 个字段是后续计算规模和论文展示上的代表集，不意味着其余 T3 字段不敏感。",
        "",
        "## 七、后续流程边界",
        "",
        "本次结果只能决定目标池怎样组织。正式流程仍应对每个选定目标分别执行："
        "目标字段剥离 → 是否初筛两条分支 → 普通 DNN 能力上限 → D-Gating 推断源定位。",
        "`03/04/05` 中移入 X 的细粒度字段不会自动进入正式默认候选发布集合。",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(Config.seeds))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", nargs="*", default=None, help="Only run matching partition prefixes")
    args = parser.parse_args()

    cfg = Config(epochs=args.epochs, seeds=tuple(args.seeds), device=args.device)
    if cfg.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("Requested MPS but torch.backends.mps.is_available() is false")
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))

    data = pd.read_csv(args.data)
    dictionary = pd.read_csv(args.dictionary, keep_default_na=False)
    if data.isna().any().any():
        raise ValueError("Input dataset contains missing values")
    if dictionary["column_name"].tolist() != data.columns.tolist():
        raise ValueError("Field dictionary does not exactly match dataset header")

    public = dictionary.loc[dictionary["role"].eq("published_candidate"), "column_name"].tolist()
    tiers = build_tiers(data, dictionary, cfg)
    tier_map = dict(zip(tiers["field"], tiers["sensitivity_tier"]))
    partitions = build_partitions(public, tiers)
    if args.only:
        partitions = [p for p in partitions if any(p["partition"].startswith(x) for x in args.only)]
    if not partitions:
        raise SystemExit("No partitions selected")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_end = int(len(data) * cfg.train_ratio)
    val_end = int(len(data) * (cfg.train_ratio + cfg.validation_ratio))
    variable_targets = tiers.loc[tiers["is_constant"].eq(0), "field"].tolist()

    tiers.to_csv(out / "sensitivity_tiers.csv", index=False)
    tiers[tiers["is_constant"].eq(1)].to_csv(out / "constant_targets.csv", index=False)
    intrinsic = intrinsic_dimension(data, variable_targets, train_end)
    intrinsic.to_csv(out / "target_intrinsic_dimension.csv", index=False)

    input_diag = pd.DataFrame(
        {
            "field": public,
            "variance": [float(data[f].var(ddof=0)) for f in public],
            "n_unique": [int(data[f].nunique(dropna=False)) for f in public],
        }
    )
    input_diag["is_constant"] = (input_diag["variance"] <= cfg.constant_tolerance).astype(int)
    input_diag.to_csv(out / "public_input_diagnostics.csv", index=False)

    config_payload = {
        **asdict(cfg),
        "seeds": list(cfg.seeds),
        "hidden": list(cfg.hidden),
        "data": str(args.data.resolve()),
        "dictionary": str(args.dictionary.resolve()),
        "n_rows": len(data),
        "train_rows": train_end,
        "validation_rows": val_end - train_end,
        "test_rows": len(data) - val_end,
        "train_time": [data["datetime_beginning"].iloc[0], data["datetime_beginning"].iloc[train_end - 1]],
        "validation_time": [data["datetime_beginning"].iloc[train_end], data["datetime_beginning"].iloc[val_end - 1]],
        "test_time": [data["datetime_beginning"].iloc[val_end], data["datetime_beginning"].iloc[-1]],
        "partitions": [
            {"partition": p["partition"], "description": p["description"], "x": p["x"], "y": p["y"]}
            for p in partitions
        ],
    }
    (out / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    all_metrics = []
    all_ridge = []
    run_records = []
    for partition in partitions:
        name = partition["partition"]
        pdir = out / name
        pdir.mkdir(parents=True, exist_ok=True)
        x_fields = partition["x"]
        y_fields = partition["y"]
        pd.DataFrame({"field": x_fields}).to_csv(pdir / "input_fields.csv", index=False)
        tiers.set_index("field").loc[y_fields].reset_index().to_csv(pdir / "target_fields.csv", index=False)
        X = data[x_fields].to_numpy(np.float32)
        Y = data[y_fields].to_numpy(np.float32)
        n_input_variable = int((X.var(axis=0) > cfg.constant_tolerance).sum())

        print(f"\n[{name}] X={len(x_fields)} ({n_input_variable} variable), Y={len(y_fields)}", flush=True)
        ridge = ridge_metrics(X, Y, y_fields, tier_map, train_end, val_end, cfg)
        ridge.insert(0, "partition", name)
        ridge["description"] = partition["description"]
        ridge.to_csv(pdir / "ridge_metrics.csv", index=False)
        all_ridge.append(ridge)

        for seed in cfg.seeds:
            sdir = pdir / f"seed_{seed}"
            sdir.mkdir(parents=True, exist_ok=True)
            metrics, history, meta = train_mlp(
                X, Y, y_fields, tier_map, train_end, val_end, cfg, seed
            )
            metrics.insert(0, "seed", seed)
            metrics.insert(0, "partition", name)
            metrics["description"] = partition["description"]
            metrics["n_input_declared"] = len(x_fields)
            metrics["n_input_variable"] = n_input_variable
            metrics.to_csv(sdir / "target_metrics.csv", index=False)
            history.to_csv(sdir / "training_history.csv", index=False)
            (sdir / "run_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            all_metrics.append(metrics)
            run_records.append({"partition": name, **meta})
            print(
                f"  seed={seed}: macro R2={metrics.r2.mean():.4f}, "
                f"median={metrics.r2.median():.4f}, best_epoch={meta['best_epoch']}, "
                f"{meta['elapsed_seconds']:.1f}s",
                flush=True,
            )

    metrics = pd.concat(all_metrics, ignore_index=True)
    ridge = pd.concat(all_ridge, ignore_index=True)
    metrics.to_csv(out / "all_runs_target_metrics.csv", index=False)
    ridge.to_csv(out / "all_partitions_ridge_metrics.csv", index=False)
    pd.DataFrame(run_records).to_csv(out / "run_timing.csv", index=False)
    target_summary = summarize_metrics(metrics, ["partition"])
    target_summary["inferability_grade"] = target_summary["r2_mean"].map(lambda x: grade_r2(x, cfg))
    target_summary.to_csv(out / "all_partitions_target_summary.csv", index=False)
    psummary = partition_summary(metrics, ridge, cfg)
    psummary.to_csv(out / "partition_summary.csv", index=False)

    base = target_summary[target_summary["partition"].eq("01_public27_joint")].copy()
    grouped = target_summary[target_summary["partition"].str.startswith("02_public27_grouped/")].copy()
    comparison = base[["target", "sensitivity_tier", "target_kind", "r2_mean", "r2_std"]].merge(
        grouped[["target", "r2_mean", "r2_std", "partition"]], on="target", suffixes=("_joint", "_grouped")
    )
    comparison["grouping_delta_r2"] = comparison["r2_mean_grouped"] - comparison["r2_mean_joint"]
    comparison.to_csv(out / "joint_vs_grouped_target_comparison.csv", index=False)

    rec = tiers.merge(
        comparison[["target", "r2_mean_joint", "r2_std_joint", "r2_mean_grouped", "r2_std_grouped", "grouping_delta_r2"]],
        left_on="field",
        right_on="target",
        how="left",
    ).drop(columns=["target"])
    rec["public27_inferability_grade"] = rec.apply(
        lambda row: "constant_not_evaluated"
        if row["sensitivity_tier"] == "T0_constant"
        else grade_r2(float(row["r2_mean_joint"]), cfg),
        axis=1,
    )
    def priority(row: pd.Series) -> str:
        if row["sensitivity_tier"] == "T0_constant":
            return "strip_before_modeling"
        score = row["r2_mean_grouped"]
        if not np.isfinite(score):
            return "test_segment_no_variance_needs_followup"
        if row["sensitivity_tier"] == "T3_high" and score >= cfg.usable_r2:
            return "primary_target"
        if row["sensitivity_tier"] == "T3_high":
            return "hard_target_or_negative_control"
        return "auxiliary_tier_experiment_not_default_public"
    rec["recommended_role"] = rec.apply(priority, axis=1)
    rec.to_csv(out / "target_recommendations.csv", index=False)

    core_def = pd.DataFrame(CORE_TARGETS)
    core = core_def.merge(rec, on="field", how="left", validate="one_to_one")
    if core["r2_mean_grouped"].isna().any() or (core["r2_mean_grouped"] < cfg.usable_r2).any():
        raise AssertionError("A recommended core target is missing or fails the configured usability floor")
    core.insert(0, "core_order", np.arange(1, len(core) + 1))
    core.to_csv(out / "recommended_core_targets.csv", index=False)

    create_plots(out, psummary, target_summary)
    write_summary(
        out, cfg, data, public, tiers, psummary, target_summary, intrinsic,
        core, rec, train_end, val_end,
    )
    print(f"\nCompleted: {out}", flush=True)


if __name__ == "__main__":
    main()
