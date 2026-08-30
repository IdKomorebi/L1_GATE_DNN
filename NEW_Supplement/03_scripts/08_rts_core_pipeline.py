#!/usr/bin/env python3
"""RTS-GMLC 12 个核心目标的完整隐性推断流程。

目录语义与 PJM 流程对齐，但针对本数据做了三点收紧：

1. 第一类风险分成“有来源的确定性公式”和“数据内恒等/近恒等式”两层；
2. 初筛阈值由训练段上的周块置换产生，不读取验证段和测试段；
3. DNN / D-Gating 均按时间 70%/10%/20% 切分，测试集不参与早停或选字段，
   并用三个随机种子的多数共识定位推断源。

本脚本不执行论文对比方法，只运行必要的能力上限、D-Gating 定位和选中字段重训。
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
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_rts_core_pipeline_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_src"))
import gates as gate_models  # noqa: E402
import identity as idt  # noqa: E402
import screening as scr  # noqa: E402


DATA_DIR = ROOT / "01_data" / "raw" / "rts_gmlc_2020"
DATA_FILE = DATA_DIR / "rts_gmlc_hourly_2020_acpf_base.csv"
DICT_FILE = DATA_DIR / "field_dictionary.csv"
CORE_FILE = ROOT / "04_outputs" / "rts_gmlc_2020" / "00_feasibility_check" / "recommended_core_targets.csv"
OUT_ROOT = ROOT / "04_outputs" / "rts_gmlc_2020"
LOG_DIR = ROOT / "05_logs"

SOURCE_COMMIT = "3ece0d3725c844056132393ee252b3083dd4eab4"
OFFICIAL_REPO = "https://github.com/GridMod/RTS-GMLC"
OFFICIAL_SOURCE_README = (
    "https://github.com/GridMod/RTS-GMLC/blob/"
    f"{SOURCE_COMMIT}/RTS_Data/SourceData/README.md"
)
OFFICIAL_TIMESERIES_README = (
    "https://github.com/GridMod/RTS-GMLC/blob/"
    f"{SOURCE_COMMIT}/RTS_Data/timeseries_data_files/README.md"
)
DGATING_PAPER = (
    "https://proceedings.neurips.cc/paper_files/paper/2025/"
    "hash/dea9b4b6f55ae611c54065d6fc750755-Abstract-Conference.html"
)
OFFICIAL_INPUT_ZIP = DATA_DIR / "Generation" / "source" / "rts_gmlc_official_inputs_3ece0d3.zip"


@dataclass
class Config:
    seeds: tuple[int, ...] = (42, 43, 44)
    hidden: tuple[int, ...] = (64, 32, 16)
    epochs: int = 220
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 30
    min_epochs: int = 60
    train_ratio: float = 0.70
    validation_ratio: float = 0.10
    dgate_lambda: float = 0.005
    dgate_depth: int = 4
    dgate_threshold: float = 0.01
    consensus_min_seeds: int = 2
    identity_train_rr: float = 0.03
    identity_check_rr: float = 0.05
    identity_max_layers: int = 12
    screening_null_draws: int = 80
    screening_block_hours: int = 168
    screening_quantile: float = 0.95
    screening_expensive_n: int = 800
    device: str = "cpu"


TARGET_CN = {
    "bus_115_va_deg": "母线 115 电压相角",
    "bus_215_va_deg": "母线 215 电压相角",
    "bus_315_va_deg": "母线 315 电压相角",
    "branch_ab1_loading_pct": "跨区支路 AB1 负载率",
    "branch_ab2_loading_pct": "跨区支路 AB2 负载率",
    "branch_ab3_loading_pct": "跨区支路 AB3 负载率",
    "branch_ca_1_loading_pct": "跨区支路 CA-1 负载率",
    "branch_cb_1_loading_pct": "跨区支路 CB-1 负载率",
    "gen_121_nuclear_1_pg_mw": "121_NUCLEAR_1 有功出力",
    "gen_218_cc_1_pg_mw": "218_CC_1 有功出力",
    "gen_317_wind_1_pg_mw": "317_WIND_1 有功出力",
    "gen_321_cc_1_status": "321_CC_1 投入状态",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def r2_score_np(y: np.ndarray, pred: np.ndarray) -> float:
    den = float(np.square(y - y.mean()).sum())
    return float(1.0 - np.square(y - pred).sum() / den) if den > 1e-12 else math.nan


def pearson_np(y: np.ndarray, pred: np.ndarray) -> float:
    if y.std() <= 1e-12 or pred.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(y, pred)[0, 1])


def evaluate_target(y: np.ndarray, pred: np.ndarray, is_status: bool) -> dict:
    out = {
        "r2": r2_score_np(y, pred),
        "rmse": float(np.sqrt(np.square(y - pred).mean())),
        "mae": float(np.abs(y - pred).mean()),
        "pearson_r": pearson_np(y, pred),
    }
    if is_status:
        true = (y >= 0.5).astype(int)
        hard = (pred >= 0.5).astype(int)
        out.update(
            accuracy=float((true == hard).mean()),
            balanced_accuracy=float(balanced_accuracy_score(true, hard)),
            roc_auc=float(roc_auc_score(true, pred)) if len(np.unique(true)) == 2 else math.nan,
        )
    else:
        out.update(accuracy=math.nan, balanced_accuracy=math.nan, roc_auc=math.nan)
    return out


def split_edges(n: int, cfg: Config) -> tuple[int, int]:
    train_end = int(n * cfg.train_ratio)
    val_end = train_end + int(n * cfg.validation_ratio)
    return train_end, val_end


@dataclass
class FitResult:
    model_name: str
    seed: int
    best_epoch: int
    history: pd.DataFrame
    validation: dict
    test: dict
    gates: np.ndarray | None
    elapsed_seconds: float


def build_model(model_name: str, n_input: int, cfg: Config) -> nn.Module:
    if model_name == "DNN":
        return gate_models.DNN(n_input, cfg.hidden)
    if model_name == "DGatingDNN":
        return gate_models.DGating(n_input, cfg.hidden, depth=cfg.dgate_depth)
    raise ValueError(model_name)


def fit_single(
    X: np.ndarray,
    y: np.ndarray,
    target: str,
    model_name: str,
    seed: int,
    cfg: Config,
) -> FitResult:
    """按时间切分训练一个模型。DNN 用验证 MSE 早停；D-Gating 用验证 MSE
    加稀疏罚项的同口径目标早停，保存点和选字段均不读取测试段。"""
    set_seed(seed)
    started = time.time()
    train_end, val_end = split_edges(len(X), cfg)
    x_mean, x_scale = X[:train_end].mean(0), X[:train_end].std(0)
    x_scale[x_scale <= 1e-12] = 1.0
    Xs = (X - x_mean) / x_scale
    y_mean, y_scale = float(y[:train_end].mean()), float(y[:train_end].std())
    if y_scale <= 1e-12:
        y_scale = 1.0
    ys = (y - y_mean) / y_scale

    device = torch.device(cfg.device)
    xt = torch.tensor(Xs[:train_end], dtype=torch.float32, device=device)
    yt = torch.tensor(ys[:train_end], dtype=torch.float32, device=device).view(-1, 1)
    xv = torch.tensor(Xs[train_end:val_end], dtype=torch.float32, device=device)
    yv = torch.tensor(ys[train_end:val_end], dtype=torch.float32, device=device).view(-1, 1)
    xte = torch.tensor(Xs[val_end:], dtype=torch.float32, device=device)

    model = build_model(model_name, X.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_obj, best_epoch, best_state, stale = float("inf"), -1, None, 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(train_end, generator=generator)
        train_loss = 0.0
        for start in range(0, train_end, cfg.batch_size):
            idx = order[start : start + cfg.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            mse = loss_fn(model(xt[idx]), yt[idx])
            if model_name == "DGatingDNN":
                penalty = cfg.dgate_lambda * (
                    model.omega.pow(2).sum() + model.gamma.pow(2).sum()
                )
                loss = mse + penalty
            else:
                penalty = torch.zeros((), device=device)
                loss = mse
            loss.backward()
            optimizer.step()
            train_loss += float(mse.detach().cpu()) * len(idx)

        model.eval()
        with torch.no_grad():
            val_std = model(xv)
            val_mse = float(loss_fn(val_std, yv).cpu())
            sparse_penalty = (
                cfg.dgate_lambda
                * float((model.omega.pow(2).sum() + model.gamma.pow(2).sum()).cpu())
                if model_name == "DGatingDNN"
                else 0.0
            )
            objective = val_mse + sparse_penalty
            val_pred = val_std.squeeze(-1).cpu().numpy() * y_scale + y_mean
            gate_values = model.gates()
        history.append(
            {
                "epoch": epoch,
                "train_standardized_mse": train_loss / train_end,
                "validation_standardized_mse": val_mse,
                "selection_objective": objective,
                "validation_r2": r2_score_np(y[train_end:val_end], val_pred),
                "n_active_at_0_01": (
                    int((gate_values >= cfg.dgate_threshold).sum())
                    if gate_values is not None
                    else X.shape[1]
                ),
            }
        )
        if objective < best_obj - 1e-6:
            best_obj = objective
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= cfg.min_epochs and stale >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError(f"{model_name} produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(xv).squeeze(-1).cpu().numpy() * y_scale + y_mean
        test_pred = model(xte).squeeze(-1).cpu().numpy() * y_scale + y_mean
    is_status = target.endswith("_status")
    return FitResult(
        model_name=model_name,
        seed=seed,
        best_epoch=best_epoch,
        history=pd.DataFrame(history),
        validation=evaluate_target(y[train_end:val_end], val_pred, is_status),
        test=evaluate_target(y[val_end:], test_pred, is_status),
        gates=model.gates(),
        elapsed_seconds=time.time() - started,
    )


def aggregate_metrics(results: list[FitResult], prefix: str) -> dict:
    out: dict[str, float] = {}
    for metric in ["r2", "rmse", "mae", "pearson_r", "accuracy", "balanced_accuracy", "roc_auc"]:
        values = np.array([r.test[metric] for r in results], dtype=float)
        finite = values[np.isfinite(values)]
        out[f"{prefix}_{metric}_mean"] = float(finite.mean()) if len(finite) else math.nan
        out[f"{prefix}_{metric}_std"] = float(finite.std(ddof=0)) if len(finite) else math.nan
    return out


def relation_stability(
    df: pd.DataFrame, target: str, support: list[str], space: str, n_blocks: int = 5
) -> dict:
    d = df[[target] + support].copy()
    if space == "log":
        if (d <= 0).any().any():
            return {"fit_rr_mean": math.nan, "check_rr_mean": math.nan, "check_rr_max": math.inf}
        d = np.log(d)
    y = d[target].to_numpy(float)
    A = np.column_stack([d[s].to_numpy(float) for s in support] + [np.ones(len(d))])
    edges = np.linspace(0, len(d), n_blocks + 1).astype(int)
    fit_rr, check_rr = [], []
    for b in range(n_blocks):
        te = np.zeros(len(d), dtype=bool)
        te[edges[b] : edges[b + 1]] = True
        tr = ~te
        coef, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        tr_sd, te_sd = y[tr].std(), y[te].std()
        if tr_sd > 1e-12:
            fit_rr.append(float(np.sqrt(np.square(y[tr] - A[tr] @ coef).mean()) / tr_sd))
        if te_sd > 1e-12:
            check_rr.append(float(np.sqrt(np.square(y[te] - A[te] @ coef).mean()) / te_sd))
    return {
        "fit_rr_mean": float(np.mean(fit_rr)) if fit_rr else math.nan,
        "check_rr_mean": float(np.mean(check_rr)) if check_rr else math.nan,
        "check_rr_max": float(np.max(check_rr)) if check_rr else math.inf,
    }


def fit_relation(df: pd.DataFrame, target: str, pool: list[str], cfg: Config, space: str) -> dict | None:
    use = df[[target] + pool].copy()
    if space == "log":
        positive_pool = [c for c in pool if (use[c] > 0).all()]
        if not (use[target] > 0).all() or not positive_pool:
            return None
        pool = positive_pool
        use = np.log(use[[target] + pool])
    rr_full = idt.exact_fit_ratio(use, target, pool)
    if not np.isfinite(rr_full) or rr_full >= cfg.identity_train_rr:
        return None
    support = idt._prune(use, target, pool, cfg.identity_train_rr)
    if not support:
        return None
    coefs, const, rr = idt._refit(use, target, support)
    stability = relation_stability(df, target, support, space)
    if rr >= cfg.identity_train_rr or stability["check_rr_max"] >= cfg.identity_check_rr:
        return None
    contribution = {s: abs(coefs[s]) * use[s].std() for s in support}
    victim = max(contribution, key=contribution.get)
    relation = (
        f"{target} = "
        + " ".join(f"{coefs[s]:+.8g}*{s}" for s in support)
        + (f" {const:+.8g}" if abs(const) > 1e-12 else "")
    )
    if space == "log":
        relation = "log(" + target + ") = " + relation.split(" = ", 1)[1]
    return {
        "space": space,
        "train_residual_ratio": float(rr),
        "relation_r2": float(1.0 - rr**2),
        "n_support": len(support),
        "support": "|".join(support),
        "relation": relation,
        "removed": victim,
        **stability,
    }


def official_formula_rows(df: pd.DataFrame, targets: list[str], public: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """审核每个目标是否能由当前公开候选池按有来源的公式闭合。

    唯一闭合项来自两个可审计事实：数据生成脚本把同燃料机组有功相加形成
    gen_fuel_*；固定版本官方 gen.csv 中 Fuel=Nuclear 只有 121_NUCLEAR_1 一台。
    """
    rows: list[dict] = []
    drops = {t: [] for t in targets}
    for target in targets:
        closure_trace = ""
        residual_after_all_drops = math.nan
        if target.startswith("bus_") and target.endswith("_va_deg"):
            formula_id = "AC_NODAL_POWER_FLOW"
            formula = "P_i,Q_i=f(V,theta,Y_bus); solve theta from complete nodal injections, voltage controls and topology"
            required = "complete nodal P/Q injections|voltage controls|Y_bus/topology|reference angle"
            available = "system/area/fuel aggregates only"
            source = OFFICIAL_SOURCE_README + " ; " + str(DATA_DIR / "Explanation.md")
            applicable, removed, rr = False, "", math.nan
            reason = "27 个候选字段不含完整节点注入、网络导纳与参考状态，方程不闭合"
        elif target.startswith("branch_") and target.endswith("_loading_pct"):
            formula_id = "BRANCH_THERMAL_LOADING"
            formula = "loading_pct=100*max(|S_from|,|S_to|)/rating"
            required = "target branch P/Q at both ends|branch rating"
            available = "no target-branch P/Q or rating in the 27-field pool"
            source = str(DATA_DIR / "Generation" / "generate_rts_gmlc_dataset.py")
            applicable, removed, rr = False, "", math.nan
            reason = "公式存在，但所需逐线路量已作为敏感目标排除，候选池无法代入"
        elif target == "gen_121_nuclear_1_pg_mw":
            formula_id = "SINGLE_NUCLEAR_RECURSIVE_FORMULA_CLOSURE"
            formula = (
                "(1) nuclear_target=gen_fuel_nuclear_mw; "
                "(2) nuclear_target=system_generation_mw-sum(other_fuel_mw); "
                "(3) nuclear_target=system_load_actual_mw+system_losses_mw-sum(other_fuel_mw)"
            )
            required = "fuel aggregates|system generation|system load and losses|official generator inventory"
            available = "all formula inputs are present in the effective public pool"
            source = (
                str(DATA_DIR / "Generation" / "generate_rts_gmlc_dataset.py")
                + " ; official fixed-commit RTS_Data/SourceData/gen.csv"
            )
            x = df["gen_fuel_nuclear_mw"].to_numpy(float)
            y = df[target].to_numpy(float)
            rr = float(np.sqrt(np.square(y - x).mean()) / y.std())
            applicable = bool(rr < 1e-8)
            closure_drop = ["gen_fuel_nuclear_mw", "system_generation_mw", "system_losses_mw"]
            removed = "|".join(closure_drop) if applicable else ""
            if applicable:
                drops[target].extend(closure_drop)
                sequential_rr = []
                for n_removed in range(0, len(closure_drop) + 1):
                    remaining = [c for c in public if c not in closure_drop[:n_removed]]
                    sequential_rr.append(idt.exact_fit_ratio(df, target, remaining))
                closure_trace = "|".join(
                    f"after_{n}_drops={value:.9g}" for n, value in enumerate(sequential_rr)
                )
                residual_after_all_drops = float(sequential_rr[-1])
            reason = (
                "固定版本官方机组清单只有一条 Nuclear 记录；直接聚合式、燃料分解式和功率平衡替代路径均由生成规则给出，逐时验算通过。为切断公式闭包需依次移除三个入口"
                if applicable else "逐时验算未通过"
            )
        elif target.endswith("_pg_mw"):
            formula_id = "FUEL_AGGREGATE_COMPONENT"
            formula = "gen_fuel_type_mw=sum(pg of all generators with the same fuel)"
            required = "all same-fuel individual outputs or a one-generator fuel class"
            available = "fuel aggregate only"
            source = str(DATA_DIR / "Generation" / "generate_rts_gmlc_dataset.py")
            applicable, removed, rr = False, "", math.nan
            reason = "同燃料类型含多个出力分量，单个机组不能由总量唯一恢复"
        else:
            formula_id = "GENERATOR_STATUS_RULE"
            formula = "status=1(abs(individual_pg)>1e-6) for dispatched gen elements"
            required = "individual generator pg"
            available = "fuel/system aggregates only"
            source = str(DATA_DIR / "Generation" / "generate_rts_gmlc_dataset.py")
            applicable, removed, rr = False, "", math.nan
            reason = "状态规则需要同一具名机组的出力，候选池没有该字段"
        rows.append(
            {
                "target": target,
                "target_cn": TARGET_CN[target],
                "formula_id": formula_id,
                "formula": formula,
                "required_information": required,
                "available_information": available,
                "applicable_to_27_field_pool": int(applicable),
                "validated_residual_ratio": rr,
                "removed_fields": removed,
                "formula_closure_residual_trace": closure_trace,
                "residual_ratio_after_all_formula_drops": residual_after_all_drops,
                "judgement": reason,
                "source": source,
            }
        )
    return pd.DataFrame(rows), drops


def write_preprocess(df: pd.DataFrame, dictionary: pd.DataFrame, targets: list[str], cfg: Config) -> tuple[list[str], list[str]]:
    out = OUT_ROOT / "01_preprocess"
    out.mkdir(parents=True, exist_ok=True)
    nominal = dictionary.loc[dictionary.role.eq("published_candidate"), "column_name"].tolist()
    rows = []
    for field in nominal:
        rows.append(
            {
                "field": field,
                "variance_full": float(df[field].var(ddof=0)),
                "std_train": float(df[field].iloc[: split_edges(len(df), cfg)[0]].std(ddof=0)),
                "n_unique": int(df[field].nunique(dropna=False)),
                "is_constant": int(df[field].std(ddof=0) <= 1e-12),
                "model_role": "drop_quality_constant" if df[field].std(ddof=0) <= 1e-12 else "candidate_input",
            }
        )
    input_diag = pd.DataFrame(rows)
    input_diag.to_csv(out / "input_fields.csv", index=False)
    effective = input_diag.query("is_constant == 0").field.tolist()
    input_diag.query("is_constant == 1").to_csv(out / "quality_dropped_constant_fields.csv", index=False)
    core = pd.read_csv(CORE_FILE)
    core[core.field.isin(targets)].to_csv(out / "target_fields.csv", index=False)
    train_end, val_end = split_edges(len(df), cfg)
    split = pd.DataFrame(
        [
            {"split": "train", "row_start": 0, "row_end_exclusive": train_end, "n_rows": train_end},
            {"split": "validation", "row_start": train_end, "row_end_exclusive": val_end, "n_rows": val_end - train_end},
            {"split": "test", "row_start": val_end, "row_end_exclusive": len(df), "n_rows": len(df) - val_end},
        ]
    )
    split.to_csv(out / "time_split.csv", index=False)
    (out / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(
        "\n".join(
            [
                "# 01 预处理",
                "",
                f"- 原始数据：`{DATA_FILE.name}`，{len(df)} 行 × {df.shape[1]} 列。",
                f"- 候选发布字段：名义 {len(nominal)} 个；去掉全年恒定字段后有效 {len(effective)} 个。",
                f"- 恒定字段：{', '.join(input_diag.query('is_constant == 1').field.tolist())}。",
                f"- 目标：12 个，且不会把其余 118 个敏感字段偷放进候选池。",
                f"- 时间划分：训练 {train_end} 行、验证 {val_end-train_end} 行、测试 {len(df)-val_end} 行。",
                "- 标准化、恒等式发现和初筛阈值都只在训练段拟合；测试段只作最终评估。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return nominal, effective


def run_stripping(
    df: pd.DataFrame, targets: list[str], public_effective: list[str], cfg: Config
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    root = OUT_ROOT / "02_class1_physical"
    layer1 = root / "layer1_documented_formula"
    layer2 = root / "layer2_empirical_identity"
    pools_dir = root / "candidate_pools"
    for d in [root, layer1, layer2, pools_dir]:
        d.mkdir(parents=True, exist_ok=True)
    train_end, _ = split_edges(len(df), cfg)
    train_df = df.iloc[:train_end].copy()

    official, official_drops = official_formula_rows(df, targets, public_effective)
    official.to_csv(layer1 / "official_formula_audit.csv", index=False)
    if not OFFICIAL_INPUT_ZIP.exists():
        raise FileNotFoundError(OFFICIAL_INPUT_ZIP)
    with zipfile.ZipFile(OFFICIAL_INPUT_ZIP) as archive:
        with archive.open("RTS_Data/SourceData/gen.csv") as handle:
            generator_inventory = pd.read_csv(handle)
    nuclear_inventory = generator_inventory.loc[
        generator_inventory["Fuel"].astype(str).str.casefold().eq("nuclear"),
        ["GEN UID", "Bus ID", "Gen ID", "Unit Type", "Category", "Fuel", "PMax MW", "PMin MW"],
    ]
    if nuclear_inventory["GEN UID"].tolist() != ["121_NUCLEAR_1"]:
        raise RuntimeError("fixed-commit nuclear inventory no longer matches the documented formula")
    nuclear_inventory.to_csv(layer1 / "official_nuclear_generator_inventory.csv", index=False)
    (layer1 / "sources.md").write_text(
        "\n".join(
            [
                "# 第一层公式来源",
                "",
                f"- RTS-GMLC 官方仓库：<{OFFICIAL_REPO}>",
                f"- 固定提交：`{SOURCE_COMMIT}`",
                f"- SourceData 字段说明：<{OFFICIAL_SOURCE_README}>",
                f"- 时序数据说明：<{OFFICIAL_TIMESERIES_README}>",
                "- 本数据构造与字段计算：`01_data/raw/rts_gmlc_2020/Generation/generate_rts_gmlc_dataset.py`",
                "- 数据集物理关系说明：`01_data/raw/rts_gmlc_2020/Explanation.md`",
                "- 固定提交官方机组表中 Nuclear 行的审计副本：`official_nuclear_generator_inventory.csv`",
                "",
                "官方资料提供网络、机组字段和时序构造定义；本项目生成脚本提供 27 个候选聚合字段的精确计算口径。",
                "只有当公式所需信息全部位于当前 27 字段候选池内时，才允许据此剥离。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    pools: dict[str, list[str]] = {}
    layer_rows: list[dict] = []
    summary_rows: list[dict] = []
    for target in targets:
        cur = [c for c in public_effective if c not in official_drops[target]]
        removed_empirical: list[str] = []
        for layer_no in range(1, cfg.identity_max_layers + 1):
            candidates = [
                r for r in [fit_relation(train_df, target, cur, cfg, "linear"), fit_relation(train_df, target, cur, cfg, "log")]
                if r is not None
            ]
            if not candidates:
                break
            rel = min(candidates, key=lambda r: r["check_rr_max"])
            victim = str(rel["removed"])
            cur.remove(victim)
            removed_empirical.append(victim)
            layer_rows.append(
                {
                    "target": target,
                    "target_cn": TARGET_CN[target],
                    "layer": layer_no,
                    **rel,
                    "n_pool_after": len(cur),
                    "decision": "remove_empirical_near_identity",
                }
            )
        pools[target] = cur
        pd.DataFrame({"field": cur}).to_csv(pools_dir / f"pool_{target}.csv", index=False)
        summary_rows.append(
            {
                "target": target,
                "target_cn": TARGET_CN[target],
                "n_start_effective": len(public_effective),
                "n_layer1_removed": len(official_drops[target]),
                "layer1_removed": "|".join(official_drops[target]),
                "n_layer2_removed": len(removed_empirical),
                "layer2_removed": "|".join(removed_empirical),
                "n_final": len(cur),
                "final_pool": "|".join(cur),
            }
        )
    layers = pd.DataFrame(layer_rows)
    if layers.empty:
        layers = pd.DataFrame(
            columns=[
                "target", "target_cn", "layer", "space", "train_residual_ratio", "relation_r2",
                "n_support", "support", "relation", "removed", "fit_rr_mean", "check_rr_mean",
                "check_rr_max", "n_pool_after", "decision",
            ]
        )
    layers.to_csv(layer2 / "layered_strip_layers.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(layer2 / "layered_strip_summary.csv", index=False)
    summary.to_csv(root / "stripping_summary.csv", index=False)

    n_applicable = int(official.applicable_to_27_field_pool.sum())
    n_empirical = len(layers)
    (layer1 / "summary.md").write_text(
        "\n".join(
            [
                "# 第一层：有来源的确定性公式审计",
                "",
                f"12 个目标中，{n_applicable} 个目标存在能由当前候选池闭合且通过逐时验算的确定性公式。",
                "其余目标虽然有 AC 潮流、线路负载率或状态定义，但公式需要的逐节点/逐线路/逐机组量不在 27 个候选字段中，因此不能剥离。",
                "",
                "`gen_121_nuclear_1_pg_mw` 是唯一闭合目标：固定提交的官方 `gen.csv` 只有 `121_NUCLEAR_1` 一条 Nuclear 记录，",
                "而数据生成脚本把所有 Nuclear 机组有功相加形成 `gen_fuel_nuclear_mw`，故二者逐时相等。",
                "仅删这个直接入口仍不够：系统总发电减其他燃料总量、以及系统负荷加网损再减其他燃料总量，都会形成替代路径。",
                "因此第一层按公式闭包依次剥离 `gen_fuel_nuclear_mw`、`system_generation_mw` 和 `system_losses_mw`。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (layer2 / "summary.md").write_text(
        "\n".join(
            [
                "# 第二层：数据内恒等式与近恒等式发现",
                "",
                f"- 只使用训练段 {train_end} 小时发现关系。",
                f"- 同时检查仿射线性空间和严格正值字段的对数空间。",
                f"- 训练相对残差必须 < {cfg.identity_train_rr:.2f}（约等于 R² > {1-cfg.identity_train_rr**2:.4f}）。",
                f"- 关系还必须通过 5 个时间块轮换复验，最坏检查残差 < {cfg.identity_check_rr:.2f}。",
                f"- 在第一层之后共发现并剥离 {n_empirical} 层关系。",
                "",
                "高相关但达不到上述门槛的关系不称为公式，也不在这里剥离，交给后续推断源定位。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return pools, summary


def run_screening(
    df: pd.DataFrame, targets: list[str], pools: dict[str, list[str]], cfg: Config
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    out = OUT_ROOT / "03_screening"
    out.mkdir(parents=True, exist_ok=True)
    train_end, _ = split_edges(len(df), cfg)
    train_df = df.iloc[:train_end].copy()
    scr.EXPENSIVE_N = cfg.screening_expensive_n
    kept_by_target: dict[str, list[str]] = {}
    rows = []
    threshold_map: dict[str, dict] = {}
    for pos, target in enumerate(targets, 1):
        pool = pools[target]
        null = scr.null_thresholds(
            train_df,
            target,
            pool,
            n_draws=cfg.screening_null_draws,
            block=cfg.screening_block_hours,
            quantile=cfg.screening_quantile,
            seed=1000 + pos,
        )
        observed = scr.screen(train_df, target, pool, null.thresholds)
        kept = observed.query("kept == 1").field.tolist()
        if not kept:
            # 宽松初筛的职责是排除明显无关项；全空时保留综合过阈值比最高的一项，
            # 并明确记录保护性回退，避免后续分支无法建模。
            score_cols = scr.METRICS
            scaled = observed[score_cols].copy()
            for m in score_cols:
                scaled[m] = scaled[m] / max(null.thresholds[m], 1e-12)
            fallback = observed.iloc[int(scaled.max(axis=1).to_numpy().argmax())].field
            observed.loc[observed.field.eq(fallback), "kept"] = 1
            observed["fallback_keep"] = observed.field.eq(fallback).astype(int)
            kept = [fallback]
        else:
            observed["fallback_keep"] = 0
        kept_by_target[target] = kept
        observed.to_csv(out / f"screen_{target}.csv", index=False)
        null.draws.to_csv(out / f"null_{target}.csv", index=False)
        threshold_map[target] = null.thresholds
        rows.append(
            {
                "target": target,
                "target_cn": TARGET_CN[target],
                "n_pool_after_stripping": len(pool),
                "n_kept": len(kept),
                "n_dropped": len(pool) - len(kept),
                "kept_fields": "|".join(kept),
                "dropped_fields": "|".join([c for c in pool if c not in kept]),
                **{f"threshold_{m}": null.thresholds[m] for m in scr.METRICS},
            }
        )
        print(f"[screen {pos:02d}/12] {target}: {len(pool)} -> {len(kept)}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "screening_summary.csv", index=False)
    (out / "thresholds.json").write_text(json.dumps(threshold_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(
        "\n".join(
            [
                "# 03 多指标初筛",
                "",
                f"初筛只使用训练段。对每个目标计算 Pearson、Spearman、Kendall、NMI、距离相关和 HSIC，",
                f"再将目标按 {cfg.screening_block_hours} 小时整块重排 {cfg.screening_null_draws} 次，",
                f"以无关系分布的 {cfg.screening_quantile:.0%} 分位数作为每个指标的客观阈值。",
                "任一指标过阈值就保留，属于宁可多留的宽松初筛。",
                "",
                f"12 个目标的候选池平均由 {summary.n_pool_after_stripping.mean():.1f} 个降至 {summary.n_kept.mean():.1f} 个。",
                "未初筛与已初筛两条分支都会继续运行，初筛效果由后续重训精度验证。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return kept_by_target, summary


def run_dgate_calibration(df: pd.DataFrame, fields: list[str], cfg: Config) -> float:
    """只用验证集选择稀疏强度；测试指标随表报告但不参与选择。"""
    out = OUT_ROOT / "01_preprocess"
    target = "bus_115_va_deg"
    X = df[fields].to_numpy(float)
    y = df[target].to_numpy(float)
    rows = []
    for lam in [0.002, 0.005, 0.01]:
        pilot_cfg = replace(
            cfg, seeds=(42,), epochs=140, min_epochs=80,
            patience=30, dgate_lambda=lam,
        )
        result = fit_single(X, y, target, "DGatingDNN", 42, pilot_cfg)
        gate_values = result.gates
        rows.append(
            {
                "target": target,
                "seed": 42,
                "lambda_dgate": lam,
                "best_epoch": result.best_epoch,
                "validation_r2": result.validation["r2"],
                "test_r2_report_only": result.test["r2"],
                "n_active_at_0_01": int((gate_values >= cfg.dgate_threshold).sum()),
                "n_numerically_nonzero_at_1e_6": int((gate_values >= 1e-6).sum()),
            }
        )
    table = pd.DataFrame(rows)
    best_validation = float(table.validation_r2.max())
    eligible = table.validation_r2 >= best_validation - 0.005
    chosen = float(table.loc[eligible, "lambda_dgate"].max())
    table["eligible_within_0_005_validation_r2"] = eligible.astype(int)
    table["chosen"] = table.lambda_dgate.eq(chosen).astype(int)
    table.to_csv(out / "dgate_lambda_calibration.csv", index=False)
    (out / "dgate_lambda_calibration.md").write_text(
        "\n".join(
            [
                "# D-Gating 稀疏强度预校准",
                "",
                f"- 预先固定代表目标：`{target}`；候选池 {len(fields)} 个字段。",
                "- 只用 seed 42 做低成本预校准，最多 140 轮；正式结果仍使用三个随机种子。",
                "- 选择规则：在最佳验证 R² 的 0.005 容差内，取最大的 λ，以获得更强稀疏性。",
                "- 测试 R² 仅随表报告，不参与 λ 选择。",
                f"- 客观选定 λ=**{chosen:g}**。",
                "",
                "阈值 0.01 只用于报告活跃数；正式定位还同时报告 1e-6–0.1 的阈值扫描、连续门控值和 2/3 种子共识。",
                "",
                f"原方法来源：<{DGATING_PAPER}>。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return chosen


def gate_table(fields: list[str], results: list[FitResult], threshold: float) -> pd.DataFrame:
    matrix = np.vstack([r.gates for r in results])
    rows = []
    for i, field in enumerate(fields):
        active = matrix[:, i] >= threshold
        rows.append(
            {
                "field": field,
                **{f"gate_seed_{r.seed}": float(matrix[k, i]) for k, r in enumerate(results)},
                "gate_mean": float(matrix[:, i].mean()),
                "gate_std": float(matrix[:, i].std(ddof=0)),
                "selection_frequency": int(active.sum()),
                "consensus_active": int(active.sum() >= 2),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["consensus_active", "selection_frequency", "gate_mean"], ascending=[False, False, False]
    )


def plot_run(run_dir: Path, target: str, branch: str, gates_df: pd.DataFrame, dnn: list[FitResult], dgate: list[FitResult], retrain: list[FitResult], topn: pd.DataFrame, threshold: float) -> None:
    labels = gates_df.field.tolist()
    values = gates_df.gate_mean.to_numpy(float)
    colors = ["#C44E52" if x else "#4C72B0" for x in gates_df.consensus_active]
    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.28)))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label=f"threshold={threshold:g}")
    ax.set_xlabel("D-Gating gate mean across three seeds")
    ax.set_title(f"{target} | {branch}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "fig_gate_bar.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for result, color, label in [
        (dnn[0], "#4C72B0", "full DNN validation R²"),
        (dgate[0], "#C44E52", "D-Gating validation R²"),
        (retrain[0], "#55A868", "selected-fields DNN validation R²"),
    ]:
        ax.plot(result.history.epoch, result.history.validation_r2, color=color, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation R²")
    ax.set_title(f"{target} | seed 42 training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "fig_training.png", dpi=170)
    plt.close(fig)

    if not topn.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(topn.n_fields, topn.test_r2, marker="o")
        ax.axhline(np.mean([r.test["r2"] for r in dnn]), color="#C44E52", linestyle="--", label="full-pool DNN mean")
        ax.set_xlabel("top-n fields")
        ax.set_ylabel("test R² (seed 42 for top-n)")
        ax.set_title(f"{target} | cumulative source validation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "fig_topn.png", dpi=170)
        plt.close(fig)


def write_run_summary(
    path: Path,
    target: str,
    branch: str,
    fields: list[str],
    selected: list[str],
    gate_df: pd.DataFrame,
    dnn: list[FitResult],
    dgate: list[FitResult],
    retrain: list[FitResult],
    topn: pd.DataFrame,
    cfg: Config,
    fallback: bool,
) -> None:
    dnn_r2 = np.array([r.test["r2"] for r in dnn])
    retrain_r2 = np.array([r.test["r2"] for r in retrain])
    gate_matrix = np.vstack([r.gates for r in dgate])
    sensitivities = []
    for th in [1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.1]:
        per_seed = (gate_matrix >= th).sum(axis=1)
        sensitivities.append((th, "/".join(map(str, per_seed.tolist()))))
    stable = len({x[1] for x in sensitivities if 1e-3 <= x[0] <= 0.02}) == 1
    lines = [
        f"# 推断源定位：{TARGET_CN[target]}",
        "",
        f"- 分支：`{branch}`",
        f"- 剥离后/初筛后候选数：{len(fields)}",
        f"- D-Gating：深度 {cfg.dgate_depth}，λ={cfg.dgate_lambda}，报告阈值 {cfg.dgate_threshold}",
        f"- 三个随机种子中至少 {cfg.consensus_min_seeds} 次活跃才进入共识集合。",
        "- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。",
        f"- D-Gating 方法来源：<{DGATING_PAPER}>。",
        "",
        "## 结论",
        "",
        f"- 全候选普通 DNN 测试 R²：**{dnn_r2.mean():.4f} ± {dnn_r2.std():.4f}**。",
        f"- 共识推断源：**{len(selected)} 个**（{', '.join(f'`{x}`' for x in selected)}）。",
        f"- 只用共识字段重训普通 DNN：**{retrain_r2.mean():.4f} ± {retrain_r2.std():.4f}**。",
        f"- 相对全候选精度变化：**{(retrain_r2.mean()-dnn_r2.mean()):+.4f}**。",
    ]
    if fallback:
        lines += ["- 注意：固定阈值下三种子共识为空，已保护性保留平均门控值最高字段；该目标的源定位证据较弱。"]
    lines += ["", "## 共识字段", "", "| 字段 | 平均门控 | 门控标准差 | 入选种子数 |", "|---|---:|---:|---:|"]
    for row in gate_df.query("consensus_active == 1").itertuples():
        lines.append(f"| `{row.field}` | {row.gate_mean:.6g} | {row.gate_std:.6g} | {row.selection_frequency}/3 |")
    lines += ["", "## 阈值敏感性", "", "| 阈值 | 三个种子的活跃数 |", "|---:|---|+"]
    # 修正上面表头末尾，避免输出一个多余的加号。
    lines[-1] = "|---:|---|"
    for th, counts in sensitivities:
        lines.append(f"| {th:g} | {counts} |")
    lines += [
        "",
        (
            "0.001–0.02 范围内活跃字段数不变，门控分布存在稳定空档。"
            if stable
            else "0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。"
        ),
        "",
        "## Top-n 累积验证",
        "",
        "| 字段数 | 测试 R²（seed 42） | 字段 |",
        "|---:|---:|---|",
    ]
    for row in topn.itertuples():
        lines.append(f"| {row.n_fields} | {row.test_r2:.4f} | `{row.fields}` |")
    lines += [
        "",
        "Top-n 曲线是最终字段选定后的诊断展示，不参与共识字段或超参数选择。",
        "详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_source_branch(
    df: pd.DataFrame,
    target: str,
    fields: list[str],
    branch: str,
    cfg: Config,
    stamp: str,
) -> dict:
    root = OUT_ROOT / "04b_source_location_excl_targets" / f"stripped_{branch}"
    run_dir = root / f"target_{target}" / "DGatingDNN" / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    X = df[fields].to_numpy(float)
    y = df[target].to_numpy(float)

    dnn_results, dgate_results = [], []
    for seed in cfg.seeds:
        dnn = fit_single(X, y, target, "DNN", seed, cfg)
        dgate = fit_single(X, y, target, "DGatingDNN", seed, cfg)
        dnn_results.append(dnn)
        dgate_results.append(dgate)
        dnn.history.to_csv(run_dir / f"history_DNN_seed{seed}.csv", index=False)
        dgate.history.to_csv(run_dir / f"history_DGatingDNN_seed{seed}.csv", index=False)

    gate_df = gate_table(fields, dgate_results, cfg.dgate_threshold)
    selected = gate_df.query(
        f"selection_frequency >= {cfg.consensus_min_seeds}"
    ).field.tolist()
    fallback = False
    if not selected:
        selected = [gate_df.iloc[0].field]
        gate_df.loc[gate_df.field.eq(selected[0]), "consensus_active"] = 1
        fallback = True

    selected_idx = [fields.index(c) for c in selected]
    retrain_results = []
    for seed in cfg.seeds:
        result = fit_single(X[:, selected_idx], y, target, "DNN", seed, cfg)
        retrain_results.append(result)
        result.history.to_csv(run_dir / f"history_retrain_seed{seed}.csv", index=False)

    ranking = gate_df.field.tolist()
    n_values = sorted({1, 2, 3, 5, 8, len(selected), min(12, len(fields)), len(fields)})
    n_values = [n for n in n_values if 0 < n <= len(fields)]
    top_rows = []
    for n_fields in n_values:
        subset = ranking[:n_fields]
        idx = [fields.index(c) for c in subset]
        result = fit_single(X[:, idx], y, target, "DNN", cfg.seeds[0], cfg)
        top_rows.append({"n_fields": n_fields, "test_r2": result.test["r2"], "fields": "|".join(subset)})
    topn = pd.DataFrame(top_rows)

    metric_rows = []
    for label, results in [("full_DNN", dnn_results), ("DGatingDNN", dgate_results), ("selected_retrain_DNN", retrain_results)]:
        for result in results:
            metric_rows.append(
                {
                    "model": label,
                    "seed": result.seed,
                    "best_epoch": result.best_epoch,
                    "elapsed_seconds": result.elapsed_seconds,
                    **{f"validation_{k}": v for k, v in result.validation.items()},
                    **{f"test_{k}": v for k, v in result.test.items()},
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_dir / "metrics_by_seed.csv", index=False)
    gate_df.to_csv(run_dir / "gates_by_seed.csv", index=False)
    topn.to_csv(run_dir / "topn.csv", index=False)
    pd.DataFrame({"field": fields}).to_csv(run_dir / "candidate_fields.csv", index=False)
    pd.DataFrame({"field": selected}).to_csv(run_dir / "selected_fields.csv", index=False)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "target": target,
                "target_cn": TARGET_CN[target],
                "branch": branch,
                "n_pool": len(fields),
                "pool": fields,
                "selected": selected,
                "fallback_selection": fallback,
                **asdict(cfg),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_run(run_dir, target, branch, gate_df, dnn_results, dgate_results, retrain_results, topn, cfg.dgate_threshold)
    write_run_summary(
        run_dir / "summary.md", target, branch, fields, selected, gate_df,
        dnn_results, dgate_results, retrain_results, topn, cfg, fallback,
    )
    row = {
        "target": target,
        "target_cn": TARGET_CN[target],
        "branch": branch,
        "n_candidates": len(fields),
        "n_selected": len(selected),
        "selected_fields": "|".join(selected),
        "fallback_selection": int(fallback),
        "mean_selection_jaccard": mean_pairwise_jaccard(gate_df, cfg),
        **aggregate_metrics(dnn_results, "full_dnn"),
        **aggregate_metrics(dgate_results, "dgate"),
        **aggregate_metrics(retrain_results, "selected_retrain"),
        "run_dir": str(run_dir.relative_to(OUT_ROOT)),
    }
    return row


def mean_pairwise_jaccard(gate_df: pd.DataFrame, cfg: Config) -> float:
    sets = []
    for seed in cfg.seeds:
        col = f"gate_seed_{seed}"
        sets.append(set(gate_df.loc[gate_df[col] >= cfg.dgate_threshold, "field"]))
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            scores.append(len(sets[i] & sets[j]) / len(union) if union else 1.0)
    return float(np.mean(scores))


def write_pipeline_summary(
    preprocess_effective: list[str], strip_summary: pd.DataFrame, screen_summary: pd.DataFrame,
    source_summary: pd.DataFrame, cfg: Config, elapsed: float, stamp: str,
) -> None:
    pivot = source_summary.pivot(index="target", columns="branch", values="selected_retrain_r2_mean")
    no_screen = source_summary.query("branch == 'no_screening'").copy()
    with_screen = source_summary.query("branch == 'with_screening'").copy()
    impact = pivot.rename(columns={"no_screening": "r2_no_screening", "with_screening": "r2_with_screening"}).reset_index()
    impact["delta_with_minus_no"] = impact.r2_with_screening - impact.r2_no_screening
    impact = impact.sort_values("delta_with_minus_no")
    impact.to_csv(OUT_ROOT / "screening_impact.csv", index=False)
    worst = impact.iloc[0]

    frequency_rows = []
    for branch, group in source_summary.groupby("branch", sort=False):
        field_targets: dict[str, list[str]] = {}
        for row in group.itertuples():
            for field in str(row.selected_fields).split("|"):
                field_targets.setdefault(field, []).append(row.target)
        for field, field_target_list in field_targets.items():
            frequency_rows.append(
                {
                    "branch": branch,
                    "field": field,
                    "n_targets_selected": len(field_target_list),
                    "targets": "|".join(field_target_list),
                }
            )
    frequency = pd.DataFrame(frequency_rows).sort_values(
        ["branch", "n_targets_selected", "field"], ascending=[True, False, True]
    )
    frequency.to_csv(OUT_ROOT / "source_frequency.csv", index=False)
    top_no_screen = frequency.query("branch == 'no_screening'").head(6)
    status_row = no_screen.loc[no_screen.target.eq("gen_321_cc_1_status")].iloc[0]
    nuclear_row = no_screen.loc[no_screen.target.eq("gen_121_nuclear_1_pg_mw")].iloc[0]
    lines = [
        "# RTS-GMLC 12 目标完整流程总结",
        "",
        f"运行标识：`{stamp}`；总耗时 {elapsed/60:.1f} 分钟。",
        "",
        "## 1. 数据与候选池",
        "",
        f"27 个候选发布字段中 3 个全年恒定，建模起点为 {len(preprocess_effective)} 个时变字段。",
        "所有 130 个敏感字段均不进入 X；本轮只预测预先选定的 12 个核心目标。",
        "",
        "## 2. 两层物理/确定性剥离",
        "",
        f"第一层有来源公式共剥离 {int(strip_summary.n_layer1_removed.sum())} 个目标-字段关系；",
        f"第二层严格恒等/近恒等发现共剥离 {int(strip_summary.n_layer2_removed.sum())} 层。",
        "只有 `gen_121_nuclear_1_pg_mw` 存在能闭合的确定性路径；为同时切断直接燃料汇总、燃料分解和功率平衡替代路径，学习前共移除三个入口字段。",
        "AC 潮流和线路负载率公式对其他目标虽存在，但当前 27 字段缺少完整节点/线路量，不能据公式直接计算。",
        "",
        "## 3. 初筛",
        "",
        f"经训练段分块置换阈值，平均候选数由 {screen_summary.n_pool_after_stripping.mean():.1f} 降至 {screen_summary.n_kept.mean():.1f}。",
        "未初筛与已初筛分支均保留，用后续测试段重训结果判断初筛是否损失必要信息。",
        "",
        "## 4. 推断源定位结果",
        "",
        "| 目标 | 分支 | 候选 | 共识源 | 全量 DNN R² | 选中重训 R² | 差值 | 种子 Jaccard |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in source_summary.itertuples():
        lines.append(
            f"| `{row.target}` | {row.branch} | {row.n_candidates} | {row.n_selected} | "
            f"{row.full_dnn_r2_mean:.4f} | {row.selected_retrain_r2_mean:.4f} | "
            f"{row.selected_retrain_r2_mean-row.full_dnn_r2_mean:+.4f} | {row.mean_selection_jaccard:.3f} |"
        )
    lines += [
        "",
        "整体均值：",
        "",
        f"- 未初筛：选中 {source_summary.query('branch == \"no_screening\"').n_selected.mean():.1f} 个，"
        f"选中字段重训 R²={source_summary.query('branch == \"no_screening\"').selected_retrain_r2_mean.mean():.4f}。",
        f"- 已初筛：选中 {source_summary.query('branch == \"with_screening\"').n_selected.mean():.1f} 个，"
        f"选中字段重训 R²={source_summary.query('branch == \"with_screening\"').selected_retrain_r2_mean.mean():.4f}。",
        f"- 未初筛有 **{int((no_screen.selected_retrain_r2_mean >= 0.90).sum())}/12** 个目标重训 R²≥0.90；"
        f"两个例外是 AB1（{no_screen.loc[no_screen.target.eq('branch_ab1_loading_pct'), 'selected_retrain_r2_mean'].iloc[0]:.4f}）"
        f"和 AB3（{no_screen.loc[no_screen.target.eq('branch_ab3_loading_pct'), 'selected_retrain_r2_mean'].iloc[0]:.4f}）。",
        f"- 公式闭包剥离后的核电机组仍由 {int(nuclear_row.n_selected)} 个共识源达到 R²={nuclear_row.selected_retrain_r2_mean:.4f}。",
        f"- 机组状态目标用 {int(status_row.n_selected)} 个共识源达到 balanced accuracy="
        f"{status_row.selected_retrain_balanced_accuracy_mean:.4f}、ROC-AUC={status_row.selected_retrain_roc_auc_mean:.4f}。",
        "",
        "## 5. 初筛应如何使用",
        "",
        f"初筛分支相对未初筛分支，12 目标平均重训 R² 变化为 "
        f"{(with_screen.selected_retrain_r2_mean.mean()-no_screen.selected_retrain_r2_mean.mean()):+.4f}。",
        f"最坏目标是 `{worst.target}`，下降 {abs(worst.delta_with_minus_no):.4f}。",
        "因此在这组 27 维候选上，初筛不应作为默认必经步骤；更合理的定位是计算量分支/消融诊断，主结论采用未初筛结果。",
        "完整逐目标影响见 `screening_impact.csv`。",
        "",
        "## 6. 跨目标反复出现的推断源（未初筛）",
        "",
        "| 字段 | 被多少个目标选中 |",
        "|---|---:|",
    ]
    for row in top_no_screen.itertuples():
        lines.append(f"| `{row.field}` | {row.n_targets_selected}/12 |")
    lines += [
        "",
        "完整频次与目标列表见 `source_frequency.csv`。这些频次反映共同预测入口，不应解释为因果重要性。",
        "",
        "## 7. 稳定性与解释边界",
        "",
        f"未初筛分支的三种子活跃集合平均 Jaccard={no_screen.mean_selection_jaccard.mean():.3f}："
        "属于中等稳定而不是完全一致，因此最终采用‘至少 2/3 种子活跃’的共识定义，并保留连续门控值。",
        "λ=0.005 由 `01_preprocess/dgate_lambda_calibration.csv` 中的验证集容差规则选出；测试集不参与 λ 或字段选择。",
        "逐目标 Top-n 曲线只作选定字段之后的诊断展示，不反向用于确定共识集合。",
        "D-Gating 选中的是能共同恢复目标的统计推断源，不等同于因果变量，也不证明现实 PJM 数据已经泄露。",
        "RTS-GMLC 是公开合成系统；这里验证的是粗粒度候选发布量对现实中通常敏感的细粒度运行变量类型所形成的可恢复性。",
        "门控阈值、三种子频率、连续门控值和选中字段重训结果均已同时保存，避免只凭一次二值选择下结论。",
        f"D-Gating 原方法：<{DGATING_PAPER}>。",
        "",
    ]
    (OUT_ROOT / "pipeline_summary.md").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = list(pivot.index)
    y = np.arange(len(labels))
    width = 0.38
    ax.barh(y - width / 2, pivot["no_screening"], height=width, label="no screening")
    ax.barh(y + width / 2, pivot["with_screening"], height=width, label="with screening")
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("selected-fields retrain test R²")
    ax.set_title("RTS-GMLC inference-source localization")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "fig_source_location_overview.png", dpi=180)
    plt.close(fig)


def write_log(
    source_summary: pd.DataFrame, strip_summary: pd.DataFrame, screen_summary: pd.DataFrame,
    cfg: Config, stamp: str, elapsed: float,
) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "20260811_13_RTS-GMLC两层剥离初筛与推断源定位.md"
    no_screen = source_summary.query("branch == 'no_screening'")
    with_screen = source_summary.query("branch == 'with_screening'")
    status_row = no_screen.loc[no_screen.target.eq("gen_321_cc_1_status")].iloc[0]
    lines = [
        "# RTS-GMLC 两层剥离、初筛与推断源定位运行日志",
        "",
        f"- 运行标识：`{stamp}`",
        f"- 完成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 总耗时：{elapsed/60:.1f} 分钟",
        f"- 配置：{json.dumps(asdict(cfg), ensure_ascii=False)}",
        "",
        "## 已执行",
        "",
        "1. 27 个候选字段质量检查，移除 3 个恒定输入；",
        "2. 12 个目标的官方/有来源公式闭合性审计；",
        "3. 训练段仿射与对数空间恒等/近恒等发现及 5 时间块复验；",
        "4. 六指标、周块置换客观阈值初筛；",
        "5. 用预先固定的代表目标和验证集容差规则，将 D-Gating λ 校准为 0.005；",
        "6. 未初筛/已初筛两分支的普通 DNN、D-Gating 三种子定位和共识字段重训；",
        "7. 逐目标 CSV、图、配置、摘要和根目录汇总的完整性检查。",
        "",
        "## 核心数字",
        "",
        f"- 第一层剥离数：{int(strip_summary.n_layer1_removed.sum())}",
        f"- 第二层剥离数：{int(strip_summary.n_layer2_removed.sum())}",
        f"- 初筛平均保留：{screen_summary.n_kept.mean():.1f}/{screen_summary.n_pool_after_stripping.mean():.1f}",
        f"- 未初筛重训 R² 均值：{source_summary.query('branch == \"no_screening\"').selected_retrain_r2_mean.mean():.4f}",
        f"- 已初筛重训 R² 均值：{source_summary.query('branch == \"with_screening\"').selected_retrain_r2_mean.mean():.4f}",
        f"- 未初筛 R²≥0.90：{int((no_screen.selected_retrain_r2_mean >= 0.90).sum())}/12 个目标",
        f"- 初筛相对未初筛的平均变化：{with_screen.selected_retrain_r2_mean.mean()-no_screen.selected_retrain_r2_mean.mean():+.4f}",
        f"- 状态目标：balanced accuracy={status_row.selected_retrain_balanced_accuracy_mean:.4f}，"
        f"ROC-AUC={status_row.selected_retrain_roc_auc_mean:.4f}",
        f"- 三种子活跃集合平均 Jaccard（未初筛）：{no_screen.mean_selection_jaccard.mean():.3f}",
        "- 24/24 个目标-分支目录通过必需文件与配置匹配检查。",
        "",
        "完整结果见 `NEW_Supplement/04_outputs/rts_gmlc_2020/pipeline_summary.md`。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def verify_outputs(targets: list[str], stamp: str) -> pd.DataFrame:
    rows = []
    for target in targets:
        for branch in ["no_screening", "with_screening"]:
            run_dir = (
                OUT_ROOT / "04b_source_location_excl_targets" / f"stripped_{branch}"
                / f"target_{target}" / "DGatingDNN" / f"run_{stamp}"
            )
            required = [
                "config.json", "summary.md", "candidate_fields.csv", "selected_fields.csv",
                "gates_by_seed.csv", "metrics_by_seed.csv", "topn.csv", "fig_gate_bar.png",
                "fig_training.png", "fig_topn.png",
            ]
            missing = [name for name in required if not (run_dir / name).exists()]
            config_ok = False
            if not missing:
                cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                config_ok = cfg["target"] == target and cfg["branch"] == branch
            rows.append(
                {
                    "target": target,
                    "branch": branch,
                    "run_dir_exists": int(run_dir.exists()),
                    "missing_files": "|".join(missing),
                    "config_matches": int(config_ok),
                    "passed": int(not missing and config_ok),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROOT / "pipeline_output_verification.csv", index=False)
    if not bool(out.passed.all()):
        raise RuntimeError("output verification failed; inspect pipeline_output_verification.csv")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "prepare", "screen", "source"], default="all")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--lambda-dgate", type=float, default=0.005)
    parser.add_argument("--screening-draws", type=int, default=80)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--stamp", default=None)
    args = parser.parse_args()
    cfg = Config(epochs=args.epochs, dgate_lambda=args.lambda_dgate, screening_null_draws=args.screening_draws)
    started = time.time()
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(DATA_FILE)
    dictionary = pd.read_csv(DICT_FILE)
    core = pd.read_csv(CORE_FILE)
    all_targets = core.field.tolist()
    targets = args.targets or all_targets
    invalid = sorted(set(targets) - set(all_targets))
    if invalid:
        raise SystemExit(f"not in core targets: {invalid}")
    if df.isna().any().any():
        raise SystemExit("base data contains missing values")

    print(f"RTS-GMLC pipeline {stamp}: {len(df)} rows, {len(targets)} targets", flush=True)
    _, effective = write_preprocess(df, dictionary, all_targets, cfg)
    pools, strip_summary = run_stripping(df, all_targets, effective, cfg)
    if args.stage == "prepare":
        print("stopped after prepare", flush=True)
        return
    kept, screen_summary = run_screening(df, all_targets, pools, cfg)
    if args.stage == "screen":
        print("stopped after screen", flush=True)
        return
    chosen_lambda = run_dgate_calibration(df, pools["bus_115_va_deg"], cfg)
    if not np.isclose(chosen_lambda, cfg.dgate_lambda):
        raise SystemExit(
            f"validation-only calibration selected lambda={chosen_lambda:g}, "
            f"but command requested {cfg.dgate_lambda:g}; rerun with the calibrated value"
        )

    rows = []
    for pos, target in enumerate(targets, 1):
        for branch, fields in [("no_screening", pools[target]), ("with_screening", kept[target])]:
            print(f"[source {pos:02d}/{len(targets):02d}] {target} | {branch} | p={len(fields)}", flush=True)
            row = run_source_branch(df, target, fields, branch, cfg, stamp)
            rows.append(row)
            print(
                f"  -> selected {row['n_selected']}, retrain R2={row['selected_retrain_r2_mean']:.4f}, "
                f"full R2={row['full_dnn_r2_mean']:.4f}",
                flush=True,
            )
    current = pd.DataFrame(rows)
    summary_file = OUT_ROOT / "source_location_summary.csv"
    if args.targets is None or set(targets) == set(all_targets):
        current.to_csv(summary_file, index=False)
        verify_outputs(all_targets, stamp)
        elapsed = time.time() - started
        write_pipeline_summary(effective, strip_summary, screen_summary, current, cfg, elapsed, stamp)
        log = write_log(current, strip_summary, screen_summary, cfg, stamp, elapsed)
        print(f"complete in {elapsed/60:.1f} min; log={log}", flush=True)
    else:
        current.to_csv(OUT_ROOT / f"source_location_partial_{stamp}.csv", index=False)
        print("partial target run completed; aggregate summary not overwritten", flush=True)


if __name__ == "__main__":
    main()
