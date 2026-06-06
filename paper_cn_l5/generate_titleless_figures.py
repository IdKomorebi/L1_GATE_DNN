from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
OUT = PAPER / "figures"
OUT.mkdir(parents=True, exist_ok=True)


TARGET_LABELS = {
    "congestion_price_da": "阻塞价DA",
    "congestion_price_rt": "阻塞价RT",
    "da_as_total_mw_primary_reserve": "主备用",
    "da_as_total_mw_thirty_minutes_reserve": "30min备用",
    "gross_actual_interchange_mw": "总实际交换",
    "marginal_loss_price_da": "边际损耗DA",
    "metered_load_mw": "计量负荷",
    "net_actual_interchange_mw": "净实际交换",
    "net_sched_interchange_mw": "净计划交换",
    "total_gen": "总发电",
    "total_lmp_da": "LMP-DA",
    "total_losses": "总损耗",
}

METHOD_ZH = {
    "DNN_AllFeatures": "全量DNN",
    "L1GateDNN": "L1DNN",
    "L1GateDNN_Selected": "L1DNN",
    "DNN_SelectedByL1": "L1选中DNN",
    "NMI": "NMI",
    "Pearson": "Pearson",
    "Spearman": "Spearman",
    "Lasso": "Lasso",
    "ElasticNet": "ElasticNet",
    "RandomForest": "RF",
    "XGBoost": "XGBoost",
}

METHOD_ORDER_ZH = ["全量 DNN", "L1GateDNN", "NMI", "Pearson", "Spearman", "Lasso", "ElasticNet", "RandomForest", "XGBoost"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "SimSun",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
        }
    )


def rel(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def common_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d6d6d6", linewidth=0.45, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_r2_curve(csv_path: str, out_name: str, ylim: tuple[float, float] | None = None) -> None:
    df = pd.read_csv(rel(csv_path))
    fig, ax = plt.subplots(figsize=(3.35, 2.18))
    ax.plot(df["epoch"], df["train_r2"], color="#1f77b4", linewidth=1.25, label="训练集")
    ax.plot(df["epoch"], df["test_r2"], color="#d62728", linewidth=1.25, label="测试集")
    best_idx = int(df["test_r2"].idxmax())
    ax.scatter([df.loc[best_idx, "epoch"]], [df.loc[best_idx, "test_r2"]], s=18, color="#d62728", zorder=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("$R^2$")
    if ylim is None:
        ymin = max(0.0, float(df[["train_r2", "test_r2"]].min().min()) - 0.05)
        ylim = (ymin, 1.02)
    ax.set_ylim(*ylim)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    common_axes(ax)
    save(fig, out_name)


def plot_train_pair_split(dnn_csv: str, l1_csv: str) -> None:
    dnn = pd.read_csv(rel(dnn_csv))
    l1 = pd.read_csv(rel(l1_csv))
    ymin = min(dnn[["train_r2", "test_r2"]].min().min(), l1[["train_r2", "test_r2"]].min().min())
    ylim = (max(0.0, float(ymin) - 0.04), 1.02)
    plot_r2_curve(dnn_csv, "net_train_dnn_r2_zh.png", ylim)
    plot_r2_curve(l1_csv, "net_train_l1_r2_zh.png", ylim)


def plot_gate_curves(csv_path: str, risk_path: str, threshold: float, out_name: str) -> None:
    gate = pd.read_csv(rel(csv_path))
    risk = pd.read_csv(rel(risk_path))
    score_col = "final_gate" if "final_gate" in risk.columns else "gate"
    selected = set(risk.loc[risk[score_col] >= threshold, "related"])
    top = risk.sort_values(score_col, ascending=False).head(10)["related"].tolist()
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    for feature, sub in gate.groupby("feature"):
        if feature in selected:
            color = "#1f77b4" if feature in top else "#6baed6"
            lw = 0.72 if feature in top else 0.52
            alpha = 0.82
        else:
            color = "#b8b8b8"
            lw = 0.34
            alpha = 0.35
        ax.plot(sub["epoch"], sub["gate"], color=color, linewidth=lw, alpha=alpha)
    ax.axhline(threshold, color="#d62728", linestyle="--", linewidth=0.8, label=f"阈值 {threshold:g}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("gate")
    ax.legend(loc="upper right", frameon=False)
    common_axes(ax)
    save(fig, out_name)


def plot_active_features(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path))
    fig, ax = plt.subplots(figsize=(3.25, 2.15))
    ax.plot(df["epoch"], df["active_features"], color="#1f77b4", linewidth=1.35)
    ax.scatter(df["epoch"].iloc[-1], df["active_features"].iloc[-1], s=18, color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("活跃字段数")
    ax.set_ylim(0, max(df["active_features"].max() + 3, 20))
    common_axes(ax)
    save(fig, out_name)


def plot_top_series(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path)).copy()
    labels = []
    for _, row in df.iterrows():
        if row["exp_type"] == "all":
            labels.append("全量")
        elif row["exp_type"] == "unselected":
            labels.append("未选中")
        else:
            labels.append(f"Top{int(row['feature_count'])}")
    df["label"] = labels
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    colors = ["#8a8a8a"] + ["#6baed6"] * 5 + ["#1f77b4", "#d62728"]
    ax.bar(np.arange(len(df)), df["best_test_r2"], color=colors[: len(df)], width=0.72)
    ax.plot(np.arange(len(df)), df["best_test_r2"], color="#333333", linewidth=0.8, marker="o", markersize=2.4)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.set_ylabel("Best Test $R^2$")
    ax.set_ylim(0, 1.05)
    common_axes(ax)
    save(fig, out_name)


def plot_meta_w(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path))
    labels = ["NMI", "Spearman", "Pearson", "Kendall", "DC", "HSIC"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#72b7b2", "#e45756"]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for i, label in enumerate(labels, start=1):
        ax.plot(df["epoch"], df[f"W_{i}"], linewidth=1.05, color=colors[i - 1], label=label)
    ax.axhline(0, color="#777777", linewidth=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("权重")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, borderaxespad=0)
    common_axes(ax)
    save(fig, out_name)


def plot_meta_b(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path))
    fig, ax = plt.subplots(figsize=(3.25, 2.1))
    ax.plot(df["epoch"], df["b_meta"], color="#1f77b4", linewidth=1.25)
    ax.axhline(0, color="#777777", linewidth=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("偏置 $b$")
    common_axes(ax)
    save(fig, out_name)


def plot_dnn_l1_compare(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path))
    wide = df.pivot(index="target_label", columns="method", values="best_test_r2")
    order = list(wide.index)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.65, 2.95))
    ax.plot(
        x,
        wide.loc[order, "DNN_AllFeatures"],
        color="#2ca02c",
        marker="o",
        markersize=3.1,
        linewidth=1.55,
        label="全量DNN",
    )
    ax.plot(
        x,
        wide.loc[order, "L1GateDNN_Selected"],
        color="#1f77b4",
        linestyle="--",
        marker="^",
        markersize=3.4,
        linewidth=1.75,
        label="L1DNN",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS.get(t, t) for t in order], rotation=55, ha="right")
    ax.set_ylabel("Best Test $R^2$")
    ax.set_ylim(0.55, 1.04)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False, ncol=2)
    common_axes(ax)
    save(fig, out_name)


def _budget_color(method: str) -> str:
    return {
        "DNN_AllFeatures": "#2ca02c",
        "全量 DNN": "#2ca02c",
        "L1GateDNN": "#1f77b4",
        "NMI": "#8a8a8a",
        "Pearson": "#9a9a9a",
        "Spearman": "#707070",
        "Lasso": "#a8a8a8",
        "ElasticNet": "#7f7f7f",
        "RandomForest": "#999999",
        "XGBoost": "#b5b5b5",
    }.get(method, "#999999")


def plot_budget_lines_one(csv_path: Path, out_name: str) -> None:
    df = pd.read_csv(csv_path)
    methods = ["DNN_AllFeatures", "L1GateDNN", "NMI", "Pearson", "Spearman", "Lasso", "ElasticNet", "RandomForest", "XGBoost"]
    markers = {"DNN_AllFeatures": "o", "L1GateDNN": "^", "NMI": "s", "Pearson": "D", "Spearman": "v", "Lasso": "P", "ElasticNet": "X", "RandomForest": "*", "XGBoost": "h"}
    order = df["target_label"].tolist()
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.7, 2.72))
    for method in methods:
        is_key = method in {"DNN_AllFeatures", "L1GateDNN"}
        ax.plot(
            x,
            df[method],
            color=_budget_color(method),
            linestyle="--" if method == "L1GateDNN" else "-",
            marker=markers[method],
            markersize=3.1 if is_key else 2.2,
            linewidth=1.7 if is_key else 0.72,
            alpha=1.0 if is_key else 0.55,
            label=METHOD_ZH[method],
            zorder=3 if is_key else 2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS.get(t, t) for t in order], rotation=55, ha="right")
    ax.set_ylabel("Test $R^2$")
    ax.set_ylim(0.15, 1.04)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), frameon=False, ncol=3, columnspacing=0.8)
    common_axes(ax)
    save(fig, out_name)


def plot_budget_average_one(csv_path: Path, out_name: str) -> None:
    df = pd.read_csv(csv_path).copy()
    method_col = next(col for col in df.columns if "方法" in col)
    value_col = next(col for col in df.columns if "Test" in col or "R" in col)
    df[method_col] = df[method_col].str.strip()
    df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    y = np.arange(len(df))
    colors = [_budget_color(method) for method in df[method_col]]
    ax.barh(y, df[value_col], color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(df[method_col])
    ax.invert_yaxis()
    ax.set_xlim(0.60, max(0.93, float(df[value_col].max()) + 0.035))
    ax.set_xlabel("平均 Test $R^2$")
    for yy, val in zip(y, df[value_col]):
        ax.text(float(val) + 0.005, yy, f"{float(val):.3f}", va="center", fontsize=6.6)
    common_axes(ax)
    save(fig, out_name)


def plot_training_noise_one(root_path: str, center: str, out_name: str) -> None:
    root = rel(root_path)
    methods = [
        ("DNN_AllFeatures", "全量DNN", "#202020"),
        ("L1GateDNN", "L1DNN", "#1f77b4"),
        ("DNN_SelectedByL1", "L1选中DNN", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(3.55, 2.55))
    all_values = []
    for method, method_label, color in methods:
        df = pd.read_csv(root / f"CenterOn_{center}" / method / "log.csv")
        all_values.extend(df["train_r2"].tolist())
        all_values.extend(df["test_r2"].tolist())
        ax.plot(df["epoch"], df["train_r2"], color=color, linestyle="--", linewidth=0.9, alpha=0.68, label=f"{method_label}训练")
        ax.plot(df["epoch"], df["test_r2"], color=color, linestyle="-", linewidth=1.25, alpha=0.98, label=f"{method_label}测试")
    finite = np.asarray(all_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    ymin = max(0.0, float(finite.min()) - 0.06) if finite.size else 0.0
    ax.set_ylim(ymin, 1.05)
    ax.set_xlabel("epoch")
    ax.set_ylabel("$R^2$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.31), frameon=False, ncol=2, columnspacing=0.8)
    common_axes(ax)
    save(fig, out_name)


def _choose_topn_csv(default_root: str) -> Path:
    l5_csv = PAPER / "topn_incremental_results_l5.csv"
    if l5_csv.exists():
        return l5_csv
    l5_runs = sorted(
        (ROOT / "NEW_PROJECT/outputs/data2025_Processed_V2/RedundantNoiseExplanation").glob("RedundantNoiseExplanation_l5*/topn_incremental_results.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if l5_runs:
        return l5_runs[0]
    return rel(default_root) / "topn_incremental_results.csv"


def plot_topn_noise_one(csv_path: Path, center: str, out_name: str) -> None:
    df = pd.read_csv(csv_path)
    sub = df[df["center"] == center].copy()
    topn = sub[sub["exp_type"] == "topn"].sort_values("feature_count")
    full_rows = sub[sub["exp_type"] == "full"]
    display_full_n = 69 if center == "da_as_total_mw_primary_reserve" else None
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    ax.plot(topn["feature_count"], topn["best_test_r2"], marker="o", markersize=2.7, linewidth=1.25, color="#1f77b4")
    best = topn.loc[topn["best_test_r2"].idxmax()]
    ax.scatter([best["feature_count"]], [best["best_test_r2"]], marker="^", s=38, color="#d62728", zorder=4)
    ax.annotate(
        f"Top{int(best['feature_count'])}\n{best['best_test_r2']:.4f}",
        xy=(best["feature_count"], best["best_test_r2"]),
        xytext=(5, -20),
        textcoords="offset points",
        fontsize=6.6,
        color="#d62728",
    )
    if not full_rows.empty:
        full = full_rows.iloc[0]
        full_x = int(display_full_n or full["feature_count"])
        ax.scatter([full_x], [full["best_test_r2"]], marker="s", s=28, color="#1f77b4", edgecolor="#d62728", linewidth=0.8, zorder=4)
        ax.annotate(
            f"Full{full_x}\n{full['best_test_r2']:.4f}",
            xy=(full_x, full["best_test_r2"]),
            xytext=(-28, 9),
            textcoords="offset points",
            fontsize=6.6,
            color="#d62728",
            ha="right",
        )
    ax.set_xlabel("字段数")
    ax.set_ylabel("Best Test $R^2$")
    finite = sub["best_test_r2"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    ymin = max(0.0, float(finite.min()) - 0.08) if finite.size else 0.0
    ax.set_ylim(ymin, 1.05)
    xmax = max(float(sub["feature_count"].max()), float(display_full_n or 0))
    ax.set_xlim(0, max(xmax + 4, 36))
    common_axes(ax)
    save(fig, out_name)


def main() -> None:
    setup_style()
    net_dnn = "NEW_PROJECT/outputs/data2025_Processed_V2/CenterOn_net_actual_interchange_mw/DNN/run_20260602_152549_combo5_DNN/log.csv"
    net_l1_root = "NEW_PROJECT/outputs/data2025_Processed_V2/CenterOn_net_actual_interchange_mw/L1GateDNN/run_20260603_l1_lr0p00065_thr0p10_combo5_L1GateDNN"
    improved_root = "NEW_PROJECT/outputs/data2025_Processed_V2/CenterOn_net_actual_interchange_mw/ImprovedGateDNN/ImprovedL1GateDNN/run_20260603_024847"
    budget_root = "NEW_PROJECT/outputs/data2025_Processed_V2/BaselineComparison/BudgetBaselineL1Selector_20260604_113307"
    redundant_root = "NEW_PROJECT/outputs/data2025_Processed_V2/RedundantNoiseExplanation/RedundantNoiseExplanation_20260604_121500"

    plot_r2_curve(net_dnn, "net_dnn_r2_zh.png")
    plot_r2_curve(f"{net_l1_root}/log.csv", "net_l1_r2_zh.png")
    plot_train_pair_split(net_dnn, f"{net_l1_root}/log.csv")
    plot_gate_curves(f"{net_l1_root}/gate_params.csv", f"{net_l1_root}/risk_map.csv", 0.10, "net_l1_gate_params_zh.png")
    plot_active_features(f"{net_l1_root}/log.csv", "net_l1_active_features_zh.png")
    plot_top_series(f"{net_l1_root}/redundancy_validation/top_series_3_6_9_12_15_18_e100/top_series_results.csv", "net_top_series_r2_zh.png")

    plot_gate_curves(f"{improved_root}/gate_params.csv", f"{improved_root}/risk_map.csv", 0.50, "improved_l1_gate_params_zh.png")
    plot_meta_w(f"{improved_root}/W_meta_evolution.csv", "improved_l1_w_meta_zh.png")
    plot_meta_b(f"{improved_root}/b_meta_evolution.csv", "improved_l1_b_meta_zh.png")

    plot_dnn_l1_compare("NEW_PROJECT/outputs/data2025_Processed_V2/BaselineComparison/DNN_vs_L1DNN_20260603_235906/method_compare_summary.csv", "dnn_vs_l1_method_compare_zh.png")
    plot_budget_lines_one(rel(budget_root) / "budget_95" / "baseline_summary_wide.csv", "budget95_baseline_lines_zh.png")
    plot_budget_lines_one(rel(budget_root) / "budget_97" / "baseline_summary_wide.csv", "budget97_baseline_lines_zh.png")
    plot_budget_average_one(rel(budget_root) / "budget_95_average_summary_zh.csv", "budget95_average_bar_zh.png")
    plot_budget_average_one(rel(budget_root) / "budget_97_average_summary_zh.csv", "budget97_average_bar_zh.png")
    plot_training_noise_one(redundant_root, "congestion_price_da", "redundant_training_congestion_zh.png")
    plot_training_noise_one(redundant_root, "da_as_total_mw_primary_reserve", "redundant_training_primary_zh.png")
    topn_csv = _choose_topn_csv(redundant_root)
    plot_topn_noise_one(topn_csv, "congestion_price_da", "topn_congestion_zh.png")
    plot_topn_noise_one(topn_csv, "da_as_total_mw_primary_reserve", "topn_primary_zh.png")


if __name__ == "__main__":
    main()
