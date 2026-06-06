from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


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
            "legend.fontsize": 7,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def common_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d8d8d8", linewidth=0.45, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def rel(path: str) -> Path:
    return ROOT / path


def plot_r2_curve(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path))
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    ax.plot(df["epoch"], df["train_r2"], color="#2a6fbb", linewidth=1.25, label="训练集")
    ax.plot(df["epoch"], df["test_r2"], color="#c44e52", linewidth=1.25, label="测试集")
    best_idx = int(df["test_r2"].idxmax())
    ax.scatter(
        [df.loc[best_idx, "epoch"]],
        [df.loc[best_idx, "test_r2"]],
        s=18,
        color="#c44e52",
        zorder=3,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("$R^2$")
    ax.set_ylim(max(0.0, df[["train_r2", "test_r2"]].min().min() - 0.05), 1.02)
    ax.legend(loc="lower right", frameon=False)
    common_axes(ax)
    save(fig, out_name)


def plot_train_pair_stack(dnn_csv: str, l1_csv: str, out_name: str) -> None:
    dnn = pd.read_csv(rel(dnn_csv))
    l1 = pd.read_csv(rel(l1_csv))
    ymin = min(dnn[["train_r2", "test_r2"]].min().min(), l1[["train_r2", "test_r2"]].min().min())
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.35), sharex=True)
    for ax, df, label in [
        (axes[0], dnn, "(a) 全量 DNN"),
        (axes[1], l1, "(b) L1GateDNN"),
    ]:
        ax.plot(df["epoch"], df["train_r2"], color="#2a6fbb", linewidth=1.1, label="训练集")
        ax.plot(df["epoch"], df["test_r2"], color="#c44e52", linewidth=1.1, label="测试集")
        best_idx = int(df["test_r2"].idxmax())
        ax.scatter([df.loc[best_idx, "epoch"]], [df.loc[best_idx, "test_r2"]], s=18, color="#c44e52", zorder=3)
        ax.text(0.02, 0.10, label, transform=ax.transAxes, fontsize=7.5, color="#333333")
        ax.set_ylabel("$R^2$")
        ax.set_ylim(max(0.0, ymin - 0.04), 1.02)
        common_axes(ax)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), frameon=False, ncol=2)
    axes[-1].set_xlabel("epoch")
    save(fig, out_name)


def plot_gate_curves(csv_path: str, risk_path: str, threshold: float, out_name: str) -> None:
    gate = pd.read_csv(rel(csv_path))
    risk = pd.read_csv(rel(risk_path))
    score_col = "final_gate" if "final_gate" in risk.columns else "gate"
    selected = set(risk.loc[risk[score_col] >= threshold, "related"])
    top = risk.sort_values(score_col, ascending=False).head(10)["related"].tolist()

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    for feature, sub in gate.groupby("feature"):
        if feature in selected:
            color = "#2a6fbb" if feature in top else "#6aaed6"
            lw = 0.75 if feature in top else 0.55
            alpha = 0.82
        else:
            color = "#b8b8b8"
            lw = 0.35
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
    ax.plot(df["epoch"], df["active_features"], color="#2a6fbb", linewidth=1.35)
    ax.scatter(df["epoch"].iloc[-1], df["active_features"].iloc[-1], s=18, color="#2a6fbb")
    ax.set_xlabel("epoch")
    ax.set_ylabel("活跃字段数")
    ax.set_ylim(0, max(df["active_features"].max() + 3, 20))
    common_axes(ax)
    save(fig, out_name)


def plot_top_series(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path)).copy()
    order = ["Full", "Top3", "Top6", "Top9", "Top12", "Top15", "Top18", "Without Top18"]
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
    colors = ["#8a8a8a"] + ["#6aaed6"] * 5 + ["#2a6fbb", "#c44e52"]
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
    ax.plot(df["epoch"], df["b_meta"], color="#2a6fbb", linewidth=1.25)
    ax.axhline(0, color="#777777", linewidth=0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("偏置 $b$")
    common_axes(ax)
    save(fig, out_name)


TARGET_LABELS = {
    "congestion_price_da": "拥塞价DA",
    "congestion_price_rt": "拥塞价RT",
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


def plot_dnn_l1_compare(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path))
    wide = df.pivot(index="target_label", columns="method", values="best_test_r2")
    order = list(wide.index)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(3.55, 2.95))
    ax.plot(
        x,
        wide.loc[order, "DNN_AllFeatures"],
        color="#8a8a8a",
        marker="o",
        markersize=3,
        linewidth=1.15,
        label="全量 DNN",
    )
    ax.plot(
        x,
        wide.loc[order, "L1GateDNN_Selected"],
        color="#2a6fbb",
        marker="^",
        markersize=3.2,
        linewidth=1.15,
        label="L1DNN",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS.get(t, t) for t in order], rotation=55, ha="right")
    ax.set_ylabel("Best Test $R^2$")
    ax.set_ylim(0.55, 1.04)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False, ncol=2)
    common_axes(ax)
    save(fig, out_name)


METHOD_ORDER = ["全量 DNN", "L1GateDNN", "NMI", "Pearson", "Spearman", "Lasso", "ElasticNet", "RandomForest", "XGBoost"]


def plot_budget_average(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path))
    df["方法"] = df["方法"].str.strip()
    df = df.set_index("方法").loc[METHOD_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    colors = ["#8a8a8a"] + ["#2a6fbb"] + ["#9ecae1"] * 3 + ["#f2a65a"] * 2 + ["#8fbf87"] * 2
    ax.bar(np.arange(len(df)), df["平均 Test R²"], color=colors, width=0.7)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["方法"], rotation=48, ha="right")
    ax.set_ylabel("平均 Test $R^2$")
    ax.set_ylim(0.60, 1.00)
    common_axes(ax)
    save(fig, out_name)


def plot_budget_lines_stack(root_path: str, out_name: str) -> None:
    root = rel(root_path)
    frames = [
        ("(a) 95% 统一预算", pd.read_csv(root / "budget_95" / "baseline_summary_wide.csv")),
        ("(b) 97% 统一预算", pd.read_csv(root / "budget_97" / "baseline_summary_wide.csv")),
    ]
    methods = ["DNN_AllFeatures", "L1GateDNN", "NMI", "Pearson", "Spearman", "Lasso", "ElasticNet", "RandomForest", "XGBoost"]
    method_labels = {
        "DNN_AllFeatures": "全量DNN",
        "L1GateDNN": "L1GateDNN",
        "NMI": "NMI",
        "Pearson": "Pearson",
        "Spearman": "Spearman",
        "Lasso": "Lasso",
        "ElasticNet": "ElasticNet",
        "RandomForest": "RF",
        "XGBoost": "XGBoost",
    }
    colors = {
        "DNN_AllFeatures": "#7f7f7f",
        "L1GateDNN": "#2a6fbb",
        "NMI": "#9ecae1",
        "Pearson": "#6baed6",
        "Spearman": "#3182bd",
        "Lasso": "#f2a65a",
        "ElasticNet": "#f28e2b",
        "RandomForest": "#59a14f",
        "XGBoost": "#86bc86",
    }
    markers = ["o", "^", "s", "D", "v", "P", "X", "*", "h"]
    order = frames[0][1]["target_label"].tolist()
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 1, figsize=(3.65, 4.45), sharex=True)
    for ax, (panel, df) in zip(axes, frames):
        for method, marker in zip(methods, markers):
            ax.plot(
                x,
                df[method],
                color=colors[method],
                marker=marker,
                markersize=2.4,
                linewidth=0.85,
                label=method_labels[method],
                alpha=0.95,
            )
        ax.text(0.02, 1.02, panel, transform=ax.transAxes, fontsize=7.5, color="#333333", va="bottom")
        ax.set_ylabel("Test $R^2$")
        ax.set_ylim(0.15, 1.04)
        common_axes(ax)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([TARGET_LABELS.get(t, t) for t in order], rotation=55, ha="right")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.38), frameon=False, ncol=3, columnspacing=0.9)
    save(fig, out_name)


def plot_budget_average_stack(root_path: str, out_name: str) -> None:
    root = rel(root_path)
    frames = [
        ("(a) 95% 统一预算", pd.read_csv(root / "budget_95_average_summary_zh.csv")),
        ("(b) 97% 统一预算", pd.read_csv(root / "budget_97_average_summary_zh.csv")),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.05), sharex=True)
    for ax, (panel, df) in zip(axes, frames):
        df = df.copy()
        df["方法"] = df["方法"].str.strip()
        df = df.set_index("方法").loc[METHOD_ORDER].reset_index()
        y = np.arange(len(df))
        colors = ["#8a8a8a"] + ["#2a6fbb"] + ["#b8cfe3"] * 3 + ["#f2a65a"] * 2 + ["#8fbf87"] * 2
        ax.barh(y, df["平均 Test R²"], color=colors, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels(df["方法"])
        ax.invert_yaxis()
        ax.set_xlim(0.60, 0.93)
        ax.set_xlabel("平均 Test $R^2$")
        ax.text(0.02, 0.08, panel, transform=ax.transAxes, fontsize=7.5, color="#333333")
        for yy, val in zip(y, df["平均 Test R²"]):
            ax.text(val + 0.006, yy, f"{val:.3f}", va="center", fontsize=6.5)
        common_axes(ax)
    save(fig, out_name)


def plot_training_noise(root_path: str, out_name: str) -> None:
    root = rel(root_path)
    centers = [
        ("congestion_price_da", "拥塞价DA"),
        ("da_as_total_mw_primary_reserve", "主备用"),
    ]
    methods = [
        ("DNN_AllFeatures", "全量DNN", "#303030"),
        ("L1GateDNN", "L1GateDNN", "#2a6fbb"),
        ("DNN_SelectedByL1", "L1选中DNN", "#f28e2b"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.0), sharex=True)
    for ax, (center, label) in zip(axes, centers):
        for method, method_label, color in methods:
            df = pd.read_csv(root / f"CenterOn_{center}" / method / "log.csv")
            ax.plot(df["epoch"], df["train_r2"], color=color, linestyle="--", linewidth=0.9, alpha=0.72, label=f"{method_label} 训练")
            ax.plot(df["epoch"], df["test_r2"], color=color, linestyle="-", linewidth=1.1, alpha=0.95, label=f"{method_label} 测试")
        ax.text(0.02, 0.10, label, transform=ax.transAxes, fontsize=7.5, color="#333333")
        ax.set_ylabel("$R^2$")
        ax.set_ylim(0.20, 1.05)
        common_axes(ax)
    axes[-1].set_xlabel("epoch")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.44), frameon=False, ncol=2, columnspacing=0.8)
    save(fig, out_name)


def plot_topn_noise(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path))
    centers = [
        ("congestion_price_da", "拥塞价DA", "#2a6fbb"),
        ("da_as_total_mw_primary_reserve", "主备用", "#c44e52"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.75), sharex=False)
    for ax, (center, label, color) in zip(axes, centers):
        sub = df[df["center"] == center].copy()
        topn = sub[sub["exp_type"] == "topn"].sort_values("feature_count")
        full = sub[sub["exp_type"] == "full"].iloc[0]
        ax.plot(topn["feature_count"], topn["best_test_r2"], marker="o", markersize=2.4, linewidth=1.05, color=color, label=label)
        ax.scatter([full["feature_count"]], [full["best_test_r2"]], marker="s", s=22, color=color, edgecolor="#333333", linewidth=0.4)
        best = topn.loc[topn["best_test_r2"].idxmax()]
        ax.scatter([best["feature_count"]], [best["best_test_r2"]], marker="^", s=36, color="#d62728", zorder=4)
        ax.annotate(
            f"Top{int(best['feature_count'])}\n{best['best_test_r2']:.4f}",
            xy=(best["feature_count"], best["best_test_r2"]),
            xytext=(4, -18),
            textcoords="offset points",
            fontsize=6.5,
            color="#d62728",
        )
        ax.text(0.02, 0.10, label, transform=ax.transAxes, fontsize=7.5, color="#333333")
        ax.set_ylabel("Best Test $R^2$")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", frameon=False)
        common_axes(ax)
    axes[-1].set_xlabel("字段数")
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
    plot_train_pair_stack(net_dnn, f"{net_l1_root}/log.csv", "net_train_r2_stack_zh.png")
    plot_gate_curves(f"{net_l1_root}/gate_params.csv", f"{net_l1_root}/risk_map.csv", 0.10, "net_l1_gate_params_zh.png")
    plot_active_features(f"{net_l1_root}/log.csv", "net_l1_active_features_zh.png")
    plot_top_series(
        f"{net_l1_root}/redundancy_validation/top_series_3_6_9_12_15_18_e100/top_series_results.csv",
        "net_top_series_r2_zh.png",
    )

    plot_gate_curves(f"{improved_root}/gate_params.csv", f"{improved_root}/risk_map.csv", 0.50, "improved_l1_gate_params_zh.png")
    plot_meta_w(f"{improved_root}/W_meta_evolution.csv", "improved_l1_w_meta_zh.png")
    plot_meta_b(f"{improved_root}/b_meta_evolution.csv", "improved_l1_b_meta_zh.png")

    plot_dnn_l1_compare(
        "NEW_PROJECT/outputs/data2025_Processed_V2/BaselineComparison/DNN_vs_L1DNN_20260603_235906/method_compare_summary.csv",
        "dnn_vs_l1_method_compare_zh.png",
    )
    plot_budget_average(f"{budget_root}/budget_95_average_summary_zh.csv", "budget95_baseline_r2_zh.png")
    plot_budget_average(f"{budget_root}/budget_97_average_summary_zh.csv", "budget97_baseline_r2_zh.png")
    plot_budget_lines_stack(budget_root, "budget_baseline_lines_stack_zh.png")
    plot_budget_average_stack(budget_root, "budget_average_stack_zh.png")
    plot_training_noise(redundant_root, "redundant_training_r2_curves_zh.png")
    plot_topn_noise(f"{redundant_root}/topn_incremental_results.csv", "redundant_topn_incremental_r2_zh.png")


if __name__ == "__main__":
    main()
