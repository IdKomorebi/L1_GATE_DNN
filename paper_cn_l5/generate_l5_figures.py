from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
OUT = PAPER / "figures"
OUT.mkdir(parents=True, exist_ok=True)

DNN_COLOR = "#2ca02c"
L1_COLOR = "#1f77b4"
PEAK_COLOR = "#d62728"
FULL_COLOR = "#174a7e"
GRAY = "#a8a8a8"

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

METHOD_LABELS = {
    "DNN_AllFeatures": "DNN",
    "L1GateDNN": "L1DNN",
    "L1GateDNN_Selected": "L1DNN",
    "NMI": "NMI",
    "Pearson": "Pearson",
    "Spearman": "Spearman",
    "Lasso": "Lasso",
    "ElasticNet": "ElasticNet",
    "RandomForest": "RF",
    "XGBoost": "XGBoost",
}


def rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def setup_style() -> None:
    preferred = [
        "PingFang SC",
        "Heiti SC",
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
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "font.size": 8.8,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
        }
    )


def common_axes(ax: plt.Axes, *, x_grid: bool = False) -> None:
    ax.grid(True, axis="both" if x_grid else "y", color="#d8d8d8", linewidth=0.45, alpha=0.78)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, out_name: str) -> None:
    fig.savefig(OUT / out_name, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dnn_l1_compare() -> None:
    df = pd.read_csv(
        rel("NEW_PROJECT/outputs/data2025_Processed_V2/BaselineComparison/DNN_vs_L1DNN_20260603_235906/method_compare_summary.csv")
    )
    wide = df.pivot(index="target_label", columns="method", values="best_test_r2")
    order = [target for target in TARGET_LABELS if target in wide.index]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(4.25, 2.9))
    ax.plot(
        x,
        wide.loc[order, "DNN_AllFeatures"].to_numpy(dtype=float),
        color=DNN_COLOR,
        marker="o",
        markersize=3.4,
        linewidth=1.85,
        label="DNN",
        zorder=3,
    )
    ax.plot(
        x,
        wide.loc[order, "L1GateDNN_Selected"].to_numpy(dtype=float),
        color=L1_COLOR,
        marker="^",
        markersize=3.6,
        linewidth=2.15,
        linestyle="--",
        label="L1DNN",
        zorder=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS.get(t, t) for t in order], rotation=48, ha="right")
    ax.set_ylabel("Best Test $R^2$")
    ax.set_ylim(0.55, 1.04)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    common_axes(ax)
    save(fig, "dnn_vs_l1_method_compare_zh.png")


def _budget_style(method: str) -> dict[str, object]:
    palette = {
        "DNN_AllFeatures": DNN_COLOR,
        "L1GateDNN": L1_COLOR,
        "NMI": "#9467bd",
        "Pearson": "#ff7f0e",
        "Spearman": "#8c564b",
        "Lasso": "#e377c2",
        "ElasticNet": "#bcbd22",
        "RandomForest": "#7f7f7f",
        "XGBoost": "#17becf",
    }
    markers = {
        "DNN_AllFeatures": "o",
        "L1GateDNN": "^",
        "NMI": "s",
        "Pearson": "D",
        "Spearman": "v",
        "Lasso": "P",
        "ElasticNet": "X",
        "RandomForest": "*",
        "XGBoost": "h",
    }
    is_key = method in {"DNN_AllFeatures", "L1GateDNN"}
    return {
        "color": palette.get(method, GRAY),
        "marker": markers.get(method, "o"),
        "linestyle": "--" if method == "L1GateDNN" else "-",
        "linewidth": 2.1 if is_key else 1.05,
        "markersize": 3.8 if is_key else 2.6,
        "alpha": 1.0 if is_key else 0.78,
        "zorder": 5 if is_key else 3,
    }


def plot_budget_lines(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path))
    methods = [
        "DNN_AllFeatures",
        "L1GateDNN",
        "NMI",
        "Pearson",
        "Spearman",
        "Lasso",
        "ElasticNet",
        "RandomForest",
        "XGBoost",
    ]
    order = df["target_label"].tolist()
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(4.35, 3.05))
    for method in methods:
        if method not in df.columns:
            continue
        ax.plot(x, df[method].to_numpy(dtype=float), label=METHOD_LABELS.get(method, method), **_budget_style(method))
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS.get(t, t) for t in order], rotation=48, ha="right")
    ax.set_ylabel("Test $R^2$")
    ax.set_ylim(0.15, 1.04)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3, frameon=False, columnspacing=0.8)
    common_axes(ax)
    save(fig, out_name)


def plot_budget_average(csv_path: str, out_name: str) -> None:
    df = pd.read_csv(rel(csv_path)).copy()
    method_col = next(col for col in df.columns if "方法" in col)
    value_col = next(col for col in df.columns if "Test" in col or "R" in col)
    df[method_col] = df[method_col].astype(str).str.strip()
    order = ["全量 DNN", "L1GateDNN", "NMI", "Pearson", "Spearman", "Lasso", "ElasticNet", "RandomForest", "XGBoost"]
    df["_order"] = df[method_col].map({name: i for i, name in enumerate(order)}).fillna(999)
    df = df.sort_values("_order").reset_index(drop=True)

    def color(method: str) -> str:
        if method == "全量 DNN":
            return DNN_COLOR
        if method == "L1GateDNN":
            return L1_COLOR
        return GRAY

    fig, ax = plt.subplots(figsize=(3.75, 2.7))
    y = np.arange(len(df))
    bars = ax.barh(y, df[value_col].to_numpy(dtype=float), color=[color(m) for m in df[method_col]], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in df[method_col]])
    ax.invert_yaxis()
    ax.set_xlim(0.60, max(0.94, float(df[value_col].max()) + 0.035))
    ax.set_xlabel("平均 Test $R^2$")
    for bar, val in zip(bars, df[value_col].to_numpy(dtype=float)):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=7.0)
    common_axes(ax)
    save(fig, out_name)


def plot_topn_single(center: str, out_name: str) -> None:
    csv_path = rel(
        "NEW_PROJECT/outputs/data2025_Processed_V2/RedundantNoiseExplanation/"
        "RedundantNoiseExplanation_20260606_topn_extended/topn_incremental_results.csv"
    )
    df = pd.read_csv(csv_path)
    sub = df[df["center"] == center].copy()
    topn = sub[sub["exp_type"] == "topn"].copy()
    full_marker = sub[sub["exp_type"] == "full_marker"].copy()
    if full_marker.empty:
        full_marker = sub[sub["exp_type"] == "full"].copy()
    full = full_marker.sort_values("plot_feature_count").tail(1)
    curve = pd.concat([topn, full], ignore_index=True).sort_values("plot_feature_count")
    x = curve["plot_feature_count"].to_numpy(dtype=float)
    y = curve["best_test_r2"].to_numpy(dtype=float)
    best = topn.loc[topn["best_test_r2"].idxmax()]
    final = full.iloc[0]

    fig, ax = plt.subplots(figsize=(3.95, 2.35))
    ax.plot(x, y, color=L1_COLOR, marker="o", markersize=3.2, linewidth=1.55, label=TARGET_LABELS.get(center, center))
    ax.scatter(
        [best["plot_feature_count"]],
        [best["best_test_r2"]],
        marker="^",
        s=58,
        color=PEAK_COLOR,
        edgecolor=PEAK_COLOR,
        zorder=5,
    )
    ax.scatter(
        [final["plot_feature_count"]],
        [final["best_test_r2"]],
        marker="s",
        s=40,
        color=FULL_COLOR,
        edgecolor=FULL_COLOR,
        zorder=5,
    )

    best_x = float(best["plot_feature_count"])
    best_y = float(best["best_test_r2"])
    final_x = float(final["plot_feature_count"])
    final_y = float(final["best_test_r2"])
    ax.annotate(
        f"Top{int(best['feature_count'])}\n{best_y:.4f}",
        xy=(best_x, best_y),
        xytext=(8, -4 if best_y > 0.95 else 8),
        textcoords="offset points",
        ha="left",
        va="top" if best_y > 0.95 else "bottom",
        color=PEAK_COLOR,
        fontsize=7.6,
    )
    ax.annotate(
        f"Full\n{final_y:.4f}",
        xy=(final_x, final_y),
        xytext=(-9, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=FULL_COLOR,
        fontsize=7.3,
    )

    ax.text(0.02, 0.10, TARGET_LABELS.get(center, center), transform=ax.transAxes, ha="left", va="bottom", fontsize=9.0, color="#333333")
    ax.set_xlabel("特征数")
    ax.set_ylabel("Best Test $R^2$")
    ax.set_xlim(-2, 72)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(np.arange(0, 71, 10))
    ax.legend(loc="lower right", frameon=False)
    common_axes(ax, x_grid=True)
    save(fig, out_name)


def main() -> None:
    setup_style()
    plot_dnn_l1_compare()
    budget_root = "NEW_PROJECT/outputs/data2025_Processed_V2/BaselineComparison/BudgetBaselineL1Selector_20260604_113307"
    plot_budget_lines(f"{budget_root}/budget_95/baseline_summary_wide.csv", "budget95_baseline_lines_zh.png")
    plot_budget_lines(f"{budget_root}/budget_97/baseline_summary_wide.csv", "budget97_baseline_lines_zh.png")
    plot_budget_average(f"{budget_root}/budget_95_average_summary_zh.csv", "budget95_average_bar_zh.png")
    plot_budget_average(f"{budget_root}/budget_97_average_summary_zh.csv", "budget97_average_bar_zh.png")
    plot_topn_single("congestion_price_da", "topn_congestion_zh.png")
    plot_topn_single("da_as_total_mw_primary_reserve", "topn_primary_zh.png")


if __name__ == "__main__":
    main()
