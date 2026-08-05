from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def _short_label(index: int, feature: str, max_len: int = 38) -> str:
    text = str(feature)
    if len(text) > max_len:
        text = f"{text[: max_len - 3]}..."
    return f"{index:02d} {text}"


def _load_effective_norms(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "dgate_effective_group_norms.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    required = {"epoch", "feature_index", "feature", "effective_group_l2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def plot_log_effective_norms(df: pd.DataFrame, run_dir: Path, eps: float = 1e-8) -> None:
    pivot = df.pivot(index="epoch", columns="feature_index", values="effective_group_l2").sort_index()
    feature_names = (
        df[["feature_index", "feature"]]
        .drop_duplicates()
        .sort_values("feature_index")
        .set_index("feature_index")["feature"]
        .to_dict()
    )
    log_values = np.log10(pivot.to_numpy(dtype=float) + eps)

    fig, ax = plt.subplots(figsize=(18, 9))
    epochs = pivot.index.to_numpy()
    for col_idx, feature_index in enumerate(pivot.columns):
        label = _short_label(int(feature_index), feature_names[int(feature_index)])
        ax.plot(epochs, log_values[:, col_idx], linewidth=1.25, alpha=0.68, label=label)

    for threshold in [1e-1, 1e-2, 1e-3, 1e-4]:
        ax.axhline(np.log10(threshold), color="black", linestyle=":", linewidth=1.0, alpha=0.35)
        ax.text(
            epochs[-1],
            np.log10(threshold),
            f" {threshold:g}",
            va="center",
            ha="left",
            fontsize=9,
            color="black",
            alpha=0.65,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\log_{10}(\|W_i^{eff}\|_2 + \epsilon)$")
    ax.set_title("D-Gating Effective Group Norms on Log Scale")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        labelspacing=0.25,
        borderaxespad=0.6,
    )
    fig.tight_layout()
    fig.savefig(run_dir / "dgate_effective_group_norms_log.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_effective_norm_heatmap(df: pd.DataFrame, run_dir: Path, eps: float = 1e-8) -> None:
    final_epoch = int(df["epoch"].max())
    final_order = (
        df[df["epoch"] == final_epoch]
        .sort_values("effective_group_l2", ascending=False)["feature_index"]
        .astype(int)
        .tolist()
    )
    pivot = df.pivot(index="feature_index", columns="epoch", values="effective_group_l2").loc[final_order]
    feature_names = (
        df[["feature_index", "feature"]]
        .drop_duplicates()
        .set_index("feature_index")["feature"]
        .to_dict()
    )
    labels = [_short_label(int(idx), feature_names[int(idx)], max_len=44) for idx in pivot.index]
    log_values = np.log10(pivot.to_numpy(dtype=float) + eps)

    fig_height = max(10, min(18, 4 + 0.22 * len(labels)))
    fig, ax = plt.subplots(figsize=(15, fig_height))
    image = ax.imshow(
        log_values,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=Normalize(vmin=-6, vmax=max(0.0, float(np.nanmax(log_values)))),
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Feature, sorted by final effective norm")
    ax.set_title(r"Heatmap of $\log_{10}(\|W_i^{eff}\|_2 + \epsilon)$")

    epochs = pivot.columns.to_numpy()
    tick_positions = np.linspace(0, len(epochs) - 1, num=min(9, len(epochs)), dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(epochs[pos])) for pos in tick_positions])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label(r"$\log_{10}(\|W_i^{eff}\|_2 + \epsilon)$")
    fig.tight_layout()
    fig.savefig(run_dir / "dgate_effective_group_norms_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_survival(df: pd.DataFrame, run_dir: Path) -> None:
    thresholds = [1e-1, 1e-2, 1e-3, 1e-4]
    rows = []
    for epoch, epoch_df in df.groupby("epoch"):
        row = {"epoch": int(epoch)}
        values = epoch_df["effective_group_l2"].to_numpy(dtype=float)
        for threshold in thresholds:
            row[f"active_gt_{threshold:g}"] = int(np.sum(values > threshold))
        rows.append(row)

    survival = pd.DataFrame(rows).sort_values("epoch")
    survival.to_csv(run_dir / "dgate_threshold_survival.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for threshold in thresholds:
        col = f"active_gt_{threshold:g}"
        ax.plot(survival["epoch"], survival[col], marker="o", markersize=2.5, linewidth=1.8, label=rf"$> {threshold:g}$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of surviving feature groups")
    ax.set_title(r"Feature Groups Surviving Effective-Norm Thresholds")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(title=r"Threshold for $\|W_i^{eff}\|_2$")
    fig.tight_layout()
    fig.savefig(run_dir / "dgate_threshold_survival.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot additional D-Gating effective-norm diagnostics.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    df = _load_effective_norms(run_dir)
    plot_log_effective_norms(df, run_dir, eps=float(args.eps))
    plot_effective_norm_heatmap(df, run_dir, eps=float(args.eps))
    plot_threshold_survival(df, run_dir)
    print(f"Saved D-Gating diagnostic plots to {run_dir}")


if __name__ == "__main__":
    main()
