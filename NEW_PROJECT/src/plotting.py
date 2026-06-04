from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd


def _zh_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(f"{p.stem}_zh{p.suffix}")


def _configure_chinese_font() -> None:
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
    plt.rcParams["axes.unicode_minus"] = False


def plot_loss_and_r2(log_df: pd.DataFrame, loss_path: str | Path, r2_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="Train loss")
    ax.plot(log_df["epoch"], log_df["test_loss"], label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Train/Test Loss")
    max_loss = float(log_df[["train_loss", "test_loss"]].max().max())
    ax.set_ylim(bottom=0.0, top=max_loss * 1.08 if max_loss > 0 else 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(log_df["epoch"], log_df["train_r2"], label=r"Train $R^2$")
    ax.plot(log_df["epoch"], log_df["test_r2"], label=r"Test $R^2$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(r"Train/Test $R^2$")
    min_r2 = float(log_df[["train_r2", "test_r2"]].min().min())
    max_r2 = float(log_df[["train_r2", "test_r2"]].max().max())
    lower = 0.0 if min_r2 >= 0 else np.floor(min_r2 * 10) / 10
    upper = 1.02 if max_r2 <= 1.02 else np.ceil(max_r2 * 10) / 10
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(r2_path, dpi=150)
    plt.close()

    _configure_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="Train loss")
    ax.plot(log_df["epoch"], log_df["test_loss"], label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("训练/测试 Loss")
    ax.set_ylim(bottom=0.0, top=max_loss * 1.08 if max_loss > 0 else 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(_zh_path(loss_path), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(log_df["epoch"], log_df["train_r2"], label=r"Train $R^2$")
    ax.plot(log_df["epoch"], log_df["test_r2"], label=r"Test $R^2$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$R^2$")
    ax.set_title(r"训练/测试 $R^2$")
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(_zh_path(r2_path), dpi=150)
    plt.close()


def plot_gate_history(
    gate_history: np.ndarray,
    epochs: Sequence[int],
    output_path: str | Path,
    feature_names: Sequence[str] | None = None,
    warmup_epoch: int | None = None,
    gate_threshold: float | None = None,
) -> None:
    if gate_history.size == 0:
        return
    feature_count = gate_history.shape[1]
    plot_width = max(16.0, min(21.0, 13.5 + 0.11 * feature_count))
    plot_height = max(5.2, min(13.2, 4.0 + 0.16 * feature_count)) * 0.85
    legend_fontsize = 8.75 if feature_count >= 45 else 10
    legend_labelspacing = 0.32 if feature_count >= 45 else 0.55
    legend_handlelength = 2.1 if feature_count >= 45 else 2.6
    axis_label_fontsize = 19.5
    title_fontsize = 22
    tick_fontsize = 17

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    for idx in range(gate_history.shape[1]):
        if feature_names and idx < len(feature_names):
            label = str(feature_names[idx])
        else:
            label = f"Feature_{idx + 1}"
        ax.plot(epochs, gate_history[:, idx], label=label, alpha=0.72, linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=axis_label_fontsize)
    ax.set_ylabel("Gate Value", fontsize=axis_label_fontsize)
    ax.set_title("Gate Values Over Epochs", fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    if warmup_epoch and warmup_epoch > 0:
        ax.axvline(warmup_epoch, color="red", linestyle="--", linewidth=1.6, alpha=0.9, label="Warm-up End")
    if gate_threshold is not None:
        ax.axhline(
            float(gate_threshold),
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            label=f"Gate = {float(gate_threshold):.2f}",
        )
    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=legend_fontsize,
        labelspacing=legend_labelspacing,
        handlelength=legend_handlelength,
        borderaxespad=0.8,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    _configure_chinese_font()
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    for idx in range(gate_history.shape[1]):
        if feature_names and idx < len(feature_names):
            label = str(feature_names[idx])
        else:
            label = f"Feature_{idx + 1}"
        ax.plot(epochs, gate_history[:, idx], label=label, alpha=0.72, linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=axis_label_fontsize)
    ax.set_ylabel("门控参数gate的值", fontsize=axis_label_fontsize)
    ax.set_title("门控参数gate随Epoch变化", fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    if warmup_epoch and warmup_epoch > 0:
        ax.axvline(warmup_epoch, color="red", linestyle="--", linewidth=1.6, alpha=0.9, label="Warm-up End")
    if gate_threshold is not None:
        ax.axhline(
            float(gate_threshold),
            color="black",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            label=f"gate阈值 = {float(gate_threshold):.2f}",
        )
    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=legend_fontsize,
        labelspacing=legend_labelspacing,
        handlelength=legend_handlelength,
        borderaxespad=0.8,
    )
    fig.tight_layout()
    fig.savefig(_zh_path(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_gate_logit_history(
    logit_history: np.ndarray,
    epochs: Sequence[int],
    output_path: str | Path,
    feature_names: Sequence[str] | None = None,
) -> None:
    if logit_history.size == 0:
        return
    plot_height = max(4.5, min(11.0, 3.0 + 0.12 * logit_history.shape[1]))
    plt.figure(figsize=(16, plot_height))
    for idx in range(logit_history.shape[1]):
        if feature_names and idx < len(feature_names):
            label = str(feature_names[idx])
        else:
            label = f"Feature_{idx + 1}"
        plt.plot(epochs, logit_history[:, idx], label=label, alpha=0.72, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel(r"Risk Score $s = W^T R + b$")
    plt.title(r"Risk-Boundary Scores $s$ Over Epochs")
    ax = plt.gca()
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.grid(True, axis="y", alpha=0.2)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        labelspacing=0.55,
        handlelength=2.6,
        borderaxespad=0.8,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    _configure_chinese_font()
    plt.figure(figsize=(16, plot_height))
    for idx in range(logit_history.shape[1]):
        if feature_names and idx < len(feature_names):
            label = str(feature_names[idx])
        else:
            label = f"Feature_{idx + 1}"
        plt.plot(epochs, logit_history[:, idx], label=label, alpha=0.72, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel(r"风险边界得分 $s = W^T R + b$")
    plt.title(r"风险边界得分 $s$ 随 Epoch 变化")
    ax = plt.gca()
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.grid(True, axis="y", alpha=0.2)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        labelspacing=0.55,
        handlelength=2.6,
        borderaxespad=0.8,
    )
    plt.tight_layout()
    plt.savefig(_zh_path(output_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_active_features(log_df: pd.DataFrame, output_path: str | Path) -> None:
    if "active_features" not in log_df.columns:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(log_df["epoch"], log_df["active_features"], marker="o", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Active Features")
    plt.title("Active Features Over Epochs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    _configure_chinese_font()
    plt.figure(figsize=(7, 4))
    plt.plot(log_df["epoch"], log_df["active_features"], marker="o", linestyle="-")
    plt.xlabel("Epoch")
    plt.ylabel("活跃特征数")
    plt.title("活跃特征数随 Epoch 变化")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(_zh_path(output_path), dpi=150)
    plt.close()


def plot_meta_history(w_history: np.ndarray, b_history: np.ndarray, epochs: Sequence[int], w_path: str | Path, b_path: str | Path) -> None:
    if w_history.size:
        plt.figure(figsize=(8, 4))
        for idx in range(w_history.shape[1]):
            plt.plot(epochs, w_history[:, idx], label=rf"$W_{{{idx + 1}}}$")
        plt.xlabel("Epoch")
        plt.ylabel("Weight W")
        plt.title("Correlation-Metric Weights W Evolution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(w_path, dpi=150)
        plt.close()

        _configure_chinese_font()
        plt.figure(figsize=(8, 4))
        for idx in range(w_history.shape[1]):
            plt.plot(epochs, w_history[:, idx], label=rf"$W_{{{idx + 1}}}$")
        plt.xlabel("Epoch")
        plt.ylabel("指标权重 W")
        plt.title("相关性指标权重 W 演化")
        plt.legend()
        plt.tight_layout()
        plt.savefig(_zh_path(w_path), dpi=150)
        plt.close()

        shifted = w_history - np.max(w_history, axis=1, keepdims=True)
        w_softmax = np.exp(shifted)
        w_softmax = w_softmax / np.sum(w_softmax, axis=1, keepdims=True)
        softmax_path = Path(w_path).with_name(f"{Path(w_path).stem}_softmax{Path(w_path).suffix}")

        plt.figure(figsize=(8, 4))
        for idx in range(w_softmax.shape[1]):
            plt.plot(epochs, w_softmax[:, idx], label=rf"$\alpha_{{{idx + 1}}}$")
        plt.xlabel("Epoch")
        plt.ylabel(r"Softmax Weight $\alpha$")
        plt.title(r"Normalized Metric Contributions $\alpha=\mathrm{softmax}(W)$")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(softmax_path, dpi=150)
        plt.close()

        _configure_chinese_font()
        plt.figure(figsize=(8, 4))
        for idx in range(w_softmax.shape[1]):
            plt.plot(epochs, w_softmax[:, idx], label=rf"$\alpha_{{{idx + 1}}}$")
        plt.xlabel("Epoch")
        plt.ylabel(r"归一化权重 $\alpha$")
        plt.title(r"归一化指标贡献 $\alpha=\mathrm{softmax}(W)$ 演化")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(_zh_path(softmax_path), dpi=150)
        plt.close()

    if b_history.size:
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, b_history)
        plt.xlabel("Epoch")
        plt.ylabel("Bias b")
        plt.title("Risk-Boundary Bias b Evolution")
        plt.tight_layout()
        plt.savefig(b_path, dpi=150)
        plt.close()

        _configure_chinese_font()
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, b_history)
        plt.xlabel("Epoch")
        plt.ylabel("偏置项 b")
        plt.title("风险边界偏置项 b 演化")
        plt.tight_layout()
        plt.savefig(_zh_path(b_path), dpi=150)
        plt.close()
