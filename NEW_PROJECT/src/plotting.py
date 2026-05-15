from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd


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


def plot_gate_history(
    gate_history: np.ndarray,
    epochs: Sequence[int],
    output_path: str | Path,
    feature_names: Sequence[str] | None = None,
) -> None:
    if gate_history.size == 0:
        return
    plot_height = max(4.5, min(11.0, 3.0 + 0.12 * gate_history.shape[1]))
    plt.figure(figsize=(16, plot_height))
    for idx in range(gate_history.shape[1]):
        if feature_names and idx < len(feature_names):
            label = str(feature_names[idx])
        else:
            label = f"Feature_{idx + 1}"
        plt.plot(epochs, gate_history[:, idx], label=label, alpha=0.72, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Gate Value")
    plt.title("Gate Values Over Epochs")
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.4))
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


def plot_meta_history(w_history: np.ndarray, b_history: np.ndarray, epochs: Sequence[int], w_path: str | Path, b_path: str | Path) -> None:
    if w_history.size:
        plt.figure(figsize=(8, 4))
        for idx in range(w_history.shape[1]):
            plt.plot(epochs, w_history[:, idx], label=f"W_{idx + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("W_meta")
        plt.title("W_meta Evolution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(w_path, dpi=150)
        plt.close()

    if b_history.size:
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, b_history)
        plt.xlabel("Epoch")
        plt.ylabel("b_meta")
        plt.title("b_meta Evolution")
        plt.tight_layout()
        plt.savefig(b_path, dpi=150)
        plt.close()