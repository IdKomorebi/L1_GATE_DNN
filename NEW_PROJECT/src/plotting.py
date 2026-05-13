from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_loss_and_r2(log_df: pd.DataFrame, loss_path: str | Path, r2_path: str | Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    plt.plot(log_df["epoch"], log_df["test_loss"], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train/Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(log_df["epoch"], log_df["train_r2"], label="train_r2")
    plt.plot(log_df["epoch"], log_df["test_r2"], label="test_r2")
    plt.xlabel("Epoch")
    plt.ylabel("R^2")
    plt.title("Train/Test R^2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(r2_path, dpi=150)
    plt.close()


def plot_gate_history(
    gate_history: np.ndarray,
    epochs: Sequence[int],
    output_path: str | Path,
) -> None:
    if gate_history.size == 0:
        return
    plt.figure(figsize=(10, 6))
    for idx in range(gate_history.shape[1]):
        plt.plot(epochs, gate_history[:, idx], label=f"Feature_{idx + 1}", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Gate Value")
    plt.title("Gate Values Over Epochs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
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
