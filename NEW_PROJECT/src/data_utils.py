from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


MODEL_OUTPUT_PARTS = {
    "DNN": ("DNN",),
    "L1GateDNN": ("L1GateDNN",),
    "ImprovedL1GateDNN": ("ImprovedGateDNN", "ImprovedL1GateDNN"),
    "ImprovedL2GateDNN": ("ImprovedGateDNN", "ImprovedL2GateDNN"),
}


@dataclass
class DatasetBundle:
    center: str
    features: List[str]
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_raw: np.ndarray
    y_train_raw: np.ndarray
    X_test_raw: np.ndarray
    y_test_raw: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float


def safe_name(value: str, max_len: int = 120) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "value")[:max_len]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_numeric_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    return numeric


def prepare_supervised_dataset(
    data_path: str | Path,
    center: str,
    features: Sequence[str],
    train_ratio: float,
    random_state: int,
) -> DatasetBundle:
    df = read_numeric_csv(data_path)
    missing = [c for c in [center, *features] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    selected = df[[center, *features]].dropna(axis=0, how="any")
    if selected.empty:
        raise ValueError("No usable rows after dropping missing values.")

    y = selected[center].to_numpy(dtype=np.float32).reshape(-1, 1)
    X = selected[list(features)].to_numpy(dtype=np.float32)

    n = len(selected)
    if n < 5:
        raise ValueError(f"Not enough rows for train/test split: {n}")

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    train_size = max(1, min(n - 1, int(n * train_ratio)))
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    X_train_raw = X[train_idx]
    y_train_raw = y[train_idx]
    X_test_raw = X[test_idx]
    y_test_raw = y[test_idx]

    x_mean = X_train_raw.mean(axis=0, keepdims=True)
    x_std = X_train_raw.std(axis=0, keepdims=True) + 1e-8
    y_mean = float(y_train_raw.mean())
    y_std = float(y_train_raw.std() + 1e-8)

    X_train = ((X_train_raw - x_mean) / x_std).astype(np.float32)
    X_test = ((X_test_raw - x_mean) / x_std).astype(np.float32)
    y_train = ((y_train_raw - y_mean) / y_std).astype(np.float32)
    y_test = ((y_test_raw - y_mean) / y_std).astype(np.float32)

    return DatasetBundle(
        center=center,
        features=list(features),
        train_idx=train_idx,
        test_idx=test_idx,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_train_raw=X_train_raw,
        y_train_raw=y_train_raw,
        X_test_raw=X_test_raw,
        y_test_raw=y_test_raw,
        x_mean=x_mean.squeeze(),
        x_std=x_std.squeeze(),
        y_mean=y_mean,
        y_std=y_std,
    )


def center_output_dir(output_root: str | Path, center: str) -> Path:
    return ensure_dir(Path(output_root) / f"CenterOn_{safe_name(center)}")


def model_output_root(center_dir: str | Path, model_name: str) -> Path:
    if model_name not in MODEL_OUTPUT_PARTS:
        raise ValueError(f"Unknown model: {model_name}")
    return ensure_dir(Path(center_dir).joinpath(*MODEL_OUTPUT_PARTS[model_name]))


def create_run_dir(center_dir: str | Path, model_name: str, run_name: str | None = None) -> Path:
    base = model_output_root(center_dir, model_name)
    name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base / safe_name(name)
    suffix = 2
    while run_dir.exists():
        run_dir = base / f"{safe_name(name)}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: str | Path, data: Dict) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_name_mapping(path: str | Path, center: str, features: Sequence[str]) -> None:
    rows = [{"index": 0, "role": "center", "name": center}]
    rows.extend({"index": i + 1, "role": "feature", "name": name} for i, name in enumerate(features))
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
