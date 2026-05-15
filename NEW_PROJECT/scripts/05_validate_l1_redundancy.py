from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import ensure_dir, prepare_supervised_dataset, read_numeric_csv, save_json
from src.config import load_config, resolve_project_path
from src.models import DNNRegressor, SimpleAdam


@dataclass
class FeatureEntry:
    drop_id: int
    feature_index: int | None
    name: str
    gate: float | None


@dataclass
class ExperimentResult:
    exp_name: str
    exp_type: str
    x_label: str
    feature_count: int
    dropped_id: int | None
    dropped_feature: str
    best_test_r2: float
    final_test_r2: float
    best_epoch: int
    output_dir: str


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested --device mps, but torch.backends.mps.is_available() is False.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested --device cuda, but torch.cuda.is_available() is False.")
    return torch.device(name)


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if float(ss_tot) <= 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def _eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            total_loss += float(loss.detach().cpu()) * xb.size(0)
            ys.append(yb.detach().cpu())
            ps.append(pred.detach().cpu())
    y_cat = torch.cat(ys, dim=0)
    p_cat = torch.cat(ps, dim=0)
    return total_loss / len(loader.dataset), _r2_score(y_cat, p_cat)


def _load_selected_features(run_dir: Path) -> List[FeatureEntry]:
    selected_path = run_dir / "selected_features.json"
    if not selected_path.exists():
        raise FileNotFoundError(f"selected_features.json not found: {selected_path}")

    payload = _load_json(selected_path)
    rows = payload.get("features", [])
    if not rows:
        raise ValueError(f"No selected features found in {selected_path}")

    entries = []
    for drop_id, row in enumerate(rows, start=1):
        entries.append(
            FeatureEntry(
                drop_id=drop_id,
                feature_index=row.get("index"),
                name=str(row["name"]),
                gate=None if row.get("gate") is None else float(row["gate"]),
            )
        )
    return entries


def _top_gate_features(run_dir: Path, top_n: int) -> List[FeatureEntry]:
    if top_n <= 0:
        raise ValueError("--top-n must be positive.")

    risk_path = run_dir / "risk_map.csv"
    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        if {"related", "best_gate"}.issubset(risk_df.columns):
            risk_df = risk_df.copy()
            risk_df["abs_gate"] = risk_df["best_gate"].abs()
            risk_df = risk_df.sort_values("abs_gate", ascending=False).head(top_n)
            return [
                FeatureEntry(
                    drop_id=i,
                    feature_index=None if pd.isna(row.get("feature_index")) else int(row.get("feature_index")),
                    name=str(row["related"]),
                    gate=float(row["best_gate"]),
                )
                for i, (_, row) in enumerate(risk_df.iterrows(), start=1)
            ]

    gate_path = run_dir / "gate_params.csv"
    if gate_path.exists():
        gate_df = pd.read_csv(gate_path)
        last_epoch = gate_df["epoch"].max()
        gate_df = gate_df[gate_df["epoch"] == last_epoch].copy()
        gate_df["abs_gate"] = gate_df["gate"].abs()
        gate_df = gate_df.sort_values("abs_gate", ascending=False).head(top_n)
        return [
            FeatureEntry(
                drop_id=i,
                feature_index=int(row["feature_index"]),
                name=str(row["feature"]),
                gate=float(row["gate"]),
            )
            for i, (_, row) in enumerate(gate_df.iterrows(), start=1)
        ]

    raise FileNotFoundError(f"Neither risk_map.csv nor gate_params.csv exists under {run_dir}")


def _all_run_features(cfg: Dict[str, Any], data_path: Path, center: str) -> List[str]:
    features = [str(f) for f in cfg.get("features", [])]
    if features:
        return features
    df = read_numeric_csv(data_path)
    return [str(c) for c in df.columns if str(c) != center]


def _check_features(data_path: Path, center: str, feature_groups: Sequence[Sequence[str]]) -> None:
    columns = set(str(c) for c in read_numeric_csv(data_path).columns)
    missing = []
    if center not in columns:
        missing.append(center)
    for features in feature_groups:
        missing.extend(f for f in features if f not in columns)
    missing = sorted(set(missing))
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")


def _config_get(cfg: Dict[str, Any], key: str, fallback: Any) -> Any:
    value = cfg.get(key, fallback)
    return fallback if value is None else value


def _train_dnn(
    data_path: Path,
    center: str,
    features: List[str],
    hidden_dims: Sequence[int],
    train_ratio: float,
    random_state: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    output_dir: Path,
) -> ExperimentResult:
    if not features:
        raise ValueError("Feature list cannot be empty.")

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_state)

    bundle = prepare_supervised_dataset(
        data_path=data_path,
        center=center,
        features=features,
        train_ratio=train_ratio,
        random_state=random_state,
    )

    train_ds = TensorDataset(torch.from_numpy(bundle.X_train), torch.from_numpy(bundle.y_train))
    test_ds = TensorDataset(torch.from_numpy(bundle.X_test), torch.from_numpy(bundle.y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = DNNRegressor(len(features), hidden_dims).to(device)
    optimizer = SimpleAdam(model.parameters(), lr=lr)

    log_rows = []
    best_test_r2 = -math.inf
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * xb.size(0)

        train_loss, train_r2 = _eval_model(model, train_loader, device)
        test_loss, test_r2 = _eval_model(model, test_loader, device)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_total_loss": train_loss_sum / len(train_ds),
                "train_r2": train_r2,
                "test_loss": test_loss,
                "test_r2": test_r2,
            }
        )
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    ensure_dir(output_dir)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "log.csv", index=False, encoding="utf-8-sig")
    save_json(output_dir / "used_features.json", {"center": center, "features": features})
    if best_state is not None:
        torch.save({"model_state": best_state, "center": center, "features": features}, output_dir / "model.pth")

    return ExperimentResult(
        exp_name=output_dir.name,
        exp_type="",
        x_label="",
        feature_count=len(features),
        dropped_id=None,
        dropped_feature="",
        best_test_r2=float(best_test_r2),
        final_test_r2=float(log_df["test_r2"].iloc[-1]),
        best_epoch=int(best_epoch),
        output_dir=str(output_dir),
    )


def _wrap_feature_line(entry: FeatureEntry, width: int = 44) -> str:
    gate = "" if entry.gate is None else f"  gate={entry.gate:.4g}"
    prefix = f"[{entry.drop_id:02d}] "
    text = f"{entry.name}{gate}"
    wrapped = textwrap.wrap(text, width=width)
    if not wrapped:
        return prefix.rstrip()
    lines = [prefix + wrapped[0]]
    lines.extend(" " * len(prefix) + part for part in wrapped[1:])
    return "\n".join(lines)


def _plot_bar_summary(
    results: List[ExperimentResult],
    selected_entries: List[FeatureEntry],
    output_path: Path,
) -> None:
    values = np.asarray([r.best_test_r2 for r in results], dtype=float)
    x = np.arange(len(results))
    labels = [r.x_label for r in results]
    colors = []
    for r in results:
        if r.exp_type == "all":
            colors.append("#7f8c8d")
        elif r.exp_type == "selected":
            colors.append("#2f80ed")
        elif r.exp_type == "unselected":
            colors.append("#7b61ff")
        else:
            colors.append("#f2994a")

    fig_height = max(6.5, min(18.0, 5.5 + 0.18 * len(selected_entries) + 0.05 * len(results)))
    fig_width = max(13.5, min(34.0, 9.5 + 0.42 * len(results) + 5.2))
    fig, (ax, info_ax) = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [max(2.7, 0.22 * len(results) + 2.7), 1.65]},
    )

    bars = ax.bar(x, values, color=colors, alpha=0.86)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel(r"Best Test $R^2$")
    ax.set_xlabel("Validation Experiment")
    ax.set_title(r"L1GateDNN Redundancy Validation by DNN $R^2$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45 if len(labels) > 8 else 0, ha="right" if len(labels) > 8 else "center")
    ax.grid(True, axis="y", alpha=0.22)

    finite = values[np.isfinite(values)]
    if finite.size:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        pad = max(0.03, (vmax - vmin) * 0.12)
        ax.set_ylim(min(0.0, vmin - pad), max(1.0 if vmax <= 1.0 else vmax + pad, vmax + pad))

    for bar, r2 in zip(bars, values):
        y = float(r2)
        va = "bottom" if y >= 0 else "top"
        offset = 3 if y >= 0 else -5
        ax.annotate(
            f"{y:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
        )

    info_ax.axis("off")
    info_lines = ["Selected Features:"]
    info_lines.extend(_wrap_feature_line(entry) for entry in selected_entries)
    info_text = "\n".join(info_lines)
    fontsize = float(np.clip(11.5 - 0.06 * len(selected_entries), 7.5, 10.8))
    info_ax.text(
        0.0,
        1.0,
        info_text,
        ha="left",
        va="top",
        fontsize=fontsize,
        linespacing=1.35,
        family="monospace",
        transform=info_ax.transAxes,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_validation(args: argparse.Namespace) -> Path:
    project_cfg = load_config(args.config)
    validation_cfg = project_cfg.get("validation", {}).get("l1_redundancy", {})

    run_dir_value = args.run_dir or validation_cfg.get("run_dir")
    if not run_dir_value:
        raise ValueError("Please set validation.l1_redundancy.run_dir in the config or pass RUN_DIR on the command line.")
    run_dir = resolve_project_path(project_cfg, run_dir_value).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    run_cfg = _load_json(run_dir / "config.json")
    if run_cfg.get("model") != "L1GateDNN":
        raise ValueError(f"Expected an L1GateDNN run, got model={run_cfg.get('model')!r}")

    center = str(run_cfg["center"])
    data_path = Path(run_cfg["data_path"]).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_path}")

    if args.top_n is None:
        selected_entries = _load_selected_features(run_dir)
        selection_tag = f"selected{len(selected_entries)}"
        selection_label = f"selected_features.json (n={len(selected_entries)})"
    else:
        selected_entries = _top_gate_features(run_dir, args.top_n)
        selection_tag = f"top{args.top_n}"
        selection_label = f"top {args.top_n} by |gate|"

    selected_features = [entry.name for entry in selected_entries]
    full_features = _all_run_features(run_cfg, data_path, center)
    selected_feature_set = set(selected_features)
    unselected_features = [feature for feature in full_features if feature not in selected_feature_set]
    _check_features(data_path, center, [full_features, selected_features, unselected_features])

    params = run_cfg.get("params", {})
    hidden_dims = [int(v) for v in _config_get(validation_cfg, "hidden_dims", params.get("hidden_dims", [64, 32, 16]))]
    epochs = int(_config_get(validation_cfg, "epochs", 100))
    train_ratio = float(_config_get(validation_cfg, "train_ratio", params.get("train_ratio", 0.8)))
    random_state = int(_config_get(validation_cfg, "random_state", params.get("random_state", 42)))
    batch_size = int(_config_get(validation_cfg, "batch_size", params.get("batch_size", 50)))
    lr = float(_config_get(validation_cfg, "lr", params.get("lr", 1e-3)))
    device = _choose_device(str(_config_get(validation_cfg, "device", "cpu")))
    include_drop_one = bool(_config_get(validation_cfg, "include_drop_one", True)) if args.include_drop_one is None else bool(args.include_drop_one)
    include_unselected = bool(_config_get(validation_cfg, "include_unselected", True))

    output_name = f"{selection_tag}_e{epochs}_{'drop' if include_drop_one else 'nodrop'}"
    output_dir_cfg = validation_cfg.get("output_dir")
    if output_dir_cfg:
        output_root = resolve_project_path(project_cfg, output_dir_cfg) / run_dir.name / output_name
    else:
        output_root = run_dir / "redundancy_validation" / output_name
    ensure_dir(output_root)

    save_json(
        output_root / "validation_config.json",
        {
            "source_run_dir": str(run_dir),
            "source_run_dir_config_value": str(run_dir_value),
            "config_path": str(Path(args.config).expanduser().resolve()),
            "center": center,
            "data_path": str(data_path),
            "selection": selection_label,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "train_ratio": train_ratio,
            "random_state": random_state,
            "device": str(device),
            "hidden_dims": hidden_dims,
            "include_drop_one": include_drop_one,
            "include_unselected": include_unselected,
        },
    )
    pd.DataFrame([entry.__dict__ for entry in selected_entries]).to_csv(
        output_root / "selected_feature_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    experiments: List[tuple[str, str, str, List[str], int | None, str]] = [
        ("DNN_full_all", "all", f"Full\nn={len(full_features)}", full_features, None, ""),
        ("DNN_selected", "selected", f"Selected\nn={len(selected_features)}", selected_features, None, ""),
    ]
    if include_drop_one:
        for entry in selected_entries:
            kept = [feature for feature in selected_features if feature != entry.name]
            experiments.append(
                (
                    f"DNN_drop_{entry.drop_id:02d}",
                    "drop_one",
                    f"Drop #{entry.drop_id:02d}\nn={len(kept)}",
                    kept,
                    entry.drop_id,
                    entry.name,
                )
            )
    if include_unselected and unselected_features:
        experiments.append(
            (
                "DNN_unselected",
                "unselected",
                f"Unselected\nn={len(unselected_features)}",
                unselected_features,
                None,
                "",
            )
        )
    elif include_unselected:
        print("Skipping DNN_unselected because Full - Selected is empty.")

    results: List[ExperimentResult] = []
    for exp_idx, (exp_name, exp_type, x_label, features, dropped_id, dropped_feature) in enumerate(experiments, start=1):
        exp_dir = output_root / exp_name
        print(f"[{exp_idx}/{len(experiments)}] {exp_name}: training DNN with {len(features)} features")
        result = _train_dnn(
            data_path=data_path,
            center=center,
            features=features,
            hidden_dims=hidden_dims,
            train_ratio=train_ratio,
            random_state=random_state,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            output_dir=exp_dir,
        )
        result.exp_type = exp_type
        result.x_label = x_label
        result.dropped_id = dropped_id
        result.dropped_feature = dropped_feature
        save_json(exp_dir / "metrics.json", result.__dict__)
        print(f"  best_test_r2={result.best_test_r2:.6f} at epoch {result.best_epoch}")
        results.append(result)

    result_df = pd.DataFrame([r.__dict__ for r in results])
    result_df.to_csv(output_root / "redundancy_validation_results.csv", index=False, encoding="utf-8-sig")
    _plot_bar_summary(
        results=results,
        selected_entries=selected_entries,
        output_path=output_root / "redundancy_validation_r2.png",
    )

    print(f"Saved validation outputs to {output_root}")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate L1GateDNN selected-feature redundancy with ordinary DNNs.")
    parser.add_argument("run_dir", nargs="?", help="Optional L1GateDNN run directory. Defaults to validation.l1_redundancy.run_dir in the config.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025.yaml"))
    parser.add_argument("--top-n", type=int, help="Use the top N features by absolute gate value instead of selected_features.json.")
    parser.add_argument(
        "--drop-one",
        dest="include_drop_one",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override validation.l1_redundancy.include_drop_one. Use --no-drop-one to skip all drop-one experiments.",
    )
    args = parser.parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()
