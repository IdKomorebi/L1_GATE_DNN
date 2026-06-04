from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_project_path
from src.data_utils import ensure_dir, normalize_column_list, save_json


def _load_script5():
    path = PROJECT_ROOT / "scripts" / "05_validate_l1_redundancy.py"
    spec = importlib.util.spec_from_file_location("validate_l1_redundancy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ranked_features(run_dir: Path) -> pd.DataFrame:
    risk_path = run_dir / "risk_map.csv"
    if not risk_path.exists():
        raise FileNotFoundError(f"risk_map.csv not found: {risk_path}")
    risk_df = pd.read_csv(risk_path)
    gate_col = next((col for col in ["final_gate", "gate", "best_epoch_gate", "best_gate"] if col in risk_df.columns), None)
    if "related" not in risk_df.columns or gate_col is None:
        raise ValueError(f"risk_map.csv must contain related and a gate column: {risk_path}")
    ranked = risk_df[["related", gate_col]].copy()
    ranked = ranked.rename(columns={gate_col: "gate"})
    ranked["abs_gate"] = ranked["gate"].abs()
    ranked = ranked.sort_values("abs_gate", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _zh_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_zh{path.suffix}")


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


def _plot_series(results: pd.DataFrame, output_path: Path, zh: bool = False) -> None:
    if zh:
        _configure_chinese_font()
    colors = ["#7f8c8d", "#2f80ed", "#2f80ed", "#2f80ed", "#2f80ed", "#2f80ed", "#7b61ff"]
    if len(results) > len(colors):
        colors = colors[:1] + ["#2f80ed"] * (len(results) - 2) + colors[-1:]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    x = np.arange(len(results))
    values = results["best_test_r2"].to_numpy(dtype=float)
    bars = ax.bar(x, values, color=colors[: len(results)], alpha=0.88)

    ax.set_ylabel(r"Best Test $R^2$", fontsize=14)
    ax.set_xlabel("Validation Experiment" if not zh else "验证实验", fontsize=14)
    ax.set_title(r"Top-N Feature Validation by DNN $R^2$" if not zh else r"Top-N 特征组合的 DNN $R^2$ 验证", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(results["x_label"].tolist(), fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", alpha=0.22)

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    pad = max(0.01, (vmax - vmin) * 0.25)
    ax.set_ylim(max(0.0, vmin - pad), min(1.0, vmax + pad))

    for bar, r2 in zip(bars, values):
        ax.annotate(
            f"{r2:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, float(r2)),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full/top-N/unselected feature sets with ordinary DNNs.")
    parser.add_argument("run_dir", help="Source L1GateDNN run directory.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument("--top-ns", nargs="+", type=int, default=[5, 10, 14, 16, 18])
    parser.add_argument("--unselected-n", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    script5 = _load_script5()
    project_cfg = load_config(args.config)
    validation_cfg = project_cfg.get("validation", {}).get("l1_redundancy", {})
    run_dir = resolve_project_path(project_cfg, args.run_dir).expanduser().resolve()
    run_cfg = _load_json(run_dir / "config.json")
    source_preprocessing = run_cfg.get("preprocessing") or project_cfg.get("preprocessing", {})
    drop_all_zero_columns = bool(source_preprocessing.get("drop_all_zero_columns", False))
    exclude_columns = normalize_column_list(source_preprocessing.get("exclude_columns"))

    center = str(run_cfg["center"])
    data_path = Path(run_cfg["data_path"]).expanduser().resolve()
    params = run_cfg.get("params", {})
    hidden_dims = [int(v) for v in validation_cfg.get("hidden_dims", params.get("hidden_dims", [64, 32, 16]))]
    epochs = int(args.epochs or validation_cfg.get("epochs") or 100)
    train_ratio = float(validation_cfg.get("train_ratio", params.get("train_ratio", 0.8)))
    random_state = int(validation_cfg.get("random_state", params.get("random_state", 42)))
    batch_size = int(validation_cfg.get("batch_size", params.get("batch_size", 50)))
    lr = float(validation_cfg.get("lr", params.get("lr", 1e-3)))
    device = script5._choose_device(str(validation_cfg.get("device", "cpu")))

    full_features = [str(f) for f in run_cfg.get("features", [])]
    if not full_features:
        full_features = script5._all_run_features(run_cfg, data_path, center, drop_all_zero_columns, exclude_columns)

    ranked = _ranked_features(run_dir)
    ranked_path_rows = ranked[["rank", "related", "gate"]].copy()

    output_name = args.output_name or f"top_series_e{epochs}"
    output_root = ensure_dir(run_dir / "redundancy_validation" / output_name)
    ranked_path_rows.to_csv(output_root / "ranked_features_by_gate.csv", index=False, encoding="utf-8-sig")

    experiments: List[tuple[str, str, str, List[str]]] = [
        ("DNN_full_all", "all", f"Full\nn={len(full_features)}", full_features),
    ]
    for top_n in args.top_ns:
        features = ranked["related"].head(top_n).astype(str).tolist()
        experiments.append((f"DNN_top{top_n:02d}", "selected", f"Top{top_n}\nn={top_n}", features))

    top_unselected = set(ranked["related"].head(args.unselected_n).astype(str).tolist())
    unselected = [feature for feature in full_features if feature not in top_unselected]
    experiments.append(
        (
            f"DNN_without_top{args.unselected_n:02d}",
            "unselected",
            f"Without Top{args.unselected_n}\nn={len(unselected)}",
            unselected,
        )
    )

    save_json(
        output_root / "validation_config.json",
        {
            "source_run_dir": str(run_dir),
            "config_path": str(Path(project_cfg["_config_path"]).expanduser().resolve()),
            "center": center,
            "combo_name": run_cfg.get("combo_name"),
            "data_path": str(data_path),
            "top_ns": args.top_ns,
            "unselected_n": args.unselected_n,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "train_ratio": train_ratio,
            "random_state": random_state,
            "device": str(device),
            "hidden_dims": hidden_dims,
            "drop_all_zero_columns": drop_all_zero_columns,
            "exclude_columns": exclude_columns,
        },
    )

    results = []
    for idx, (exp_name, exp_type, x_label, features) in enumerate(experiments, start=1):
        exp_dir = output_root / exp_name
        print(f"[{idx}/{len(experiments)}] {exp_name}: training DNN with {len(features)} features")
        result = script5._train_dnn(
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
            drop_all_zero_columns=drop_all_zero_columns,
            exclude_columns=exclude_columns,
        )
        result.exp_type = exp_type
        result.x_label = x_label
        save_json(exp_dir / "metrics.json", result.__dict__)
        print(f"  best_test_r2={result.best_test_r2:.6f} at epoch {result.best_epoch}")
        results.append(result.__dict__)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_root / "top_series_results.csv", index=False, encoding="utf-8-sig")
    plot_path = output_root / "top_series_r2.png"
    _plot_series(result_df, plot_path)
    _plot_series(result_df, _zh_path(plot_path), zh=True)
    print(f"Saved validation outputs to {output_root}")


if __name__ == "__main__":
    main()
