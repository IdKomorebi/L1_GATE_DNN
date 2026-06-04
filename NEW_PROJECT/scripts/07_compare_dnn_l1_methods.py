from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, merged_training_params, resolve_project_path
from src.data_utils import ensure_dir, normalize_column_list, read_numeric_csv, safe_name, save_json
from src.trainer import train_center_model


def _load_script6():
    path = PROJECT_ROOT / "scripts" / "06_run_baselines.py"
    spec = importlib.util.spec_from_file_location("run_baselines_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _params_from_run(path: Path) -> Dict[str, Any]:
    cfg = _load_json(path / "config.json")
    params = cfg.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"Invalid params in {path / 'config.json'}")
    return dict(params)


def _parse_int_overrides(values: Sequence[str] | None) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected center=n override, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty center name in override {value!r}")
        overrides[key] = int(raw)
    return overrides


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


def _selected_features_from_l1(run_dir: Path) -> List[str]:
    selected_path = run_dir / "selected_features.json"
    if selected_path.exists():
        payload = _load_json(selected_path)
        rows = payload.get("features", [])
        if rows:
            return [str(row["name"]) for row in rows]

    risk_path = run_dir / "risk_map.csv"
    config_path = run_dir / "config.json"
    if not risk_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Cannot load selected L1 features under {run_dir}")
    cfg = _load_json(config_path)
    threshold = float((cfg.get("params") or {}).get("active_threshold", 0.0))
    df = pd.read_csv(risk_path)
    gate_col = next((col for col in ["final_gate", "gate", "best_epoch_gate", "best_gate"] if col in df.columns), None)
    if gate_col is None or "related" not in df.columns:
        raise ValueError(f"Invalid risk_map.csv: {risk_path}")
    selected = df[df[gate_col].abs() > threshold].copy()
    selected["abs_gate"] = selected[gate_col].abs()
    return [str(v) for v in selected.sort_values("abs_gate", ascending=False)["related"].tolist()]


def _top_gate_features_from_l1(run_dir: Path, top_n: int) -> List[str]:
    risk_path = run_dir / "risk_map.csv"
    if not risk_path.exists():
        raise FileNotFoundError(f"risk_map.csv not found: {risk_path}")
    risk_df = pd.read_csv(risk_path)
    gate_col = next((col for col in ["final_gate", "gate", "best_epoch_gate", "best_gate"] if col in risk_df.columns), None)
    if "related" not in risk_df.columns or gate_col is None:
        raise ValueError(f"risk_map.csv must contain related and a gate column: {risk_path}")
    ranked = risk_df[["related", gate_col]].copy()
    ranked["abs_gate"] = ranked[gate_col].abs()
    return [str(v) for v in ranked.sort_values("abs_gate", ascending=False)["related"].head(int(top_n)).tolist()]


def _feature_universe_from_l1(run_dir: Path) -> List[str]:
    cfg = _load_json(run_dir / "config.json")
    features = cfg.get("features") or []
    if not isinstance(features, list) or not features:
        raise ValueError(f"Cannot load feature universe from {run_dir / 'config.json'}")
    return [str(feature) for feature in features]


def _all_features(data_path: Path, center: str, drop_all_zero_columns: bool, exclude_columns: Sequence[str]) -> List[str]:
    df = read_numeric_csv(data_path, drop_all_zero_columns=drop_all_zero_columns, exclude_columns=exclude_columns)
    return [str(col) for col in df.columns if str(col) != center]


def _plot_summary(summary_df: pd.DataFrame, output_path: Path, zh: bool = False) -> None:
    if zh:
        _configure_chinese_font()

    labels = summary_df["plot_label"].drop_duplicates().tolist()
    methods = ["DNN_AllFeatures", "L1GateDNN_Selected"]
    x = np.arange(len(labels))
    fig_width = max(14.5, len(labels) * 1.02)
    fig, ax = plt.subplots(figsize=(fig_width, 6.6))
    styles = {
        "DNN_AllFeatures": {"color": "#4d4d4d", "linestyle": "--", "marker": "s", "linewidth": 2.8, "label": "DNN_AllFeatures"},
        "L1GateDNN_Selected": {"color": "#1f77b4", "linestyle": "-", "marker": "o", "linewidth": 3.2, "label": "L1GateDNN_Selected"},
    }

    pivot = summary_df.pivot(index="plot_label", columns="method", values="best_test_r2")
    for method in methods:
        values = pivot.reindex(labels)[method].to_numpy(dtype=float)
        ax.plot(x, values, markersize=6.5, **styles[method])

    for xi in x:
        ys = {method: float(pivot.reindex(labels).iloc[int(xi)][method]) for method in methods}
        close = abs(ys["DNN_AllFeatures"] - ys["L1GateDNN_Selected"]) < 0.018
        offsets = {
            "DNN_AllFeatures": (-4, 10 if close else 8),
            "L1GateDNN_Selected": (4, -17 if close else 8),
        }
        vas = {"DNN_AllFeatures": "bottom", "L1GateDNN_Selected": "top" if close else "bottom"}
        for method in methods:
            yi = ys[method]
            if not np.isfinite(yi):
                continue
            ax.annotate(
                f"{yi:.3f}",
                (xi, yi),
                textcoords="offset points",
                xytext=offsets[method],
                ha="center",
                va=vas[method],
                fontsize=8.5,
                color=styles[method]["color"],
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
            )

    ax.set_title(r"DNN and L1GateDNN Method Comparison by Test $R^2$" if not zh else r"DNN 与 L1GateDNN 方法的 Test $R^2$ 对比", fontsize=15)
    ax.set_xlabel("Center Target" if not zh else "中心目标变量", fontsize=13)
    ax.set_ylabel(r"Best Test $R^2$" if not zh else r"最佳 Test $R^2$", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=42, ha="right", fontsize=9.3)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)

    finite = summary_df["best_test_r2"].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        ax.set_ylim(max(0.0, float(finite.min()) - 0.08), min(1.02, float(finite.max()) + 0.04))

    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full DNN against selected-feature L1GateDNN DNN evaluation.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument("--source-baseline", default="outputs/data2025_Processed_V2/BaselineComparison/manual_exclude_n_v2_script6")
    parser.add_argument("--run-name", default="dnn_l1_method_compare_current")
    parser.add_argument(
        "--l1-param-run",
        default=(
            "outputs/data2025_Processed_V2/CenterOn_net_actual_interchange_mw/L1GateDNN/"
            "run_20260603_l1_lr0p00065_thr0p10_combo5_L1GateDNN"
        ),
        help="L1GateDNN run directory whose config.json params should be reused.",
    )
    parser.add_argument(
        "--full-feature-source",
        choices=["all-candidates", "l1-config"],
        default="all-candidates",
        help="Use all non-excluded candidate columns or the feature universe recorded by the L1 source run.",
    )
    parser.add_argument(
        "--selected-top-n-overrides",
        nargs="*",
        default=[],
        help="Optional center=n rules. Example: congestion_price_rt=31 uses top-31 gates instead of threshold selection.",
    )
    parser.add_argument("--quiet-l1", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    script6 = _load_script6()
    cfg = load_config(args.config)
    source_dir = resolve_project_path(cfg, args.source_baseline).expanduser().resolve()
    source_cfg = _load_json(source_dir / "baseline_config.json")
    dataset_cfg = cfg["dataset"]
    data_path = resolve_project_path(cfg, dataset_cfg["processed_csv"])
    output_root = ensure_dir(resolve_project_path(cfg, dataset_cfg["output_root"]) / "BaselineComparison" / safe_name(args.run_name))

    preprocessing = cfg.get("preprocessing", {})
    drop_all_zero_columns = bool(preprocessing.get("drop_all_zero_columns", False))
    global_excludes = normalize_column_list(preprocessing.get("exclude_columns"))
    dnn_params = merged_training_params(cfg, "DNN", (cfg.get("baseline_comparison", {}).get("dnn_training_overrides") or {}))
    l1_param_run = resolve_project_path(cfg, args.l1_param_run).expanduser().resolve() if args.l1_param_run else None
    l1_params = _params_from_run(l1_param_run) if l1_param_run else merged_training_params(cfg, "L1GateDNN", {})
    selected_top_n_overrides = _parse_int_overrides(args.selected_top_n_overrides)
    device = script6._choose_device(str((cfg.get("baseline_comparison", {}) or {}).get("device", "auto")))

    save_json(
        output_root / "method_compare_config.json",
        {
            "config_path": str(Path(cfg["_config_path"]).resolve()),
            "source_baseline": str(source_dir),
            "data_path": str(data_path),
            "dnn_params": dnn_params,
            "l1_params": l1_params,
            "l1_param_run": str(l1_param_run) if l1_param_run else "",
            "full_feature_source": args.full_feature_source,
            "selected_top_n_overrides": selected_top_n_overrides,
            "preprocessing": {"drop_all_zero_columns": drop_all_zero_columns, "exclude_columns": global_excludes},
            "centers": source_cfg["centers"],
        },
    )

    rows: List[Dict[str, Any]] = []
    for idx, target in enumerate(source_cfg["centers"], start=1):
        center = str(target["center"])
        label = str(target.get("label") or center)
        target_excludes = normalize_column_list([*global_excludes, *normalize_column_list(target.get("baseline_exclude_columns"))])
        center_dir = ensure_dir(output_root / f"CenterOn_{safe_name(label)}")
        print(f"[{idx}/{len(source_cfg['centers'])}] center={center}")
        if target_excludes:
            print(f"  exclude: {', '.join(target_excludes)}")

        l1_source_dir = center_dir / "L1GateDNN_source"
        train_kwargs = dict(
            center=center,
            model_name="L1GateDNN",
            overrides=l1_params,
            run_name=f"method_l1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name(label, 80)}",
            force_relations=False,
            exclude_columns=target_excludes,
            combo_name=None,
            output_run_dir=l1_source_dir,
        )
        if args.quiet_l1:
            with contextlib.redirect_stdout(io.StringIO()):
                train_center_model(cfg, **train_kwargs)
        else:
            train_center_model(cfg, **train_kwargs)

        all_candidate_features = _all_features(data_path, center, drop_all_zero_columns, target_excludes)
        if args.full_feature_source == "l1-config":
            available_features = set(all_candidate_features)
            full_features = [feature for feature in _feature_universe_from_l1(l1_source_dir) if feature in available_features]
        else:
            full_features = all_candidate_features
        if not full_features:
            raise ValueError(f"L1GateDNN source feature universe is empty for {center}")
        full_result = script6._train_dnn(
            data_path=data_path,
            center=center,
            features=full_features,
            params=dnn_params,
            output_dir=center_dir / "DNN_AllFeatures",
            device=device,
            drop_all_zero_columns=drop_all_zero_columns,
            exclude_columns=target_excludes,
        )
        print(f"  DNN_AllFeatures: n={full_result.feature_count}, best_test_r2={full_result.best_test_r2:.6f}")

        selected_source = "threshold"
        if center in selected_top_n_overrides:
            selected_source = f"top_gate_{selected_top_n_overrides[center]}"
            selected_features = _top_gate_features_from_l1(l1_source_dir, selected_top_n_overrides[center])
        else:
            selected_features = _selected_features_from_l1(l1_source_dir)
        if not selected_features:
            raise ValueError(f"L1GateDNN selected no features for {center}")
        selected_features = [feature for feature in selected_features if feature in set(full_features)]
        l1_eval_result = script6._train_dnn(
            data_path=data_path,
            center=center,
            features=selected_features,
            params=dnn_params,
            output_dir=center_dir / "L1GateDNN_Selected",
            device=device,
            drop_all_zero_columns=drop_all_zero_columns,
            exclude_columns=target_excludes,
        )
        save_json(
            center_dir / "L1GateDNN_Selected" / "selected_from_l1.json",
            {"features": selected_features, "l1_source_dir": str(l1_source_dir), "selection_source": selected_source},
        )
        plot_label = f"{label}\nn={len(selected_features)}"
        full_row = full_result.__dict__.copy()
        full_row.update(
            {
                "method": "DNN_AllFeatures",
                "target_label": label,
                "plot_label": plot_label,
                "exclude_columns": "|".join(target_excludes),
                "l1_selected_count": len(selected_features),
                "selection_source": selected_source,
            }
        )
        rows.append(full_row)
        l1_row = l1_eval_result.__dict__.copy()
        l1_row.update(
            {
                "method": "L1GateDNN_Selected",
                "target_label": label,
                "plot_label": plot_label,
                "exclude_columns": "|".join(target_excludes),
                "l1_selected_count": len(selected_features),
                "selection_source": selected_source,
            }
        )
        rows.append(l1_row)
        print(f"  L1GateDNN_Selected: n={l1_eval_result.feature_count}, best_test_r2={l1_eval_result.best_test_r2:.6f} ({selected_source})")

        save_json(
            center_dir / "center_config.json",
            {
                "center": center,
                "target_label": label,
                "exclude_columns": target_excludes,
                "full_feature_count": len(full_features),
                "l1_selected_count": len(selected_features),
                "selection_source": selected_source,
                "l1_source_dir": str(l1_source_dir),
            },
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_root / "method_compare_summary.csv", index=False, encoding="utf-8-sig")
    _plot_summary(summary_df, output_root / "method_compare_r2.png", zh=False)
    _plot_summary(summary_df, _zh_path(output_root / "method_compare_r2.png"), zh=True)
    print(f"Saved method comparison outputs to {output_root}")


if __name__ == "__main__":
    main()
