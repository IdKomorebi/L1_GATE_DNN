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


DEFAULT_CENTERS = ["congestion_price_da", "da_as_total_mw_primary_reserve"]


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


def _ranked_gate_features(run_dir: Path) -> pd.DataFrame:
    risk_path = run_dir / "risk_map.csv"
    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        gate_col = next((col for col in ["final_gate", "gate", "best_epoch_gate", "best_gate"] if col in risk_df.columns), None)
        if "related" in risk_df.columns and gate_col is not None:
            ranked = risk_df[["related", gate_col]].copy()
            ranked = ranked.rename(columns={gate_col: "gate"})
            ranked["abs_gate"] = ranked["gate"].abs()
            ranked = ranked.sort_values("abs_gate", ascending=False).reset_index(drop=True)
            ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
            return ranked

    gate_path = run_dir / "gate_params.csv"
    if not gate_path.exists():
        raise FileNotFoundError(f"Neither risk_map.csv nor gate_params.csv exists under {run_dir}")
    gate_df = pd.read_csv(gate_path)
    last_epoch = int(gate_df["epoch"].max())
    ranked = gate_df[gate_df["epoch"] == last_epoch][["feature", "gate"]].copy()
    ranked = ranked.rename(columns={"feature": "related"})
    ranked["abs_gate"] = ranked["gate"].abs()
    ranked = ranked.sort_values("abs_gate", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def _selected_features_from_l1(run_dir: Path) -> List[str]:
    selected_path = run_dir / "selected_features.json"
    if selected_path.exists():
        payload = _load_json(selected_path)
        rows = payload.get("features", [])
        if rows:
            return [str(row["name"]) for row in rows]
    run_cfg = _load_json(run_dir / "config.json")
    threshold = float((run_cfg.get("params") or {}).get("active_threshold", 0.10))
    ranked = _ranked_gate_features(run_dir)
    selected = ranked[ranked["gate"].abs() > threshold]
    return [str(v) for v in selected["related"].tolist()]


def _all_features(data_path: Path, center: str, drop_all_zero_columns: bool, exclude_columns: Sequence[str]) -> List[str]:
    df = read_numeric_csv(data_path, drop_all_zero_columns=drop_all_zero_columns, exclude_columns=exclude_columns)
    return [str(col) for col in df.columns if str(col) != center]


def _target_map(script6: Any, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    baseline_cfg = cfg.get("baseline_comparison", {})
    targets = script6._resolve_targets(cfg, baseline_cfg, baseline_cfg.get("centers", []))
    return {str(target["center"]): target for target in targets}


def _plot_training_curves(
    log_specs: List[Dict[str, Any]],
    output_path: Path,
    zh: bool = False,
) -> None:
    if zh:
        _configure_chinese_font()

    fig, axes = plt.subplots(1, len(log_specs), figsize=(8.6 * len(log_specs), 6.2), sharey=False)
    if len(log_specs) == 1:
        axes = [axes]

    styles = {
        "DNN_AllFeatures_train": {"color": "#4d4d4d", "linestyle": "--", "linewidth": 2.2, "label": "DNN_AllFeatures Train"},
        "DNN_AllFeatures_test": {"color": "#111111", "linestyle": "-", "linewidth": 2.4, "label": "DNN_AllFeatures Test"},
        "L1GateDNN_train": {"color": "#6baed6", "linestyle": "--", "linewidth": 2.4, "label": "L1GateDNN Train"},
        "L1GateDNN_test": {"color": "#1f77b4", "linestyle": "-", "linewidth": 3.0, "label": "L1GateDNN Test"},
        "DNN_Selected_train": {"color": "#fdae6b", "linestyle": "--", "linewidth": 2.0, "label": "DNN_Selected Train"},
        "DNN_Selected_test": {"color": "#e6550d", "linestyle": "-", "linewidth": 2.4, "label": "DNN_Selected Test"},
    }

    for ax, spec in zip(axes, log_specs):
        center = spec["center"]
        for method in ["DNN_AllFeatures", "L1GateDNN", "DNN_Selected"]:
            log_df = pd.read_csv(spec[f"{method}_log"])
            for split in ["train", "test"]:
                key = f"{method}_{split}"
                ax.plot(log_df["epoch"], log_df[f"{split}_r2"], **styles[key])

        ax.set_title(center, fontsize=15)
        ax.set_xlabel("Epoch", fontsize=13)
        ax.set_ylabel(r"$R^2$" if not zh else r"$R^2$", fontsize=13)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=11)

        y_values = []
        for method in ["DNN_AllFeatures", "L1GateDNN", "DNN_Selected"]:
            log_df = pd.read_csv(spec[f"{method}_log"])
            y_values.extend(log_df["train_r2"].to_numpy(dtype=float))
            y_values.extend(log_df["test_r2"].to_numpy(dtype=float))
        finite = np.asarray(y_values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            ax.set_ylim(max(-0.1, float(finite.min()) - 0.06), min(1.03, float(finite.max()) + 0.04))

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=True, fontsize=11)
    fig.suptitle(
        r"Training and Test $R^2$ Curves for Redundancy Explanation"
        if not zh
        else r"冗余噪声解释实验：训练集与测试集 $R^2$ 曲线",
        fontsize=17,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 0.91, 0.95))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_topn_curve(results: pd.DataFrame, output_path: Path, zh: bool = False) -> None:
    if zh:
        _configure_chinese_font()

    centers = results["center"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(centers), figsize=(9.6 * len(centers), 6.4), sharey=False)
    if len(centers) == 1:
        axes = [axes]

    for ax, center in zip(axes, centers):
        df = results[results["center"] == center].copy()
        df = df.sort_values(["feature_count", "series_order"])
        x = np.arange(len(df))
        values = df["best_test_r2"].to_numpy(dtype=float)
        colors = ["#1f77b4" if row["exp_type"] == "topn" else "#4d4d4d" for _, row in df.iterrows()]
        ax.plot(x, values, color="#1f77b4", linewidth=2.7, marker="o", markersize=6.2)
        ax.scatter(x, values, c=colors, s=46, zorder=3)

        for xi, yi in zip(x, values):
            if float(yi) >= 0.96:
                offset_y = -13 - 11 * (int(xi) % 2)
                va = "top"
            else:
                offset_y = 7 + 9 * (int(xi) % 2)
                va = "bottom"
            ax.annotate(
                f"{yi:.4f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, offset_y),
                ha="center",
                va=va,
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
            )

        ax.set_title(center, fontsize=15)
        ax.set_xlabel("Feature Count n" if not zh else "特征数量 n", fontsize=13)
        ax.set_ylabel(r"Best Test $R^2$" if not zh else r"最佳 Test $R^2$", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(df["x_label"].tolist(), rotation=35, ha="right", fontsize=10)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        finite = values[np.isfinite(values)]
        if finite.size:
            pad = max(0.02, (float(finite.max()) - float(finite.min())) * 0.22)
            ax.set_ylim(max(0.0, float(finite.min()) - pad), min(1.05, float(finite.max()) + pad + 0.02))

    fig.suptitle(
        r"Top-N Feature Incremental Validation"
        if not zh
        else "Top-N 特征增量验证",
        fontsize=17,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_topn_table(results: pd.DataFrame, output_path: Path, zh: bool = False) -> None:
    if zh:
        _configure_chinese_font()

    table_df = results[["center", "x_label", "feature_count", "best_test_r2", "best_epoch", "final_test_r2"]].copy()
    table_df = table_df.rename(
        columns={
            "center": "Center" if not zh else "中心变量",
            "x_label": "Experiment" if not zh else "实验",
            "feature_count": "n",
            "best_test_r2": "Best Test R2",
            "best_epoch": "Best Epoch" if not zh else "最佳Epoch",
            "final_test_r2": "Final Test R2",
        }
    )
    for col in ["Best Test R2", "Final Test R2"]:
        table_df[col] = table_df[col].map(lambda v: f"{float(v):.4f}")

    row_count = len(table_df)
    fig_height = max(5.8, 1.0 + 0.33 * row_count)
    fig_width = 14.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns.tolist(),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.35)
    try:
        table.auto_set_column_width(col=list(range(len(table_df.columns))))
    except Exception:
        pass

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2f4057")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f6fa")
        else:
            cell.set_facecolor("white")
        cell.set_edgecolor("#d0d6de")

    ax.set_title(
        r"Top-N Feature Incremental Validation Results"
        if not zh
        else "Top-N 特征增量验证结果表",
        fontsize=16,
        pad=18,
    )
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _dedupe_top_ns(values: Sequence[int], full_n: int) -> List[int]:
    seen = set()
    top_ns = []
    for value in values:
        n = int(value)
        if 0 < n < full_n and n not in seen:
            seen.add(n)
            top_ns.append(n)
    return top_ns


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain why selected features can outperform full DNN under redundant noise.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument("--centers", nargs="+", default=DEFAULT_CENTERS)
    parser.add_argument("--run-name", default=f"RedundantNoiseExplanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument(
        "--l1-param-run",
        default=(
            "outputs/data2025_Processed_V2/CenterOn_net_actual_interchange_mw/L1GateDNN/"
            "run_20260603_l1_lr0p00065_thr0p10_combo5_L1GateDNN"
        ),
    )
    parser.add_argument("--top-ns", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 24, 32])
    parser.add_argument("--quiet-l1", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    script6 = _load_script6()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    data_path = resolve_project_path(cfg, dataset_cfg["processed_csv"])
    output_root = ensure_dir(resolve_project_path(cfg, dataset_cfg["output_root"]) / "RedundantNoiseExplanation" / safe_name(args.run_name))

    preprocessing = cfg.get("preprocessing", {})
    drop_all_zero_columns = bool(preprocessing.get("drop_all_zero_columns", False))
    global_excludes = normalize_column_list(preprocessing.get("exclude_columns"))
    targets_by_center = _target_map(script6, cfg)
    dnn_params = merged_training_params(cfg, "DNN", (cfg.get("baseline_comparison", {}).get("dnn_training_overrides") or {}))
    l1_param_run = resolve_project_path(cfg, args.l1_param_run).expanduser().resolve()
    l1_params = _params_from_run(l1_param_run)
    device = script6._choose_device(str((cfg.get("baseline_comparison", {}) or {}).get("device", "auto")))

    save_json(
        output_root / "explanation_config.json",
        {
            "config_path": str(Path(cfg["_config_path"]).resolve()),
            "data_path": str(data_path),
            "centers": args.centers,
            "top_ns": args.top_ns,
            "dnn_params": dnn_params,
            "l1_params": l1_params,
            "l1_param_run": str(l1_param_run),
            "device": str(device),
        },
    )

    training_specs: List[Dict[str, Any]] = []
    topn_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for idx, center in enumerate(args.centers, start=1):
        if center not in targets_by_center:
            raise ValueError(f"Center {center!r} is not present in baseline_comparison.centers.")
        target = targets_by_center[center]
        label = str(target.get("label") or center)
        target_excludes = normalize_column_list([*global_excludes, *normalize_column_list(target.get("baseline_exclude_columns"))])
        center_dir = ensure_dir(output_root / f"CenterOn_{safe_name(label)}")
        print(f"[{idx}/{len(args.centers)}] center={center}")
        if target_excludes:
            print(f"  exclude: {', '.join(target_excludes)}")

        l1_dir = center_dir / "L1GateDNN"
        train_kwargs = dict(
            center=center,
            model_name="L1GateDNN",
            overrides=l1_params,
            run_name=f"explain_l1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name(label, 80)}",
            force_relations=False,
            exclude_columns=target_excludes,
            combo_name=None,
            output_run_dir=l1_dir,
        )
        if args.quiet_l1:
            with contextlib.redirect_stdout(io.StringIO()):
                train_center_model(cfg, **train_kwargs)
        else:
            train_center_model(cfg, **train_kwargs)

        full_features = _all_features(data_path, center, drop_all_zero_columns, target_excludes)
        ranked = _ranked_gate_features(l1_dir)
        ranked.to_csv(center_dir / "ranked_features_by_gate.csv", index=False, encoding="utf-8-sig")
        selected_features = [feature for feature in _selected_features_from_l1(l1_dir) if feature in set(full_features)]
        if not selected_features:
            selected_features = ranked["related"].head(max(1, int(target.get("n_features_override") or 1))).astype(str).tolist()

        dnn_full = script6._train_dnn(
            data_path=data_path,
            center=center,
            features=full_features,
            params=dnn_params,
            output_dir=center_dir / "DNN_AllFeatures",
            device=device,
            drop_all_zero_columns=drop_all_zero_columns,
            exclude_columns=target_excludes,
        )
        dnn_selected = script6._train_dnn(
            data_path=data_path,
            center=center,
            features=selected_features,
            params=dnn_params,
            output_dir=center_dir / "DNN_SelectedByL1",
            device=device,
            drop_all_zero_columns=drop_all_zero_columns,
            exclude_columns=target_excludes,
        )

        l1_log = pd.read_csv(l1_dir / "log.csv")
        l1_best_idx = int(l1_log["test_r2"].idxmax())
        l1_best_row = l1_log.loc[l1_best_idx]
        save_json(
            center_dir / "center_config.json",
            {
                "center": center,
                "target_label": label,
                "exclude_columns": target_excludes,
                "full_feature_count": len(full_features),
                "l1_selected_count": len(selected_features),
                "l1_source_dir": str(l1_dir),
            },
        )

        training_specs.append(
            {
                "center": center,
                "DNN_AllFeatures_log": str(center_dir / "DNN_AllFeatures" / "log.csv"),
                "L1GateDNN_log": str(l1_dir / "log.csv"),
                "DNN_Selected_log": str(center_dir / "DNN_SelectedByL1" / "log.csv"),
            }
        )

        summary_rows.extend(
            [
                {
                    "center": center,
                    "method": "DNN_AllFeatures",
                    "feature_count": dnn_full.feature_count,
                    "best_test_r2": dnn_full.best_test_r2,
                    "best_train_r2": dnn_full.best_train_r2,
                    "best_epoch": dnn_full.best_epoch,
                    "final_test_r2": dnn_full.final_test_r2,
                    "final_train_r2": dnn_full.final_train_r2,
                    "generalization_gap_final": dnn_full.final_train_r2 - dnn_full.final_test_r2,
                },
                {
                    "center": center,
                    "method": "L1GateDNN",
                    "feature_count": len(full_features),
                    "best_test_r2": float(l1_log["test_r2"].max()),
                    "best_train_r2": float(l1_log["train_r2"].max()),
                    "best_epoch": int(l1_best_row["epoch"]),
                    "final_test_r2": float(l1_log["test_r2"].iloc[-1]),
                    "final_train_r2": float(l1_log["train_r2"].iloc[-1]),
                    "generalization_gap_final": float(l1_log["train_r2"].iloc[-1] - l1_log["test_r2"].iloc[-1]),
                },
                {
                    "center": center,
                    "method": "DNN_SelectedByL1",
                    "feature_count": dnn_selected.feature_count,
                    "best_test_r2": dnn_selected.best_test_r2,
                    "best_train_r2": dnn_selected.best_train_r2,
                    "best_epoch": dnn_selected.best_epoch,
                    "final_test_r2": dnn_selected.final_test_r2,
                    "final_train_r2": dnn_selected.final_train_r2,
                    "generalization_gap_final": dnn_selected.final_train_r2 - dnn_selected.final_test_r2,
                },
            ]
        )

        full_n = len(full_features)
        top_ns = _dedupe_top_ns([*args.top_ns, len(selected_features), int(target.get("n_features_override") or 0)], full_n)
        available = set(full_features)
        for series_order, top_n in enumerate(top_ns, start=1):
            features = [feature for feature in ranked["related"].head(top_n).astype(str).tolist() if feature in available]
            if len(features) != top_n:
                features = [*features, *[f for f in full_features if f not in set(features)][: top_n - len(features)]]
            exp_name = f"DNN_top{top_n:02d}"
            print(f"  top-n {top_n}: training DNN")
            result = script6._train_dnn(
                data_path=data_path,
                center=center,
                features=features,
                params=dnn_params,
                output_dir=center_dir / "TopNIncremental" / exp_name,
                device=device,
                drop_all_zero_columns=drop_all_zero_columns,
                exclude_columns=target_excludes,
            )
            topn_rows.append(
                {
                    "center": center,
                    "exp_type": "topn",
                    "series_order": series_order,
                    "x_label": f"Top{top_n}",
                    "feature_count": top_n,
                    "best_test_r2": result.best_test_r2,
                    "best_epoch": result.best_epoch,
                    "final_test_r2": result.final_test_r2,
                    "output_dir": result.output_dir,
                }
            )

        topn_rows.append(
            {
                "center": center,
                "exp_type": "full",
                "series_order": 9999,
                "x_label": "Full",
                "feature_count": full_n,
                "best_test_r2": dnn_full.best_test_r2,
                "best_epoch": dnn_full.best_epoch,
                "final_test_r2": dnn_full.final_test_r2,
                "output_dir": dnn_full.output_dir,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "training_curve_summary.csv", index=False, encoding="utf-8-sig")
    topn_df = pd.DataFrame(topn_rows)
    topn_df.to_csv(output_root / "topn_incremental_results.csv", index=False, encoding="utf-8-sig")

    _plot_training_curves(training_specs, output_root / "training_r2_curves.png", zh=False)
    _plot_training_curves(training_specs, _zh_path(output_root / "training_r2_curves.png"), zh=True)
    _plot_topn_curve(topn_df, output_root / "topn_incremental_r2.png", zh=False)
    _plot_topn_curve(topn_df, _zh_path(output_root / "topn_incremental_r2.png"), zh=True)
    _plot_topn_table(topn_df, output_root / "topn_incremental_table.png", zh=False)
    _plot_topn_table(topn_df, _zh_path(output_root / "topn_incremental_table.png"), zh=True)

    print(f"Saved redundant-noise explanation outputs to {output_root}")


if __name__ == "__main__":
    main()
