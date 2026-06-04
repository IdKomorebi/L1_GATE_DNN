from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
from src.data_utils import ensure_dir, normalize_column_list, safe_name, save_json


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


def _zh_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_zh{path.suffix}")


def _metric_from_dir(path: Path) -> Dict[str, Any] | None:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return None
    return _load_json(metrics_path)


def _result_row_from_metrics(
    metrics: Dict[str, Any],
    *,
    center: str,
    target_label: str,
    plot_label: str,
    method: str,
    n_features: int,
    output_dir: Path,
    full_r2: float,
    budget_kind: str | None = None,
) -> Dict[str, Any]:
    return {
        "center": center,
        "target_label": target_label,
        "plot_label": plot_label,
        "method": method,
        "n_features": int(n_features),
        "budget_kind": budget_kind or "",
        "feature_count": int(metrics.get("feature_count", n_features)),
        "best_test_r2": float(metrics.get("best_test_r2", np.nan)),
        "best_train_r2": float(metrics.get("best_train_r2", np.nan)),
        "min_test_loss": float(metrics.get("min_test_loss", np.nan)),
        "min_train_loss": float(metrics.get("min_train_loss", np.nan)),
        "final_test_r2": float(metrics.get("final_test_r2", np.nan)),
        "final_train_r2": float(metrics.get("final_train_r2", np.nan)),
        "final_test_loss": float(metrics.get("final_test_loss", np.nan)),
        "final_train_loss": float(metrics.get("final_train_loss", np.nan)),
        "best_epoch": int(metrics.get("best_epoch", 0)),
        "full_dnn_r2": float(full_r2),
        "ratio_to_full": float(metrics.get("best_test_r2", np.nan)) / float(full_r2) if full_r2 else np.nan,
        "output_dir": str(output_dir),
    }


def _choose_budget(
    rows: pd.DataFrame,
    n_values: Sequence[int],
    target_ratio: float,
    fallback_ratio: float,
    plateau_steps: int,
    plateau_epsilon: float,
) -> Dict[str, Any]:
    by_n = rows.groupby("n_features")["best_test_r2"].max().reindex(list(n_values))
    full_r2 = float(rows["full_dnn_r2"].dropna().iloc[0])
    target_value = target_ratio * full_r2
    fallback_value = fallback_ratio * full_r2

    for n, value in by_n.items():
        if np.isfinite(value) and float(value) >= target_value:
            return {
                "selected_n": int(n),
                "selection_rule": f"ratio_{target_ratio:.2f}",
                "target_ratio": target_ratio,
                "target_value": target_value,
                "best_r2_at_n": float(value),
                "ratio_at_n": float(value) / full_r2 if full_r2 else np.nan,
            }

    finite = by_n.dropna()
    if not finite.empty:
        for idx in range(0, len(finite) - plateau_steps):
            n = int(finite.index[idx])
            value = float(finite.iloc[idx])
            future = finite.iloc[idx + 1 : idx + 1 + plateau_steps].to_numpy(dtype=float)
            if value >= fallback_value and future.size >= plateau_steps and float(np.nanmax(future) - value) < plateau_epsilon:
                return {
                    "selected_n": n,
                    "selection_rule": f"plateau_{fallback_ratio:.2f}",
                    "target_ratio": target_ratio,
                    "target_value": target_value,
                    "best_r2_at_n": value,
                    "ratio_at_n": value / full_r2 if full_r2 else np.nan,
                }

    if finite.empty:
        raise ValueError("Cannot choose budget because no finite sweep results exist.")
    best_idx = int(finite.idxmax())
    best_value = float(finite.loc[best_idx])
    return {
        "selected_n": best_idx,
        "selection_rule": "best_available",
        "target_ratio": target_ratio,
        "target_value": target_value,
        "best_r2_at_n": best_value,
        "ratio_at_n": best_value / full_r2 if full_r2 else np.nan,
    }


def _local_n_values(anchor: int | None, candidate_count: int, fallback_values: Sequence[int], offsets: Sequence[int]) -> List[int]:
    if anchor is None:
        values = [int(v) for v in fallback_values]
    else:
        values = [int(anchor) + int(offset) for offset in offsets]
    values = [value for value in values if 1 <= value <= candidate_count]
    return sorted(set(values))


def _plot_summary(summary_df: pd.DataFrame, methods: Sequence[str], output_path: Path, zh: bool = False) -> None:
    if zh:
        _configure_chinese_font()
    script6 = _load_script6()
    full_name = script6.FULL_DNN_NAME
    main_method = "L1GateDNN"
    x_col = "plot_label"
    labels = summary_df[x_col].drop_duplicates().tolist()
    pivot = summary_df.pivot(index=x_col, columns="method", values="best_test_r2")
    x = np.arange(len(labels))

    fig_width = max(14.5, len(labels) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))
    for method in methods:
        if method not in pivot.columns:
            continue
        values = pivot.reindex(labels)[method].to_numpy(dtype=float)
        is_main = method == main_method
        is_full = method == full_name
        ax.plot(
            x,
            values,
            marker="o",
            linestyle="--" if is_full else "-",
            color="black" if is_full else ("#1f77b4" if is_main else None),
            linewidth=3.2 if is_main else (2.4 if is_full else 1.6),
            markersize=6 if is_main else (5 if is_full else 3.5),
            alpha=1.0 if is_main else (0.88 if is_full else 0.58),
            zorder=5 if is_main else (4 if is_full else 2),
            label=method,
        )

    title = "Baseline Comparison under Unified Feature Budget"
    xlabel = "Center Target"
    ylabel = r"Best Test $R^2$"
    if zh:
        title = "统一特征预算下的Baseline方法对比"
        xlabel = "中心目标变量"
        ylabel = r"最佳 Test $R^2$"
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline comparisons with automatically selected unified feature budgets.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument("--source-baseline", default="outputs/data2025_Processed_V2/BaselineComparison/manual_exclude_n_v2_script6")
    parser.add_argument("--run-name", default="budget_baseline_sweep")
    parser.add_argument("--n-values", nargs="*", type=int, default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30])
    parser.add_argument(
        "--local-window-from-source",
        action="store_true",
        help="Use each source baseline n_features_override as the anchor and sweep nearby n values.",
    )
    parser.add_argument("--window-offsets", nargs="*", type=int, default=[-4, -2, 0, 2, 4])
    parser.add_argument("--ratios", nargs="*", type=float, default=[0.95, 0.97])
    parser.add_argument("--fallback-ratio", type=float, default=0.94)
    parser.add_argument("--plateau-steps", type=int, default=3)
    parser.add_argument("--plateau-epsilon", type=float, default=0.005)
    parser.add_argument("--baselines", nargs="*", default=None)
    parser.add_argument("--selector-baselines", nargs="*", default=["L1GateDNN"])
    args = parser.parse_args()

    script6 = _load_script6()
    cfg = load_config(args.config)
    source_dir = resolve_project_path(cfg, args.source_baseline).expanduser().resolve()
    source_cfg = _load_json(source_dir / "baseline_config.json")
    dataset_cfg = cfg["dataset"]
    data_path = resolve_project_path(cfg, dataset_cfg["processed_csv"])
    output_root = ensure_dir(resolve_project_path(cfg, dataset_cfg["output_root"]) / "BaselineComparison" / safe_name(args.run_name))

    baseline_cfg = cfg.get("baseline_comparison", {})
    methods = script6._resolve_methods(args.baselines or baseline_cfg.get("baselines", "all"))
    selector_methods = script6._resolve_methods(args.selector_baselines)
    display_methods = [script6.FULL_DNN_NAME, *methods]
    dnn_params = merged_training_params(cfg, "DNN", baseline_cfg.get("dnn_training_overrides") or {})
    random_state = int(dnn_params.get("random_state", baseline_cfg.get("random_state", 42)))
    device = script6._choose_device(str(baseline_cfg.get("device", "auto")))
    preprocessing = cfg.get("preprocessing", {})
    drop_all_zero_columns = bool(preprocessing.get("drop_all_zero_columns", False))
    global_excludes = normalize_column_list(preprocessing.get("exclude_columns"))
    skip_existing = True

    save_json(
        output_root / "budget_sweep_config.json",
        {
            "config_path": str(Path(cfg["_config_path"]).resolve()),
            "source_baseline": str(source_dir),
            "n_values": args.n_values,
            "local_window_from_source": args.local_window_from_source,
            "window_offsets": args.window_offsets,
            "ratios": args.ratios,
            "fallback_ratio": args.fallback_ratio,
            "plateau_steps": args.plateau_steps,
            "plateau_epsilon": args.plateau_epsilon,
            "methods": methods,
            "selector_methods": selector_methods,
            "dnn_params": dnn_params,
        },
    )

    sweep_rows: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    final_rows_by_ratio: Dict[float, List[Dict[str, Any]]] = {ratio: [] for ratio in args.ratios}

    for center_idx, target in enumerate(source_cfg["centers"], start=1):
        center = str(target["center"])
        target_label = str(target.get("label") or center)
        target_excludes = normalize_column_list([*global_excludes, *normalize_column_list(target.get("baseline_exclude_columns"))])
        center_dir = ensure_dir(output_root / f"CenterOn_{safe_name(target_label)}")
        print(f"[{center_idx}/{len(source_cfg['centers'])}] center={center}")
        if target_excludes:
            print(f"  exclude: {', '.join(target_excludes)}")

        l1_run_dir = Path(target.get("l1_run_dir") or source_dir / f"CenterOn_{safe_name(target_label)}" / "L1GateDNN_source")
        if not l1_run_dir.exists():
            l1_run_dir = script6._ensure_l1_run(
                cfg,
                center,
                None,
                resolve_project_path(cfg, dataset_cfg["output_root"]),
                data_path,
                drop_all_zero_columns,
                target_excludes,
                baseline_cfg,
                center_dir / "L1GateDNN_source",
            )
        features, X_train, y_train = script6._ranking_frame(
            data_path,
            center,
            drop_all_zero_columns,
            target_excludes,
            train_ratio=float(dnn_params.get("train_ratio", 0.8)),
            random_state=random_state,
        )
        full_dir = ensure_dir(center_dir / script6.FULL_DNN_NAME)
        full_context = script6._dnn_run_context(
            data_path=data_path,
            center=center,
            method=script6.FULL_DNN_NAME,
            features=features,
            params=dnn_params,
            drop_all_zero_columns=drop_all_zero_columns,
            exclude_columns=target_excludes,
        )
        if skip_existing and script6._can_reuse_dnn(full_dir, full_context):
            full_metrics = _load_json(full_dir / "metrics.json")
        else:
            full_result = script6._train_dnn(
                data_path=data_path,
                center=center,
                features=features,
                params=dnn_params,
                output_dir=full_dir,
                device=device,
                drop_all_zero_columns=drop_all_zero_columns,
                exclude_columns=target_excludes,
            )
            save_json(full_dir / script6.RUN_CONTEXT_FILE, full_context)
            full_metrics = full_result.__dict__
        full_r2 = float(full_metrics["best_test_r2"])
        center_n_values = _local_n_values(
            int(target["n_features_override"]) if args.local_window_from_source and target.get("n_features_override") else None,
            len(features),
            args.n_values,
            args.window_offsets,
        )
        full_row_base = _result_row_from_metrics(
            full_metrics,
            center=center,
            target_label=target_label,
            plot_label=target_label,
            method=script6.FULL_DNN_NAME,
            n_features=len(features),
            output_dir=full_dir,
            full_r2=full_r2,
        )
        print(f"  Full DNN: all={len(features)}, best_test_r2={full_r2:.6f}; n_values={center_n_values}")

        for n in center_n_values:
            if n > len(features):
                continue
            print(f"  n={n}")
            for method in selector_methods:
                method_dir = ensure_dir(center_dir / "Sweep" / f"n{n:02d}" / safe_name(method))
                selected, scores = script6._baseline_features(method, n, l1_run_dir, features, X_train, y_train, random_state, baseline_cfg)
                if len(selected) < n:
                    remaining = [feature for feature in features if feature not in selected]
                    selected = [*selected, *remaining[: n - len(selected)]]
                context = script6._dnn_run_context(
                    data_path=data_path,
                    center=center,
                    method=method,
                    features=selected,
                    params=dnn_params,
                    drop_all_zero_columns=drop_all_zero_columns,
                    exclude_columns=target_excludes,
                    extra={"n_features": n, "l1_run_dir": str(l1_run_dir)},
                )
                metrics = _metric_from_dir(method_dir) if skip_existing and script6._can_reuse_dnn(method_dir, context) else None
                if metrics is None:
                    score_rows = [
                        {"feature": feature, "score": float(scores.get(feature, np.nan)), "selected": feature in selected}
                        for feature in features
                    ]
                    pd.DataFrame(score_rows).sort_values(["selected", "score"], ascending=[False, False]).to_csv(
                        method_dir / "feature_scores.csv", index=False, encoding="utf-8-sig"
                    )
                    result = script6._train_dnn(
                        data_path=data_path,
                        center=center,
                        features=selected,
                        params=dnn_params,
                        output_dir=method_dir,
                        device=device,
                        drop_all_zero_columns=drop_all_zero_columns,
                        exclude_columns=target_excludes,
                    )
                    save_json(method_dir / script6.RUN_CONTEXT_FILE, context)
                    metrics = result.__dict__
                row = _result_row_from_metrics(
                    metrics,
                    center=center,
                    target_label=target_label,
                    plot_label=f"{target_label}\nn={n}",
                    method=method,
                    n_features=n,
                    output_dir=method_dir,
                    full_r2=full_r2,
                )
                sweep_rows.append(row)
                print(f"    {method}: {row['best_test_r2']:.6f}")
                pd.DataFrame(sweep_rows).to_csv(output_root / "budget_sweep_long.csv", index=False, encoding="utf-8-sig")

        center_sweep = pd.DataFrame([row for row in sweep_rows if row["center"] == center])
        for ratio in args.ratios:
            budget = _choose_budget(
                center_sweep,
                center_n_values,
                ratio,
                args.fallback_ratio,
                args.plateau_steps,
                args.plateau_epsilon,
            )
            budget_row = {
                "center": center,
                "target_label": target_label,
                "budget_ratio": ratio,
                "full_feature_count": len(features),
                "full_dnn_r2": full_r2,
                **budget,
            }
            budget_rows.append(budget_row)
            n_selected = int(budget["selected_n"])
            plot_label = f"{target_label}\nn={n_selected}"
            full_for_plot = dict(full_row_base)
            full_for_plot.update({"plot_label": plot_label, "n_features": n_selected, "budget_kind": f"{ratio:.2f}"})
            final_rows_by_ratio[ratio].append(full_for_plot)
            selected_rows = center_sweep[center_sweep["n_features"] == n_selected].copy()
            selected_by_method = {str(row["method"]): row for row in selected_rows.to_dict(orient="records")}
            for method in methods:
                if method in selected_by_method:
                    row = dict(selected_by_method[method])
                    row["plot_label"] = plot_label
                    row["budget_kind"] = f"{ratio:.2f}"
                    final_rows_by_ratio[ratio].append(row)
                    continue

                method_dir = ensure_dir(center_dir / f"Budget_{int(round(ratio * 100))}" / safe_name(method))
                selected, scores = script6._baseline_features(method, n_selected, l1_run_dir, features, X_train, y_train, random_state, baseline_cfg)
                if len(selected) < n_selected:
                    remaining = [feature for feature in features if feature not in selected]
                    selected = [*selected, *remaining[: n_selected - len(selected)]]
                context = script6._dnn_run_context(
                    data_path=data_path,
                    center=center,
                    method=method,
                    features=selected,
                    params=dnn_params,
                    drop_all_zero_columns=drop_all_zero_columns,
                    exclude_columns=target_excludes,
                    extra={"n_features": n_selected, "budget_ratio": ratio, "l1_run_dir": str(l1_run_dir)},
                )
                metrics = _metric_from_dir(method_dir) if skip_existing and script6._can_reuse_dnn(method_dir, context) else None
                if metrics is None:
                    score_rows = [
                        {"feature": feature, "score": float(scores.get(feature, np.nan)), "selected": feature in selected}
                        for feature in features
                    ]
                    pd.DataFrame(score_rows).sort_values(["selected", "score"], ascending=[False, False]).to_csv(
                        method_dir / "feature_scores.csv", index=False, encoding="utf-8-sig"
                    )
                    result = script6._train_dnn(
                        data_path=data_path,
                        center=center,
                        features=selected,
                        params=dnn_params,
                        output_dir=method_dir,
                        device=device,
                        drop_all_zero_columns=drop_all_zero_columns,
                        exclude_columns=target_excludes,
                    )
                    save_json(method_dir / script6.RUN_CONTEXT_FILE, context)
                    metrics = result.__dict__
                row = _result_row_from_metrics(
                    metrics,
                    center=center,
                    target_label=target_label,
                    plot_label=plot_label,
                    method=method,
                    n_features=n_selected,
                    output_dir=method_dir,
                    full_r2=full_r2,
                    budget_kind=f"{ratio:.2f}",
                )
                final_rows_by_ratio[ratio].append(row)
            print(f"  budget {ratio:.2f}: n={n_selected}, rule={budget['selection_rule']}, ratio={budget['ratio_at_n']:.3f}")
        pd.DataFrame(budget_rows).to_csv(output_root / "budget_selection_summary.csv", index=False, encoding="utf-8-sig")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(output_root / "budget_sweep_long.csv", index=False, encoding="utf-8-sig")
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(output_root / "budget_selection_summary.csv", index=False, encoding="utf-8-sig")
    for ratio, rows in final_rows_by_ratio.items():
        ratio_dir = ensure_dir(output_root / f"budget_{int(round(ratio * 100))}")
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(ratio_dir / "baseline_summary_long.csv", index=False, encoding="utf-8-sig")
        summary_df.pivot(index="target_label", columns="method", values="best_test_r2").to_csv(
            ratio_dir / "baseline_summary_wide.csv", encoding="utf-8-sig"
        )
        _plot_summary(summary_df, display_methods, ratio_dir / "baseline_test_r2.png", zh=False)
        _plot_summary(summary_df, display_methods, _zh_path(ratio_dir / "baseline_test_r2.png"), zh=True)
    print(f"Saved budget baseline sweep to {output_root}")


if __name__ == "__main__":
    main()
