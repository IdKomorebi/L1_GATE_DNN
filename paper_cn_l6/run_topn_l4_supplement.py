from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "NEW_PROJECT"
PAPER = Path(__file__).resolve().parent
BASE_RUN = PROJECT / "outputs" / "data2025_Processed_V2" / "RedundantNoiseExplanation" / "RedundantNoiseExplanation_20260604_121500"
OUT_CSV = PAPER / "topn_incremental_results_l5.csv"

sys.path.insert(0, str(PROJECT))

from src.config import load_config, merged_training_params, resolve_project_path
from src.data_utils import normalize_column_list


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_existing() -> pd.DataFrame:
    frames = [pd.read_csv(BASE_RUN / "topn_incremental_results.csv")]
    if OUT_CSV.exists():
        frames.append(pd.read_csv(OUT_CSV))
    merged = pd.concat(frames, ignore_index=True)
    merged["_priority"] = merged["output_dir"].astype(str).str.contains("TopNIncremental_L5").astype(int)
    merged = merged.sort_values(["center", "exp_type", "feature_count", "_priority"])
    merged = merged.drop_duplicates(["center", "exp_type", "feature_count"], keep="last").drop(columns=["_priority"])
    return renumber(merged)


def renumber(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for center, sub in df.groupby("center", sort=False):
        topn = sub[sub["exp_type"] == "topn"].sort_values("feature_count").copy()
        topn["series_order"] = range(1, len(topn) + 1)
        full = sub[sub["exp_type"] == "full"].copy()
        full["series_order"] = 9999
        rows.extend(topn.to_dict("records"))
        rows.extend(full.to_dict("records"))
    return pd.DataFrame(rows).sort_values(["center", "series_order", "feature_count"]).reset_index(drop=True)


def save_results(df: pd.DataFrame) -> None:
    renumber(df).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"saved {OUT_CSV}", flush=True)


def center_label(center: str) -> str:
    return f"CenterOn_{center}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Supplement l5 Top-N points without rerunning the whole original experiment.")
    parser.add_argument("--top-ns", nargs="+", type=int, default=[40, 50, 60])
    parser.add_argument("--centers", nargs="+", default=["congestion_price_da", "da_as_total_mw_primary_reserve"])
    parser.add_argument("--epochs", type=int, default=None, help="Optional DNN epoch override for debugging only.")
    args = parser.parse_args()

    script6 = load_module("run_baselines_l5", PROJECT / "scripts" / "06_run_baselines.py")
    script10 = load_module("explain_noise_l5", PROJECT / "scripts" / "10_explain_redundant_noise.py")

    cfg = load_config(PROJECT / "configs" / "data2025_v2.yaml")
    dataset_cfg = cfg["dataset"]
    data_path = resolve_project_path(cfg, dataset_cfg["processed_csv"])
    preprocessing = cfg.get("preprocessing", {})
    drop_all_zero_columns = bool(preprocessing.get("drop_all_zero_columns", False))
    global_excludes = normalize_column_list(preprocessing.get("exclude_columns"))
    targets_by_center = script10._target_map(script6, cfg)
    dnn_params = merged_training_params(cfg, "DNN", (cfg.get("baseline_comparison", {}).get("dnn_training_overrides") or {}))
    if args.epochs is not None:
        dnn_params["epochs"] = int(args.epochs)
    device = script6._choose_device(str((cfg.get("baseline_comparison", {}) or {}).get("device", "auto")))

    results = read_existing()
    save_results(results)

    for center in args.centers:
        if center not in targets_by_center:
            raise ValueError(f"Unknown center: {center}")
        target = targets_by_center[center]
        target_excludes = normalize_column_list([*global_excludes, *normalize_column_list(target.get("baseline_exclude_columns"))])
        full_features = script10._all_features(data_path, center, drop_all_zero_columns, target_excludes)
        full_n = len(full_features)
        ranked = script10._ranked_gate_features(BASE_RUN / center_label(center) / "L1GateDNN")
        available = set(full_features)

        for top_n in args.top_ns:
            if not (0 < int(top_n) < full_n):
                print(f"skip {center} Top{top_n}: outside 1..{full_n - 1}", flush=True)
                continue
            exists = results[
                (results["center"] == center)
                & (results["exp_type"] == "topn")
                & (results["feature_count"].astype(int) == int(top_n))
            ]
            if not exists.empty:
                print(f"skip {center} Top{top_n}: already exists", flush=True)
                continue

            selected = [f for f in ranked["related"].head(int(top_n)).astype(str).tolist() if f in available]
            if len(selected) != int(top_n):
                selected = [*selected, *[f for f in full_features if f not in set(selected)][: int(top_n) - len(selected)]]

            output_dir = BASE_RUN / center_label(center) / "TopNIncremental_L5" / f"DNN_top{int(top_n):02d}"
            print(f"train {center} Top{top_n} -> {output_dir}", flush=True)
            result = script6._train_dnn(
                data_path=data_path,
                center=center,
                features=selected,
                params=dnn_params,
                output_dir=output_dir,
                device=device,
                drop_all_zero_columns=drop_all_zero_columns,
                exclude_columns=target_excludes,
            )
            row = {
                "center": center,
                "exp_type": "topn",
                "series_order": 0,
                "x_label": f"Top{int(top_n)}",
                "feature_count": int(top_n),
                "best_test_r2": result.best_test_r2,
                "best_epoch": result.best_epoch,
                "final_test_r2": result.final_test_r2,
                "output_dir": str(result.output_dir),
            }
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            results = results.drop_duplicates(["center", "exp_type", "feature_count"], keep="last")
            save_results(results)

    print("done", flush=True)


if __name__ == "__main__":
    main()
