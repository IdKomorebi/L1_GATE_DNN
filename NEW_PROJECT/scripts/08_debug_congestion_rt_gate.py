from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, merged_training_params, resolve_project_path
from src.data_utils import ensure_dir, read_numeric_csv, save_json
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


def _selected(run_dir: Path) -> List[str]:
    rows = _load_json(run_dir / "selected_features.json").get("features", [])
    return [str(row["name"]) for row in rows]


def main() -> None:
    script6 = _load_script6()
    cfg = load_config(PROJECT_ROOT / "configs" / "data2025_v2.yaml")
    data_path = resolve_project_path(cfg, cfg["dataset"]["processed_csv"])
    output_root = ensure_dir(
        resolve_project_path(cfg, cfg["dataset"]["output_root"])
        / "BaselineComparison"
        / "congestion_rt_gate_debug_grid"
    )
    center = "congestion_price_rt"
    dnn_params = merged_training_params(cfg, "DNN", (cfg.get("baseline_comparison", {}).get("dnn_training_overrides") or {}))
    base_l1 = {
        "epochs": 200,
        "batch_size": 50,
        "lr": 0.00065,
        "train_ratio": 0.8,
        "random_state": 42,
        "hidden_dims": [64, 32, 16],
        "key_epochs": [10, 20, 50, 100, 200],
        "lambda_l1": 0.002,
        "active_threshold": 0.1,
    }
    df = read_numeric_csv(data_path, drop_all_zero_columns=False, exclude_columns=[])
    all_features = [str(col) for col in df.columns if str(col) != center]
    device = script6._choose_device(str((cfg.get("baseline_comparison", {}) or {}).get("device", "auto")))
    combos = [
        {"tag": "lam002_thr010", "lambda_l1": 0.002, "active_threshold": 0.10},
        {"tag": "lam003_thr010", "lambda_l1": 0.003, "active_threshold": 0.10},
        {"tag": "lam0035_thr010", "lambda_l1": 0.0035, "active_threshold": 0.10},
        {"tag": "lam004_thr010", "lambda_l1": 0.004, "active_threshold": 0.10},
        {"tag": "lam0045_thr010", "lambda_l1": 0.0045, "active_threshold": 0.10},
        {"tag": "lam004_thr008", "lambda_l1": 0.004, "active_threshold": 0.08},
        {"tag": "lam004_thr012", "lambda_l1": 0.004, "active_threshold": 0.12},
        {"tag": "lam003_thr015", "lambda_l1": 0.003, "active_threshold": 0.15},
        {"tag": "lam004_thr015", "lambda_l1": 0.004, "active_threshold": 0.15},
        {"tag": "lam005_thr015", "lambda_l1": 0.005, "active_threshold": 0.15},
    ]
    rows = []
    for combo in combos:
        params = {**base_l1, "lambda_l1": combo["lambda_l1"], "active_threshold": combo["active_threshold"]}
        run_dir = ensure_dir(output_root / str(combo["tag"]) / "L1GateDNN_source")
        dnn_dir = output_root / str(combo["tag"]) / "DNN_on_selected"
        print(f"Training {combo['tag']}")
        if not (run_dir / "selected_features.json").exists():
            with contextlib.redirect_stdout(io.StringIO()):
                train_center_model(
                    cfg,
                    center=center,
                    model_name="L1GateDNN",
                    overrides=params,
                    run_name=f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{combo['tag']}",
                    output_run_dir=run_dir,
                )
        selected = _selected(run_dir)
        if (dnn_dir / "metrics.json").exists():
            dnn_metrics = _load_json(dnn_dir / "metrics.json")
            best_test_r2 = float(dnn_metrics.get("best_test_r2", 0.0))
            best_epoch = int(dnn_metrics.get("best_epoch", 0))
        else:
            result = script6._train_dnn(
                data_path=data_path,
                center=center,
                features=selected,
                params=dnn_params,
                output_dir=dnn_dir,
                device=device,
                drop_all_zero_columns=False,
                exclude_columns=[],
            )
            best_test_r2 = result.best_test_r2
            best_epoch = result.best_epoch
        metrics = _load_json(run_dir / "metrics.json")
        row = {
            **combo,
            "source_feature_count": int(metrics.get("feature_count", 0)),
            "selected_count": len(selected),
            "source_best_test_r2": float(metrics.get("best_test_r2", 0.0)),
            "selected_dnn_best_test_r2": best_test_r2,
            "selected_dnn_best_epoch": best_epoch,
            "run_dir": str(run_dir),
        }
        rows.append(row)
        print(
            f"  selected={row['selected_count']} "
            f"source_r2={row['source_best_test_r2']:.6f} "
            f"dnn_r2={row['selected_dnn_best_test_r2']:.6f}"
        )
    pd.DataFrame(rows).to_csv(output_root / "grid_summary.csv", index=False, encoding="utf-8-sig")
    save_json(output_root / "grid_config.json", {"dnn_params": dnn_params, "base_l1": base_l1, "combos": combos})
    print(f"Saved to {output_root}")


if __name__ == "__main__":
    main()
