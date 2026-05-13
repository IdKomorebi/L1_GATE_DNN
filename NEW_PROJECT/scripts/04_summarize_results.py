from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize saved training runs.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = resolve_project_path(cfg, cfg["dataset"]["output_root"])
    rows = []
    for metrics_path in output_root.glob("CenterOn_*/**/metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["metrics_path"] = str(metrics_path)
        rows.append(row)

    if not rows:
        print(f"No metrics.json files found under {output_root}")
        return

    summary = pd.DataFrame(rows).sort_values(["center", "model", "best_test_r2"], ascending=[True, True, False])
    summary_path = output_root / "summary_all_centers.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    for center, group in summary.groupby("center"):
        center_dirs = list(output_root.glob(f"CenterOn_{center}"))
        if not center_dirs:
            safe_dirs = list(output_root.glob("CenterOn_*"))
            center_dir = next((d for d in safe_dirs if group.iloc[0]["run_dir"].startswith(str(d))), None)
        else:
            center_dir = center_dirs[0]
        if center_dir is not None:
            group.to_csv(center_dir / "compare_models.csv", index=False, encoding="utf-8-sig")

    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
