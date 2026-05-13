from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_project_path
from src.data_utils import center_output_dir
from src.relation_analyzer import analyze_all_relationships, analyze_center_relationships


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze field relationships for a dataset.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025.yaml"))
    parser.add_argument("--center", help="If provided, only analyze this center against all other columns.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    rel_cfg = cfg.get("relations", {})
    data_path = resolve_project_path(cfg, dataset["processed_csv"])
    output_root = resolve_project_path(cfg, dataset["output_root"])

    if args.center:
        out_dir = center_output_dir(output_root, args.center)
        output_csv = out_dir / "center_relationships.csv"
        result = analyze_center_relationships(
            data_path=data_path,
            center=args.center,
            output_csv=output_csv,
            metrics=rel_cfg.get("metrics", []),
            thresholds=rel_cfg.get("thresholds", {}),
            sample_size=int(rel_cfg.get("sample_size", 3000)),
            expensive_sample_size=int(rel_cfg.get("expensive_sample_size", 1200)),
            random_state=int(cfg.get("training", {}).get("random_state", 42)),
        )
    else:
        output_csv = output_root / "relationships.csv"
        result = analyze_all_relationships(
            data_path=data_path,
            output_csv=output_csv,
            metrics=rel_cfg.get("metrics", []),
            thresholds=rel_cfg.get("thresholds", {}),
            sample_size=int(rel_cfg.get("sample_size", 3000)),
            expensive_sample_size=int(rel_cfg.get("expensive_sample_size", 1200)),
            random_state=int(cfg.get("training", {}).get("random_state", 42)),
        )

    print(f"Saved {len(result)} relationships to {output_csv}")


if __name__ == "__main__":
    main()
