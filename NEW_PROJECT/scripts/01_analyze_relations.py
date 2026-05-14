from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_project_path
from src.knowledge_graph import generate_knowledge_graph
from src.relation_analyzer import (
    analyze_all_relationships,
    analyze_center_relationships,
    center_relation_analysis_dir,
    relation_analysis_dir,
    relation_set_name,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze field relationships for a dataset.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025.yaml"))
    parser.add_argument("--center", help="If provided, only analyze this center against all other columns.")
    parser.add_argument("--no-graph", action="store_true", help="Skip knowledge graph HTML generation.")
    parser.add_argument("--progress-every", type=int, help="Print progress every N pairs. Use 0 to disable.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    rel_cfg = cfg.get("relations", {})
    data_path = resolve_project_path(cfg, dataset["processed_csv"])
    output_root = resolve_project_path(cfg, dataset["output_root"])
    metrics = rel_cfg.get("metrics", [])
    thresholds = rel_cfg.get("thresholds", {})
    sample_size = int(rel_cfg.get("sample_size", 3000))
    expensive_sample_size = int(rel_cfg.get("expensive_sample_size", 1200))
    random_state = int(cfg.get("training", {}).get("random_state", 42))
    tag = relation_set_name(metrics, thresholds, sample_size, expensive_sample_size)
    print(f"Relationship setting: {tag}")

    if args.center:
        out_dir = center_relation_analysis_dir(output_root, args.center, metrics, thresholds, sample_size, expensive_sample_size)
        output_csv = out_dir / "center_relationships.csv"
        result = analyze_center_relationships(
            data_path=data_path,
            center=args.center,
            output_csv=output_csv,
            metrics=metrics,
            thresholds=thresholds,
            sample_size=sample_size,
            expensive_sample_size=expensive_sample_size,
            random_state=random_state,
            progress_every=10 if args.progress_every is None else args.progress_every,
        )
        if not args.no_graph:
            graph_path = generate_knowledge_graph(
                result,
                out_dir / "center_knowledge_graph.html",
                metrics=metrics,
                thresholds=thresholds,
                title=f"Center Relationship Graph - {args.center}",
                center=args.center,
            )
            print(f"Saved knowledge graph to {graph_path}")
    else:
        out_dir = relation_analysis_dir(output_root, metrics, thresholds, sample_size, expensive_sample_size)
        output_csv = out_dir / "relationships.csv"
        result = analyze_all_relationships(
            data_path=data_path,
            output_csv=output_csv,
            metrics=metrics,
            thresholds=thresholds,
            sample_size=sample_size,
            expensive_sample_size=expensive_sample_size,
            random_state=random_state,
            progress_every=100 if args.progress_every is None else args.progress_every,
        )
        if not args.no_graph:
            graph_path = generate_knowledge_graph(
                result,
                out_dir / "knowledge_graph.html",
                metrics=metrics,
                thresholds=thresholds,
                title="All Pair Relationship Knowledge Graph",
            )
            print(f"Saved knowledge graph to {graph_path}")

    print(f"Saved {len(result)} relationships to {output_csv}")


if __name__ == "__main__":
    main()
