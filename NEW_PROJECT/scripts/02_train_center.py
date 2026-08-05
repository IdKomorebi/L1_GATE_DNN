from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import column_combinations, load_config, resolve_center_spec
from src.data_utils import normalize_column_list, safe_name
from src.trainer import train_center_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one center variable with one model.")
    parser.add_argument("--config", help="Config YAML path. Defaults to configs/active_config.yaml.")
    parser.add_argument("--center")
    parser.add_argument("--combo", help="Column combination id/name from column_combinations in the config.")
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        choices=["DNN", "L1GateDNN", "DGatingDNN", "ImprovedL1GateDNN", "ImprovedL2GateDNN"],
        help="One or more model names.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lambda-l1", type=float)
    parser.add_argument("--lambda-l2", type=float)
    parser.add_argument("--lambda-dgate", type=float)
    parser.add_argument("--dgate-depth", type=int, choices=[2, 3, 4, 5, 6])
    parser.add_argument(
        "--dgate-normalize-lambda-by-depth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether DGatingDNN divides lambda_dgate by dgate_depth.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        help="After threshold filtering, rank features by the absolute-sum of the six relationship metrics and keep Top-N.",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--force-relations", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    combo_spec = None
    if args.combo:
        combinations = column_combinations(cfg)
        if args.combo not in combinations:
            available = ", ".join(sorted(combinations)) or "(none)"
            raise ValueError(f"Unknown combo: {args.combo}. Available combos: {available}")
        combo_spec = resolve_center_spec(cfg, args.combo)

    center = combo_spec["center"] if combo_spec else args.center
    if args.combo and args.center and args.center != center:
        raise ValueError(f"--center {args.center!r} does not match combo {args.combo!r} center {center!r}.")
    if not center:
        raise ValueError("Please pass --center or use --combo with a center.")

    exclude_columns = normalize_column_list(combo_spec["exclude_columns"] if combo_spec else None)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "lambda_dgate": args.lambda_dgate,
        "dgate_depth": args.dgate_depth,
        "dgate_normalize_lambda_by_depth": args.dgate_normalize_lambda_by_depth,
    }
    for model_name in args.model:
        run_name = args.run_name
        combo_name = None
        if combo_spec:
            combo_name = f"combo{combo_spec['id']}"
            if not run_name:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name(combo_name, 60)}"
        elif args.top_n and not run_name:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_top{args.top_n}"
        if combo_spec and args.top_n and not args.run_name and run_name:
            run_name = f"{run_name}_top{args.top_n}"
        if run_name and len(args.model) > 1:
            run_name = f"{run_name}_{model_name}"
        if args.combo:
            print(f"=== Training combo={combo_name} center={center} model={model_name} ===")
            if exclude_columns:
                print(f"Excluded columns: {', '.join(exclude_columns)}")
        else:
            print(f"=== Training center={center} model={model_name} ===")
        run_dir = train_center_model(
            cfg,
            center=center,
            model_name=model_name,
            overrides=overrides,
            run_name=run_name,
            force_relations=args.force_relations,
            exclude_columns=exclude_columns,
            combo_name=combo_name,
            feature_top_n=args.top_n,
        )
        print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
