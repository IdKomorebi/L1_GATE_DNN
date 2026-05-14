from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.trainer import train_center_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one center variable with one model.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025.yaml"))
    parser.add_argument("--center", required=True)
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        choices=["DNN", "L1GateDNN", "ImprovedL1GateDNN", "ImprovedL2GateDNN"],
        help="One or more model names.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lambda-l1", type=float)
    parser.add_argument("--lambda-l2", type=float)
    parser.add_argument("--run-name")
    parser.add_argument("--force-relations", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
    }
    for model_name in args.model:
        run_name = args.run_name
        if run_name and len(args.model) > 1:
            run_name = f"{run_name}_{model_name}"
        print(f"=== Training center={args.center} model={model_name} ===")
        run_dir = train_center_model(
            cfg,
            center=args.center,
            model_name=model_name,
            overrides=overrides,
            run_name=run_name,
            force_relations=args.force_relations,
        )
        print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
