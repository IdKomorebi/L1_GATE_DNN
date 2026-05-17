from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.trainer import train_center_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured centers and models.")
    parser.add_argument("--config", help="Config YAML path. Defaults to configs/active_config.yaml.")
    parser.add_argument("--centers", nargs="*")
    parser.add_argument("--models", nargs="*", choices=["DNN", "L1GateDNN", "ImprovedL1GateDNN", "ImprovedL2GateDNN"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force-relations", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    centers = args.centers or cfg.get("experiment", {}).get("centers", [])
    models = args.models or cfg.get("experiment", {}).get("models", [])
    overrides = {"epochs": args.epochs}

    if not centers:
        raise ValueError("No centers configured.")
    if not models:
        raise ValueError("No models configured.")

    for center in centers:
        for model_name in models:
            print(f"=== Training center={center} model={model_name} ===")
            run_dir = train_center_model(
                cfg,
                center=center,
                model_name=model_name,
                overrides=overrides,
                force_relations=args.force_relations,
            )
            print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
