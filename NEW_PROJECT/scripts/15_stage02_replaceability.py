from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_project_path
from src.data_utils import ensure_dir, normalize_column_list, prepare_supervised_dataset, save_json
from src.models import make_mlp


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if float(ss_tot.detach().cpu()) <= 0:
        return 0.0
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def _eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    ys = []
    preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            total_loss += float(nn.functional.mse_loss(pred, yb).detach().cpu()) * xb.size(0)
            ys.append(yb.detach().cpu())
            preds.append(pred.detach().cpu())
    y_cat = torch.cat(ys, dim=0)
    pred_cat = torch.cat(preds, dim=0)
    return total_loss / len(loader.dataset), _r2_score(y_cat, pred_cat)


def _train_subset_worker(payload: dict[str, Any]) -> dict[str, Any]:
    features = list(payload["features"])
    all_features = list(payload["all_features"])
    idx = [all_features.index(feature) for feature in features]
    device = torch.device(payload["device"])
    seed = int(payload["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_train_all = payload["X_train"]
    x_test_all = payload["X_test"]
    x_train = x_train_all[:, idx] if idx else x_train_all[:, :0]
    x_test = x_test_all[:, idx] if idx else x_test_all[:, :0]
    y_train = payload["y_train"]
    y_test = payload["y_test"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=int(payload["batch_size"]),
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32))),
        batch_size=int(payload["batch_size"]),
        shuffle=False,
    )

    model = make_mlp(len(features), payload["hidden_dims"], dropout=float(payload["dropout"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(payload["lr"]))
    best = {"r2": -math.inf, "mse": math.inf, "epoch": 0}
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    patience = int(payload["patience"])
    min_delta = float(payload["min_delta"])

    for epoch in range(1, int(payload["epochs"]) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
        test_mse, test_r2 = _eval_model(model, test_loader, device)
        if test_r2 > best["r2"] + min_delta:
            best = {"r2": float(test_r2), "mse": float(test_mse), "epoch": int(epoch)}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if patience > 0 and epoch >= 50 and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_mse, train_r2 = _eval_model(model, train_loader, device)
    test_mse, test_r2 = _eval_model(model, test_loader, device)
    return {
        "case_id": payload["case_id"],
        "target_feature": payload.get("target_feature"),
        "case_type": payload["case_type"],
        "feature_count": len(features),
        "features": ";".join(features),
        "best_test_r2": best["r2"],
        "best_test_mse": best["mse"],
        "best_epoch": best["epoch"],
        "final_train_r2": float(train_r2),
        "final_train_mse": float(train_mse),
        "final_test_r2": float(test_r2),
        "final_test_mse": float(test_mse),
    }


def _safe_run_dir(stage_root: Path, run_name: str | None) -> Path:
    run_dir = stage_root / (run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def _plot_replaceability(summary_df: pd.DataFrame, tau: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    order = np.arange(len(summary_df))
    width = 0.35
    ax.bar(order - width / 2, summary_df["c_only_r2"], width=width, label="C_i only")
    ax.bar(order + width / 2, summary_df["c_plus_r_r2"], width=width, label="C_i + R")
    ax.axhline(tau, color="#d62728", linestyle="--", linewidth=1.5, label=f"tau={tau:g}")
    ax.set_xticks(order)
    ax.set_xticklabels(summary_df["target_feature"], rotation=45, ha="right")
    ax.set_ylabel("Best test R2")
    ax.set_title("Stage 02 Replaceability Precheck")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_reprune_steps(steps_df: pd.DataFrame, tau: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_df["step"], steps_df["best_test_r2"], marker="o", linewidth=2)
    ax.axhline(tau, color="#d62728", linestyle="--", linewidth=1.5, label=f"tau={tau:g}")
    labels = [
        "baseline" if pd.isna(row.get("try_drop_feature")) else str(row.get("try_drop_feature"))
        for _, row in steps_df.iterrows()
    ]
    ax.set_xticks(steps_df["step"])
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Cumulative pruning step / attempted dropped field")
    ax.set_ylabel("Best test R2")
    ax.set_title("Stage 02 Cumulative Re-pruning")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _classify(c_r2: float, c_plus_r_r2: float, r_only_r2: float, tau: float) -> str:
    if c_r2 >= tau:
        return "main_path_redundant"
    if c_plus_r_r2 < tau:
        return "not_replaceable_under_current_universe"
    if r_only_r2 >= tau:
        return "independent_alternative_path"
    return "conditionally_replaceable"


def _fit_subset_case(
    *,
    case_id: str,
    case_type: str,
    target_feature: str | None,
    features: Sequence[str],
    all_features: Sequence[str],
    bundle: Any,
    args: argparse.Namespace,
    device: str,
    seed_text: str,
) -> dict[str, Any]:
    seed = int(args.seed) + zlib.adler32(seed_text.encode("utf-8")) % 100000
    return _train_subset_worker(
        {
            "case_id": case_id,
            "case_type": case_type,
            "target_feature": target_feature,
            "features": list(features),
            "all_features": list(all_features),
            "X_train": bundle.X_train,
            "y_train": bundle.y_train,
            "X_test": bundle.X_test,
            "y_test": bundle.y_test,
            "hidden_dims": list(args.hidden_dims),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "dropout": float(args.dropout),
            "patience": int(args.patience),
            "min_delta": float(args.min_delta),
            "seed": seed,
            "device": device,
        }
    )


def _run_cumulative_reprune(
    *,
    summary_df: pd.DataFrame,
    main_path: list[str],
    all_features: list[str],
    bundle: Any,
    args: argparse.Namespace,
    device: str,
    out_dir: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    redundant = summary_df[summary_df["replaceability_status"] == "main_path_redundant"].copy()
    redundant = redundant.sort_values("C_i_r2", ascending=False)
    candidate_order = redundant["target_feature"].astype(str).tolist()

    current = list(main_path)
    dropped: list[str] = []
    rows: list[dict[str, Any]] = []

    baseline = _fit_subset_case(
        case_id="reprune_baseline_main_path",
        case_type="reprune_baseline",
        target_feature=None,
        features=current,
        all_features=all_features,
        bundle=bundle,
        args=args,
        device=device,
        seed_text="reprune_baseline_main_path",
    )
    rows.append(
        {
            **baseline,
            "step": 0,
            "try_drop_feature": None,
            "decision": "baseline",
            "kept_dropped": False,
            "current_path_features": ";".join(current),
            "remaining_path_count": len(current),
        }
    )

    for step, feature in enumerate(candidate_order, start=1):
        trial = [item for item in current if item != feature]
        result = _fit_subset_case(
            case_id=f"reprune_try_drop_{feature}",
            case_type="reprune_try_drop",
            target_feature=feature,
            features=trial,
            all_features=all_features,
            bundle=bundle,
            args=args,
            device=device,
            seed_text=f"reprune_step_{step}_{feature}",
        )
        can_drop = bool(float(result["best_test_r2"]) >= float(args.tau))
        if can_drop:
            current = trial
            dropped.append(feature)
            decision = "drop_and_continue"
        else:
            decision = "stop_first_below_tau"
        rows.append(
            {
                **result,
                "step": step,
                "try_drop_feature": feature,
                "decision": decision,
                "kept_dropped": can_drop,
                "current_path_features": ";".join(current),
                "remaining_path_count": len(current),
            }
        )
        if not can_drop:
            break

    steps_df = pd.DataFrame(rows)
    steps_df.to_csv(out_dir / "reprune_steps.csv", index=False, encoding="utf-8-sig")
    save_json(
        out_dir / "repruned_main_path.json",
        {
            "tau": args.tau,
            "candidate_order": candidate_order,
            "dropped_features": dropped,
            "final_main_path_count": len(current),
            "final_main_path_features": current,
            "stopped_after_first_below_tau": bool(rows[-1]["decision"] == "stop_first_below_tau") if rows else False,
        },
    )
    _plot_reprune_steps(steps_df, float(args.tau), out_dir / "reprune_steps.png")
    return steps_df, current, dropped


def _make_precheck_jobs(
    *,
    main_path: Sequence[str],
    residual_features: Sequence[str],
    all_features: Sequence[str],
    bundle: Any,
    args: argparse.Namespace,
    devices: Sequence[str],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    def add_job(case_id: str, case_type: str, features: Sequence[str], target_feature: str | None = None) -> None:
        seed_text = f"{case_id}|{case_type}|{target_feature or ''}"
        seed = int(args.seed) + zlib.adler32(seed_text.encode("utf-8")) % 100000
        jobs.append(
            {
                "case_id": case_id,
                "case_type": case_type,
                "target_feature": target_feature,
                "features": list(features),
                "all_features": list(all_features),
                "X_train": bundle.X_train,
                "y_train": bundle.y_train,
                "X_test": bundle.X_test,
                "y_test": bundle.y_test,
                "hidden_dims": list(args.hidden_dims),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "dropout": float(args.dropout),
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
                "seed": seed,
                "device": devices[len(jobs) % len(devices)],
            }
        )

    add_job("full_all_features", "full_all_features", all_features)
    add_job("r_only", "R_only", residual_features)
    for feature in main_path:
        c_features = [item for item in main_path if item != feature]
        c_plus_r_features = [item for item in all_features if item != feature]
        add_job(f"{feature}__C_only", "C_only", c_features, target_feature=feature)
        add_job(f"{feature}__C_plus_R", "C_plus_R", c_plus_r_features, target_feature=feature)
    return jobs


def _run_jobs(
    jobs: Sequence[dict[str, Any]],
    *,
    out_csv: Path,
    workers: int,
) -> pd.DataFrame:
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        if len(existing) >= len(jobs):
            return existing

    rows: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    if workers <= 1:
        rows = [_train_subset_worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as executor:
            future_map = {executor.submit(_train_subset_worker, job): job["case_id"] for job in jobs}
            for future in as_completed(future_map):
                rows.append(future.result())
                pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    cases_df = pd.DataFrame(rows).sort_values(["case_type", "target_feature"], na_position="first")
    cases_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return cases_df


def _summarize_replaceability(
    *,
    cases_df: pd.DataFrame,
    main_path: Sequence[str],
    tau: float,
) -> pd.DataFrame:
    r_only = cases_df[cases_df["case_type"] == "R_only"].iloc[0]
    full = cases_df[cases_df["case_type"] == "full_all_features"].iloc[0]
    summary_rows = []
    for feature in main_path:
        c_row = cases_df[(cases_df["target_feature"] == feature) & (cases_df["case_type"] == "C_only")].iloc[0]
        cr_row = cases_df[(cases_df["target_feature"] == feature) & (cases_df["case_type"] == "C_plus_R")].iloc[0]
        c_r2 = float(c_row["best_test_r2"])
        cr_r2 = float(cr_row["best_test_r2"])
        r_r2 = float(r_only["best_test_r2"])
        status = _classify(c_r2, cr_r2, r_r2, float(tau))
        summary_rows.append(
            {
                "target_feature": feature,
                "C_i_feature_count": int(c_row["feature_count"]),
                "C_i_r2": c_r2,
                "C_i_meets_tau": bool(c_r2 >= tau),
                "C_i_plus_R_feature_count": int(cr_row["feature_count"]),
                "C_i_plus_R_r2": cr_r2,
                "C_i_plus_R_meets_tau": bool(cr_r2 >= tau),
                "R_only_feature_count": int(r_only["feature_count"]),
                "R_only_r2": r_r2,
                "R_only_meets_tau": bool(r_r2 >= tau),
                "full_all_features_r2": float(full["best_test_r2"]),
                "replaceability_status": status,
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 02: main-path field replaceability precheck.")
    parser.add_argument(
        "--stage01-interface",
        default=str(
            PROJECT_ROOT
            / "conditional_residual_compensation_outputs"
            / "CenterOn_net_actual_interchange_mw"
            / "stage01_main_path"
            / "run_20260705_175709"
            / "stage01_main_path_interface.json"
        ),
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data2025_v2.yaml"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "conditional_residual_compensation_outputs"))
    parser.add_argument("--stage-dir", default="stage02_replaceability")
    parser.add_argument("--run-name")
    parser.add_argument("--resume-run-dir")
    parser.add_argument("--force-precheck", action="store_true")
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32, 16])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-devices", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    with Path(args.stage01_interface).open("r", encoding="utf-8") as f:
        stage01 = json.load(f)

    center = stage01["center"]
    combo = stage01.get("combo")
    main_path = list(stage01["main_path_features"])
    all_features = list(stage01["all_available_features"])
    residual_features = [feature for feature in all_features if feature not in set(main_path)]
    exclude_columns = normalize_column_list(
        _read_json(Path(stage01["dgate_run_dir"]) / "config.json").get("preprocessing", {}).get("exclude_columns")
    )

    output_root = Path(args.output_root).resolve()
    stage_root = output_root / f"CenterOn_{center}" / args.stage_dir
    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
    else:
        run_dir = _safe_run_dir(stage_root, args.run_name)
    precheck_dir = ensure_dir(run_dir / "01_precheck")
    summary_dir = ensure_dir(run_dir / "02_summary")
    reprune_dir = ensure_dir(run_dir / "03_reprune")
    recheck_dir = ensure_dir(run_dir / "04_recheck_after_reprune")

    dataset_cfg = cfg["dataset"]
    training = cfg.get("training", {})
    preprocessing = cfg.get("preprocessing", {})
    bundle = prepare_supervised_dataset(
        data_path=resolve_project_path(cfg, dataset_cfg["processed_csv"]),
        center=center,
        features=all_features,
        train_ratio=float(training.get("train_ratio", 0.8)),
        random_state=int(training.get("random_state", 42)),
        drop_all_zero_columns=bool(preprocessing.get("drop_all_zero_columns", False)),
        exclude_columns=exclude_columns,
    )

    if args.parallel_devices:
        devices = [item.strip() for item in args.parallel_devices.split(",") if item.strip()]
    elif torch.cuda.is_available():
        devices = [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    workers = int(args.num_workers) if int(args.num_workers) > 0 else len(devices)

    jobs: list[dict[str, Any]] = []

    def add_job(case_id: str, case_type: str, features: Sequence[str], target_feature: str | None = None) -> None:
        seed_text = f"{case_id}|{case_type}|{target_feature or ''}"
        seed = int(args.seed) + zlib.adler32(seed_text.encode("utf-8")) % 100000
        jobs.append(
            {
                "case_id": case_id,
                "case_type": case_type,
                "target_feature": target_feature,
                "features": list(features),
                "all_features": all_features,
                "X_train": bundle.X_train,
                "y_train": bundle.y_train,
                "X_test": bundle.X_test,
                "y_test": bundle.y_test,
                "hidden_dims": list(args.hidden_dims),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "dropout": float(args.dropout),
                "patience": int(args.patience),
                "min_delta": float(args.min_delta),
                "seed": seed,
                "device": devices[len(jobs) % len(devices)],
            }
        )

    add_job("full_all_features", "full_all_features", all_features)
    add_job("r_only", "R_only", residual_features)
    for feature in main_path:
        c_features = [item for item in main_path if item != feature]
        c_plus_r_features = [item for item in all_features if item != feature]
        add_job(f"{feature}__C_only", "C_only", c_features, target_feature=feature)
        add_job(f"{feature}__C_plus_R", "C_plus_R", c_plus_r_features, target_feature=feature)

    save_json(
        run_dir / "stage_config.json",
        {
            "stage": "stage02_replaceability",
            "stage01_interface": str(Path(args.stage01_interface).resolve()),
            "center": center,
            "combo": combo,
            "main_path_count": len(main_path),
            "residual_feature_count": len(residual_features),
            "tau": args.tau,
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden_dims": args.hidden_dims,
                "dropout": args.dropout,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "devices": devices,
                "workers": workers,
            },
        },
    )

    precheck_path = precheck_dir / "replaceability_precheck_cases.csv"
    if precheck_path.exists() and not args.force_precheck:
        cases_df = pd.read_csv(precheck_path)
    else:
        rows: list[dict[str, Any]] = []
        ctx = mp.get_context("spawn")
        if workers <= 1:
            rows = [_train_subset_worker(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as executor:
                future_map = {executor.submit(_train_subset_worker, job): job["case_id"] for job in jobs}
                for future in as_completed(future_map):
                    rows.append(future.result())
                    pd.DataFrame(rows).to_csv(precheck_path, index=False, encoding="utf-8-sig")

        cases_df = pd.DataFrame(rows).sort_values(["case_type", "target_feature"], na_position="first")
        cases_df.to_csv(precheck_path, index=False, encoding="utf-8-sig")

    r_only = cases_df[cases_df["case_type"] == "R_only"].iloc[0]
    full = cases_df[cases_df["case_type"] == "full_all_features"].iloc[0]
    summary_rows = []
    for feature in main_path:
        c_row = cases_df[(cases_df["target_feature"] == feature) & (cases_df["case_type"] == "C_only")].iloc[0]
        cr_row = cases_df[(cases_df["target_feature"] == feature) & (cases_df["case_type"] == "C_plus_R")].iloc[0]
        c_r2 = float(c_row["best_test_r2"])
        cr_r2 = float(cr_row["best_test_r2"])
        r_r2 = float(r_only["best_test_r2"])
        status = _classify(c_r2, cr_r2, r_r2, float(args.tau))
        summary_rows.append(
            {
                "target_feature": feature,
                "C_i_feature_count": int(c_row["feature_count"]),
                "C_i_r2": c_r2,
                "C_i_meets_tau": bool(c_r2 >= args.tau),
                "C_i_plus_R_feature_count": int(cr_row["feature_count"]),
                "C_i_plus_R_r2": cr_r2,
                "C_i_plus_R_meets_tau": bool(cr_r2 >= args.tau),
                "R_only_feature_count": int(r_only["feature_count"]),
                "R_only_r2": r_r2,
                "R_only_meets_tau": bool(r_r2 >= args.tau),
                "full_all_features_r2": float(full["best_test_r2"]),
                "replaceability_status": status,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_dir / "replaceability_summary.csv", index=False, encoding="utf-8-sig")
    _plot_replaceability(summary_df.rename(columns={"C_i_r2": "c_only_r2", "C_i_plus_R_r2": "c_plus_r_r2"}), args.tau, summary_dir / "replaceability_precheck.png")
    reprune_steps_df, repruned_main_path, reprune_dropped = _run_cumulative_reprune(
        summary_df=summary_df,
        main_path=main_path,
        all_features=all_features,
        bundle=bundle,
        args=args,
        device=devices[0],
        out_dir=reprune_dir,
    )
    repruned_residual_features = [feature for feature in all_features if feature not in set(repruned_main_path)]
    recheck_jobs = _make_precheck_jobs(
        main_path=repruned_main_path,
        residual_features=repruned_residual_features,
        all_features=all_features,
        bundle=bundle,
        args=args,
        devices=devices,
    )
    recheck_cases_df = _run_jobs(
        recheck_jobs,
        out_csv=recheck_dir / "replaceability_precheck_cases.csv",
        workers=workers,
    )
    recheck_summary_df = _summarize_replaceability(
        cases_df=recheck_cases_df,
        main_path=repruned_main_path,
        tau=float(args.tau),
    )
    recheck_summary_df.to_csv(recheck_dir / "replaceability_summary.csv", index=False, encoding="utf-8-sig")
    _plot_replaceability(
        recheck_summary_df.rename(columns={"C_i_r2": "c_only_r2", "C_i_plus_R_r2": "c_plus_r_r2"}),
        args.tau,
        recheck_dir / "replaceability_precheck.png",
    )

    interface = {
        "schema_version": 1,
        "stage": "stage02_replaceability",
        "center": center,
        "target": center,
        "source_stage01_interface": str(Path(args.stage01_interface).resolve()),
        "source_run_dir": str(run_dir),
        "tau": args.tau,
        "main_path_features": main_path,
        "residual_candidate_features": residual_features,
        "repruned_main_path_features": repruned_main_path,
        "repruned_main_path_count": len(repruned_main_path),
        "repruned_dropped_features": reprune_dropped,
        "repruned_residual_candidate_features": repruned_residual_features,
        "replaceability_summary_csv": str(summary_dir / "replaceability_summary.csv"),
        "precheck_cases_csv": str(precheck_dir / "replaceability_precheck_cases.csv"),
        "reprune_steps_csv": str(reprune_dir / "reprune_steps.csv"),
        "repruned_main_path_json": str(reprune_dir / "repruned_main_path.json"),
        "repruned_replaceability_summary_csv": str(recheck_dir / "replaceability_summary.csv"),
        "repruned_precheck_cases_csv": str(recheck_dir / "replaceability_precheck_cases.csv"),
        "replaceable_features": summary_df[summary_df["replaceability_status"] == "conditionally_replaceable"]["target_feature"].tolist(),
        "non_replaceable_features": summary_df[summary_df["replaceability_status"] == "not_replaceable_under_current_universe"]["target_feature"].tolist(),
        "main_path_redundant_features": summary_df[summary_df["replaceability_status"] == "main_path_redundant"]["target_feature"].tolist(),
        "independent_alternative_features": summary_df[summary_df["replaceability_status"] == "independent_alternative_path"]["target_feature"].tolist(),
        "repruned_replaceable_features": recheck_summary_df[recheck_summary_df["replaceability_status"] == "conditionally_replaceable"]["target_feature"].tolist(),
        "repruned_non_replaceable_features": recheck_summary_df[recheck_summary_df["replaceability_status"] == "not_replaceable_under_current_universe"]["target_feature"].tolist(),
        "repruned_main_path_redundant_features": recheck_summary_df[recheck_summary_df["replaceability_status"] == "main_path_redundant"]["target_feature"].tolist(),
        "repruned_independent_alternative_features": recheck_summary_df[recheck_summary_df["replaceability_status"] == "independent_alternative_path"]["target_feature"].tolist(),
    }
    save_json(run_dir / "stage02_replaceability_interface.json", interface)

    lines = [
        "# 阶段 02：主路径字段可替代性预检",
        "",
        "## 目标",
        "",
        "对阶段 01 主路径中的每个字段 `x_i`，验证路径外字段 `R` 是否能在固定上下文 `C_i = P \\ {x_i}` 下补偿它。",
        "",
        "## 判定规则",
        "",
        f"- 阈值 tau: `{args.tau}`",
        "- 如果 `C_i` 单独达标，则该字段在当前主路径里是冗余字段。",
        "- 如果 `C_i ∪ R` 不达标，则该字段在当前字段宇宙下不可替代。",
        "- 如果 `R` 单独达标，则存在独立替代路径。",
        "- 如果 `C_i` 不达标、`C_i ∪ R` 达标、`R` 不达标，则进入后续残差补偿阶段。",
        "",
        "## 全局基准",
        "",
        f"- full all features R2: `{float(full['best_test_r2']):.6f}`",
        f"- R only R2: `{float(r_only['best_test_r2']):.6f}`",
        "",
        "## 分类结果",
        "",
    ]
    counts = summary_df["replaceability_status"].value_counts().to_dict()
    for key in ["conditionally_replaceable", "not_replaceable_under_current_universe", "main_path_redundant", "independent_alternative_path"]:
        lines.append(f"- {key}: {int(counts.get(key, 0))}")
    last_reprune = reprune_steps_df.iloc[-1]
    lines.extend(
        [
            "",
            "## 03 累计剪枝",
            "",
            "对 `main_path_redundant` 字段按 `C_i_r2` 从高到低累计尝试删除；每一步重新训练当前主路径子集，只有 R2 仍达到 tau 才真正删除。",
            "",
            f"- 原始主路径字段数: `{len(main_path)}`",
            f"- 累计删除字段数: `{len(reprune_dropped)}`",
            f"- 删除字段: `{'; '.join(reprune_dropped) if reprune_dropped else '(none)'}`",
            f"- 剪枝后主路径字段数: `{len(repruned_main_path)}`",
            f"- 最后一步 R2: `{float(last_reprune['best_test_r2']):.6f}`",
            f"- 最后一步决策: `{last_reprune['decision']}`",
        ]
    )
    recheck_counts = recheck_summary_df["replaceability_status"].value_counts().to_dict()
    lines.extend(
        [
            "",
            "## 04 剪枝后重新预检",
            "",
            "累计剪枝后，主路径 `P`、上下文 `C_i` 和路径外字段 `R` 都发生变化，因此重新执行阶段 02 的三项预检。",
            "",
        ]
    )
    for key in ["conditionally_replaceable", "not_replaceable_under_current_universe", "main_path_redundant", "independent_alternative_path"]:
        lines.append(f"- {key}: {int(recheck_counts.get(key, 0))}")
    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- `01_precheck/replaceability_precheck_cases.csv`: 所有子集模型训练结果。",
            "- `02_summary/replaceability_summary.csv`: 每个主路径字段的三项预检和分类。",
            "- `02_summary/replaceability_precheck.png`: `C_i` 与 `C_i ∪ R` 的 R2 对比图。",
            "- `03_reprune/reprune_steps.csv`: 冗余候选的累计剪枝过程。",
            "- `03_reprune/repruned_main_path.json`: 剪枝后的主路径接口片段。",
            "- `03_reprune/reprune_steps.png`: 累计剪枝 R2 曲线。",
            "- `04_recheck_after_reprune/replaceability_summary.csv`: 剪枝后主路径的重新可替代性预检。",
            "- `04_recheck_after_reprune/replaceability_precheck.png`: 剪枝后 `C_i` 与 `C_i ∪ R` 的 R2 对比图。",
            "- `stage02_replaceability_interface.json`: 后续阶段标准接口。",
        ]
    )
    (run_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Stage 02 run saved to {run_dir}")


if __name__ == "__main__":
    main()
