from __future__ import annotations

"""Lambda-path scan for L1GateDNN: warm-start continuation over an increasing
sparsity-strength schedule. Records each field's exit point (critical lambda at
which its gate collapses), per-stage gate trajectories/stability stats, and
verifies each stage's surviving subset (and its complement) with a plain DNN.

Reuses the exact feature list, split seed and hyperparameters of a reference
L1GateDNN run so results are directly comparable.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_project_path
from src.data_utils import prepare_supervised_dataset
from src.models import DNNRegressor, L1GateRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def eval_r2(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(DEVICE)).cpu())
            trues.append(yb)
    return r2_score(torch.cat(trues), torch.cat(preds))


def train_epochs(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lam: float,
    log_rows: list,
    stage: int,
    gate_epoch_rows: list,
    features: list[str],
) -> tuple[float, float]:
    loss_fn = nn.MSELoss()
    best_r2 = -np.inf
    final_r2 = -np.inf
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if lam > 0 and hasattr(model, "get_gates"):
                loss = loss + lam * model.get_gates().abs().sum()
            loss.backward()
            optimizer.step()
        final_r2 = eval_r2(model, test_loader)
        best_r2 = max(best_r2, final_r2)
        log_rows.append({"stage": stage, "lambda": lam, "epoch": ep, "test_r2": round(final_r2, 6)})
        if hasattr(model, "get_gates"):
            g = model.get_gates().detach().cpu().numpy()
            for i, name in enumerate(features):
                gate_epoch_rows.append(
                    {"stage": stage, "lambda": lam, "epoch": ep, "feature": name, "gate": float(g[i])}
                )
    return best_r2, final_r2


def main() -> None:
    parser = argparse.ArgumentParser(description="Lambda-path scan for L1GateDNN.")
    parser.add_argument(
        "--reference-run",
        default="outputs/data2025_Processed_V2/CenterOn_net_actual_interchange_mw/L1GateDNN/"
        "run_20260603_l1_lr0p00065_thr0p10_combo5_L1GateDNN",
    )
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032])
    parser.add_argument("--first-stage-epochs", type=int, default=200)
    parser.add_argument("--stage-epochs", type=int, default=80)
    parser.add_argument("--collapse-tol", type=float, default=0.02)
    parser.add_argument("--stability-window", type=int, default=20)
    parser.add_argument("--verify-epochs", type=int, default=100)
    parser.add_argument("--run-name", default="run_20260714_lambdapath_combo5_L1GateDNN")
    args = parser.parse_args()

    ref_dir = PROJECT_ROOT / args.reference_run
    ref_cfg = json.loads((ref_dir / "config.json").read_text(encoding="utf-8"))
    features: list[str] = ref_cfg["features"]
    params = ref_cfg["params"]
    center = ref_cfg["center"]
    exclude_columns = ref_cfg["preprocessing"]["exclude_columns"]
    hidden = params["hidden_dims"]
    lr = float(params["lr"])
    batch_size = int(params["batch_size"])
    seed = int(params["random_state"])
    train_ratio = float(params["train_ratio"])

    cfg = load_config(None)
    data_path = resolve_project_path(cfg, cfg["dataset"]["processed_csv"])
    bundle = prepare_supervised_dataset(
        data_path=data_path,
        center=center,
        features=features,
        train_ratio=train_ratio,
        random_state=seed,
        drop_all_zero_columns=False,
        exclude_columns=exclude_columns,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = TensorDataset(torch.from_numpy(bundle.X_train), torch.from_numpy(bundle.y_train))
    test_ds = TensorDataset(torch.from_numpy(bundle.X_test), torch.from_numpy(bundle.y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    out_dir = ref_dir.parent / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "reference_run": str(ref_dir),
                "center": center,
                "features": features,
                "lambdas": args.lambdas,
                "first_stage_epochs": args.first_stage_epochs,
                "stage_epochs": args.stage_epochs,
                "collapse_tol": args.collapse_tol,
                "stability_window": args.stability_window,
                "verify_epochs": args.verify_epochs,
                "params": params,
                "device": str(DEVICE),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    q = len(features)
    model = L1GateRegressor(q, hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_rows: list = []
    gate_epoch_rows: list = []
    stage_rows: list = []
    gate_path_rows: list = []
    exit_lambda: dict[str, float | None] = {name: None for name in features}
    prev_active = set(features)

    for s, lam in enumerate(args.lambdas, start=1):
        epochs = args.first_stage_epochs if s == 1 else args.stage_epochs
        best_r2, final_r2 = train_epochs(
            model, optimizer, train_loader, test_loader, epochs, lam, log_rows, s, gate_epoch_rows, features
        )
        gates = model.get_gates().detach().cpu().numpy()
        window = [
            r for r in gate_epoch_rows if r["stage"] == s and r["epoch"] > epochs - args.stability_window
        ]
        stats: dict[str, list[float]] = {name: [] for name in features}
        for r in window:
            stats[r["feature"]].append(abs(r["gate"]))
        active = set()
        for i, name in enumerate(features):
            g_abs = abs(float(gates[i]))
            g_mean = float(np.mean(stats[name]))
            g_std = float(np.std(stats[name]))
            is_active = g_mean >= args.collapse_tol
            if is_active:
                active.add(name)
            elif exit_lambda[name] is None:
                exit_lambda[name] = lam
            gate_path_rows.append(
                {
                    "stage": s,
                    "lambda": lam,
                    "feature": name,
                    "gate_final": round(g_abs, 6),
                    "gate_mean_window": round(g_mean, 6),
                    "gate_std_window": round(g_std, 6),
                    "active": int(is_active),
                }
            )
        stage_rows.append(
            {
                "stage": s,
                "lambda": lam,
                "epochs": epochs,
                "best_test_r2": round(best_r2, 6),
                "final_test_r2": round(final_r2, 6),
                "active_count": len(active),
                "newly_exited": ";".join(sorted(prev_active - active)),
            }
        )
        prev_active = active
        print(f"[stage {s}] lambda={lam} final_r2={final_r2:.4f} active={len(active)}", flush=True)

    exit_rows = []
    for name in features:
        lam_star = exit_lambda[name]
        exit_rows.append(
            {
                "feature": name,
                "exit_lambda": lam_star if lam_star is not None else "survived_max",
                "contribution_rank_metric": lam_star if lam_star is not None else args.lambdas[-1] * 2,
            }
        )

    def write_csv(path: Path, rows: list, fieldnames: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    write_csv(out_dir / "path_stages.csv", stage_rows, list(stage_rows[0].keys()))
    write_csv(out_dir / "gate_path.csv", gate_path_rows, list(gate_path_rows[0].keys()))
    write_csv(out_dir / "exit_points.csv", exit_rows, list(exit_rows[0].keys()))
    write_csv(out_dir / "log.csv", log_rows, list(log_rows[0].keys()))
    write_csv(out_dir / "gate_epochs.csv", gate_epoch_rows, list(gate_epoch_rows[0].keys()))

    # Verification: retrain a plain DNN on each stage's surviving subset and its complement.
    name_to_idx = {n: i for i, n in enumerate(features)}
    verify_rows = []
    for row in stage_rows:
        s, lam = row["stage"], row["lambda"]
        subset = [r["feature"] for r in gate_path_rows if r["stage"] == s and r["active"] == 1]
        for mode, cols in (("subset", subset), ("complement", [n for n in features if n not in subset])):
            if not cols:
                continue
            idx = [name_to_idx[c] for c in cols]
            torch.manual_seed(seed)
            sub_train = TensorDataset(torch.from_numpy(bundle.X_train[:, idx]), torch.from_numpy(bundle.y_train))
            sub_test = TensorDataset(torch.from_numpy(bundle.X_test[:, idx]), torch.from_numpy(bundle.y_test))
            sub_train_loader = DataLoader(sub_train, batch_size=batch_size, shuffle=True)
            sub_test_loader = DataLoader(sub_test, batch_size=batch_size, shuffle=False)
            dnn = DNNRegressor(len(idx), hidden).to(DEVICE)
            opt = torch.optim.Adam(dnn.parameters(), lr=lr)
            dummy_log: list = []
            best, final = train_epochs(
                dnn, opt, sub_train_loader, sub_test_loader, args.verify_epochs, 0.0, dummy_log, -s, [], []
            )
            verify_rows.append(
                {
                    "stage": s,
                    "lambda": lam,
                    "mode": mode,
                    "n_features": len(idx),
                    "best_test_r2": round(best, 6),
                    "final_test_r2": round(final, 6),
                }
            )
            print(f"[verify stage {s} {mode}] n={len(idx)} best_r2={best:.4f}", flush=True)

    write_csv(out_dir / "verification.csv", verify_rows, list(verify_rows[0].keys()))
    print(f"Saved lambda-path run to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
