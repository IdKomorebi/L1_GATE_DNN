from __future__ import annotations

"""Lambda-path scan v2 for L1GateDNN, addressing two mechanism gaps of v1:

1. Scale compensation: after every optimizer step, the first hidden layer's
   weight columns are renormalized to unit L2 norm, so the per-field input
   scale is carried by the gate coefficient alone.
2. Real exit: a field judged collapsed (window mean <= eps AND window std <=
   eps_s) has its gate hard-set to zero and frozen for all later stages, so
   survival sets are nested by construction and stage-end R2 reflects a model
   with exited channels truly closed.

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
DELTA = 1e-8


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


def renorm_first_layer(model: nn.Module) -> None:
    with torch.no_grad():
        w = model.net[0].weight
        w.div_(w.norm(dim=0, keepdim=True) + DELTA)


def train_stage(
    model: L1GateRegressor,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lam: float,
    frozen: torch.Tensor,
    log_rows: list,
    stage: int,
    gate_epoch_rows: list,
    features: list[str],
) -> float:
    loss_fn = nn.MSELoss()
    final_r2 = -np.inf
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb) + lam * model.gate.abs().sum()
            loss.backward()
            if frozen.any() and model.gate.grad is not None:
                model.gate.grad[frozen] = 0.0
            optimizer.step()
            renorm_first_layer(model)
            if frozen.any():
                with torch.no_grad():
                    model.gate.data[frozen] = 0.0
        final_r2 = eval_r2(model, test_loader)
        log_rows.append({"stage": stage, "lambda": lam, "epoch": ep, "test_r2": round(final_r2, 6)})
        g = model.gate.detach().cpu().numpy()
        for i, name in enumerate(features):
            gate_epoch_rows.append(
                {"stage": stage, "lambda": lam, "epoch": ep, "feature": name, "gate": float(g[i])}
            )
    return final_r2


def train_plain_dnn(
    in_idx: list[int],
    bundle,
    hidden,
    lr: float,
    batch_size: int,
    seed: int,
    epochs: int,
) -> float:
    torch.manual_seed(seed)
    tr = TensorDataset(torch.from_numpy(bundle.X_train[:, in_idx]), torch.from_numpy(bundle.y_train))
    te = TensorDataset(torch.from_numpy(bundle.X_test[:, in_idx]), torch.from_numpy(bundle.y_test))
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)
    dnn = DNNRegressor(len(in_idx), hidden).to(DEVICE)
    opt = torch.optim.Adam(dnn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best = -np.inf
    for _ in range(epochs):
        dnn.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(dnn(xb), yb).backward()
            opt.step()
        best = max(best, eval_r2(dnn, te_loader))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Lambda-path scan v2 (renorm + freeze) for L1GateDNN.")
    parser.add_argument(
        "--reference-run",
        default="outputs/data2025_Processed_V2/CenterOn_net_actual_interchange_mw/L1GateDNN/"
        "run_20260603_l1_lr0p00065_thr0p10_combo5_L1GateDNN",
    )
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032])
    parser.add_argument("--first-stage-epochs", type=int, default=200)
    parser.add_argument("--stage-epochs", type=int, default=80)
    parser.add_argument("--collapse-tol", type=float, default=0.02)
    parser.add_argument("--stability-tol", type=float, default=0.01)
    parser.add_argument("--stability-window", type=int, default=20)
    parser.add_argument("--verify-epochs", type=int, default=100)
    parser.add_argument("--run-name", default="run_20260717_lambdapath_v2_combo5_L1GateDNN")
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
                "stability_tol": args.stability_tol,
                "stability_window": args.stability_window,
                "verify_epochs": args.verify_epochs,
                "mechanism": "first-layer column unit-norm + hard freeze on exit",
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
    renorm_first_layer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    frozen = torch.zeros(q, dtype=torch.bool, device=DEVICE)

    log_rows: list = []
    gate_epoch_rows: list = []
    stage_rows: list = []
    gate_path_rows: list = []
    exit_lambda: dict[str, float | None] = {name: None for name in features}

    for s, lam in enumerate(args.lambdas, start=1):
        epochs = args.first_stage_epochs if s == 1 else args.stage_epochs
        final_r2 = train_stage(
            model, optimizer, train_loader, test_loader, epochs, lam, frozen,
            log_rows, s, gate_epoch_rows, features,
        )
        window = [
            r for r in gate_epoch_rows if r["stage"] == s and r["epoch"] > epochs - args.stability_window
        ]
        stats: dict[str, list[float]] = {name: [] for name in features}
        for r in window:
            stats[r["feature"]].append(abs(r["gate"]))
        newly_exited: list[str] = []
        for i, name in enumerate(features):
            if bool(frozen[i]):
                m, sd, active = 0.0, 0.0, 0
            else:
                m = float(np.mean(stats[name]))
                sd = float(np.std(stats[name]))
                if m <= args.collapse_tol and sd <= args.stability_tol:
                    newly_exited.append(name)
                    frozen[i] = True
                    exit_lambda[name] = lam
                    active = 0
                else:
                    active = 1
            gate_path_rows.append(
                {
                    "stage": s,
                    "lambda": lam,
                    "feature": name,
                    "gate_mean_window": round(m, 6),
                    "gate_std_window": round(sd, 6),
                    "active": active,
                }
            )
        with torch.no_grad():
            model.gate.data[frozen] = 0.0
        r2_after_freeze = eval_r2(model, test_loader)
        active_count = int(q - int(frozen.sum()))
        stage_rows.append(
            {
                "stage": s,
                "lambda": lam,
                "epochs": epochs,
                "final_test_r2": round(final_r2, 6),
                "r2_after_freeze": round(r2_after_freeze, 6),
                "active_count": active_count,
                "newly_exited": ";".join(sorted(newly_exited)),
            }
        )
        print(
            f"[stage {s}] lambda={lam} final_r2={final_r2:.4f} "
            f"after_freeze={r2_after_freeze:.4f} active={active_count}",
            flush=True,
        )

    exit_rows = [
        {"feature": name, "exit_lambda": exit_lambda[name] if exit_lambda[name] is not None else "survived_max"}
        for name in features
    ]

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

    name_to_idx = {n: i for i, n in enumerate(features)}
    verify_rows = []
    for row in stage_rows:
        s, lam = row["stage"], row["lambda"]
        subset = [r["feature"] for r in gate_path_rows if r["stage"] == s and r["active"] == 1]
        for mode, cols in (("subset", subset), ("complement", [n for n in features if n not in subset])):
            if not cols:
                continue
            idx = [name_to_idx[c] for c in cols]
            best = train_plain_dnn(idx, bundle, hidden, lr, batch_size, seed, args.verify_epochs)
            verify_rows.append(
                {"stage": s, "lambda": lam, "mode": mode, "n_features": len(idx), "best_test_r2": round(best, 6)}
            )
            print(f"[verify stage {s} {mode}] n={len(idx)} best_r2={best:.4f}", flush=True)

    write_csv(out_dir / "verification.csv", verify_rows, list(verify_rows[0].keys()))
    print(f"Saved lambda-path v2 run to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
