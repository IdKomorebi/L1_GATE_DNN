from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import merged_training_params, resolve_project_path
from .data_utils import (
    center_output_dir,
    create_run_dir,
    prepare_supervised_dataset,
    save_json,
    write_name_mapping,
)
from .knowledge_graph import generate_knowledge_graph
from .models import DNNRegressor, ImprovedGateRegressor, L1GateRegressor, SimpleAdam
from .plotting import plot_active_features, plot_gate_history, plot_loss_and_r2, plot_meta_history
from .relation_analyzer import analyze_center_relationships, center_relation_analysis_dir, correlation_vectors, select_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if float(ss_tot) <= 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def _model_params_for_optimizer(model: nn.Module, model_name: str, params: Dict[str, Any]):
    if model_name.startswith("Improved") and isinstance(model, ImprovedGateRegressor):
        return [
            {
                "params": model.net.parameters(),
                "lr": float(params.get("lr", 1e-3)),
                "weight_decay": float(params.get("net_weight_decay", 0.0)),
            },
            {
                "params": [model.W_meta, model.b_meta],
                "lr": float(params.get("meta_lr", params.get("lr", 1e-3))),
                "weight_decay": float(params.get("meta_weight_decay", 0.0)),
            },
        ]
    return model.parameters()


def _build_model(model_name: str, in_dim: int, params: Dict[str, Any], corr_vectors: np.ndarray | None) -> nn.Module:
    hidden_dims = params.get("hidden_dims", [64, 32, 16])
    if model_name == "DNN":
        return DNNRegressor(in_dim, hidden_dims)
    if model_name == "L1GateDNN":
        return L1GateRegressor(in_dim, hidden_dims)
    if model_name in {"ImprovedL1GateDNN", "ImprovedL2GateDNN"}:
        if corr_vectors is None:
            raise ValueError(f"{model_name} requires correlation vectors.")
        return ImprovedGateRegressor(
            in_dim,
            hidden_dims,
            correlation_vectors=corr_vectors,
            dropout=float(params.get("dropout", 0.0)),
        )
    raise ValueError(f"Unknown model: {model_name}")


def _loss_for_model(
    model_name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    model: nn.Module,
    params: Dict[str, Any],
    epoch: int,
) -> tuple[torch.Tensor, float, float]:
    mse = nn.functional.mse_loss(pred, target)
    warmup = int(params.get("warmup_epochs", 0) or 0)
    if model_name == "L1GateDNN":
        gate = model.get_gates()  # type: ignore[attr-defined]
        reg = torch.sum(torch.abs(gate))
        return mse + float(params.get("lambda_l1", 0.0)) * reg, float(mse.detach()), float(reg.detach())
    if model_name == "ImprovedL1GateDNN":
        gate = model.get_gates()  # type: ignore[attr-defined]
        reg = torch.mean(torch.abs(gate))
        lam = 0.0 if epoch <= warmup else float(params.get("lambda_l1", 0.0))
        return mse + lam * reg, float(mse.detach()), float(reg.detach())
    if model_name == "ImprovedL2GateDNN":
        gate = model.get_gates()  # type: ignore[attr-defined]
        reg = torch.mean(gate**2)
        lam = 0.0 if epoch <= warmup else float(params.get("lambda_l2", 0.0))
        return mse + lam * reg, float(mse.detach()), float(reg.detach())
    return mse, float(mse.detach()), 0.0


def _eval_model(model: nn.Module, loader: DataLoader) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            total_loss += float(loss.detach()) * xb.size(0)
            ys.append(yb.cpu())
            ps.append(pred.cpu())
    y_cat = torch.cat(ys, dim=0)
    p_cat = torch.cat(ps, dim=0)
    return total_loss / len(loader.dataset), r2_score(y_cat, p_cat), y_cat, p_cat


def _active_count(model_name: str, gates: np.ndarray, params: Dict[str, Any]) -> int:
    if model_name == "L1GateDNN":
        threshold = float(params.get("active_threshold", 0.06))
        return int(np.sum(np.abs(gates) > threshold))
    if model_name == "ImprovedL2GateDNN":
        threshold = float(params.get("risk_threshold", 0.5))
        return int(np.sum(gates >= threshold))
    threshold = float(params.get("active_threshold", 0.06))
    return int(np.sum(gates >= threshold))


def _selected_features(model_name: str, features: List[str], gates: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    if model_name == "L1GateDNN":
        threshold = float(params.get("active_threshold", 0.06))
        mask = np.abs(gates) > threshold
    elif model_name == "ImprovedL2GateDNN":
        threshold = float(params.get("risk_threshold", 0.5))
        mask = gates >= threshold
    else:
        threshold = float(params.get("active_threshold", 0.06))
        mask = gates >= threshold
    selected = [{"index": i + 1, "name": f, "gate": float(gates[i])} for i, f in enumerate(features) if mask[i]]
    return {"threshold": threshold, "count": len(selected), "features": selected}


def train_center_model(
    cfg: Dict[str, Any],
    center: str,
    model_name: str,
    overrides: Dict[str, Any] | None = None,
    run_name: str | None = None,
    force_relations: bool = False,
) -> Path:
    dataset_cfg = cfg["dataset"]
    relation_cfg = cfg.get("relations", {})
    metrics = relation_cfg.get("metrics", [])
    thresholds = relation_cfg.get("thresholds", {})
    sample_size = int(relation_cfg.get("sample_size", 3000))
    expensive_sample_size = int(relation_cfg.get("expensive_sample_size", 1200))
    params = merged_training_params(cfg, model_name, overrides)

    data_path = resolve_project_path(cfg, dataset_cfg["processed_csv"])
    output_root = resolve_project_path(cfg, dataset_cfg["output_root"])
    center_dir = center_output_dir(output_root, center)
    center_rel_dir = center_relation_analysis_dir(output_root, center, metrics, thresholds, sample_size, expensive_sample_size)
    center_rel_path = center_rel_dir / "center_relationships.csv"
    center_graph_path = center_rel_dir / "center_knowledge_graph.html"

    if force_relations or not center_rel_path.exists():
        center_rel = analyze_center_relationships(
            data_path=data_path,
            center=center,
            output_csv=center_rel_path,
            metrics=metrics,
            thresholds=thresholds,
            sample_size=sample_size,
            expensive_sample_size=expensive_sample_size,
            random_state=int(params.get("random_state", 42)),
            progress_every=10,
        )
    else:
        center_rel = pd.read_csv(center_rel_path)

    if force_relations or not center_graph_path.exists():
        generate_knowledge_graph(
            center_rel,
            center_graph_path,
            metrics=metrics,
            thresholds=thresholds,
            title=f"Center Relationship Graph - {center}",
            center=center,
        )

    features = select_features(center_rel, cfg.get("feature_selection", {}))
    if not features:
        raise ValueError(f"No features selected for center: {center}")

    corr_vectors = None
    if model_name.startswith("Improved"):
        corr_vectors = correlation_vectors(center_rel, features, metrics)

    bundle = prepare_supervised_dataset(
        data_path=data_path,
        center=center,
        features=features,
        train_ratio=float(params.get("train_ratio", 0.8)),
        random_state=int(params.get("random_state", 42)),
    )

    torch.manual_seed(int(params.get("random_state", 42)))
    np.random.seed(int(params.get("random_state", 42)))

    train_ds = TensorDataset(torch.from_numpy(bundle.X_train), torch.from_numpy(bundle.y_train))
    test_ds = TensorDataset(torch.from_numpy(bundle.X_test), torch.from_numpy(bundle.y_test))
    train_loader = DataLoader(train_ds, batch_size=int(params.get("batch_size", 50)), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=int(params.get("batch_size", 50)), shuffle=False)

    model = _build_model(model_name, len(features), params, corr_vectors).to(DEVICE)
    optimizer = SimpleAdam(_model_params_for_optimizer(model, model_name, params), lr=float(params.get("lr", 1e-3)))

    run_dir = create_run_dir(center_dir, model_name, run_name)
    write_name_mapping(run_dir / "name_mapping.csv", center, features)

    save_json(
        run_dir / "config.json",
        {
            "dataset": dataset_cfg.get("name"),
            "center": center,
            "model": model_name,
            "data_path": str(data_path),
            "center_relationships": str(center_rel_path),
            "relationship_analysis_dir": str(center_rel_dir),
            "features": features,
            "params": params,
        },
    )

    log_rows: List[Dict[str, Any]] = []
    gate_history: List[np.ndarray] = []
    w_history: List[np.ndarray] = []
    b_history: List[float] = []
    keydata: Dict[str, Any] = {}
    key_epochs = set(int(v) for v in params.get("key_epochs", []) if int(v) <= int(params.get("epochs", 1)))
    key_epochs.add(int(params.get("epochs", 1)))

    best_r2 = -1e18
    best_state: Dict[str, Any] | None = None
    best_gate_values: np.ndarray | None = None
    no_improve = 0
    min_delta = float(params.get("early_stopping_min_delta", 0.0) or 0.0)
    patience = int(params.get("early_stopping_patience", 0) or 0)
    warmup = int(params.get("warmup_epochs", 0) or 0)

    for epoch in range(1, int(params.get("epochs", 200)) + 1):
        model.train()
        total_loss_sum = 0.0
        mse_sum = 0.0
        reg_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss, mse_value, reg_value = _loss_for_model(model_name, pred, yb, model, params, epoch)
            loss.backward()
            optimizer.step()
            total_loss_sum += float(loss.detach()) * xb.size(0)
            mse_sum += mse_value * xb.size(0)
            reg_sum += reg_value * xb.size(0)

        train_loss, train_r2, _, _ = _eval_model(model, train_loader)
        test_loss, test_r2, _, _ = _eval_model(model, test_loader)

        gates_np = None
        active_features = None
        if hasattr(model, "get_gates"):
            gates_np = model.get_gates().detach().cpu().numpy()  # type: ignore[attr-defined]
            gate_history.append(gates_np.copy())
            active_features = _active_count(model_name, gates_np, params)
            if epoch in key_epochs:
                keydata[str(epoch)] = _selected_features(model_name, features, gates_np, params)

        if isinstance(model, ImprovedGateRegressor):
            w_history.append(model.W_meta.detach().cpu().numpy().reshape(-1).copy())
            b_history.append(float(model.b_meta.detach().cpu().numpy().reshape(-1)[0]))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_r2": train_r2,
            "test_loss": test_loss,
            "test_r2": test_r2,
            "train_total_loss": total_loss_sum / len(train_ds),
            "train_mse_loss": mse_sum / len(train_ds),
            "train_reg_loss": reg_sum / len(train_ds),
        }
        if active_features is not None:
            row["active_features"] = active_features
        log_rows.append(row)

        if test_r2 > best_r2 + min_delta:
            best_r2 = test_r2
            no_improve = 0
            best_gate_values = gates_np.copy() if gates_np is not None else None
            best_state = {
                "model_state": model.state_dict(),
                "center": center,
                "features": features,
                "x_mean": bundle.x_mean,
                "x_std": bundle.x_std,
                "y_mean": bundle.y_mean,
                "y_std": bundle.y_std,
                "params": params,
            }
            if corr_vectors is not None:
                best_state["correlation_vectors"] = corr_vectors
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} train_r2={train_r2:.5f} | "
            f"test_loss={test_loss:.5f} test_r2={test_r2:.5f}"
        )

        if patience > 0 and epoch > warmup and no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(run_dir / "log.csv", index=False, encoding="utf-8-sig")
    plot_loss_and_r2(log_df, run_dir / "loss.png", run_dir / "r2.png")

    if gate_history:
        gate_arr = np.asarray(gate_history)
        epochs = log_df["epoch"].tolist()
        gate_long = []
        for e_idx, epoch in enumerate(epochs):
            for f_idx, feature in enumerate(features):
                gate_long.append({"epoch": epoch, "feature_index": f_idx + 1, "feature": feature, "gate": gate_arr[e_idx, f_idx]})
        pd.DataFrame(gate_long).to_csv(run_dir / "gate_params.csv", index=False, encoding="utf-8-sig")
        plot_gate_history(gate_arr, epochs, run_dir / "gate_params.png", feature_names=features)
        plot_active_features(log_df, run_dir / "active_features.png")
        if keydata:
            save_json(run_dir / "keydata_for_pointdata.json", keydata)

    if w_history:
        w_arr = np.asarray(w_history)
        b_arr = np.asarray(b_history)
        pd.DataFrame(w_arr, columns=[f"W_{i + 1}" for i in range(w_arr.shape[1])]).assign(epoch=log_df["epoch"]).to_csv(
            run_dir / "W_meta_evolution.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame({"epoch": log_df["epoch"], "b_meta": b_arr}).to_csv(run_dir / "b_meta_evolution.csv", index=False, encoding="utf-8-sig")
        plot_meta_history(w_arr, b_arr, log_df["epoch"].tolist(), run_dir / "W_meta_evolution.png", run_dir / "b_meta_evolution.png")

    if best_state is not None:
        torch.save(best_state, run_dir / "model.pth")

    if best_gate_values is not None:
        save_json(run_dir / "selected_features.json", _selected_features(model_name, features, best_gate_values, params))
        risk_df = center_rel.set_index("related").loc[features].reset_index()
        risk_df.insert(0, "feature_index", range(1, len(features) + 1))
        risk_df["best_gate"] = best_gate_values
        risk_df.to_csv(run_dir / "risk_map.csv", index=False, encoding="utf-8-sig")

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    _, final_test_r2, y_test_std, pred_test_std = _eval_model(model, test_loader)
    _, final_train_r2, y_train_std, pred_train_std = _eval_model(model, train_loader)

    pred_rows = []
    for split, idxs, y_std, p_std in [
        ("train", bundle.train_idx, y_train_std.numpy(), pred_train_std.numpy()),
        ("test", bundle.test_idx, y_test_std.numpy(), pred_test_std.numpy()),
    ]:
        y_true = y_std.reshape(-1) * bundle.y_std + bundle.y_mean
        y_pred = p_std.reshape(-1) * bundle.y_std + bundle.y_mean
        for row_idx, actual, pred in zip(idxs, y_true, y_pred):
            pred_rows.append({"split": split, "row_index": int(row_idx), "y_true": float(actual), "y_pred": float(pred)})
    pd.DataFrame(pred_rows).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    metrics_out = {
        "center": center,
        "model": model_name,
        "run_dir": str(run_dir),
        "feature_count": len(features),
        "best_test_r2": float(best_r2),
        "final_train_r2": float(final_train_r2),
        "final_test_r2": float(final_test_r2),
        "epochs_completed": int(log_df["epoch"].max()),
    }
    save_json(run_dir / "metrics.json", metrics_out)
    return run_dir
