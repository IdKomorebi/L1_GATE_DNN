from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import merged_training_params, resolve_project_path
from .data_utils import (
    center_output_dir,
    create_run_dir,
    ensure_dir,
    normalize_column_list,
    prepare_supervised_dataset,
    save_json,
    write_name_mapping,
)
from .knowledge_graph import generate_knowledge_graph
from .models import DNNRegressor, ImprovedGateRegressor, L1GateRegressor, SimpleAdam
from .plotting import plot_active_features, plot_gate_history, plot_gate_logit_history, plot_loss_and_r2, plot_meta_history
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
        meta_lr = float(params.get("meta_lr", params.get("lr", 1e-3)))
        return [
            {
                "name": "net",
                "params": model.net.parameters(),
                "lr": float(params.get("lr", 1e-3)),
                "weight_decay": float(params.get("net_weight_decay", 0.0)),
            },
            {
                "name": "w_meta",
                "params": [model.W_meta],
                "lr": float(params.get("w_meta_lr", meta_lr)),
                "weight_decay": float(params.get("w_meta_weight_decay", params.get("meta_weight_decay", 0.0))),
            },
            {
                "name": "b_meta",
                "params": [model.b_meta],
                "lr": float(params.get("b_meta_lr", meta_lr)),
                "weight_decay": float(params.get("b_meta_weight_decay", params.get("meta_weight_decay", 0.0))),
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
            meta_init_scale=float(params.get("meta_init_scale", 0.0)),
            b_meta_init=float(params.get("b_meta_init", 0.0)),
            gate_temperature=float(params.get("gate_temperature", 1.0)),
        )
    raise ValueError(f"Unknown model: {model_name}")


def _regularization_lambda(params: Dict[str, Any], key: str, epoch: int) -> float:
    target = float(params.get(key, 0.0))
    warmup = int(params.get("warmup_epochs", 0) or 0)
    if epoch <= warmup:
        return 0.0
    ramp_epochs = int(params.get("lambda_ramp_epochs", 0) or 0)
    if ramp_epochs <= 0:
        return target
    progress = min(max(epoch - warmup, 1), ramp_epochs) / ramp_epochs
    return target * progress


def _decay_b_meta_lr(optimizer: SimpleAdam, params: Dict[str, Any]) -> None:
    if bool(params.get("use_epoch_lr_schedule", False)):
        return
    decay = float(params.get("b_meta_lr_decay", 1.0) or 1.0)
    if decay <= 0 or decay >= 1.0:
        return
    for group in optimizer.param_groups:
        if group.get("name") == "b_meta":
            group["lr"] *= decay


def _initialize_lr_schedule(optimizer: SimpleAdam) -> None:
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group.get("lr", 0.0))


def _apply_warmup_meta_lr_scale(optimizer: SimpleAdam, params: Dict[str, Any], epoch: int) -> None:
    warmup = int(params.get("warmup_epochs", 0) or 0)
    scale = float(params.get("warmup_meta_lr_scale", 1.0) or 1.0)
    start_scale = float(params.get("warmup_meta_lr_start_scale", scale) or scale)
    ramp_epochs = int(params.get("warmup_meta_lr_ramp_epochs", 0) or 0)
    if scale <= 0 or start_scale <= 0:
        return
    b_decay = float(params.get("b_meta_lr_decay", 1.0) or 1.0)
    has_warmup_schedule = warmup > 0 and (scale != 1.0 or start_scale != 1.0 or ramp_epochs > 0)
    use_schedule = bool(params.get("use_epoch_lr_schedule", has_warmup_schedule))
    if not use_schedule:
        return
    if use_schedule:
        params["use_epoch_lr_schedule"] = True
    for group in optimizer.param_groups:
        name = group.get("name")
        if name not in {"w_meta", "b_meta"}:
            continue
        base_lr = float(group.get("base_lr", group.get("lr", 0.0)))
        lr = base_lr
        if warmup > 0 and epoch <= warmup:
            current_scale = scale
            if ramp_epochs > 0:
                progress = min(max(epoch - 1, 0), ramp_epochs) / ramp_epochs
                current_scale = start_scale + (scale - start_scale) * progress
            lr *= current_scale
        if use_schedule and name == "b_meta" and 0 < b_decay < 1.0:
            lr *= b_decay ** max(epoch - 1, 0)
        group["lr"] = lr


def _capture_gate_state(
    model: nn.Module,
    model_name: str,
    params: Dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None, int | None]:
    gates_np = None
    logits_np = None
    w_np = None
    b_value = None
    active_features = None
    if hasattr(model, "get_gates"):
        gates_np = model.get_gates().detach().cpu().numpy()  # type: ignore[attr-defined]
        active_features = _active_count(model_name, gates_np, params)
    if isinstance(model, ImprovedGateRegressor):
        logits_np = model.get_gate_logits().detach().cpu().numpy().copy()
        w_np = model.W_meta.detach().cpu().numpy().reshape(-1).copy()
        b_value = float(model.b_meta.detach().cpu().numpy().reshape(-1)[0])
    return gates_np, logits_np, w_np, b_value, active_features


def _loss_for_model(
    model_name: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    model: nn.Module,
    params: Dict[str, Any],
    epoch: int,
    high_gate_anchor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, float]:
    mse = nn.functional.mse_loss(pred, target)
    if model_name == "L1GateDNN":
        gate = model.get_gates()  # type: ignore[attr-defined]
        reg = torch.sum(torch.abs(gate))
        return mse + float(params.get("lambda_l1", 0.0)) * reg, float(mse.detach()), float(reg.detach())
    if model_name == "ImprovedL1GateDNN":
        gate = model.get_gates()  # type: ignore[attr-defined]
        reg = _improved_l1_gate_regularizer(gate, model, params)
        lam = _regularization_lambda(params, "lambda_l1", epoch)
        binary_beta = float(params.get("binary_gate_beta", 0.0) or 0.0)
        binary_reg = torch.mean(gate * (1.0 - gate)) if binary_beta > 0 else torch.zeros((), device=gate.device)
        anchor_reg = _high_gate_anchor_regularizer(gate, high_gate_anchor, params)
        return mse + lam * reg + binary_beta * binary_reg + anchor_reg, float(mse.detach()), float(reg.detach())
    if model_name == "ImprovedL2GateDNN":
        gate = model.get_gates()  # type: ignore[attr-defined]
        reg = torch.mean(gate**2)
        lam = _regularization_lambda(params, "lambda_l2", epoch)
        binary_beta = float(params.get("binary_gate_beta", 0.0) or 0.0)
        binary_reg = torch.mean(gate * (1.0 - gate)) if binary_beta > 0 else torch.zeros((), device=gate.device)
        anchor_reg = _high_gate_anchor_regularizer(gate, high_gate_anchor, params)
        return mse + lam * reg + binary_beta * binary_reg + anchor_reg, float(mse.detach()), float(reg.detach())
    return mse, float(mse.detach()), 0.0


def _improved_l1_gate_regularizer(gate: torch.Tensor, model: nn.Module, params: Dict[str, Any]) -> torch.Tensor:
    if not bool(params.get("adaptive_gate_l1", False)):
        return torch.mean(torch.abs(gate))
    if not hasattr(model, "get_gate_logits"):
        return torch.mean(torch.abs(gate))
    logits = model.get_gate_logits().detach()  # type: ignore[attr-defined]
    weight_min = float(params.get("adaptive_l1_min", 0.4))
    weight_max = float(params.get("adaptive_l1_max", 1.6))
    tau = float(params.get("adaptive_l1_tau", 0.0))
    sharpness = float(params.get("adaptive_l1_k", 2.0))
    weights = weight_min + (weight_max - weight_min) * torch.sigmoid(sharpness * (tau - logits))
    return torch.mean(weights * torch.abs(gate))


def _high_gate_anchor_regularizer(
    gate: torch.Tensor,
    high_gate_anchor: torch.Tensor | None,
    params: Dict[str, Any],
) -> torch.Tensor:
    beta = float(params.get("high_gate_anchor_beta", 0.0) or 0.0)
    if beta <= 0 or high_gate_anchor is None:
        return torch.zeros((), device=gate.device)
    threshold = float(params.get("high_gate_anchor_threshold", 0.65))
    margin = float(params.get("high_gate_anchor_margin", 0.08))
    anchor = high_gate_anchor.to(device=gate.device, dtype=gate.dtype)
    high_mask = anchor >= threshold
    if not bool(torch.any(high_mask)):
        return torch.zeros((), device=gate.device)
    drop = torch.relu(anchor[high_mask] - gate[high_mask] - margin)
    return beta * torch.mean(drop)


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


def _selected_features(
    model_name: str,
    features: List[str],
    gates: np.ndarray,
    params: Dict[str, Any],
    epoch: int | None = None,
    source: str | None = None,
) -> Dict[str, Any]:
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
    payload: Dict[str, Any] = {"threshold": threshold, "count": len(selected), "features": selected}
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if source is not None:
        payload["source"] = source
    return payload


def train_center_model(
    cfg: Dict[str, Any],
    center: str,
    model_name: str,
    overrides: Dict[str, Any] | None = None,
    run_name: str | None = None,
    force_relations: bool = False,
    exclude_columns: Sequence[str] | None = None,
    combo_name: str | None = None,
    output_run_dir: str | Path | None = None,
    feature_top_n: int | None = None,
) -> Path:
    dataset_cfg = cfg["dataset"]
    relation_cfg = cfg.get("relations", {})
    metrics = relation_cfg.get("metrics", [])
    thresholds = relation_cfg.get("thresholds", {})
    sample_size = int(relation_cfg.get("sample_size", 3000))
    expensive_sample_size = int(relation_cfg.get("expensive_sample_size", 1200))
    params = merged_training_params(cfg, model_name, overrides)
    preprocessing = cfg.get("preprocessing", {})
    drop_all_zero_columns = bool(preprocessing.get("drop_all_zero_columns", False))
    merged_exclude_columns = normalize_column_list(
        [*normalize_column_list(preprocessing.get("exclude_columns")), *normalize_column_list(exclude_columns)]
    )
    if center in merged_exclude_columns:
        raise ValueError(f"Center column cannot be excluded: {center}")

    data_path = resolve_project_path(cfg, dataset_cfg["processed_csv"])
    output_root = resolve_project_path(cfg, dataset_cfg["output_root"])
    center_dir = center_output_dir(output_root, center)
    center_rel_dir = center_relation_analysis_dir(
        output_root,
        center,
        metrics,
        thresholds,
        sample_size,
        expensive_sample_size,
        drop_all_zero_columns,
        merged_exclude_columns,
    )
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
            drop_all_zero_columns=drop_all_zero_columns,
            exclude_columns=merged_exclude_columns,
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

    feature_cfg = deepcopy(cfg.get("feature_selection", {}))
    if feature_top_n is not None:
        if int(feature_top_n) <= 0:
            raise ValueError("feature_top_n must be positive.")
        feature_cfg["top_n_by_sum"] = int(feature_top_n)
    features = select_features(center_rel, feature_cfg)
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
        drop_all_zero_columns=drop_all_zero_columns,
        exclude_columns=merged_exclude_columns,
    )

    torch.manual_seed(int(params.get("random_state", 42)))
    np.random.seed(int(params.get("random_state", 42)))

    train_ds = TensorDataset(torch.from_numpy(bundle.X_train), torch.from_numpy(bundle.y_train))
    test_ds = TensorDataset(torch.from_numpy(bundle.X_test), torch.from_numpy(bundle.y_test))
    train_loader = DataLoader(train_ds, batch_size=int(params.get("batch_size", 50)), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=int(params.get("batch_size", 50)), shuffle=False)

    model = _build_model(model_name, len(features), params, corr_vectors).to(DEVICE)
    optimizer = SimpleAdam(_model_params_for_optimizer(model, model_name, params), lr=float(params.get("lr", 1e-3)))
    _initialize_lr_schedule(optimizer)

    run_dir = ensure_dir(output_run_dir) if output_run_dir is not None else create_run_dir(center_dir, model_name, run_name)
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
            "combo_name": combo_name,
            "preprocessing": {
                "drop_all_zero_columns": drop_all_zero_columns,
                "exclude_columns": merged_exclude_columns,
            },
            "feature_selection": feature_cfg,
            "feature_top_n": feature_top_n,
            "features": features,
            "params": params,
        },
    )

    log_rows: List[Dict[str, Any]] = []
    gate_history: List[np.ndarray] = []
    gate_epochs: List[int] = []
    gate_logit_history: List[np.ndarray] = []
    gate_logit_epochs: List[int] = []
    w_history: List[np.ndarray] = []
    w_epochs: List[int] = []
    b_history: List[float] = []
    keydata: Dict[str, Any] = {}
    key_epochs = set(int(v) for v in params.get("key_epochs", []) if int(v) <= int(params.get("epochs", 1)))
    key_epochs.add(int(params.get("epochs", 1)))

    best_r2 = -1e18
    best_epoch = 0
    best_state: Dict[str, Any] | None = None
    best_gate_values: np.ndarray | None = None
    no_improve = 0
    min_delta = float(params.get("early_stopping_min_delta", 0.0) or 0.0)
    patience = int(params.get("early_stopping_patience", 0) or 0)
    warmup = int(params.get("warmup_epochs", 0) or 0)
    high_gate_anchor: torch.Tensor | None = None

    if bool(params.get("record_epoch0_gate", False)):
        gates_np, logits_np, w_np, b_value, active_features = _capture_gate_state(model, model_name, params)
        if gates_np is not None:
            gate_history.append(gates_np.copy())
            gate_epochs.append(0)
            keydata["0"] = _selected_features(model_name, features, gates_np, params, epoch=0, source="initial_state")
            log_rows.append(
                {
                    "epoch": 0,
                    "train_loss": np.nan,
                    "train_r2": np.nan,
                    "test_loss": np.nan,
                    "test_r2": np.nan,
                    "train_total_loss": np.nan,
                    "train_mse_loss": np.nan,
                    "train_reg_loss": np.nan,
                    "active_features": active_features,
                }
            )
        if logits_np is not None:
            gate_logit_history.append(logits_np.copy())
            gate_logit_epochs.append(0)
        if w_np is not None and b_value is not None:
            w_history.append(w_np.copy())
            w_epochs.append(0)
            b_history.append(b_value)

    for epoch in range(1, int(params.get("epochs", 200)) + 1):
        if model_name.startswith("Improved"):
            _apply_warmup_meta_lr_scale(optimizer, params, epoch)
        model.train()
        total_loss_sum = 0.0
        mse_sum = 0.0
        reg_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss, mse_value, reg_value = _loss_for_model(model_name, pred, yb, model, params, epoch, high_gate_anchor)
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
            gate_epochs.append(epoch)
            active_features = _active_count(model_name, gates_np, params)
            if epoch in key_epochs:
                keydata[str(epoch)] = _selected_features(model_name, features, gates_np, params, epoch=epoch, source="epoch_snapshot")
            if (
                model_name.startswith("Improved")
                and high_gate_anchor is None
                and float(params.get("high_gate_anchor_beta", 0.0) or 0.0) > 0
                and epoch >= warmup
            ):
                high_gate_anchor = model.get_gates().detach().clone()  # type: ignore[attr-defined]

        if isinstance(model, ImprovedGateRegressor):
            gate_logit_history.append(model.get_gate_logits().detach().cpu().numpy().copy())
            gate_logit_epochs.append(epoch)
            w_history.append(model.W_meta.detach().cpu().numpy().reshape(-1).copy())
            w_epochs.append(epoch)
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
            best_epoch = epoch
            no_improve = 0
            best_gate_values = gates_np.copy() if gates_np is not None else None
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "center": center,
                "features": features,
                "x_mean": bundle.x_mean,
                "x_std": bundle.x_std,
                "y_mean": bundle.y_mean,
                "y_std": bundle.y_std,
                "params": params,
                "preprocessing": {
                    "drop_all_zero_columns": drop_all_zero_columns,
                    "exclude_columns": merged_exclude_columns,
                },
            }
            if corr_vectors is not None:
                best_state["correlation_vectors"] = corr_vectors
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} train_r2={train_r2:.5f} | "
            f"test_loss={test_loss:.5f} test_r2={test_r2:.5f}"
        )

        if model_name.startswith("Improved"):
            _decay_b_meta_lr(optimizer, params)

        if patience > 0 and epoch > warmup and no_improve >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(run_dir / "log.csv", index=False, encoding="utf-8-sig")
    plot_loss_and_r2(log_df, run_dir / "loss.png", run_dir / "r2.png")

    if gate_history:
        gate_arr = np.asarray(gate_history)
        epochs = gate_epochs
        gate_long = []
        for e_idx, epoch in enumerate(epochs):
            for f_idx, feature in enumerate(features):
                gate_long.append({"epoch": epoch, "feature_index": f_idx + 1, "feature": feature, "gate": gate_arr[e_idx, f_idx]})
        pd.DataFrame(gate_long).to_csv(run_dir / "gate_params.csv", index=False, encoding="utf-8-sig")
        plot_gate_history(
            gate_arr,
            epochs,
            run_dir / "gate_params.png",
            feature_names=features,
            warmup_epoch=warmup,
            gate_threshold=float(params.get("active_threshold", params.get("risk_threshold", 0.5))),
        )
        plot_active_features(log_df, run_dir / "active_features.png")
        if keydata:
            save_json(run_dir / "keydata_for_pointdata.json", keydata)

    if gate_logit_history:
        logit_arr = np.asarray(gate_logit_history)
        epochs = gate_logit_epochs
        logit_long = []
        for e_idx, epoch in enumerate(epochs):
            for f_idx, feature in enumerate(features):
                logit_long.append(
                    {
                        "epoch": epoch,
                        "feature_index": f_idx + 1,
                        "feature": feature,
                        "gate_logit": logit_arr[e_idx, f_idx],
                    }
                )
        pd.DataFrame(logit_long).to_csv(run_dir / "gate_logits.csv", index=False, encoding="utf-8-sig")
        plot_gate_logit_history(logit_arr, epochs, run_dir / "gate_logits.png", feature_names=features)

    if w_history:
        w_arr = np.asarray(w_history)
        b_arr = np.asarray(b_history)
        pd.DataFrame(w_arr, columns=[f"W_{i + 1}" for i in range(w_arr.shape[1])]).assign(epoch=w_epochs).to_csv(
            run_dir / "W_meta_evolution.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame({"epoch": w_epochs, "b_meta": b_arr}).to_csv(run_dir / "b_meta_evolution.csv", index=False, encoding="utf-8-sig")
        plot_meta_history(w_arr, b_arr, w_epochs, run_dir / "W_meta_evolution.png", run_dir / "b_meta_evolution.png")

    if best_state is not None:
        torch.save(best_state, run_dir / "model.pth")

    if best_gate_values is not None:
        if gate_history:
            final_epoch = int(log_df["epoch"].iloc[-1])
            final_gate_values = np.asarray(gate_history[-1])
            save_json(
                run_dir / "selected_features.json",
                _selected_features(model_name, features, final_gate_values, params, epoch=final_epoch, source="final_epoch"),
            )
            save_json(
                run_dir / "best_epoch_selected_features.json",
                _selected_features(model_name, features, best_gate_values, params, epoch=best_epoch, source="best_test_r2_epoch"),
            )
        risk_df = center_rel.set_index("related").loc[features].reset_index()
        risk_df.insert(0, "feature_index", range(1, len(features) + 1))
        if gate_history:
            final_gate_values = np.asarray(gate_history[-1])
            risk_df["gate"] = final_gate_values
            risk_df["final_gate"] = final_gate_values
        risk_df["best_epoch_gate"] = best_gate_values
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
        "best_epoch": int(best_epoch),
        "last_epoch_train_r2": float(log_df["train_r2"].iloc[-1]),
        "last_epoch_test_r2": float(log_df["test_r2"].iloc[-1]),
        "final_train_r2": float(final_train_r2),
        "final_test_r2": float(final_test_r2),
        "epochs_completed": int(log_df["epoch"].max()),
    }
    save_json(run_dir / "metrics.json", metrics_out)
    return run_dir
