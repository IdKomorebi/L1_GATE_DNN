from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, merged_training_params, resolve_center_spec, resolve_project_path
from src.data_utils import center_output_dir, ensure_dir, normalize_column_list, prepare_supervised_dataset, read_numeric_csv, safe_name, save_json
from src.relation_analyzer import normalized_mutual_info
from src.models import DNNRegressor, SimpleAdam
from src.trainer import train_center_model


BASELINE_NAMES = ["L1GateDNN", "NMI", "Pearson", "Spearman", "Lasso", "ElasticNet", "RandomForest", "XGBoost"]
FULL_DNN_NAME = "DNN_AllFeatures"
RUN_CONTEXT_FILE = "run_context.json"


@dataclass
class DNNResult:
    center: str
    method: str
    feature_count: int
    best_test_r2: float
    best_train_r2: float
    min_test_loss: float
    min_train_loss: float
    final_test_r2: float
    final_train_r2: float
    final_test_loss: float
    final_train_loss: float
    best_epoch: int
    output_dir: str


def _choose_device(name: str) -> torch.device:
    name = str(name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA but torch.cuda.is_available() is false.")
    return torch.device(name)


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if float(ss_tot) <= 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


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
            loss = nn.functional.mse_loss(pred, yb)
            total_loss += float(loss.detach().cpu()) * xb.size(0)
            ys.append(yb.detach().cpu())
            preds.append(pred.detach().cpu())
    y_cat = torch.cat(ys, dim=0)
    p_cat = torch.cat(preds, dim=0)
    return total_loss / len(loader.dataset), _r2_score(y_cat, p_cat)


def _train_dnn(
    data_path: Path,
    center: str,
    features: List[str],
    params: Dict[str, Any],
    output_dir: Path,
    device: torch.device,
    drop_all_zero_columns: bool,
    exclude_columns: Sequence[str],
) -> DNNResult:
    if not features:
        raise ValueError(f"No features for {center}.")

    random_state = int(params.get("random_state", 42))
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_state)

    bundle = prepare_supervised_dataset(
        data_path=data_path,
        center=center,
        features=features,
        train_ratio=float(params.get("train_ratio", 0.8)),
        random_state=random_state,
        drop_all_zero_columns=drop_all_zero_columns,
        exclude_columns=exclude_columns,
    )
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.X_train), torch.from_numpy(bundle.y_train)),
        batch_size=int(params.get("batch_size", 50)),
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.X_test), torch.from_numpy(bundle.y_test)),
        batch_size=int(params.get("batch_size", 50)),
        shuffle=False,
    )

    model = DNNRegressor(len(features), [int(v) for v in params.get("hidden_dims", [64, 32, 16])]).to(device)
    optimizer = SimpleAdam(model.parameters(), lr=float(params.get("lr", 1e-3)))

    best_test_r2 = -math.inf
    best_epoch = 0
    best_state = None
    log_rows = []
    for epoch in range(1, int(params.get("epochs", 200)) + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * xb.size(0)

        train_loss, train_r2 = _eval_model(model, train_loader, device)
        test_loss, test_r2 = _eval_model(model, test_loader, device)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_total_loss": train_loss_sum / len(train_loader.dataset),
                "train_r2": train_r2,
                "test_loss": test_loss,
                "test_r2": test_r2,
            }
        )
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ensure_dir(output_dir)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(output_dir / "log.csv", index=False, encoding="utf-8-sig")
    save_json(output_dir / "used_features.json", {"center": center, "features": features})
    if best_state is not None:
        torch.save({"model_state": best_state, "center": center, "features": features}, output_dir / "model.pth")

    result = DNNResult(
        center=center,
        method=output_dir.name,
        feature_count=len(features),
        best_test_r2=float(best_test_r2),
        best_train_r2=float(log_df["train_r2"].max()),
        min_test_loss=float(log_df["test_loss"].min()),
        min_train_loss=float(log_df["train_loss"].min()),
        final_test_r2=float(log_df["test_r2"].iloc[-1]),
        final_train_r2=float(log_df["train_r2"].iloc[-1]),
        final_test_loss=float(log_df["test_loss"].iloc[-1]),
        final_train_loss=float(log_df["train_loss"].iloc[-1]),
        best_epoch=int(best_epoch),
        output_dir=str(output_dir),
    )
    save_json(output_dir / "metrics.json", result.__dict__)
    return result


def _ranking_frame(
    data_path: Path,
    center: str,
    drop_all_zero_columns: bool,
    exclude_columns: Sequence[str],
    train_ratio: float,
    random_state: int,
) -> tuple[List[str], np.ndarray, np.ndarray]:
    df = read_numeric_csv(data_path, drop_all_zero_columns=drop_all_zero_columns, exclude_columns=exclude_columns)
    if center not in df.columns:
        raise ValueError(f"Center column not found: {center}")
    features = [str(c) for c in df.columns if str(c) != center]
    clean = df[[center, *features]].dropna(axis=0, how="any")
    if len(clean) < 5:
        raise ValueError(f"Not enough usable rows for {center}.")

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(clean))
    train_size = max(1, min(len(clean) - 1, int(len(clean) * train_ratio)))
    train = clean.iloc[perm[:train_size]]
    X = train[features].to_numpy(dtype=np.float64)
    y = train[center].to_numpy(dtype=np.float64)
    return features, X, y


def _score_abs_correlation(features: List[str], X: np.ndarray, y: np.ndarray, method: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if method == "Pearson":
        for idx, feature in enumerate(features):
            x = X[:, idx]
            if np.std(x) <= 0 or np.std(y) <= 0:
                scores[feature] = 0.0
            else:
                scores[feature] = abs(float(np.corrcoef(x, y)[0, 1]))
    elif method == "Spearman":
        ranked_x = pd.DataFrame(X, columns=features).rank(method="average").to_numpy(dtype=float)
        ranked_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
        return _score_abs_correlation(features, ranked_x, ranked_y, "Pearson")
    else:
        raise ValueError(method)
    return scores


def _score_nmi(features: List[str], X: np.ndarray, y: np.ndarray, random_state: int) -> Dict[str, float]:
    return {feature: float(normalized_mutual_info(X[:, idx], y)) for idx, feature in enumerate(features)}


def _sklearn_imports() -> Dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import ElasticNet, Lasso
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise ImportError(
            "Lasso, ElasticNet and RandomForest baselines require scikit-learn/scipy. "
            "The current environment cannot import them. Try reinstalling scipy and scikit-learn in Pytorch310."
        ) from exc
    return {
        "RandomForestRegressor": RandomForestRegressor,
        "ElasticNet": ElasticNet,
        "Lasso": Lasso,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
    }


def _score_linear_model(features: List[str], X: np.ndarray, y: np.ndarray, method: str, baseline_cfg: Dict[str, Any]) -> Dict[str, float]:
    imports = _sklearn_imports()
    Lasso = imports["Lasso"]
    ElasticNet = imports["ElasticNet"]
    make_pipeline = imports["make_pipeline"]
    StandardScaler = imports["StandardScaler"]
    if method == "Lasso":
        model = Lasso(
            alpha=float(baseline_cfg.get("lasso_alpha", 0.001)),
            max_iter=int(baseline_cfg.get("linear_max_iter", 20000)),
            random_state=int(baseline_cfg.get("random_state", 42)),
        )
    elif method == "ElasticNet":
        model = ElasticNet(
            alpha=float(baseline_cfg.get("elasticnet_alpha", 0.001)),
            l1_ratio=float(baseline_cfg.get("elasticnet_l1_ratio", 0.5)),
            max_iter=int(baseline_cfg.get("linear_max_iter", 20000)),
            random_state=int(baseline_cfg.get("random_state", 42)),
        )
    else:
        raise ValueError(method)
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(X, y)
    coefs = pipe.named_steps[type(model).__name__.lower()].coef_
    return {feature: abs(float(coefs[idx])) for idx, feature in enumerate(features)}


def _score_random_forest(features: List[str], X: np.ndarray, y: np.ndarray, baseline_cfg: Dict[str, Any]) -> Dict[str, float]:
    RandomForestRegressor = _sklearn_imports()["RandomForestRegressor"]
    model = RandomForestRegressor(
        n_estimators=int(baseline_cfg.get("rf_n_estimators", 300)),
        max_depth=baseline_cfg.get("rf_max_depth"),
        min_samples_leaf=int(baseline_cfg.get("rf_min_samples_leaf", 1)),
        n_jobs=int(baseline_cfg.get("n_jobs", -1)),
        random_state=int(baseline_cfg.get("random_state", 42)),
    )
    model.fit(X, y)
    return {feature: float(model.feature_importances_[idx]) for idx, feature in enumerate(features)}


def _score_xgboost(features: List[str], X: np.ndarray, y: np.ndarray, baseline_cfg: Dict[str, Any]) -> Dict[str, float]:
    try:
        import xgboost as xgb
    except Exception as exc:
        raise ImportError("XGBoost baseline requires xgboost in the active Python environment.") from exc

    dtrain = xgb.DMatrix(X, label=y, feature_names=features)
    params = {
        "objective": "reg:squarederror",
        "max_depth": int(baseline_cfg.get("xgb_max_depth", 4)),
        "eta": float(baseline_cfg.get("xgb_learning_rate", 0.05)),
        "subsample": float(baseline_cfg.get("xgb_subsample", 0.9)),
        "colsample_bytree": float(baseline_cfg.get("xgb_colsample_bytree", 0.9)),
        "seed": int(baseline_cfg.get("random_state", 42)),
        "nthread": int(baseline_cfg.get("n_jobs", -1)),
        "verbosity": 0,
    }
    booster = xgb.train(params, dtrain, num_boost_round=int(baseline_cfg.get("xgb_n_estimators", 300)), verbose_eval=False)
    score = booster.get_score(importance_type=str(baseline_cfg.get("xgb_importance_type", "gain")))
    return {feature: float(score.get(feature, 0.0)) for feature in features}


def _top_features_from_scores(scores: Dict[str, float], n: int) -> List[str]:
    return [
        feature
        for feature, _ in sorted(
            scores.items(),
            key=lambda item: (-abs(float(item[1])) if np.isfinite(item[1]) else 0.0, item[0]),
        )[:n]
    ]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _preprocessing_matches(run_cfg: Dict[str, Any], data_path: Path, drop_all_zero_columns: bool, exclude_columns: Sequence[str]) -> bool:
    if Path(run_cfg.get("data_path", "")).resolve() != data_path.resolve():
        return False
    prep = run_cfg.get("preprocessing") or {}
    return bool(prep.get("drop_all_zero_columns", False)) == drop_all_zero_columns and normalize_column_list(prep.get("exclude_columns")) == list(exclude_columns)


def _latest_l1_run(output_root: Path, center: str, data_path: Path, drop_all_zero_columns: bool, exclude_columns: Sequence[str]) -> Path | None:
    base = center_output_dir(output_root, center) / "L1GateDNN"
    if not base.exists():
        return None
    candidates = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in candidates:
        config_path = run_dir / "config.json"
        selected_path = run_dir / "selected_features.json"
        if not config_path.exists() or not selected_path.exists():
            continue
        try:
            run_cfg = _load_json(config_path)
        except Exception:
            continue
        if run_cfg.get("center") == center and run_cfg.get("model") == "L1GateDNN" and _preprocessing_matches(
            run_cfg, data_path, drop_all_zero_columns, exclude_columns
        ):
            return run_dir
    return None


def _l1_ranked_features(run_dir: Path) -> List[str]:
    risk_path = run_dir / "risk_map.csv"
    if risk_path.exists():
        df = pd.read_csv(risk_path)
        gate_col = next((col for col in ["final_gate", "gate", "best_epoch_gate", "best_gate"] if col in df.columns), None)
        if "related" in df.columns and gate_col is not None:
            df = df.copy()
            df["abs_gate"] = df[gate_col].abs()
            return [str(v) for v in df.sort_values("abs_gate", ascending=False)["related"].tolist()]

    selected_path = run_dir / "selected_features.json"
    payload = _load_json(selected_path)
    rows = payload.get("features", [])
    return [str(row["name"]) for row in sorted(rows, key=lambda row: -abs(float(row.get("gate", 0.0))))]


def _ensure_l1_run(
    cfg: Dict[str, Any],
    center: str,
    combo_name: str | None,
    output_root: Path,
    data_path: Path,
    drop_all_zero_columns: bool,
    exclude_columns: Sequence[str],
    baseline_cfg: Dict[str, Any],
    output_run_dir: Path,
) -> Path:
    local_config_path = output_run_dir / "config.json"
    local_selected_path = output_run_dir / "selected_features.json"
    if bool(baseline_cfg.get("resume_l1_sources", True)) and local_config_path.exists() and local_selected_path.exists():
        try:
            run_cfg = _load_json(local_config_path)
        except Exception:
            run_cfg = {}
        if run_cfg.get("center") == center and run_cfg.get("model") == "L1GateDNN" and _preprocessing_matches(
            run_cfg, data_path, drop_all_zero_columns, exclude_columns
        ):
            print(f"  L1GateDNN source: {output_run_dir}")
            return output_run_dir

    if bool(baseline_cfg.get("reuse_l1_runs", False)):
        existing = _latest_l1_run(output_root, center, data_path, drop_all_zero_columns, exclude_columns)
        if existing is not None:
            print(f"  L1GateDNN source: {existing}")
            return existing

    if not bool(baseline_cfg.get("train_l1_if_missing", True)):
        raise FileNotFoundError(f"No matching L1GateDNN run found for center={center}")

    overrides = baseline_cfg.get("l1_training_overrides") or {}
    print(f"  Training L1GateDNN source for {center}")
    run_tag = safe_name(combo_name or center, 80)
    train_kwargs = dict(
        center=center,
        model_name="L1GateDNN",
        overrides=overrides,
        run_name=f"baseline_l1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_tag}",
        force_relations=bool(baseline_cfg.get("force_l1_relations", False)),
        exclude_columns=exclude_columns,
        combo_name=combo_name,
        output_run_dir=output_run_dir,
    )
    if bool(baseline_cfg.get("quiet_l1_training", True)):
        with contextlib.redirect_stdout(io.StringIO()):
            run_dir = train_center_model(cfg, **train_kwargs)
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            metrics = _load_json(metrics_path)
            print(f"  L1GateDNN source trained: best_test_r2={float(metrics.get('best_test_r2', np.nan)):.6f}")
        return run_dir
    return train_center_model(cfg, **train_kwargs)


def _baseline_features(
    method: str,
    n_features: int,
    l1_run_dir: Path,
    all_features: List[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    baseline_cfg: Dict[str, Any],
) -> tuple[List[str], Dict[str, float]]:
    if method == "L1GateDNN":
        ranked = [feature for feature in _l1_ranked_features(l1_run_dir) if feature in set(all_features)]
        return ranked[:n_features], {feature: float(len(ranked) - idx) for idx, feature in enumerate(ranked)}
    if method == "NMI":
        scores = _score_nmi(all_features, X_train, y_train, random_state)
    elif method in {"Pearson", "Spearman"}:
        scores = _score_abs_correlation(all_features, X_train, y_train, method)
    elif method in {"Lasso", "ElasticNet"}:
        scores = _score_linear_model(all_features, X_train, y_train, method, baseline_cfg)
    elif method == "RandomForest":
        scores = _score_random_forest(all_features, X_train, y_train, baseline_cfg)
    elif method == "XGBoost":
        scores = _score_xgboost(all_features, X_train, y_train, baseline_cfg)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    return _top_features_from_scores(scores, n_features), scores


def _row_from_metrics(
    metrics: Dict[str, Any],
    method: str,
    output_dir: Path,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    row = dict(context)
    row.update(
        {
            "method": method,
            "feature_count": int(metrics.get("feature_count", 0)),
            "best_test_r2": float(metrics.get("best_test_r2", np.nan)),
            "best_train_r2": float(metrics.get("best_train_r2", np.nan)),
            "min_test_loss": float(metrics.get("min_test_loss", np.nan)),
            "min_train_loss": float(metrics.get("min_train_loss", np.nan)),
            "final_test_r2": float(metrics.get("final_test_r2", np.nan)),
            "final_train_r2": float(metrics.get("final_train_r2", np.nan)),
            "final_test_loss": float(metrics.get("final_test_loss", np.nan)),
            "final_train_loss": float(metrics.get("final_train_loss", np.nan)),
            "best_epoch": int(metrics.get("best_epoch", 0)),
            "output_dir": str(output_dir),
        }
    )
    return row


def _row_from_result(result: DNNResult, context: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(context)
    row.update(result.__dict__)
    return row


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dnn_run_context(
    *,
    data_path: Path,
    center: str,
    method: str,
    features: Sequence[str],
    params: Dict[str, Any],
    drop_all_zero_columns: bool,
    exclude_columns: Sequence[str],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context = {
        "data_path": str(Path(data_path).resolve()),
        "center": center,
        "method": method,
        "features": list(features),
        "training_params": params,
        "preprocessing": {
            "drop_all_zero_columns": bool(drop_all_zero_columns),
            "exclude_columns": normalize_column_list(exclude_columns),
        },
    }
    if extra:
        context.update(extra)
    return context


def _can_reuse_dnn(output_dir: Path, expected_context: Dict[str, Any]) -> bool:
    if not (output_dir / "metrics.json").exists() or not (output_dir / "used_features.json").exists():
        return False
    context_path = output_dir / RUN_CONTEXT_FILE
    if not context_path.exists():
        return False
    try:
        current_context = _load_json(context_path)
    except Exception:
        return False
    return _stable_json(current_context) == _stable_json(expected_context)


def _run_remaining_validation(
    *,
    method: str,
    selected_features: Sequence[str],
    all_features: Sequence[str],
    center_dir: Path,
    data_path: Path,
    center: str,
    dnn_params: Dict[str, Any],
    device: torch.device,
    drop_all_zero_columns: bool,
    exclude_columns: Sequence[str],
    skip_existing: bool,
    context: Dict[str, Any],
) -> Dict[str, Any] | None:
    selected_set = set(selected_features)
    remaining_features = [feature for feature in all_features if feature not in selected_set]
    if not remaining_features:
        return None

    output_dir = ensure_dir(center_dir / "RemainingValidation" / safe_name(method))
    metrics_path = output_dir / "metrics.json"
    run_context = _dnn_run_context(
        data_path=data_path,
        center=center,
        method=f"RemainingValidation::{method}",
        features=remaining_features,
        params=dnn_params,
        drop_all_zero_columns=drop_all_zero_columns,
        exclude_columns=exclude_columns,
        extra={"removed_features": list(selected_features)},
    )
    if skip_existing and _can_reuse_dnn(output_dir, run_context):
        result_row = _row_from_metrics(_load_json(metrics_path), method, output_dir, context)
    else:
        save_json(
            output_dir / "removed_features.json",
            {
                "center": center,
                "method": method,
                "removed_features": list(selected_features),
                "remaining_features": remaining_features,
            },
        )
        result = _train_dnn(
            data_path=data_path,
            center=center,
            features=remaining_features,
            params=dnn_params,
            output_dir=output_dir,
            device=device,
                drop_all_zero_columns=drop_all_zero_columns,
                exclude_columns=exclude_columns,
            )
        save_json(output_dir / RUN_CONTEXT_FILE, run_context)
        result_row = _row_from_result(result, context)

    result_row["method"] = method
    result_row["removed_feature_count"] = len(selected_features)
    result_row["remaining_feature_count"] = len(remaining_features)
    result_row["removed_features"] = "|".join(str(v) for v in selected_features)
    return result_row


def _plot_summary(
    summary_df: pd.DataFrame,
    methods: Sequence[str],
    output_path: Path,
    main_method: str = "L1GateDNN",
    title: str = r"Baseline Comparison by Test $R^2$",
    ylabel: str = r"Best Test $R^2$",
) -> None:
    x_col = "plot_label" if "plot_label" in summary_df.columns else ("target_label" if "target_label" in summary_df.columns else "center")
    pivot = summary_df.pivot(index=x_col, columns="method", values="best_test_r2")
    labels = summary_df[x_col].drop_duplicates().tolist()
    x = np.arange(len(labels))

    fig_width = max(14, len(labels) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))
    for method in methods:
        if method not in pivot.columns:
            continue
        values = pivot.reindex(labels)[method].to_numpy(dtype=float)
        is_main = method == main_method
        is_full = method == FULL_DNN_NAME
        ax.plot(
            x,
            values,
            marker="o",
            linestyle="--" if is_full else "-",
            color="black" if is_full else None,
            linewidth=3.2 if is_main else (2.4 if is_full else 1.6),
            markersize=6 if is_main else (5 if is_full else 3.5),
            alpha=1.0 if is_main else (0.88 if is_full else 0.58),
            zorder=5 if is_main else (4 if is_full else 2),
            label=method,
        )

    ax.set_title(title)
    ax.set_xlabel("Center Target")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _resolve_methods(value: Any) -> List[str]:
    values = normalize_column_list(value)
    if not values or any(v.lower() == "all" for v in values):
        return BASELINE_NAMES.copy()
    invalid = [v for v in values if v not in BASELINE_NAMES]
    if invalid:
        raise ValueError(f"Unknown baselines: {invalid}. Available: {BASELINE_NAMES}")
    return values


def _positive_int_or_none(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    out = int(value)
    if out <= 0:
        raise ValueError("Feature count overrides must be positive.")
    return out


def _raw_target_items(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, int)):
        return [{"value": str(value)}]
    items: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            raw_value = item.get("combo", item.get("target", item.get("center", item.get("value"))))
            if raw_value is None:
                raise ValueError(f"Baseline center item is missing center/combo/value: {item}")
            cfg = dict(item)
            cfg["value"] = str(raw_value)
            items.append(cfg)
        else:
            items.append({"value": str(item)})
    return items


def _target_override_keys(target: Dict[str, Any], raw_value: str) -> List[str]:
    keys = [
        raw_value,
        str(target.get("label", "")),
        str(target.get("center", "")),
        str(target.get("name", "")),
        str(target.get("id", "")),
    ]
    if target.get("kind") == "combo":
        keys.append(f"combo{target['id']}_{target['name']}")
    seen = set()
    return [key for key in keys if key and not (key in seen or seen.add(key))]


def _resolve_targets(cfg: Dict[str, Any], baseline_cfg: Dict[str, Any], centers_value: Any) -> List[Dict[str, Any]]:
    overrides = baseline_cfg.get("target_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("baseline_comparison.target_overrides must be a mapping.")

    targets = []
    for item in _raw_target_items(centers_value):
        raw_value = str(item["value"])
        target = resolve_center_spec(cfg, raw_value)
        inline_exclude = normalize_column_list(item.get("exclude_columns") or item.get("exclude"))
        inline_n = item.get("n_features", item.get("feature_count", item.get("n")))

        override_cfg: Dict[str, Any] = {}
        for key in _target_override_keys(target, raw_value):
            if isinstance(overrides.get(key), dict):
                override_cfg = dict(overrides[key])
                break

        override_exclude = normalize_column_list(override_cfg.get("exclude_columns") or override_cfg.get("exclude"))
        override_n = override_cfg.get("n_features", override_cfg.get("feature_count", override_cfg.get("n", inline_n)))
        target["raw_value"] = raw_value
        target["baseline_exclude_columns"] = normalize_column_list([*inline_exclude, *override_exclude])
        target["n_features_override"] = _positive_int_or_none(override_n)
        target["override_note"] = str(override_cfg.get("note", item.get("note", "")) or "")
        targets.append(target)
    return targets


def _feature_count(
    l1_count: int,
    candidate_count: int,
    min_features: int,
    max_features: int | None,
    fixed_feature_count: int | None,
    target_n_features: int | None = None,
) -> int:
    if target_n_features is not None:
        return max(1, min(int(target_n_features), candidate_count))
    if fixed_feature_count is not None:
        return max(1, min(int(fixed_feature_count), candidate_count))
    n = max(int(l1_count), int(min_features))
    if max_features is not None:
        n = min(n, int(max_features))
    return max(1, min(n, candidate_count))


def _line_data(log_path: Path) -> Dict[str, List[float]]:
    if not log_path.exists():
        return {}
    df = pd.read_csv(log_path)
    out = {"epoch": [int(v) for v in df["epoch"].tolist()]}
    for col in ["train_loss", "test_loss", "train_r2", "test_r2"]:
        if col in df.columns:
            out[col] = [float(v) for v in df[col].tolist()]
    return out


def _write_html_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    methods: Sequence[str],
    main_method: str = "L1GateDNN",
    remaining_df: pd.DataFrame | None = None,
) -> Path:
    payload = {
        "summary": summary_df.to_dict(orient="records"),
        "remaining": [] if remaining_df is None or remaining_df.empty else remaining_df.to_dict(orient="records"),
        "methods": list(methods),
        "main_method": main_method,
        "full_method": FULL_DNN_NAME,
        "logs": {},
    }
    for row in payload["summary"]:
        key = f"{row.get('target_label') or row['center']}::{row['method']}"
        payload["logs"][key] = _line_data(Path(row["output_dir"]) / "log.csv")

    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Baseline Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #222; }
    .controls { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    select { min-width: 220px; min-height: 34px; }
    canvas { width: 100%; height: 360px; border: 1px solid #ddd; margin: 10px 0 22px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
    th { background: #f4f4f4; }
  </style>
</head>
<body>
  <h2>Baseline Report</h2>
  <div class="controls">
    <label>Center<br><select id="center"></select></label>
    <label>Methods<br><select id="methods" multiple size="8"></select></label>
    <label>Metric<br><select id="metric">
      <option value="test_r2">test R²</option>
      <option value="train_r2">train R²</option>
      <option value="test_loss">test loss</option>
      <option value="train_loss">train loss</option>
    </select></label>
  </div>
  <h3>Single Center Curves</h3>
  <canvas id="curve" width="1200" height="380"></canvas>
  <h3>Across Centers</h3>
  <canvas id="summary" width="1200" height="380"></canvas>
  <h3>Remaining-Feature Validation</h3>
  <canvas id="remaining" width="1200" height="380"></canvas>
  <h3>Summary</h3>
  <div id="table"></div>
  <h3>Remaining Summary</h3>
  <div id="remainingTable"></div>
<script>
const DATA = __DATA__;
const colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f"];
const targets = [];
DATA.summary.forEach(r => {
  const key = r.target_label || r.center;
  if (!targets.some(t => t.key === key)) targets.push({key, label: r.plot_label || key});
});
const centers = targets.map(t => t.key);
const centerLabels = targets.map(t => t.label);
const centerSel = document.getElementById("center");
const methodSel = document.getElementById("methods");
const metricSel = document.getElementById("metric");
targets.forEach(t => centerSel.add(new Option(t.label, t.key)));
DATA.methods.forEach((m,i) => { const o = new Option(m, m); o.selected = true; methodSel.add(o); });
function selectedMethods(){ return [...methodSel.selectedOptions].map(o => o.value); }
function isMain(m){ return m === DATA.main_method; }
function isFull(m){ return m === DATA.full_method; }
function drawLine(canvas, series, xLabels, yLabel){
  const ctx = canvas.getContext("2d"); ctx.clearRect(0,0,canvas.width,canvas.height);
  const pad = {l:70,r:25,t:30,b:70}, w=canvas.width-pad.l-pad.r, h=canvas.height-pad.t-pad.b;
  const vals = series.flatMap(s => s.y).filter(Number.isFinite); if(!vals.length) return;
  let min=Math.min(...vals), max=Math.max(...vals); if(min===max){ min-=0.01; max+=0.01; }
  const y = v => pad.t + h - (v-min)/(max-min)*h; const x = i => pad.l + (xLabels.length<=1 ? w/2 : i*w/(xLabels.length-1));
  ctx.strokeStyle="#ddd"; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+h); ctx.lineTo(pad.l+w,pad.t+h); ctx.stroke();
  ctx.fillStyle="#333"; ctx.font="13px Arial"; ctx.fillText(yLabel, 10, 20); ctx.fillText(max.toFixed(4), 8, pad.t+5); ctx.fillText(min.toFixed(4), 8, pad.t+h);
  xLabels.forEach((lab,i)=>{ if(i % Math.ceil(xLabels.length/12)===0){ ctx.save(); ctx.translate(x(i), pad.t+h+16); ctx.rotate(-0.55); ctx.fillText(String(lab),0,0); ctx.restore(); }});
  series.forEach((s,si)=>{ ctx.strokeStyle=s.full?"#000":colors[si%colors.length]; ctx.lineWidth=s.main?4:(s.full?3:2); ctx.globalAlpha=s.main?1:(s.full?0.9:0.65); ctx.setLineDash(s.full?[8,5]:[]); ctx.beginPath(); s.y.forEach((v,i)=>{ if(i===0) ctx.moveTo(x(i), y(v)); else ctx.lineTo(x(i), y(v)); }); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha=1; ctx.fillText(s.name, pad.l+8, pad.t+18+si*16); });
}
function update(){
  const c = centerSel.value, ms = selectedMethods(), metric = metricSel.value;
  const curveSeries = ms.map(m => { const key = `${c}::${m}`; const log = DATA.logs[key] || {}; return {name:m, y:log[metric]||[], main:isMain(m), full:isFull(m)}; }).filter(s=>s.y.length);
  const epochs = curveSeries[0] ? (DATA.logs[`${c}::${curveSeries[0].name}`].epoch || []) : [];
  drawLine(document.getElementById("curve"), curveSeries, epochs, metric);
  const summaryMetric = {test_r2:"best_test_r2", train_r2:"best_train_r2", test_loss:"min_test_loss", train_loss:"min_train_loss"}[metric] || "best_test_r2";
  const summaryLabel = {test_r2:"best test R²", train_r2:"best train R²", test_loss:"min test loss", train_loss:"min train loss"}[metric] || summaryMetric;
  const summarySeries = ms.map(m => ({name:m, main:isMain(m), full:isFull(m), y:centers.map(cn => { const r=DATA.summary.find(r=>(r.target_label||r.center)===cn && r.method===m); return r ? Number(r[summaryMetric]) : NaN; })}));
  drawLine(document.getElementById("summary"), summarySeries, centerLabels, summaryLabel);
  const remMethods = ms.filter(m => !isFull(m));
  const remSeries = remMethods.map(m => ({name:m, main:isMain(m), full:false, y:centers.map(cn => { const r=DATA.remaining.find(r=>(r.target_label||r.center)===cn && r.method===m); return r ? Number(r.best_test_r2) : NaN; })}));
  drawLine(document.getElementById("remaining"), remSeries, centerLabels, "remaining best test R²");
}
function table(){
  const rows = DATA.summary.map(r => `<tr><td>${r.plot_label||r.target_label||r.center}</td><td>${r.method}</td><td>${r.feature_count}</td><td>${Number(r.best_test_r2).toFixed(6)}</td><td>${Number(r.min_test_loss).toExponential(3)}</td><td>${r.best_epoch}</td></tr>`).join("");
  document.getElementById("table").innerHTML = `<table><tr><th>Center</th><th>Method</th><th>X count</th><th>Best test R²</th><th>Min test loss</th><th>Best epoch</th></tr>${rows}</table>`;
  const remRows = DATA.remaining.map(r => `<tr><td>${r.plot_label||r.target_label||r.center}</td><td>${r.method}</td><td>${r.removed_feature_count}</td><td>${r.remaining_feature_count}</td><td>${Number(r.best_test_r2).toFixed(6)}</td><td>${r.best_epoch}</td></tr>`).join("");
  document.getElementById("remainingTable").innerHTML = `<table><tr><th>Center</th><th>Method</th><th>Removed Top-n</th><th>Remaining X</th><th>Best test R²</th><th>Best epoch</th></tr>${remRows}</table>`;
}
centerSel.onchange=update; methodSel.onchange=update; metricSel.onchange=update; table(); update();
</script>
</body>
</html>"""
    path = output_dir / "baseline_report.html"
    path.write_text(html.replace("__DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    return path


def run_baselines(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    baseline_cfg = cfg.get("baseline_comparison", {})
    dataset_cfg = cfg["dataset"]
    preprocessing = cfg.get("preprocessing", {})
    drop_all_zero_columns = bool(preprocessing.get("drop_all_zero_columns", False))
    exclude_columns = normalize_column_list(preprocessing.get("exclude_columns"))

    centers_value = args.centers if args.centers else baseline_cfg.get("centers")
    if not centers_value:
        raise ValueError("No centers configured. Set baseline_comparison.centers or pass --centers.")
    targets = _resolve_targets(cfg, baseline_cfg, centers_value)
    methods = _resolve_methods(args.baselines or baseline_cfg.get("baselines", "all"))
    min_features = int(args.min_features or baseline_cfg.get("min_features", 3))
    if min_features <= 0:
        raise ValueError("min_features must be positive.")
    max_features_value = args.max_features if args.max_features is not None else baseline_cfg.get("max_features")
    max_features = None if max_features_value in {None, ""} else int(max_features_value)
    fixed_value = args.fixed_feature_count if args.fixed_feature_count is not None else baseline_cfg.get("fixed_feature_count")
    fixed_feature_count = None if fixed_value in {None, ""} else int(fixed_value)
    if max_features is not None and max_features < min_features and fixed_feature_count is None:
        raise ValueError("baseline_comparison.max_features must be >= min_features unless fixed_feature_count is set.")

    data_path = resolve_project_path(cfg, dataset_cfg["processed_csv"])
    output_root = resolve_project_path(cfg, dataset_cfg["output_root"])
    if args.check_only:
        all_excludes = sorted(
            {
                col
                for target in targets
                for col in [
                    *exclude_columns,
                    *normalize_column_list(target["exclude_columns"]),
                    *normalize_column_list(target.get("baseline_exclude_columns")),
                ]
            }
        )
        columns = set(read_numeric_csv(data_path, drop_all_zero_columns=drop_all_zero_columns, exclude_columns=[]).columns)
        missing = [target["center"] for target in targets if target["center"] not in columns]
        if missing:
            raise ValueError(f"Center columns not found in data: {missing}")
        print(
            f"check_only_ok: centers={len(targets)}, baselines={methods}, "
            f"min_features={min_features}, max_features={max_features}, fixed_feature_count={fixed_feature_count}"
        )
        print(f"data_path={data_path}")
        print(f"output_root={output_root}")
        for target in targets:
            print(
                f"  target={target['label']} center={target['center']} "
                f"n_override={target.get('n_features_override')} "
                f"extra_exclude={normalize_column_list(target.get('baseline_exclude_columns'))}"
            )
        if all_excludes:
            print(f"combination/global excluded columns: {all_excludes}")
        return output_root

    run_name = args.run_name or baseline_cfg.get("run_name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = ensure_dir(output_root / str(baseline_cfg.get("output_dir", "BaselineComparison")) / safe_name(run_name))

    dnn_params = merged_training_params(cfg, "DNN", baseline_cfg.get("dnn_training_overrides") or {})
    if args.epochs is not None:
        dnn_params["epochs"] = args.epochs
    if args.batch_size is not None:
        dnn_params["batch_size"] = args.batch_size
    if args.lr is not None:
        dnn_params["lr"] = args.lr
    random_state = int(dnn_params.get("random_state", baseline_cfg.get("random_state", 42)))
    device = _choose_device(str(baseline_cfg.get("device", "auto")))
    skip_existing = bool(baseline_cfg.get("skip_existing", True))
    remaining_cfg = baseline_cfg.get("remaining_validation") or {}
    remaining_enabled = bool(remaining_cfg.get("enabled", True))

    save_json(
        output_dir / "baseline_config.json",
        {
            "config_path": str(Path(cfg["_config_path"]).resolve()),
            "dataset": dataset_cfg.get("name"),
            "data_path": str(data_path),
            "centers": targets,
            "baselines": methods,
            "control_method": FULL_DNN_NAME,
            "target_overrides": baseline_cfg.get("target_overrides") or {},
            "remaining_validation": {"enabled": remaining_enabled},
            "min_features": min_features,
            "max_features": max_features,
            "fixed_feature_count": fixed_feature_count,
            "dnn_params": dnn_params,
            "skip_existing": skip_existing,
            "preprocessing": {"drop_all_zero_columns": drop_all_zero_columns, "exclude_columns": exclude_columns},
        },
    )

    all_results: List[Dict[str, Any]] = []
    remaining_results: List[Dict[str, Any]] = []
    for center_idx, target in enumerate(targets, start=1):
        center = str(target["center"])
        combo_name = None
        if target["kind"] == "combo":
            combo_name = f"combo{target['id']}_{target['name']}"
        target_exclude_columns = normalize_column_list(
            [
                *exclude_columns,
                *normalize_column_list(target["exclude_columns"]),
                *normalize_column_list(target.get("baseline_exclude_columns")),
            ]
        )
        target_label = str(target["label"])
        print(f"[{center_idx}/{len(targets)}] Center={center}" + (f" ({combo_name})" if combo_name else ""))
        if target_exclude_columns:
            print(f"  Excluded columns: {', '.join(target_exclude_columns)}")
        center_dir = ensure_dir(output_dir / f"CenterOn_{safe_name(target_label)}")
        l1_run_dir = _ensure_l1_run(
            cfg,
            center,
            combo_name,
            output_root,
            data_path,
            drop_all_zero_columns,
            target_exclude_columns,
            baseline_cfg,
            center_dir / "L1GateDNN_source",
        )
        l1_selected_count = len(_load_json(l1_run_dir / "selected_features.json").get("features", []))
        features, X_train, y_train = _ranking_frame(
            data_path,
            center,
            drop_all_zero_columns,
            target_exclude_columns,
            train_ratio=float(dnn_params.get("train_ratio", 0.8)),
            random_state=random_state,
        )
        n_override = target.get("n_features_override")
        n_features = _feature_count(l1_selected_count, len(features), min_features, max_features, fixed_feature_count, n_override)
        n_source = f"target_override={n_override}" if n_override is not None else "global_rule"
        print(f"  L1 selected={l1_selected_count}; protected n_i={n_features}; candidates={len(features)} ({n_source})")

        plot_label = f"{target_label}\nn={n_features}/all={len(features)}"
        row_context = {
            "center": center,
            "target_label": target_label,
            "plot_label": plot_label,
            "combo_name": combo_name or "",
            "protected_feature_count": n_features,
            "l1_selected_count": l1_selected_count,
            "n_features_source": n_source,
            "l1_run_dir": str(l1_run_dir),
        }
        save_json(
            center_dir / "center_config.json",
            {
                "center": center,
                "target": target,
                "exclude_columns": target_exclude_columns,
                "l1_run_dir": str(l1_run_dir),
                "l1_selected_count": l1_selected_count,
                "protected_feature_count": n_features,
                "n_features_source": n_source,
                "override_note": target.get("override_note", ""),
            },
        )

        full_method_dir = ensure_dir(center_dir / FULL_DNN_NAME)
        full_metrics_path = full_method_dir / "metrics.json"
        print(f"  [control] {FULL_DNN_NAME} with all {len(features)} candidate features")
        full_run_context = _dnn_run_context(
            data_path=data_path,
            center=center,
            method=FULL_DNN_NAME,
            features=features,
            params=dnn_params,
            drop_all_zero_columns=drop_all_zero_columns,
            exclude_columns=target_exclude_columns,
            extra={"target_label": target_label, "protected_feature_count": n_features},
        )
        if skip_existing and _can_reuse_dnn(full_method_dir, full_run_context):
            full_metrics = _load_json(full_metrics_path)
            full_row = {
                "center": center,
                "target_label": target_label,
                "plot_label": plot_label,
                "combo_name": combo_name or "",
                "method": FULL_DNN_NAME,
                "protected_feature_count": n_features,
                "l1_selected_count": l1_selected_count,
                "feature_count": int(full_metrics.get("feature_count", 0)),
                "best_test_r2": float(full_metrics.get("best_test_r2", np.nan)),
                "best_train_r2": float(full_metrics.get("best_train_r2", np.nan)),
                "min_test_loss": float(full_metrics.get("min_test_loss", np.nan)),
                "min_train_loss": float(full_metrics.get("min_train_loss", np.nan)),
                "final_test_r2": float(full_metrics.get("final_test_r2", np.nan)),
                "final_train_r2": float(full_metrics.get("final_train_r2", np.nan)),
                "final_test_loss": float(full_metrics.get("final_test_loss", np.nan)),
                "final_train_loss": float(full_metrics.get("final_train_loss", np.nan)),
                "best_epoch": int(full_metrics.get("best_epoch", 0)),
                "output_dir": str(full_method_dir),
                "l1_run_dir": str(l1_run_dir),
            }
            print(f"    reused existing metrics: best_test_r2={full_row['best_test_r2']:.6f}")
        else:
            pd.DataFrame({"feature": features, "score": [np.nan] * len(features), "selected": [True] * len(features)}).to_csv(
                full_method_dir / "feature_scores.csv", index=False, encoding="utf-8-sig"
            )
            full_result = _train_dnn(
                data_path=data_path,
                center=center,
                features=features,
                params=dnn_params,
                output_dir=full_method_dir,
                device=device,
                drop_all_zero_columns=drop_all_zero_columns,
                exclude_columns=target_exclude_columns,
            )
            save_json(full_method_dir / RUN_CONTEXT_FILE, full_run_context)
            full_row = {
                "center": center,
                "target_label": target_label,
                "plot_label": plot_label,
                "combo_name": combo_name or "",
                "method": FULL_DNN_NAME,
                "protected_feature_count": n_features,
                "l1_selected_count": l1_selected_count,
                "feature_count": full_result.feature_count,
                "best_test_r2": full_result.best_test_r2,
                "best_train_r2": full_result.best_train_r2,
                "min_test_loss": full_result.min_test_loss,
                "min_train_loss": full_result.min_train_loss,
                "final_test_r2": full_result.final_test_r2,
                "final_train_r2": full_result.final_train_r2,
                "final_test_loss": full_result.final_test_loss,
                "final_train_loss": full_result.final_train_loss,
                "best_epoch": full_result.best_epoch,
                "output_dir": full_result.output_dir,
                "l1_run_dir": str(l1_run_dir),
            }
            print(f"    best_test_r2={full_result.best_test_r2:.6f} at epoch {full_result.best_epoch}")
        all_results.append(full_row)
        pd.DataFrame(all_results).to_csv(output_dir / "baseline_summary_long.csv", index=False, encoding="utf-8-sig")

        for method_idx, method in enumerate(methods, start=1):
            print(f"  [{method_idx}/{len(methods)}] {method}")
            method_dir = ensure_dir(center_dir / safe_name(method))
            metrics_path = method_dir / "metrics.json"
            selected, scores = _baseline_features(method, n_features, l1_run_dir, features, X_train, y_train, random_state, baseline_cfg)
            if len(selected) < n_features:
                remaining = [feature for feature in features if feature not in selected]
                selected = [*selected, *remaining[: n_features - len(selected)]]
            run_context = _dnn_run_context(
                data_path=data_path,
                center=center,
                method=method,
                features=selected,
                params=dnn_params,
                drop_all_zero_columns=drop_all_zero_columns,
                exclude_columns=target_exclude_columns,
                extra={
                    "target_label": target_label,
                    "protected_feature_count": n_features,
                    "l1_run_dir": str(l1_run_dir),
                },
            )
            if skip_existing and _can_reuse_dnn(method_dir, run_context):
                metrics = _load_json(metrics_path)
                row = {
                    "center": center,
                    "target_label": target_label,
                    "plot_label": plot_label,
                    "combo_name": combo_name or "",
                    "method": method,
                    "protected_feature_count": n_features,
                    "l1_selected_count": l1_selected_count,
                    "feature_count": int(metrics.get("feature_count", 0)),
                    "best_test_r2": float(metrics.get("best_test_r2", np.nan)),
                    "best_train_r2": float(metrics.get("best_train_r2", np.nan)),
                    "min_test_loss": float(metrics.get("min_test_loss", np.nan)),
                    "min_train_loss": float(metrics.get("min_train_loss", np.nan)),
                    "final_test_r2": float(metrics.get("final_test_r2", np.nan)),
                    "final_train_r2": float(metrics.get("final_train_r2", np.nan)),
                    "final_test_loss": float(metrics.get("final_test_loss", np.nan)),
                    "final_train_loss": float(metrics.get("final_train_loss", np.nan)),
                    "best_epoch": int(metrics.get("best_epoch", 0)),
                    "output_dir": str(method_dir),
                    "l1_run_dir": str(l1_run_dir),
                }
                all_results.append(row)
                pd.DataFrame(all_results).to_csv(output_dir / "baseline_summary_long.csv", index=False, encoding="utf-8-sig")
                print(f"    reused existing metrics: best_test_r2={row['best_test_r2']:.6f}")
                if remaining_enabled:
                    rem_row = _run_remaining_validation(
                        method=method,
                        selected_features=selected,
                        all_features=features,
                        center_dir=center_dir,
                        data_path=data_path,
                        center=center,
                        dnn_params=dnn_params,
                        device=device,
                        drop_all_zero_columns=drop_all_zero_columns,
                        exclude_columns=target_exclude_columns,
                        skip_existing=skip_existing,
                        context=row_context,
                    )
                    if rem_row is not None:
                        remaining_results.append(rem_row)
                        pd.DataFrame(remaining_results).to_csv(
                            output_dir / "remaining_validation_summary_long.csv", index=False, encoding="utf-8-sig"
                        )
                continue
            score_rows = [{"feature": feature, "score": float(scores.get(feature, np.nan)), "selected": feature in selected} for feature in features]
            pd.DataFrame(score_rows).sort_values(["selected", "score"], ascending=[False, False]).to_csv(
                method_dir / "feature_scores.csv", index=False, encoding="utf-8-sig"
            )
            result = _train_dnn(
                data_path=data_path,
                center=center,
                features=selected,
                params=dnn_params,
                output_dir=method_dir,
                device=device,
                drop_all_zero_columns=drop_all_zero_columns,
                exclude_columns=target_exclude_columns,
            )
            save_json(method_dir / RUN_CONTEXT_FILE, run_context)
            row = {
                "center": center,
                "target_label": target_label,
                "plot_label": plot_label,
                "combo_name": combo_name or "",
                "method": method,
                "protected_feature_count": n_features,
                "l1_selected_count": l1_selected_count,
                "feature_count": result.feature_count,
                "best_test_r2": result.best_test_r2,
                "best_train_r2": result.best_train_r2,
                "min_test_loss": result.min_test_loss,
                "min_train_loss": result.min_train_loss,
                "final_test_r2": result.final_test_r2,
                "final_train_r2": result.final_train_r2,
                "final_test_loss": result.final_test_loss,
                "final_train_loss": result.final_train_loss,
                "best_epoch": result.best_epoch,
                "output_dir": result.output_dir,
                "l1_run_dir": str(l1_run_dir),
            }
            all_results.append(row)
            pd.DataFrame(all_results).to_csv(output_dir / "baseline_summary_long.csv", index=False, encoding="utf-8-sig")
            print(f"    best_test_r2={result.best_test_r2:.6f} at epoch {result.best_epoch}")
            if remaining_enabled:
                rem_row = _run_remaining_validation(
                    method=method,
                    selected_features=selected,
                    all_features=features,
                    center_dir=center_dir,
                    data_path=data_path,
                    center=center,
                    dnn_params=dnn_params,
                    device=device,
                    drop_all_zero_columns=drop_all_zero_columns,
                    exclude_columns=target_exclude_columns,
                    skip_existing=skip_existing,
                    context=row_context,
                )
                if rem_row is not None:
                    remaining_results.append(rem_row)
                    pd.DataFrame(remaining_results).to_csv(
                        output_dir / "remaining_validation_summary_long.csv", index=False, encoding="utf-8-sig"
                    )
                    print(f"    remaining_best_test_r2={rem_row['best_test_r2']:.6f}")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(output_dir / "baseline_summary_long.csv", index=False, encoding="utf-8-sig")
    pivot_index = "target_label" if "target_label" in summary_df.columns else "center"
    summary_df.pivot(index=pivot_index, columns="method", values="best_test_r2").to_csv(
        output_dir / "baseline_summary_wide.csv", encoding="utf-8-sig"
    )
    remaining_df = pd.DataFrame(remaining_results)
    if not remaining_df.empty:
        remaining_df.to_csv(output_dir / "remaining_validation_summary_long.csv", index=False, encoding="utf-8-sig")
        remaining_df.pivot(index=pivot_index, columns="method", values="best_test_r2").to_csv(
            output_dir / "remaining_validation_summary_wide.csv", encoding="utf-8-sig"
        )
    display_methods = [FULL_DNN_NAME, *methods]
    main_method = str(baseline_cfg.get("main_method", "L1GateDNN"))
    _plot_summary(summary_df, display_methods, output_dir / "baseline_test_r2.png", main_method=main_method)
    if not remaining_df.empty:
        _plot_summary(
            remaining_df,
            methods,
            output_dir / "remaining_validation_r2.png",
            main_method=main_method,
            title=r"Remaining-Feature Validation by Test $R^2$",
            ylabel=r"Best Test $R^2$ After Removing Top-n",
        )
    _write_html_report(output_dir, summary_df, display_methods, main_method=main_method, remaining_df=remaining_df)
    print(f"Saved baseline comparison to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare L1GateDNN-selected feature count against baseline feature rankings.")
    parser.add_argument("--config", help="Config YAML path. Defaults to configs/active_config.yaml.")
    parser.add_argument("--centers", nargs="*", help="Override baseline_comparison.centers.")
    parser.add_argument("--baselines", nargs="*", help="Override baseline_comparison.baselines. Use all for every baseline.")
    parser.add_argument("--min-features", type=int, help="Minimum protected feature count n_i.")
    parser.add_argument("--max-features", type=int, help="Maximum protected feature count n_i when fixed count is not set.")
    parser.add_argument("--fixed-feature-count", type=int, help="Use the same fixed feature count for every center.")
    parser.add_argument("--epochs", type=int, help="Override DNN baseline epochs.")
    parser.add_argument("--batch-size", type=int, help="Override DNN baseline batch size.")
    parser.add_argument("--lr", type=float, help="Override DNN baseline learning rate.")
    parser.add_argument("--run-name", help="Output run directory name under BaselineComparison.")
    parser.add_argument("--check-only", action="store_true", help="Validate config and center columns without training.")
    args = parser.parse_args()
    run_baselines(args)


if __name__ == "__main__":
    main()
