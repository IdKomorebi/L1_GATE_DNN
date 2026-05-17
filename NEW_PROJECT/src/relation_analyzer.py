from __future__ import annotations

import hashlib
import itertools
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

try:
    from scipy.stats import kendalltau as scipy_kendalltau
except Exception:
    scipy_kendalltau = None

try:
    from sklearn.feature_selection import mutual_info_regression
except Exception:
    mutual_info_regression = None

try:
    import dcor  # type: ignore

    DCOR_AVAILABLE = True
except Exception:
    dcor = None
    DCOR_AVAILABLE = False

from .data_utils import center_output_dir, ensure_dir, normalize_column_list, read_numeric_csv


DEFAULT_METRICS = ["nmi", "spearman", "pearson", "kendall", "distance_corr", "hsic"]
METRIC_ABBR = {
    "nmi": "nmi",
    "spearman": "sp",
    "pearson": "pe",
    "kendall": "ke",
    "distance_corr": "dc",
    "hsic": "hsic",
}


def _value_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _exclude_tag(exclude_columns: Sequence[str] | None) -> str:
    columns = sorted(normalize_column_list(exclude_columns))
    if not columns:
        return ""
    digest = hashlib.sha1("\n".join(columns).encode("utf-8")).hexdigest()[:8]
    return f"_excl{digest}"


def relation_set_name(
    metrics: Sequence[str],
    thresholds: Dict[str, float],
    sample_size: int,
    expensive_sample_size: int,
    drop_all_zero_columns: bool = False,
    exclude_columns: Sequence[str] | None = None,
) -> str:
    parts = []
    for metric in metrics:
        if metric in thresholds:
            parts.append(f"{METRIC_ABBR.get(metric, metric)}{_value_tag(float(thresholds[metric]))}")
    threshold_part = "_".join(parts) if parts else "no_thresholds"
    zero_part = "_dropzero" if drop_all_zero_columns else ""
    return f"thr_{threshold_part}_s{sample_size}_e{expensive_sample_size}{zero_part}{_exclude_tag(exclude_columns)}"


def relation_analysis_dir(
    output_root: str | Path,
    metrics: Sequence[str],
    thresholds: Dict[str, float],
    sample_size: int,
    expensive_sample_size: int,
    drop_all_zero_columns: bool = False,
    exclude_columns: Sequence[str] | None = None,
) -> Path:
    return ensure_dir(
        Path(output_root)
        / "RelationshipAnalysis"
        / relation_set_name(metrics, thresholds, sample_size, expensive_sample_size, drop_all_zero_columns, exclude_columns)
    )


def center_relation_analysis_dir(
    output_root: str | Path,
    center: str,
    metrics: Sequence[str],
    thresholds: Dict[str, float],
    sample_size: int,
    expensive_sample_size: int,
    drop_all_zero_columns: bool = False,
    exclude_columns: Sequence[str] | None = None,
) -> Path:
    return ensure_dir(
        center_output_dir(output_root, center)
        / "RelationshipAnalysis"
        / relation_set_name(metrics, thresholds, sample_size, expensive_sample_size, drop_all_zero_columns, exclude_columns)
    )


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask].astype(float), y[mask].astype(float)


def _sample_pair(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(x) <= max_samples:
        return x, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(x), size=max_samples, replace=False)
    return x[idx], y[idx]


def normalized_mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    if mutual_info_regression is not None:
        mi = mutual_info_regression(x.reshape(-1, 1), y, random_state=42)[0]

        def estimated_entropy(arr: np.ndarray, bins: int = 20) -> float:
            hist, _ = np.histogram(arr, bins=bins, density=False)
            prob = hist.astype(float) / max(float(hist.sum()), 1.0)
            prob = prob[prob > 0]
            return float(-np.sum(prob * np.log(prob))) if len(prob) else 0.0

        h_min = min(estimated_entropy(x), estimated_entropy(y))
        if h_min <= 0 or not np.isfinite(mi):
            return 0.0
        return float(np.clip(mi / h_min, 0.0, 1.0))

    bins = 20
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_xy / max(hist_xy.sum(), 1.0)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    h_min = min(hx, hy)
    return float(np.clip(mi / h_min, 0.0, 1.0)) if h_min > 0 else 0.0


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return pearson_corr(rx, ry)


def kendall_corr(x: np.ndarray, y: np.ndarray) -> float:
    if scipy_kendalltau is not None:
        return float(scipy_kendalltau(x, y).statistic)
    n = len(x)
    if n < 2:
        return 0.0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    tri = np.triu_indices(n, k=1)
    prod = np.sign(dx[tri] * dy[tri])
    denom = len(prod)
    if denom == 0:
        return 0.0
    return float(np.sum(prod) / denom)


def fallback_distance_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov = np.sqrt(np.mean(A * B))
    dvar_x = np.sqrt(np.mean(A * A))
    dvar_y = np.sqrt(np.mean(B * B))
    denom = np.sqrt(dvar_x * dvar_y)
    if denom <= 0:
        return 0.0
    return float(dcov / denom)


def normalized_hsic(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    x2 = x.reshape(-1, 1)
    y2 = y.reshape(-1, 1)
    K = np.exp(-((x2 - x2.T) ** 2))
    L = np.exp(-((y2 - y2.T) ** 2))
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    KH = H @ K @ H
    LH = H @ L @ H
    hsic = np.trace(KH @ LH) / ((n - 1) ** 2)
    norm = np.sqrt(np.trace(KH @ KH) * np.trace(LH @ LH)) / ((n - 1) ** 2)
    if norm <= 0:
        return 0.0
    return float(hsic / norm)


def analyze_pair(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    metrics: Sequence[str],
    sample_size: int,
    expensive_sample_size: int,
    random_state: int,
) -> Dict[str, float]:
    x, y = _finite_pair(np.asarray(x_raw), np.asarray(y_raw))
    if len(x) < 3:
        return {metric: np.nan for metric in metrics}

    x_std, y_std = _sample_pair(x, y, sample_size, random_state)
    x_exp, y_exp = _sample_pair(x, y, expensive_sample_size, random_state)

    result: Dict[str, float] = {}
    for metric in metrics:
        try:
            if metric == "nmi":
                result[metric] = normalized_mutual_info(x_std, y_std)
            elif metric == "spearman":
                result[metric] = spearman_corr(x_std, y_std)
            elif metric == "pearson":
                result[metric] = pearson_corr(x_std, y_std)
            elif metric == "kendall":
                result[metric] = kendall_corr(x_exp, y_exp)
            elif metric == "distance_corr":
                if DCOR_AVAILABLE and dcor is not None:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="Falling back to uncompiled AVL.*")
                        result[metric] = float(dcor.distance_correlation(x_exp, y_exp))
                else:
                    result[metric] = fallback_distance_corr(x_exp, y_exp)
            elif metric == "hsic":
                result[metric] = normalized_hsic(x_exp, y_exp)
            else:
                result[metric] = np.nan
        except Exception:
            result[metric] = np.nan
    return result


def add_threshold_columns(df: pd.DataFrame, metrics: Sequence[str], thresholds: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    pass_cols: List[str] = []
    for metric in metrics:
        if metric not in out.columns:
            continue
        threshold = float(thresholds.get(metric, 0.0))
        col = f"{metric}_pass"
        out[col] = (out[metric].abs() >= threshold).astype(int)
        pass_cols.append(col)
    out["pass_count"] = out[pass_cols].sum(axis=1) if pass_cols else 0
    metric_cols = [m for m in metrics if m in out.columns]
    out["max_abs_score"] = out[metric_cols].abs().max(axis=1) if metric_cols else np.nan
    return out


def analyze_all_relationships(
    data_path: str | Path,
    output_csv: str | Path,
    metrics: Sequence[str] = DEFAULT_METRICS,
    thresholds: Dict[str, float] | None = None,
    sample_size: int = 3000,
    expensive_sample_size: int = 1200,
    random_state: int = 42,
    progress_every: int = 100,
    drop_all_zero_columns: bool = False,
    exclude_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    df = read_numeric_csv(data_path, drop_all_zero_columns=drop_all_zero_columns, exclude_columns=exclude_columns)
    pairs = list(itertools.combinations(df.columns, 2))
    total = len(pairs)
    if progress_every:
        print(f"Analyzing {total} column pairs from {df.shape[1]} columns...")
    rows = []
    for idx, (col_a, col_b) in enumerate(pairs, start=1):
        values = analyze_pair(
            df[col_a].to_numpy(),
            df[col_b].to_numpy(),
            metrics=metrics,
            sample_size=sample_size,
            expensive_sample_size=expensive_sample_size,
            random_state=random_state + idx,
        )
        rows.append({"index": idx, "column_a": col_a, "column_b": col_b, **values})
        if progress_every and (idx == 1 or idx % progress_every == 0 or idx == total):
            print(f"  processed {idx}/{total} pairs")
    result = add_threshold_columns(pd.DataFrame(rows), metrics, thresholds or {})
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result


def analyze_center_relationships(
    data_path: str | Path,
    center: str,
    output_csv: str | Path,
    metrics: Sequence[str] = DEFAULT_METRICS,
    thresholds: Dict[str, float] | None = None,
    sample_size: int = 3000,
    expensive_sample_size: int = 1200,
    random_state: int = 42,
    progress_every: int = 10,
    drop_all_zero_columns: bool = False,
    exclude_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    df = read_numeric_csv(data_path, drop_all_zero_columns=drop_all_zero_columns, exclude_columns=exclude_columns)
    if center not in df.columns:
        raise ValueError(f"Center column not found: {center}")

    rows = []
    related_cols: List[str] = [c for c in df.columns if c != center]
    total = len(related_cols)
    if progress_every:
        print(f"Analyzing center '{center}' against {total} columns...")
    for idx, related in enumerate(related_cols, start=1):
        values = analyze_pair(
            df[center].to_numpy(),
            df[related].to_numpy(),
            metrics=metrics,
            sample_size=sample_size,
            expensive_sample_size=expensive_sample_size,
            random_state=random_state + idx,
        )
        rows.append({"index": idx, "center": center, "related": related, **values})
        if progress_every and (idx == 1 or idx % progress_every == 0 or idx == total):
            print(f"  processed {idx}/{total} center pairs")
    result = add_threshold_columns(pd.DataFrame(rows), metrics, thresholds or {})
    result = result.sort_values(["pass_count", "max_abs_score"], ascending=[False, False]).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return result


def select_features(center_rel: pd.DataFrame, feature_cfg: Dict) -> List[str]:
    if "related" not in center_rel.columns:
        raise ValueError("Center relationship table must contain a related column.")

    require_any = bool(feature_cfg.get("require_any_metric", True))
    fallback_top_k = int(feature_cfg.get("fallback_top_k") or 15)
    max_features = feature_cfg.get("max_features")
    top_n_by_sum = feature_cfg.get("top_n_by_sum")

    rel = center_rel.copy()
    if require_any and "pass_count" in rel.columns:
        selected = rel[rel["pass_count"] > 0].copy()
    else:
        selected = rel.copy()

    if selected.empty:
        selected = rel.sort_values("max_abs_score", ascending=False).head(fallback_top_k).copy()

    if top_n_by_sum:
        metric_cols = [metric for metric in DEFAULT_METRICS if metric in selected.columns]
        if not metric_cols:
            raise ValueError("Cannot apply top_n_by_sum because no relationship metric columns are available.")
        selected = selected.copy()
        selected["metric_abs_sum"] = selected[metric_cols].abs().sum(axis=1)
        selected = selected.sort_values(["metric_abs_sum", "pass_count", "max_abs_score"], ascending=[False, False, False])
        selected = selected.head(int(top_n_by_sum))
    else:
        selected = selected.sort_values(["pass_count", "max_abs_score"], ascending=[False, False])
    if max_features:
        selected = selected.head(int(max_features))
    return selected["related"].astype(str).tolist()


def correlation_vectors(center_rel: pd.DataFrame, features: Sequence[str], metrics: Sequence[str]) -> np.ndarray:
    indexed = center_rel.set_index("related", drop=False)
    rows = []
    for feature in features:
        if feature in indexed.index:
            row = indexed.loc[feature]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            rows.append([float(row.get(metric, np.nan)) for metric in metrics])
        else:
            rows.append([np.nan for _ in metrics])
    arr = np.asarray(rows, dtype=np.float32)
    if np.isnan(arr).any():
        col_means = np.nanmean(arr, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_means, inds[1])
    return arr
