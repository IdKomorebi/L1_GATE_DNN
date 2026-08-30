"""恒等式发现：零空间分析 + 逐目标稀疏恢复 + 对数空间派生关系检测。

第一类（公式型/显式推断）判定的实现原型：
  1) 全局零空间分析给出"数据中共有多少条独立的精确仿射恒等式"
  2) 逐目标前向选择给出每条恒等式的可读形式与最小支撑集
  3) 对数空间重跑同一套流程，捕获比值/乘积型派生关系
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(
    "NEW_PROJECT/data/processed/data2025_Processed_V2/"
    "pjm_rto_hourly_2025_aligned_processed_one_header.csv"
)
TIME_COLS = ["datetime_beginning_utc", "datetime_beginning_ept"]

# 恒等式判定阈值：残差比 RMSE/sigma_y 低于此值判为精确关系
EPS_ID = 1e-6
# 奇异值相对阈值：sigma_k/sigma_1 低于此值判为零空间方向
EPS_SVD = 1e-10
MAX_SUPPORT = 6


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df.drop(columns=[c for c in TIME_COLS if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def nullspace_report(X: np.ndarray, names: list[str], tag: str) -> int:
    """列归一化后做 SVD，报告奇异值谱与零空间维数。"""
    M = np.column_stack([X, np.ones(len(X))])
    norms = np.linalg.norm(M, axis=0)
    norms[norms == 0] = 1.0
    M = M / norms
    s = np.linalg.svd(M, compute_uv=False)
    ratio = s / s[0]
    dim = int(np.sum(ratio < EPS_SVD))
    print(f"\n[{tag}] 矩阵 {M.shape}, 奇异值比最小 10 个:")
    print("   " + "  ".join(f"{v:.2e}" for v in ratio[-10:]))
    print(f"[{tag}] 零空间维数 (sigma_k/sigma_1 < {EPS_SVD:.0e}) = {dim}")
    return dim


def forward_identity(
    y: np.ndarray, X: np.ndarray, names: list[str], max_k: int = MAX_SUPPORT
) -> tuple[list[int], np.ndarray, float] | None:
    """前向选择：逐个加入字段直到残差比低于 EPS_ID。返回 (支撑集, 系数, 残差比)。"""
    n = len(y)
    sigma = y.std()
    if sigma == 0:
        return None
    A = np.column_stack([X, np.ones(n)])
    chosen: list[int] = []
    resid = y.copy()
    for _ in range(max_k):
        # 选与当前残差相关性最强的未选字段
        best, best_score = -1, -1.0
        for j in range(X.shape[1]):
            if j in chosen:
                continue
            col = X[:, j]
            if col.std() == 0:
                continue
            score = abs(np.dot(col - col.mean(), resid - resid.mean())) / (
                np.linalg.norm(col - col.mean()) * np.linalg.norm(resid - resid.mean()) + 1e-30
            )
            if score > best_score:
                best, best_score = j, score
        if best < 0:
            break
        chosen.append(best)
        sub = np.column_stack([X[:, chosen], np.ones(n)])
        coef, *_ = np.linalg.lstsq(sub, y, rcond=None)
        resid = y - sub @ coef
        ratio = np.sqrt(np.mean(resid**2)) / sigma
        if ratio < EPS_ID:
            return chosen, coef, ratio
    return None


def fmt_identity(target: str, idx: list[int], coef: np.ndarray, names: list[str]) -> str:
    parts = []
    for k, j in enumerate(idx):
        c = coef[k]
        parts.append(f"{c:+.6g}*{names[j]}")
    b = coef[-1]
    if abs(b) > 1e-9:
        parts.append(f"{b:+.6g}")
    return f"{target} = " + " ".join(parts)


def coef_cleanliness(coef: np.ndarray) -> str:
    """系数整洁性：接近整数或简单有理数是会计恒等式的指纹。"""
    c = coef[:-1]
    dev = np.abs(c - np.round(c))
    if np.all(dev < 1e-6):
        return "整数系数"
    if np.all(np.abs(c * 100 - np.round(c * 100)) < 1e-4):
        return "两位小数系数"
    return "一般实数系数"


def scan(df: pd.DataFrame, tag: str) -> list[str]:
    names = list(df.columns)
    D = df.to_numpy(float)
    ok = ~np.isnan(D).any(axis=1)
    D = D[ok]
    print(f"\n[{tag}] 完整样本行数 = {len(D)} / {len(df)}")
    nullspace_report(D, names, tag)

    found: list[str] = []
    print(f"\n[{tag}] 逐目标稀疏恒等式恢复 (残差比 < {EPS_ID:.0e}):")
    for i, tgt in enumerate(names):
        y = D[:, i]
        X = np.delete(D, i, axis=1)
        sub_names = [n for k, n in enumerate(names) if k != i]
        res = forward_identity(y, X, sub_names)
        if res is None:
            continue
        idx, coef, ratio = res
        line = fmt_identity(tgt, idx, coef, sub_names)
        clean = coef_cleanliness(coef)
        print(f"  残差比={ratio:.2e}  支撑={len(idx)}  [{clean}]")
        print(f"    {line}")
        found.append(line)
    if not found:
        print("  （未发现）")
    return found


def main() -> None:
    df = load()
    print(f"字段数 = {df.shape[1]}, 行数 = {df.shape[0]}")

    print("\n" + "=" * 70)
    print("A. 原始空间：精确线性/仿射恒等式")
    print("=" * 70)
    scan(df, "线性")

    print("\n" + "=" * 70)
    print("B. 对数空间：比值/乘积型派生关系")
    print("=" * 70)
    pos = df.loc[:, (df > 0).all(axis=0, skipna=True) & df.notna().all(axis=0)]
    print(f"全正字段数 = {pos.shape[1]}")
    if pos.shape[1] >= 3:
        scan(np.log(pos), "对数")
    else:
        print("全正字段不足，跳过")


if __name__ == "__main__":
    main()
