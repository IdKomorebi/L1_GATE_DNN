"""多指标初筛：算六类依赖指标，并用置换检验给出客观阈值。

相对原项目 `NEW_PROJECT/src/relation_analyzer.py` 修了三处问题：

1. **HSIC 的核宽度写死成 1，导致结果只反映字段量纲、不反映依赖强度。**
   实测：同一对相关系数 0.96 的变量，仅把量纲乘以不同倍数，旧实现算出的 HSIC
   在 0.21 到 0.93 之间乱跳。而真实字段的量纲跨了 11 个数量级。
   改法：先把变量标准化，再用"两两距离的中位数"作为核宽度（惯用做法）。

2. **归一化互信息把两种不同的估计方式混用。**
   互信息用 k 近邻估计（连续型），熵用 20 格直方图估计（离散型），两者不在同一
   尺度上，相除得到的数没有明确含义，而且直方图熵还随格子数变化。
   改法：互信息和熵都用同一套等频分箱来估计，口径一致；等频分箱还顺带让结果
   不受量纲影响。

3. **阈值靠人工指定**（原来是 NMI 0.06、Spearman 0.15、Pearson 0.15、Kendall 0.12、
   距离相关 0.20、HSIC 0.25，没有依据）。
   改法：用分块置换构造"两个字段之间没有真实关系"时各指标的取值分布，
   取其上分位数作为阈值。分块而不是逐点打乱，是为了保住电力数据本身的时间结构——
   逐点打乱会把自相关破坏掉，得到的零分布过于宽松，几乎什么字段都能通过。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

METRICS = ["pearson", "spearman", "kendall", "nmi", "distance_corr", "hsic"]
METRIC_CN = {
    "pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall",
    "nmi": "归一化互信息", "distance_corr": "距离相关", "hsic": "HSIC",
}
# 距离相关和 HSIC 要算两两距离矩阵，代价随样本数平方增长，先抽样
EXPENSIVE_N = 1500


def _z(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / s if s > 0 else v - v.mean()


def _sub(x: np.ndarray, y: np.ndarray, n: int, seed: int):
    if len(x) <= n:
        return x, y
    idx = np.random.default_rng(seed).choice(len(x), n, replace=False)
    return x[idx], y[idx]


# --------------------------------------------------------------------------
# 六类指标
# --------------------------------------------------------------------------

def pearson(x, y) -> float:
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return abs(float(np.corrcoef(x, y)[0, 1]))


def spearman(x, y) -> float:
    rx = pd.Series(x).rank().to_numpy(float)
    ry = pd.Series(y).rank().to_numpy(float)
    return pearson(rx, ry)


def kendall(x, y, n: int = 2000, seed: int = 42) -> float:
    from scipy.stats import kendalltau
    xs, ys = _sub(x, y, n, seed)
    t = kendalltau(xs, ys).statistic
    return abs(float(t)) if np.isfinite(t) else 0.0


def nmi(x, y, bins: int = 16) -> float:
    """归一化互信息。互信息和熵都用同一套等频分箱估计，口径一致。

    等频分箱（按分位数切）比等宽分箱更适合电力数据——很多字段分布很偏，
    等宽分箱会把绝大多数样本挤进一两个格子里。
    """
    def disc(v):
        q = np.quantile(v, np.linspace(0, 1, bins + 1))
        q = np.unique(q)
        if len(q) < 3:
            return np.zeros(len(v), int)
        return np.clip(np.digitize(v, q[1:-1]), 0, len(q) - 2)

    a, b = disc(x), disc(y)
    ka, kb = a.max() + 1, b.max() + 1
    if ka < 2 or kb < 2:
        return 0.0
    joint = np.zeros((ka, kb))
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / (px[:, None] * py[None, :])[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    h = min(hx, hy)
    return float(np.clip(mi / h, 0.0, 1.0)) if h > 0 else 0.0


def distance_corr(x, y, seed: int = 42) -> float:
    """距离相关系数。变量先标准化，使结果不受量纲影响。"""
    xs, ys = _sub(_z(x), _z(y), EXPENSIVE_N, seed)
    n = len(xs)
    if n < 10:
        return 0.0
    a = np.abs(xs[:, None] - xs[None, :])
    b = np.abs(ys[:, None] - ys[None, :])
    A = a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
    B = b - b.mean(0, keepdims=True) - b.mean(1, keepdims=True) + b.mean()
    dcov = np.mean(A * B)
    dvx, dvy = np.mean(A * A), np.mean(B * B)
    d = np.sqrt(np.sqrt(dvx * dvy))
    return float(np.sqrt(max(dcov, 0.0)) / d) if d > 0 else 0.0


def hsic(x, y, seed: int = 42) -> float:
    """HSIC。核宽度用"两两距离的中位数"，而不是写死成 1。

    写死核宽度是原实现的主要问题：字段量纲一大，指数项就全部下溢成 0，
    核矩阵退化成单位阵；量纲一小，核矩阵又全是 1。两种情况下算出的都不是依赖强度。
    """
    xs, ys = _sub(_z(x), _z(y), EXPENSIVE_N, seed)
    n = len(xs)
    if n < 10:
        return 0.0

    def kern(v):
        d2 = (v[:, None] - v[None, :]) ** 2
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        return np.exp(-d2 / (med if med > 0 else 1.0))

    K, L = kern(xs), kern(ys)
    # H @ K @ H 与下面的双中心化完全等价，但显式矩阵乘法是 O(n^3)。
    # 置换检验会调用本函数上千次，直接按行列均值中心化可把复杂度降到 O(n^2)，
    # 同时避免构造额外的 n×n 中心矩阵。
    KH = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    LH = L - L.mean(axis=0, keepdims=True) - L.mean(axis=1, keepdims=True) + L.mean()
    num = float(np.sum(KH * LH))
    den = float(np.sqrt(np.sum(KH * KH) * np.sum(LH * LH)))
    return float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0


def all_metrics(x: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 20 or x.std() == 0 or y.std() == 0:
        return {k: 0.0 for k in METRICS}
    return {"pearson": pearson(x, y), "spearman": spearman(x, y),
            "kendall": kendall(x, y, seed=seed), "nmi": nmi(x, y),
            "distance_corr": distance_corr(x, y, seed), "hsic": hsic(x, y, seed)}


# --------------------------------------------------------------------------
# 置换检验定阈值
# --------------------------------------------------------------------------

def block_shuffle(v: np.ndarray, block: int, rng) -> np.ndarray:
    """按块打乱：切成若干段整段重排，保住序列自身的时间结构。"""
    n = len(v)
    idx = np.arange(n)
    blocks = [idx[i:i + block] for i in range(0, n, block)]
    rng.shuffle(blocks)
    return v[np.concatenate(blocks)]


@dataclass
class NullResult:
    thresholds: dict
    draws: pd.DataFrame


def null_thresholds(
    df: pd.DataFrame, target: str, pool: list[str],
    n_draws: int = 120, block: int = 168, quantile: float = 0.95, seed: int = 0
) -> NullResult:
    """构造"字段与目标之间没有真实关系"时各指标的取值分布，取上分位数作阈值。

    每次抽一个真实字段、把目标按周（168 小时）整段打乱，再算六个指标。
    轮换不同字段，让零分布反映数据里实际的自相关强弱分布。
    """
    rng = np.random.default_rng(seed)
    y = df[target].to_numpy(float)
    rows = []
    for i in range(n_draws):
        c = pool[i % len(pool)]
        x = df[c].to_numpy(float)
        rows.append(all_metrics(x, block_shuffle(y, block, rng), seed=i))
    draws = pd.DataFrame(rows)
    return NullResult(
        thresholds={k: float(np.quantile(draws[k], quantile)) for k in METRICS},
        draws=draws,
    )


def screen(
    df: pd.DataFrame, target: str, pool: list[str], thresholds: dict
) -> pd.DataFrame:
    """算每个候选字段对目标的六个指标，标出各自是否过阈值。
    采用"任一指标过阈值即保留"的宽松规则。"""
    rows = []
    y = df[target].to_numpy(float)
    for c in pool:
        v = all_metrics(df[c].to_numpy(float), y)
        v["field"] = c
        n_pass = 0
        for k in METRICS:
            p = v[k] >= thresholds[k]
            v[f"{k}_pass"] = int(p)
            n_pass += int(p)
        v["n_pass"] = n_pass
        v["kept"] = int(n_pass > 0)
        rows.append(v)
    cols = ["field"] + METRICS + [f"{k}_pass" for k in METRICS] + ["n_pass", "kept"]
    return pd.DataFrame(rows)[cols].sort_values("n_pass", ascending=False)
