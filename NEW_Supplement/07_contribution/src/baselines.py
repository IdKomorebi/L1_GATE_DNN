"""对照方法：现有做法在同一道题上给出什么答案。

全部只读地调用 02_src/gates.py，不修改那边任何东西。

五种对照：
  分解式门控   本项目要改进的对象。它的输出是"选中/未选中"加一个门控强度
  随机门控     另一种稀疏门控，同类做法
  两两相关性   最朴素的做法，用来展示"边际相关和联合必要性无关"
  置换重要性   随机森林那一路的常规做法，打乱某列看精度掉多少
  Lasso 系数   线性方法的代表

这些方法和贡献值的根本差别在于：它们都是在**全部字段都在场**的前提下
问"这个字段重不重要"。字段一旦有替身，在场时抽掉它不痛，就会被判成不重要；
但它其实完全有能力独立把目标推出来。贡献值是对所有字段组合取平均，
不预设"其他字段都在场"这个前提，所以不会掉进这个坑。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_src"))
import gates as G          # noqa: E402


def dgating_scores(X: np.ndarray, y: np.ndarray, seed: int = 42,
                   epochs: int = 200, lambda_dgate: float = 0.005,
                   depth: int = 4) -> dict:
    """分解式门控（D-Gating）。返回门控值和按阈值判定的选中集合。"""
    cfg = G.TrainConfig(seed=seed, epochs=epochs, lambda_dgate=lambda_dgate,
                        dgate_depth=depth)
    res = G.train("DGatingDNN", X, y, cfg)
    g = np.asarray(res.final_gates, dtype=float)
    return {"score": g,
            "selected": np.where(g >= cfg.dgate_threshold)[0].tolist(),
            "n_exact_zero": int((g == 0).sum()),
            "full_r2": float(res.best_test_r2)}


def stg_scores(X: np.ndarray, y: np.ndarray, seed: int = 42,
               epochs: int = 200) -> dict:
    """随机门控（STG）。报告未裁剪的 μ——裁剪后所有选中字段都等于 1，排序就没了。"""
    cfg = G.TrainConfig(seed=seed, epochs=epochs)
    res = G.train("STG", X, y, cfg)
    g = np.asarray(res.final_gates, dtype=float)
    return {"score": g,
            "selected": np.where(g >= cfg.stg_threshold)[0].tolist(),
            "full_r2": float(res.best_test_r2)}


def pearson_scores(X: np.ndarray, y: np.ndarray) -> dict:
    """两两相关性绝对值。"""
    s = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if X[:, j].std() < 1e-12:
            continue
        s[j] = abs(float(np.corrcoef(X[:, j], y)[0, 1]))
    return {"score": np.nan_to_num(s)}


def permutation_scores(X: np.ndarray, y: np.ndarray, seed: int = 42,
                       n_repeat: int = 5) -> dict:
    """置换重要性：训练一个模型，逐列打乱看 R² 掉多少。

    注意它和"不可替代性"是近亲——都预设其余字段在场，
    所以在有替身的字段上会同样给出接近 0 的分数。
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    n = len(X)
    k = int(n * 0.8)
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed,
                               min_samples_leaf=3)
    rf.fit(X[:k], y[:k])
    r = permutation_importance(rf, X[k:], y[k:], n_repeats=n_repeat,
                               random_state=seed, n_jobs=-1)
    return {"score": np.asarray(r.importances_mean, dtype=float)}


def lasso_scores(X: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """Lasso 系数绝对值（标准化后）。"""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    xs = StandardScaler().fit_transform(X)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    m = LassoCV(cv=5, random_state=seed, n_alphas=60, max_iter=5000).fit(xs, ys)
    return {"score": np.abs(m.coef_), "alpha": float(m.alpha_)}


ALL = {
    "分解式门控": dgating_scores,
    "随机门控": stg_scores,
    "两两相关性": pearson_scores,
    "置换重要性": permutation_scores,
    "Lasso系数": lasso_scores,
}


def run_all(X: np.ndarray, y: np.ndarray, seed: int = 42,
            which: list[str] | None = None) -> dict[str, np.ndarray]:
    """跑全部对照方法，返回 {方法名: 每个字段的分数}。"""
    out = {}
    for name, fn in ALL.items():
        if which and name not in which:
            continue
        try:
            kw = {"seed": seed} if name != "两两相关性" else {}
            out[name] = np.asarray(fn(X, y, **kw)["score"], dtype=float)
        except Exception as e:                                  # noqa: BLE001
            print(f"    [跳过] {name}：{type(e).__name__} {e}")
    return out
