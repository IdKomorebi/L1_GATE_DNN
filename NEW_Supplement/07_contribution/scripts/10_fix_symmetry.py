"""修对称性：两条有机制依据的办法，实测哪条管用。

问题
----
候选池里两列数字**逐个数完全相同**（全系统核电出力 与 1 区核电出力），
代理模型却给出 v(只发布甲)=0.105、v(只发布乙)=0.176。被求解的那个函数自己
就不对称，于是贡献值也不对称。加大抽样量压不下去——那不是抽样误差。

为什么会这样：两个一模一样的输入，网络把权重怎么分给它们都不影响输出
（8 分 2 分和 5 分 5 分给出同样的预测）。训练损失对这件事没有偏好，
于是随机初始化落在哪里就停在哪里。

两条办法
--------
**一 加大权重衰减。** 这一条有确切的机制：两个相同的输入 w₁x + w₂x = (w₁+w₂)x，
在总和固定的前提下，让 w₁²+w₂² 最小的解**必然是 w₁ = w₂**。
也就是说权重衰减本身就在把网络往对称解上推。现在用的 1e-5 太小，几乎不起作用。
代价是衰减太大会伤精度，所以要扫一遍找拐点。

**二 多个种子平均。** 每个种子把权重分偏的方向是随机的，平均之后互相抵消。
代价是训练时间乘以种子数。

两条可以叠加。判据是同时看两个量——对称性误差要降，而校准偏差不能变差。
只看前者会调出一个又对称又不准的模型。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import datasets as ds          # noqa: E402
from src import surrogate as sg         # noqa: E402
from src import attribution as att      # noqa: E402
from src import truthsets as ts         # noqa: E402


class Ensemble:
    """把若干个独立训练的代理模型平均成一个 v(S)。"""

    def __init__(self, vfs):
        self.vfs = vfs
        self.p = vfs[0].p
        self.idx = vfs[0].idx

    def __call__(self, masks, batch: int = 64):
        return np.mean([vf(masks, batch) for vf in self.vfs], axis=0)

    def of_sets(self, sets):
        return np.mean([vf.of_sets(sets) for vf in self.vfs], axis=0)

    def full(self):
        return self(np.ones((1, self.p), dtype=np.float32))[0]

    def empty(self):
        return self(np.zeros((1, self.p), dtype=np.float32))[0]


def sym_metrics(vf, i, j, p, probes):
    """量 v 自身的对称程度：只发布单个、抽掉单个、以及随机组合上的互换差。"""
    eye = np.eye(p, dtype=np.float32)
    solo = vf(eye[[i, j]])
    loo = vf(1.0 - eye[[i, j]])
    A = [sorted(s + [i]) for s in probes]
    B = [sorted(s + [j]) for s in probes]
    dd = np.abs(np.asarray(vf.of_sets(A)) - np.asarray(vf.of_sets(B)))
    return dict(单发布差=float(abs(solo[0] - solo[1])),
                抽掉差=float(abs(loo[0] - loo[1])),
                互换差均值=float(dd.mean()), 互换差最大=float(dd.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--target", default="bus_215_va_deg")
    ap.add_argument("--hidden", default="384,256,192,128")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--wds", type=float, nargs="*",
                    default=[1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--ens-seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--n-coalitions", type=int, default=16384)
    ap.add_argument("--n-probe", type=int, default=300)
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    base, figd, datd = ds.run_dir(a.dataset, "L0_fix_symmetry")
    print(f"输出目录 {base}")

    pool = d.pool(a.target, drop_constants=False)
    dups = ts.duplicate_groups_in_pool(d.df, pool)
    g = dups[0]
    i, j = pool.index(g[0]), pool.index(g[1])
    p = len(pool)
    print(f"目标 {d.label(a.target)}，候选池 {p}")
    print(f"完全重复的一对：{g[0]} 与 {g[1]}")

    X = d.df[pool].to_numpy(float)
    y = d.df[a.target].to_numpy(float)

    rng = np.random.default_rng(5)
    others = [q for q in range(p) if q not in (i, j)]
    probes = [sorted(rng.choice(others,
                                size=int(rng.integers(0, len(others) + 1)),
                                replace=False).tolist())
              for _ in range(a.n_probe)]

    # 校准用的参照答案：一小批组合的真实重训 R²，用来确认精度没被衰减拖垮
    cfg0 = sg.SurrogateConfig(seed=42)
    _, _, i_te = sg._split_idx(len(X), cfg0)
    eval_idx = np.sort(np.random.default_rng(0).choice(
        i_te, size=min(a.n_eval, len(i_te)), replace=False))
    rng2 = np.random.default_rng(31)
    cal_sets = []
    for k in [1, 2, 4, 10, 24, 40, p]:
        for _ in range(2):
            cal_sets.append(sorted(rng2.choice(p, size=min(k, p),
                                               replace=False).tolist()))
    print(f"\n算 {len(cal_sets)} 个校准参照答案（真实重训）…")
    cal_true = np.array([sg.retrain_reference(X, y, s, cfg0, eval_idx)
                         for s in cal_sets])
    print(f"  完成（{time.time()-t0:.0f}s）")

    rows = []
    print(f"\n{'='*80}\n扫权重衰减（单个种子）")
    for wd in a.wds:
        cfg = sg.SurrogateConfig(hidden=tuple(int(x) for x in a.hidden.split(",")),
                                 epochs=a.epochs, min_epochs=150, patience=150,
                                 seed=42, weight_decay=wd)
        t1 = time.time()
        res = sg.fit(X, y, pool, cfg)
        vf = sg.ValueFunction(res, X, y, n_eval=a.n_eval, seed=0)
        m = sym_metrics(vf, i, j, p, probes)
        r = att.kernel_shap(vf, p, n_coalitions=a.n_coalitions, seed=7)
        phi = np.asarray(r.phi)
        cal_hat = np.asarray(vf.of_sets(cal_sets))
        rec = dict(权重衰减=wd, 种子数=1, v_全部=float(vf.full()),
                   贡献值对称误差=float(abs(phi[i] - phi[j])),
                   贡献值对称相对误差=float(abs(phi[i] - phi[j]) /
                                    max(np.abs(phi).max(), 1e-12)),
                   校准平均绝对偏差=float(np.abs(cal_hat - cal_true).mean()),
                   耗时秒=round(time.time() - t1, 1), **m)
        rows.append(rec)
        print(f"  wd={wd:<8.0e} v(全部)={rec['v_全部']:.4f}  "
              f"单发布差 {m['单发布差']:.4f}  互换差均值 {m['互换差均值']:.4f}  "
              f"贡献值对称误差 {rec['贡献值对称误差']:.4f}"
              f"（相对 {rec['贡献值对称相对误差']:.1%}）  "
              f"校准偏差 {rec['校准平均绝对偏差']:.4f}  {rec['耗时秒']:.0f}s",
              flush=True)

    # 挑一个衰减：对称性明显改善、校准偏差不劣化太多
    df = pd.DataFrame(rows)
    baseline_cal = float(df.iloc[0]["校准平均绝对偏差"])
    okd = df[df["校准平均绝对偏差"] <= baseline_cal * 1.15]
    best_wd = float(okd.sort_values("互换差均值").iloc[0]["权重衰减"]) \
        if len(okd) else float(df.iloc[0]["权重衰减"])
    print(f"\n在校准偏差不超过基准 1.15 倍的前提下，互换差最小的衰减 = {best_wd:.0e}")

    print(f"\n{'='*80}\n多种子平均（衰减固定在 {best_wd:.0e}）")
    for nseed in range(1, len(a.ens_seeds) + 1):
        seeds = a.ens_seeds[:nseed]
        t1 = time.time()
        vfs = []
        for sd in seeds:
            cfg = sg.SurrogateConfig(
                hidden=tuple(int(x) for x in a.hidden.split(",")),
                epochs=a.epochs, min_epochs=150, patience=150,
                seed=sd, weight_decay=best_wd)
            res = sg.fit(X, y, pool, cfg)
            vfs.append(sg.ValueFunction(res, X, y, n_eval=a.n_eval, seed=0))
        ens = Ensemble(vfs)
        m = sym_metrics(ens, i, j, p, probes)
        r = att.kernel_shap(ens, p, n_coalitions=a.n_coalitions, seed=7)
        phi = np.asarray(r.phi)
        cal_hat = np.asarray(ens.of_sets(cal_sets))
        rec = dict(权重衰减=best_wd, 种子数=nseed, v_全部=float(ens.full()),
                   贡献值对称误差=float(abs(phi[i] - phi[j])),
                   贡献值对称相对误差=float(abs(phi[i] - phi[j]) /
                                    max(np.abs(phi).max(), 1e-12)),
                   校准平均绝对偏差=float(np.abs(cal_hat - cal_true).mean()),
                   耗时秒=round(time.time() - t1, 1), **m)
        rows.append(rec)
        print(f"  {nseed} 个种子：单发布差 {m['单发布差']:.4f}  "
              f"互换差均值 {m['互换差均值']:.4f}  "
              f"贡献值对称误差 {rec['贡献值对称误差']:.4f}"
              f"（相对 {rec['贡献值对称相对误差']:.1%}）  "
              f"校准偏差 {rec['校准平均绝对偏差']:.4f}  {rec['耗时秒']:.0f}s",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(datd / "fix_symmetry.csv", index=False)
    print(f"\n{'='*80}")
    cols = ["权重衰减", "种子数", "v_全部", "单发布差", "互换差均值",
            "贡献值对称误差", "贡献值对称相对误差", "校准平均绝对偏差", "耗时秒"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    best = df.sort_values(["贡献值对称相对误差"]).iloc[0]
    print(f"\n最佳组合：权重衰减 {best['权重衰减']:.0e}，"
          f"{int(best['种子数'])} 个种子 → "
          f"对称相对误差 {best['贡献值对称相对误差']:.1%}，"
          f"校准偏差 {best['校准平均绝对偏差']:.4f}")

    (datd / "config.json").write_text(json.dumps(
        {"数据集": a.dataset, "目标": a.target, "重复字段": g,
         "衰减档": a.wds, "种子": a.ens_seeds, "抽取组合数": a.n_coalitions,
         "推荐": {"权重衰减": float(best["权重衰减"]),
                "种子数": int(best["种子数"])},
         "总耗时秒": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base}")


if __name__ == "__main__":
    main()
