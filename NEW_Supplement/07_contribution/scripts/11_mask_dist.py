"""训练时的掩码规模分布：把力气花在真正被用到的地方。

诊断出的病根
------------
贡献值全部由 Shapley 核加权算出来，这个核**极度偏向两头**——
只含一两个字段的组合、以及只差一两个字段的组合，权重最高。

可训练时掩码规模是均匀抽的。"只发布某一个指定字段"这种情形在训练样本里
的占比约 1/(p-1)/p，p=46 时是 0.05%——每轮六千个样本里只摊到两三个。
**模型最没训到的地方，恰恰是估计量最依赖的地方。**

实测佐证：两列逐个数完全相同的字段，模型给出的"只发布它一个"的还原能力
差了 0.071，而"全部发布再抽掉它"只差 0.0038。不对称完全集中在小组合上。

试三种分布
----------
  uniform  各规模等概率（原来的做法）
  kernel   完全按 Shapley 核，两头重中间轻
  mixture  两者各一半

判据同时看三个量，缺一不可：
  对称性   两个完全相同的字段之间差多少（理论为 0）
  校准偏差 代理模型和真实重训对不对得上（别调出一个又对称又不准的模型）
  中间规模 中间档的 v(S) 有没有被牺牲掉（纯 kernel 的风险就在这里）
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--target", default="bus_215_va_deg")
    ap.add_argument("--hidden", default="384,256,192,128")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--n-coalitions", type=int, default=16384)
    ap.add_argument("--n-probe", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43])
    ap.add_argument("--weight-decay", type=float, default=3e-3)
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    base, figd, datd = ds.run_dir(a.dataset, "L0_mask_dist")
    print(f"输出目录 {base}")

    pool = d.pool(a.target, drop_constants=False)
    g = ts.duplicate_groups_in_pool(d.df, pool)[0]
    i, j = pool.index(g[0]), pool.index(g[1])
    p = len(pool)
    print(f"目标 {d.label(a.target)}，候选池 {p}")
    print(f"完全重复的一对：{g[0]} 与 {g[1]}\n")

    for dist in ("uniform", "mixture", "kernel"):
        pr = sg.size_probs(p, dist)
        print(f"  {dist:<8} 规模 1 的概率 {pr[0]:.4f}，规模 {p//2} 的概率 "
              f"{pr[p//2-1]:.4f}，比值 {pr[0]/pr[p//2-1]:.1f}")

    X = d.df[pool].to_numpy(float)
    y = d.df[a.target].to_numpy(float)

    rng = np.random.default_rng(5)
    others = [q for q in range(p) if q not in (i, j)]
    # 互换检验的探针按规模分层，好分别看小/中/大组合上的对称性
    probes_by_band = {}
    for band, (lo, hi) in {"小(1~4)": (0, 4), "中(15~30)": (15, 30),
                           "大(40~44)": (40, 44)}.items():
        ps = []
        for _ in range(a.n_probe // 3):
            k = int(rng.integers(lo, min(hi, len(others)) + 1))
            ps.append(sorted(rng.choice(others, size=k, replace=False).tolist()))
        probes_by_band[band] = ps

    cfg0 = sg.SurrogateConfig(seed=42)
    _, _, i_te = sg._split_idx(len(X), cfg0)
    eval_idx = np.sort(np.random.default_rng(0).choice(
        i_te, size=min(a.n_eval, len(i_te)), replace=False))
    rng2 = np.random.default_rng(31)
    cal_sets, cal_bands = [], []
    for k in [1, 1, 1, 2, 2, 3, 3, 6, 8, 12, 16, 24, 30, 36, 40, 44, p]:
        for _ in range(2):
            s = sorted(rng2.choice(p, size=min(k, p), replace=False).tolist())
            cal_sets.append(s)
            cal_bands.append("小" if k <= 3 else ("中" if k <= 30 else "大"))
    print(f"\n算 {len(cal_sets)} 个校准参照答案（真实重训）…")
    cal_true = np.array([sg.retrain_reference(X, y, s, cfg0, eval_idx)
                         for s in cal_sets])
    print(f"  完成（{time.time()-t0:.0f}s）\n{'='*84}")

    rows = []
    for dist in ("uniform", "mixture", "kernel"):
        for sd in a.seeds:
            cfg = sg.SurrogateConfig(
                hidden=tuple(int(x) for x in a.hidden.split(",")),
                epochs=a.epochs, min_epochs=150, patience=150, seed=sd,
                mask_dist=dist, weight_decay=a.weight_decay)
            t1 = time.time()
            res = sg.fit(X, y, pool, cfg)
            vf = sg.ValueFunction(res, X, y, n_eval=a.n_eval, seed=0)

            eye = np.eye(p, dtype=np.float32)
            solo = vf(eye[[i, j]])
            rec = dict(掩码分布=dist, 种子=sd, v_全部=float(vf.full()),
                       单发布差=float(abs(solo[0] - solo[1])))
            for band, ps in probes_by_band.items():
                A = [sorted(s + [i]) for s in ps]
                B = [sorted(s + [j]) for s in ps]
                dd = np.abs(np.asarray(vf.of_sets(A)) - np.asarray(vf.of_sets(B)))
                rec[f"互换差_{band}"] = float(dd.mean())

            r = att.kernel_shap(vf, p, n_coalitions=a.n_coalitions, seed=7)
            phi = np.asarray(r.phi)
            rec["贡献值对称误差"] = float(abs(phi[i] - phi[j]))
            rec["贡献值对称相对误差"] = float(abs(phi[i] - phi[j]) /
                                       max(np.abs(phi).max(), 1e-12))

            cal_hat = np.asarray(vf.of_sets(cal_sets))
            err = np.abs(cal_hat - cal_true)
            rec["校准偏差"] = float(err.mean())
            for b in ("小", "中", "大"):
                m = [k for k, bb in enumerate(cal_bands) if bb == b]
                rec[f"校准偏差_{b}"] = float(err[m].mean())
            rec["耗时秒"] = round(time.time() - t1, 1)
            rows.append(rec)
            print(f"  {dist:<8} 种子{sd}  单发布差 {rec['单发布差']:.4f}  "
                  f"互换差 小{rec['互换差_小(1~4)']:.4f}/"
                  f"中{rec['互换差_中(15~30)']:.4f}/"
                  f"大{rec['互换差_大(40~44)']:.4f}  "
                  f"对称相对误差 {rec['贡献值对称相对误差']:.1%}  "
                  f"校准 {rec['校准偏差']:.4f}"
                  f"(小{rec['校准偏差_小']:.3f}/中{rec['校准偏差_中']:.3f}/"
                  f"大{rec['校准偏差_大']:.3f})  {rec['耗时秒']:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(datd / "mask_dist.csv", index=False)

    print(f"\n{'='*84}\n按分布聚合（跨种子平均）")
    g2 = df.groupby("掩码分布").mean(numeric_only=True).drop(columns=["种子"])
    print(g2.to_string(float_format=lambda x: f"{x:9.4f}"))

    best = g2["贡献值对称相对误差"].idxmin()
    print(f"\n对称性最好的分布：{best}")
    print(f"校准偏差最小的分布：{g2['校准偏差'].idxmin()}")
    print("\n读法：如果 kernel 或 mixture 把对称性显著压下来而校准偏差没变差，"
          "就说明病根确实是小组合训练不足，改分布是对症的；"
          "如果对称性降了但中间规模的校准偏差涨了，那是拆东墙补西墙，要取折中。")

    (datd / "config.json").write_text(json.dumps(
        {"数据集": a.dataset, "目标": a.target, "重复字段": g,
         "网络": a.hidden, "轮次": a.epochs, "种子": a.seeds,
         "抽取组合数": a.n_coalitions, "最优分布": best,
         "权重衰减": a.weight_decay,
         "总耗时秒": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base}")


if __name__ == "__main__":
    main()
