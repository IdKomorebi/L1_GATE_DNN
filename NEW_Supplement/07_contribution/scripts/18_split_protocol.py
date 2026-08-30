"""两种划分口径下，每个敏感目标到底推不推得出来。

为什么这件事要单独查
--------------------
"划分方式"听起来像个实现细节，其实它就是**攻击者假设**：

  随机划分  训练集和测试集在时间上交错。相当于假设外部**已经掌握了同一批数据里
            的一部分配对样本**——知道某些时刻的公开字段和对应的敏感值，
            要去补出剩下那些时刻。
  时序划分  用前 70% 训练、后 20% 测试。相当于假设外部**只有历史数据**，
            要去推断**新发生**的运行状态。

同一批字段在这两种假设下的风险完全可以不一样。发布前审查关心的多半是后者
（历史数据已经发出去了，问的是接下来会不会被推出来），而仓库既有的主结果
用的是前者。两个数不能混着引。

量的是什么，要非常小心
--------------------------
"这个目标推不推得出来"必须用**专门为这个字段组合训练的普通网络**来量，
**不能用随机屏蔽的代理模型**。代理模型要同时应付所有字段组合，
在任何单个组合上都不如专训的模型——这个亏空叫摊销代价。

第一版脚本在这里踩了坑：报的是代理模型全 1 掩码下的 v(全部)，
在 PJM 的交换功率类目标上得到 −1.21 / −0.33 / −0.92，据此写下了
"这些目标推不出来"的结论。**这是错的。** 同样口径下专训普通网络能到
0.34 / −0.04 / 0.36——它们是**弱可推**，不是不可推。
那些负数量的是代理模型在难目标上的能力上限，不是可推断性。

（对照：容易的目标上两者几乎一致，总发电量 0.988 对 0.970，
计量负荷 0.998。摊销代价只在目标本身难预测时才变大。）

所以本脚本一律用 `sg.retrain_reference` 在全部候选字段上训一个普通网络，
不碰代理模型。这样也更便宜。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["pjm_2025", "rts_v2"])
    ap.add_argument("--epochs", type=int, default=600)  # 保留但已不用（改用专训普通网络）
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43])
    a = ap.parse_args()

    t0 = time.time()
    rows = []
    for dsname in a.datasets:
        d = ds.load(dsname)
        print(f"\n{'='*78}\n数据集 {dsname}：{d.df.shape}，敏感目标 {len(d.targets)} 个")
        for tgt in d.targets:
            pool = d.pool(tgt)
            X = d.df[pool].to_numpy(float)
            y = d.df[tgt].to_numpy(float)
            rec = {"数据集": dsname, "目标": d.label(tgt), "字段名": tgt,
                   "候选数": len(pool)}
            for split in ("random", "temporal"):
                vs = []
                for sd in a.seeds:
                    cfg = sg.SurrogateConfig(seed=sd, split=split)
                    _, _, i_te = sg._split_idx(len(X), cfg)
                    ev = np.sort(np.random.default_rng(0).choice(
                        i_te, size=min(a.n_eval, len(i_te)), replace=False))
                    vs.append(sg.retrain_reference(
                        X, y, list(range(len(pool))), cfg, ev, seed=sd))
                rec[f"{split}_可推断性"] = float(np.mean(vs))
                rec[f"{split}_种子间标准差"] = float(np.std(vs))
            rec["两口径之差"] = rec["random_可推断性"] - rec["temporal_可推断性"]
            rows.append(rec)
            print(f"  {rec['目标']:<24} 随机 {rec['random_可推断性']:7.4f}   "
                  f"时序 {rec['temporal_可推断性']:7.4f}   "
                  f"差 {rec['两口径之差']:7.4f}", flush=True)

    df = pd.DataFrame(rows)
    base, figd, datd = ds.run_dir("_cross", "split_protocol")
    df.to_csv(datd / "split_protocol.csv", index=False)

    print(f"\n{'='*78}\n汇总")
    for dsname, g in df.groupby("数据集", sort=False):
        n_bad_t = int((g["temporal_可推断性"] < 0.3).sum())
        n_bad_r = int((g["random_可推断性"] < 0.3).sum())
        print(f"\n{dsname}：共 {len(g)} 个目标")
        print(f"  随机划分下 v(全部) < 0.3 的：{n_bad_r} 个")
        print(f"  时序划分下 v(全部) < 0.3 的：{n_bad_t} 个")
        print(f"  两口径平均差（随机 − 时序）：{g['两口径之差'].mean():.4f}")
        bad = g[g["temporal_可推断性"] < 0.3]
        if len(bad):
            print("  时序划分下推不出来的目标：")
            for _, r in bad.iterrows():
                print(f"    {r['目标']:<24} 随机 {r['random_可推断性']:7.4f} → "
                      f"时序 {r['temporal_可推断性']:7.4f}")

    # 出图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    for dsname, g in df.groupby("数据集", sort=False):
        ax.scatter(g["random_可推断性"], g["temporal_可推断性"], s=42, alpha=0.85,
                   label=dsname, edgecolors="white", linewidths=0.6)
    lo = float(min(df["random_可推断性"].min(), df["temporal_可推断性"].min(), 0)) - 0.1
    ax.plot([lo, 1.05], [lo, 1.05], color="#444444", ls="--", lw=0.9)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.axhline(0.3, color="#E15759", ls=":", lw=1.0)
    ax.set_xlabel("随机划分下的可推断性（外部已有同批配对样本）")
    ax.set_ylabel("时序划分下的可推断性（外部只有历史数据）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(figd / "fig_两种划分口径.png", bbox_inches="tight")
    plt.close(fig)

    (datd / "config.json").write_text(json.dumps(
        {"数据集": a.datasets, "轮次": a.epochs, "种子": a.seeds,
         "评估行数": a.n_eval, "总耗时秒": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base}")


if __name__ == "__main__":
    main()
