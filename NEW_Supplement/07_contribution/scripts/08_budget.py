"""抽多少个字段组合才够：给采样量找一个有依据的数字。

贡献值是靠抽若干个字段组合、再解一个加权最小二乘算出来的。
抽得越多越准，但也越慢。"抽多少算够"如果拍脑袋定，
就等于在批评别人阈值主观的同时自己引入一个更主观的数字。

这里有一个很便宜的办法把它变成客观的：**用一个已知答案的量当标尺**。
RTS-GMLC 的候选池里，"全系统核电出力"和"1 区核电出力"这两列数字
**逐个数完全相同**。任何讲得通的分法都必须给它们**完全相等**的份额，
理论差值就是 0。于是它们实际差了多少，就是这个采样量下的绝对误差。

这比常见的"看相邻两档结果稳不稳"强，因为稳定不等于正确——
抽样有系统偏差时，两个错得一样的结果之间也很稳。有了确切为 0 的对照，
量出来的就是真误差。

同时报三个量：
  对称性绝对误差   两个完全相同的字段之间实际差了多少（理论应为 0）
  恒定字段的份额   信息量为零的字段实际拿到多少（理论应为 0）
  与上一档的秩相关 排序还在不在变
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import datasets as ds          # noqa: E402
from src import surrogate as sg         # noqa: E402
from src import attribution as att      # noqa: E402
from src import truthsets as ts         # noqa: E402
from src import report as rp            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--targets", nargs="*",
                    default=["bus_215_va_deg", "branch_ab1_loading_pct",
                             "gen_218_cc_1_pg_mw"])
    ap.add_argument("--hidden", default="384,256,192,128")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--budgets", type=int, nargs="*",
                    default=[1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    base, figd, datd = ds.run_dir(a.dataset, "L0_budget")
    print(f"输出目录 {base}")

    cfg = sg.SurrogateConfig(hidden=tuple(int(x) for x in a.hidden.split(",")),
                             epochs=a.epochs, min_epochs=150, patience=150, seed=42)

    rows = []
    for tgt in a.targets:
        pool = d.pool(tgt, drop_constants=False)
        dups = ts.duplicate_groups_in_pool(d.df, pool)
        consts = [c for c in pool if c in d.constants]
        idx = {f: i for i, f in enumerate(pool)}
        print(f"\n{'='*74}\n目标 {d.label(tgt)}，候选池 {len(pool)}，"
              f"完全重复组 {len(dups)}，恒定字段 {len(consts)}")
        if not dups:
            print("  这个目标的池子里没有完全重复的字段，跳过（没有绝对标尺）")
            continue

        X = d.df[pool].to_numpy(float)
        y = d.df[tgt].to_numpy(float)
        print("  训练代理模型…")
        res = sg.fit(X, y, pool, cfg)
        vf = sg.ValueFunction(res, X, y, n_eval=a.n_eval, seed=0)
        print(f"  完成，v(全部)={float(vf.full()):.4f}")

        prev = None
        for nb in a.budgets:
            sym, zer, phis = [], [], []
            t1 = time.time()
            for rep in range(a.repeats):
                r = att.kernel_shap(vf, len(pool), n_coalitions=nb, seed=100 + rep)
                phi = np.asarray(r.phi)
                phis.append(phi)
                for g in dups:
                    v = phi[[idx[f] for f in g]]
                    sym.append(float(v.max() - v.min()))
                if consts:
                    zer.append(float(np.abs(phi[[idx[c] for c in consts]]).max()))
            phim = np.mean(phis, axis=0)
            scale = float(np.abs(phim).max())
            rec = dict(目标=d.label(tgt), 抽取组合数=nb,
                       对称性绝对误差=float(np.mean(sym)),
                       对称性相对误差=float(np.mean(sym)) / max(scale, 1e-12),
                       重复间秩相关=float(np.mean([
                           spearmanr(phis[i], phis[j]).statistic
                           for i in range(len(phis)) for j in range(i + 1, len(phis))
                       ])) if a.repeats > 1 else np.nan,
                       恒定字段最大份额=float(np.mean(zer)) if zer else np.nan,
                       与上档秩相关=(float(spearmanr(phim, prev).statistic)
                                if prev is not None else np.nan),
                       最大份额=scale,
                       耗时秒=round(time.time() - t1, 1))
            prev = phim
            rows.append(rec)
            print(f"  {nb:>6} 组合：对称性绝对误差 {rec['对称性绝对误差']:.5f}"
                  f"（相对 {rec['对称性相对误差']:.2%}）"
                  f"  重复间秩相关 {rec['重复间秩相关']:.4f}"
                  f"  与上档 {rec['与上档秩相关']:.4f}"
                  f"  {rec['耗时秒']:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(datd / "budget_scan.csv", index=False)

    print(f"\n{'='*74}\n汇总（跨目标平均）")
    g = df.groupby("抽取组合数").agg(
        对称性绝对误差=("对称性绝对误差", "mean"),
        对称性相对误差=("对称性相对误差", "mean"),
        重复间秩相关=("重复间秩相关", "mean"),
        与上档秩相关=("与上档秩相关", "mean"),
        单次耗时秒=("耗时秒", "mean")).reset_index()
    print(g.to_string(index=False, float_format=lambda x: f"{x:10.5f}"))

    # 选一个数：对称性相对误差首次降到 2% 以下
    ok = g[g["对称性相对误差"] < 0.02]
    pick = int(ok["抽取组合数"].iloc[0]) if len(ok) else int(g["抽取组合数"].iloc[-1])
    print(f"\n建议采样量 = {pick}"
          f"（对称性相对误差首次降到 2% 以下的那一档）")
    if not len(ok):
        print("  [注意] 没有任何一档达到 2%，这里给的是最大的一档，"
              "结论里必须写明这一点")

    # 出图
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for tgt, sub in df.groupby("目标"):
        axes[0].plot(sub["抽取组合数"], sub["对称性相对误差"], marker="o",
                     ms=4, label=tgt)
        axes[1].plot(sub["抽取组合数"], sub["重复间秩相关"], marker="s", ms=4,
                     label=tgt)
    axes[0].axhline(0.02, color="#E15759", ls="--", lw=1.0)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("抽取的字段组合数")
    axes[0].set_ylabel("对称性相对误差（理论应为 0）")
    axes[0].legend(fontsize=7.5)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("抽取的字段组合数")
    axes[1].set_ylabel("独立重复之间的秩相关")
    axes[1].legend(fontsize=7.5)
    fig.savefig(figd / "fig_采样量收敛.png", bbox_inches="tight")
    plt.close(fig)

    (datd / "config.json").write_text(json.dumps(
        {"数据集": a.dataset, "目标": a.targets, "代理模型": cfg.to_dict(),
         "档位": a.budgets, "每档重复": a.repeats, "建议采样量": pick,
         "总耗时秒": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base}")


if __name__ == "__main__":
    main()
