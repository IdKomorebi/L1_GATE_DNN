"""诊断：代理模型对"加一个字段带来多少增量"的判断，什么时候可信。

上一步扫参数时冒出一个刺眼的数字：**增量的符号只有 55% 说对**，几乎和抛硬币一样。
如果这是真的，整套方法就废了——贡献值完全由增量算出来。

但先别急着下结论，有一个更平常的解释：往一个已经有 40 个字段的组合里再加一个字段，
真实增量本来就可能只有万分之几。而"真实增量"是拿两次**独立训练**的网络的
测试 R² 相减得到的，每次训练自己就有随机波动。两个小于波动的数相减，
得到的符号本来就是随机的——这时候符号不一致不是代理模型的错，
是那个"真值"本身就测不出来。

所以这一步把增量按大小分档，逐档看符号一致率和相对误差。
判据很清楚：**增量大到超过参照答案自身的噪声时，符号必须说对**；
小于噪声的那部分说不对，是问题本身没有分辨率，不是方法的缺陷。
这两件事必须分开报，否则一个 0.55 会把结论带到完全错误的方向。
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
from src import report as rp            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--target", default="branch_ab1_loading_pct")
    ap.add_argument("--hidden", default="384,256,192,128")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--ref-seeds", type=int, nargs="*", default=[42, 43, 44])
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    base, figd, datd = ds.run_dir(a.dataset, "L0_diff_diagnosis")
    print(f"输出目录 {base}")

    pool = d.pool(a.target)
    X = d.df[pool].to_numpy(float)
    y = d.df[a.target].to_numpy(float)
    p = len(pool)
    print(f"目标 {d.label(a.target)}，候选池 {p} 个字段")

    cfg = sg.SurrogateConfig(hidden=tuple(int(x) for x in a.hidden.split(",")),
                             epochs=a.epochs, min_epochs=150, patience=150, seed=42)
    _, _, i_te = sg._split_idx(len(X), cfg)
    eval_idx = np.sort(np.random.default_rng(0).choice(
        i_te, size=min(a.n_eval, len(i_te)), replace=False))

    print("\n训练代理模型…")
    res = sg.fit(X, y, pool, cfg)
    vf = sg.ValueFunction(res, X, y, n_eval=a.n_eval, seed=0)
    print(f"  完成，{time.time()-t0:.0f}s，最佳轮次 {res.best_epoch}")

    # 构造成对组合：底集规模从很小到很大，每档若干对
    rng = np.random.default_rng(23)
    pairs = []
    for k in [0, 1, 2, 4, 8, 16, 30, p - 1]:
        if k >= p:
            continue
        for _ in range(a.n_pairs):
            b0 = sorted(rng.choice(p, size=k, replace=False).tolist()) if k else []
            rest = [j for j in range(p) if j not in b0]
            j = int(rng.choice(rest))
            pairs.append((b0, sorted(b0 + [j]), j, k))

    print(f"\n{len(pairs)} 对组合，每对的两端各用 {len(a.ref_seeds)} 个种子重训")
    rows = []
    for i, (b0, b1, j, k) in enumerate(pairs):
        vh = vf.of_sets([b0, b1])
        d_hat = float(vh[1] - vh[0])
        t_lo = [sg.retrain_reference(X, y, b0, cfg, eval_idx, seed=s)
                for s in a.ref_seeds]
        t_hi = [sg.retrain_reference(X, y, b1, cfg, eval_idx, seed=s)
                for s in a.ref_seeds]
        d_true = float(np.mean(t_hi) - np.mean(t_lo))
        # 真值增量自身的不确定度：两端各自的种子标准差合成
        sd = float(np.sqrt(np.var(t_lo) / len(t_lo) + np.var(t_hi) / len(t_hi)))
        rows.append(dict(底集规模=k, 加入字段=pool[j],
                         代理增量=d_hat, 真实增量=d_true,
                         真值不确定度=sd,
                         可分辨=abs(d_true) > 2 * sd,
                         符号一致=np.sign(d_hat) == np.sign(d_true),
                         绝对误差=abs(d_hat - d_true)))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pairs)}  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(datd / "diff_diagnosis.csv", index=False)

    print(f"\n{'='*74}")
    print("按真实增量的大小分档")
    bins = [0, 0.005, 0.02, 0.05, 0.15, 1e9]
    labels = ["<0.005", "0.005~0.02", "0.02~0.05", "0.05~0.15", ">0.15"]
    df["增量档"] = pd.cut(df["真实增量"].abs(), bins=bins, labels=labels,
                        include_lowest=True)
    g = df.groupby("增量档", observed=True).agg(
        对数=("符号一致", "size"),
        符号一致率=("符号一致", "mean"),
        平均绝对误差=("绝对误差", "mean"),
        真值不确定度=("真值不确定度", "mean"),
        可分辨比例=("可分辨", "mean"))
    print(g.to_string(float_format=lambda x: f"{x:10.4f}"))

    print(f"\n{'='*74}")
    print("只看真实增量确实可分辨的那些（增量 > 2 倍自身不确定度）")
    sub = df[df["可分辨"]]
    if len(sub):
        print(f"  共 {len(sub)}/{len(df)} 对")
        print(f"  符号一致率 {sub['符号一致'].mean():.3f}")
        print(f"  平均绝对误差 {sub['绝对误差'].mean():.4f}")
        print(f"  增量相关系数 {sub['代理增量'].corr(sub['真实增量']):.4f}")
    unres = df[~df["可分辨"]]
    if len(unres):
        print(f"\n不可分辨的那些（{len(unres)} 对）：")
        print(f"  符号一致率 {unres['符号一致'].mean():.3f}"
              f"（这里接近 0.5 是正常的——真值本身就测不出来）")
        print(f"  真实增量绝对值中位数 {unres['真实增量'].abs().median():.5f}，"
              f"而真值不确定度中位数 {unres['真值不确定度'].median():.5f}")

    print(f"\n{'='*74}")
    print("按底集规模分档（看代理模型在哪一段最吃力）")
    g2 = df.groupby("底集规模").agg(
        对数=("符号一致", "size"), 符号一致率=("符号一致", "mean"),
        平均绝对误差=("绝对误差", "mean"),
        真实增量绝对值均值=("真实增量", lambda s: s.abs().mean()))
    print(g2.to_string(float_format=lambda x: f"{x:10.4f}"))

    # 出图
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lo = float(min(df["真实增量"].min(), df["代理增量"].min()))
    hi = float(max(df["真实增量"].max(), df["代理增量"].max()))
    ax.plot([lo, hi], [lo, hi], color="#E15759", lw=1.2)
    ax.axhline(0, color="#999999", lw=0.7)
    ax.axvline(0, color="#999999", lw=0.7)
    m = df["可分辨"]
    ax.scatter(df.loc[~m, "真实增量"], df.loc[~m, "代理增量"], s=26,
               color="#BAB0AC", label="真值本身测不出来", alpha=0.8)
    ax.scatter(df.loc[m, "真实增量"], df.loc[m, "代理增量"], s=32,
               color="#4E79A7", label="真值可分辨", alpha=0.9,
               edgecolors="white", linewidths=0.5)
    ax.set_xlabel("真实增量（多种子平均后的重训 R² 之差）")
    ax.set_ylabel("代理模型给出的增量")
    ax.legend(fontsize=8)
    fig.savefig(figd / "fig_增量诊断.png", bbox_inches="tight")
    plt.close(fig)

    meta = dict(数据集=a.dataset, 目标=a.target, 代理模型=cfg.to_dict(),
                对数=len(df), 参照种子=a.ref_seeds,
                可分辨对数=int(m.sum()),
                可分辨符号一致率=float(sub["符号一致"].mean()) if len(sub) else None,
                总耗时秒=round(time.time() - t0, 1))
    (datd / "config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base}")


if __name__ == "__main__":
    main()
