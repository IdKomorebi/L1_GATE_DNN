"""L0 校准：代理模型说的话到底算不算数。

为什么这一步必须最先做
----------------------
整套方法的逻辑是：训练一个随机屏蔽的代理模型 → 用它查任意字段组合的还原能力
→ 由这些查询结果算出每个字段的贡献值。第一环如果不准，后面全部作废，
而且在真实数据上是**看不出来**的——没有答案可以对照，错的数字看起来和对的一样。

所以要做一件笨但可靠的事：抽若干个字段组合，**真的按这个组合从零训练一个普通网络**，
把真实的 R² 和代理模型给出的 v(S) 摆在一起比。两者贴合，代理模型才可信。

抽组合的时候要覆盖各种规模，尤其是**小组合**——只有一两个字段的时候数据信息最少、
任务最难，代理模型最容易在这里失准，而这一段恰恰是衡量"字段独立能力"的地方，
权重很高。只在中等规模上验证是自欺欺人。

顺带做的事：先扫一下训练轮次，看代理模型收敛需要多少轮，
免得后面全量实验用一个没收敛的模型跑几个小时。
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

DEFAULT_TARGETS = [
    "gen_121_nuclear_1_pg_mw",     # 有完美替身，v(全部) 极高
    "branch_ab1_loading_pct",      # 中等难度，方法最能体现差别的档
    "bus_215_va_deg",              # 相角类
]


def sample_probe_sets(p: int, n_per_band: int, rng) -> list[list[int]]:
    """抽用于校准的字段组合，规模要盖住从 1 个到全部。

    小规模那几档单独加密，因为代理模型在那里最容易失准。
    """
    bands = [1, 2, 3, 5, 8, 12, 18, 26, 34, p - 2, p]
    bands = sorted({b for b in bands if 1 <= b <= p})
    out = []
    for k in bands:
        for _ in range(n_per_band):
            out.append(sorted(rng.choice(p, size=k, replace=False).tolist()))
    # 去重
    seen, uniq = set(), []
    for s in out:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            uniq.append(s)
    return uniq


def epoch_scan(X, y, fields, seeds=(42,), grid=(100, 200, 400, 700)) -> pd.DataFrame:
    """扫训练轮次：看代理模型要多少轮才收敛。"""
    rows = []
    for ep in grid:
        for sd in seeds:
            cfg = sg.SurrogateConfig(epochs=ep, min_epochs=ep, patience=10 ** 9,
                                     seed=sd)
            t0 = time.time()
            res = sg.fit(X, y, fields, cfg)
            vf = sg.ValueFunction(res, X, y, n_eval=1500, seed=0)
            rows.append(dict(轮次=ep, 种子=sd,
                             验证损失=res.history["val_loss"][-1],
                             v_全部=float(vf.full()), v_空=float(vf.empty()),
                             耗时秒=round(time.time() - t0, 1)))
            print(f"    轮次 {ep:4d} 种子 {sd}: 验证损失 {rows[-1]['验证损失']:.5f}  "
                  f"v(全部) {rows[-1]['v_全部']:.4f}  {rows[-1]['耗时秒']}s")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--per-band", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--skip-scan", action="store_true")
    a = ap.parse_args()

    t_start = time.time()
    d = ds.load(a.dataset)
    base, figd, datd = ds.run_dir(a.dataset, "L0_calibration")
    print(f"输出目录 {base}")

    meta = {"数据集": a.dataset, "目标": a.targets, "轮次": a.epochs,
            "每档抽样数": a.per_band, "评估行数": a.n_eval}
    all_probe, scans = [], []

    for ti, tgt in enumerate(a.targets):
        print(f"\n{'='*66}\n[{ti+1}/{len(a.targets)}] 目标 {d.label(tgt)}（{tgt}）")
        pool = d.pool(tgt)
        X = d.df[pool].to_numpy(float)
        y = d.df[tgt].to_numpy(float)
        p = len(pool)
        print(f"候选池 {p} 个字段")

        if not a.skip_scan:
            print("\n-- 扫训练轮次 --")
            sc = epoch_scan(X, y, pool)
            sc.insert(0, "目标", d.label(tgt))
            scans.append(sc)

        print(f"\n-- 训练正式代理模型（{a.epochs} 轮上限，早停开启）--")
        cfg = sg.SurrogateConfig(epochs=a.epochs, seed=42)
        t0 = time.time()
        res = sg.fit(X, y, pool, cfg, verbose=True)
        print(f"最佳轮次 {res.best_epoch}，实跑 {len(res.history['epoch'])} 轮，"
              f"{time.time()-t0:.0f}s")
        rp.training_curve(res.history, figd / f"fig_训练曲线_{tgt}.png")

        vf = sg.ValueFunction(res, X, y, n_eval=a.n_eval, seed=0)
        eval_idx = vf.idx

        rng = np.random.default_rng(7)
        probes = sample_probe_sets(p, a.per_band, rng)
        print(f"\n-- 校准：{len(probes)} 个字段组合，每个都真的重训一遍 --")

        rows = []
        for i, s in enumerate(probes):
            v_hat = float(vf.of_sets([s])[0])
            t1 = time.time()
            v_true = sg.retrain_reference(X, y, s, cfg, eval_idx)
            rows.append(dict(目标=d.label(tgt), 字段数=len(s),
                             代理模型_vS=v_hat, 真实重训R2=v_true,
                             偏差=v_hat - v_true, 重训耗时=round(time.time()-t1, 1)))
            if (i + 1) % 10 == 0 or i == len(probes) - 1:
                sofar = pd.DataFrame(rows)
                print(f"  {i+1:3d}/{len(probes)}  "
                      f"平均绝对偏差 {sofar['偏差'].abs().mean():.4f}  "
                      f"（已用 {time.time()-t0:.0f}s）")

        pr = pd.DataFrame(rows)
        pr = pr.rename(columns={"代理模型_vS": "代理模型v(S)"})
        all_probe.append(pr)

        rp.calibration(pr, figd / f"fig_校准_{tgt}.png")

        mae = pr["偏差"].abs().mean()
        corr = pr["代理模型v(S)"].corr(pr["真实重训R2"])
        small = pr[pr["字段数"] <= 3]
        print(f"\n  平均绝对偏差 {mae:.4f}，相关系数 {corr:.4f}")
        print(f"  小组合（≤3 字段，{len(small)} 个）平均绝对偏差 "
              f"{small['偏差'].abs().mean():.4f}")
        print("  按字段数分档：")
        print(pr.groupby("字段数")[["代理模型v(S)", "真实重训R2", "偏差"]]
              .mean().to_string(float_format=lambda x: f"{x:8.4f}"))

    probe = pd.concat(all_probe, ignore_index=True)
    probe.to_csv(datd / "calibration_probes.csv", index=False)
    if scans:
        pd.concat(scans, ignore_index=True).to_csv(datd / "epoch_scan.csv", index=False)
    rp.calibration(probe, figd / "fig_校准_全部.png")

    summ = probe.groupby("目标").apply(
        lambda g: pd.Series({
            "组合数": len(g),
            "平均绝对偏差": g["偏差"].abs().mean(),
            "中位绝对偏差": g["偏差"].abs().median(),
            "相关系数": g["代理模型v(S)"].corr(g["真实重训R2"]),
            "小组合平均绝对偏差": g[g["字段数"] <= 3]["偏差"].abs().mean(),
            "系统性偏差": g["偏差"].mean(),
        }), include_groups=False).reset_index()
    summ.to_csv(datd / "calibration_summary.csv", index=False)

    print(f"\n{'='*66}\n汇总")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))

    meta["总耗时秒"] = round(time.time() - t_start, 1)
    meta["整体平均绝对偏差"] = float(probe["偏差"].abs().mean())
    meta["整体相关系数"] = float(probe["代理模型v(S)"].corr(probe["真实重训R2"]))
    (datd / "config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物写入 {base}")
    print(f"总耗时 {meta['总耗时秒']/60:.1f} 分钟")


if __name__ == "__main__":
    main()
