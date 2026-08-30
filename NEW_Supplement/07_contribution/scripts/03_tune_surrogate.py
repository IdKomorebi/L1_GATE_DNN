"""给代理模型定参数：多大的网络、训多少轮，才算把话说准了。

思路上的一个便宜
----------------
判断代理模型准不准，要拿它和"真的按这个字段组合从零训练"的结果比。
真训练很慢，但有一件事很关键：**真实答案和代理模型的参数无关**。
所以先把若干个字段组合的真实 R² 算一遍存下来，之后每换一组参数，
只要查代理模型（几乎不要钱），拿去和缓存的答案比就行。
一次投入，反复使用。

判据用什么
----------
不能用验证损失。验证损失低只说明"在随机抽的掩码上预测得准"，
但我们真正要的是"查任意字段组合时给出的 R² 对不对"，这是两回事。
所以判据直接就是校准偏差：代理模型给的 v(S) 和真实重训 R² 的平均绝对差。

还要单独看**差值**准不准。Shapley 值全部由 v(S∪{j}) − v(S) 这样的差算出来，
如果代理模型对所有组合都低估同样多，差值反而是准的，贡献值不受影响。
所以除了绝对偏差，还要报"配对差值的偏差"——这才是真正致命的那个量。
"""

from __future__ import annotations

import argparse
import itertools
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


CACHE = Path(__file__).resolve().parents[1] / "outputs" / "_truth_cache"


def build_truth(X, y, cfg, eval_idx, p, n_per_band=3, seed=7, cache_key=""):
    """抽字段组合并算出它们的真实 R²，存成缓存。

    额外抽一批**成对**的组合：S 和 S∪{j}，用来单独检验差值准不准。

    真实答案只和(数据集,目标,划分,评估行)有关，与代理模型的参数无关，
    所以落盘缓存起来，换参数重跑时直接读，不用再等几十次重训。
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{cache_key}_b{n_per_band}_s{seed}.json"
    rng = np.random.default_rng(seed)
    bands = sorted({b for b in [1, 2, 3, 5, 8, 12, 18, 26, 34, p - 2, p] if 1 <= b <= p})
    sets = []
    for k in bands:
        for _ in range(n_per_band):
            sets.append(sorted(rng.choice(p, size=k, replace=False).tolist()))

    # 成对组合：在若干个基底上各加一个字段
    pairs = []
    for k in [1, 3, 8, 20, 40]:
        if k >= p:
            continue
        for _ in range(4):
            base = sorted(rng.choice(p, size=k, replace=False).tolist())
            rest = [j for j in range(p) if j not in base]
            j = int(rng.choice(rest))
            sets.append(base)
            sets.append(sorted(base + [j]))
            pairs.append((tuple(base), tuple(sorted(base + [j])), j))

    seen, uniq = set(), []
    for s in sets:
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            uniq.append(s)

    cached = {}
    if cf.exists():
        raw = json.loads(cf.read_text())
        cached = {tuple(json.loads(k)): v for k, v in raw.items()}
        print(f"    命中缓存 {cf.name}：{len(cached)} 个组合")

    truth = {}
    t0 = time.time()
    todo = [s for s in uniq if tuple(s) not in cached]
    for i, s in enumerate(uniq):
        t = tuple(s)
        truth[t] = cached[t] if t in cached else \
            sg.retrain_reference(X, y, s, cfg, eval_idx)
        if t not in cached and (i + 1) % 10 == 0:
            print(f"    真实答案 {i+1}/{len(uniq)}  ({time.time()-t0:.0f}s)",
                  flush=True)
    if todo:
        print(f"    新算了 {len(todo)} 个组合，{time.time()-t0:.0f}s")
        cf.write_text(json.dumps({json.dumps(list(k)): v
                                  for k, v in truth.items()}))
    return truth, pairs


def score(vf, truth, pairs):
    """把代理模型的回答和缓存的真实答案比。"""
    sets = list(truth.keys())
    vhat = vf.of_sets([list(s) for s in sets])
    vtrue = np.array([truth[s] for s in sets])
    m = {t: float(v) for t, v in zip(sets, vhat)}
    err = vhat - vtrue

    # 差值检验：加一个字段带来的增量，代理模型说的和真的差多少
    dh = np.array([m[b] - m[a] for a, b, _ in pairs])
    dt = np.array([truth[b] - truth[a] for a, b, _ in pairs])

    from scipy.stats import spearmanr
    return dict(
        平均绝对偏差=float(np.abs(err).mean()),
        中位绝对偏差=float(np.median(np.abs(err))),
        系统性偏差=float(err.mean()),
        相关系数=float(np.corrcoef(vhat, vtrue)[0, 1]),
        秩相关=float(spearmanr(vhat, vtrue).statistic),
        差值平均绝对偏差=float(np.abs(dh - dt).mean()),
        差值相关系数=float(np.corrcoef(dh, dt)[0, 1]) if len(dh) > 2 else np.nan,
        差值符号一致率=float((np.sign(dh) == np.sign(dt)).mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--target", default="branch_ab1_loading_pct")
    ap.add_argument("--n-per-band", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=1500)
    a = ap.parse_args()

    t_start = time.time()
    d = ds.load(a.dataset)
    base, figd, datd = ds.run_dir(a.dataset, "L0_tune")
    print(f"输出目录 {base}")
    print(f"目标 {d.label(a.target)}（{a.target}）")

    pool = d.pool(a.target)
    X = d.df[pool].to_numpy(float)
    y = d.df[a.target].to_numpy(float)
    p = len(pool)
    print(f"候选池 {p} 个字段\n")

    ref_cfg = sg.SurrogateConfig(seed=42)
    _, _, i_te = sg._split_idx(len(X), ref_cfg)
    eval_idx = np.sort(np.random.default_rng(0).choice(
        i_te, size=min(a.n_eval, len(i_te)), replace=False))

    print("-- 第一步：算真实答案（一次投入，后面反复用）--")
    truth, pairs = build_truth(X, y, ref_cfg, eval_idx, p, a.n_per_band,
                               cache_key=f"{a.dataset}__{a.target}")

    grid = [
        dict(hidden=(64, 48), epochs=200),
        dict(hidden=(128, 96, 64), epochs=200),
        dict(hidden=(128, 96, 64), epochs=500),
        dict(hidden=(128, 96, 64), epochs=1000),
        dict(hidden=(256, 192, 128), epochs=500),
        dict(hidden=(256, 192, 128), epochs=1000),
        dict(hidden=(256, 192, 128), epochs=1500),
        dict(hidden=(384, 256, 192, 128), epochs=1000),
        dict(hidden=(384, 256, 192, 128), epochs=1500),
    ]

    print("\n-- 第二步：扫参数 --")
    rows = []
    for g in grid:
        cfg = sg.SurrogateConfig(hidden=tuple(g["hidden"]), epochs=g["epochs"],
                                 min_epochs=min(120, g["epochs"]), patience=150,
                                 seed=42)
        t0 = time.time()
        res = sg.fit(X, y, pool, cfg)
        vf = sg.ValueFunction(res, X, y, n_eval=a.n_eval, seed=0)
        s = score(vf, truth, pairs)
        s.update(网络="-".join(map(str, g["hidden"])), 轮次上限=g["epochs"],
                 实跑轮次=len(res.history["epoch"]), 最佳轮次=res.best_epoch,
                 验证损失=res.history["val_loss"][res.best_epoch],
                 耗时秒=round(time.time() - t0, 1))
        rows.append(s)
        print(f"  {s['网络']:<20} {g['epochs']:>5}轮(实跑{s['实跑轮次']:>4}) "
              f"偏差 {s['平均绝对偏差']:.4f}  差值偏差 {s['差值平均绝对偏差']:.4f}  "
              f"秩相关 {s['秩相关']:.4f}  {s['耗时秒']:.0f}s")

    df = pd.DataFrame(rows)
    cols = ["网络", "轮次上限", "实跑轮次", "最佳轮次", "验证损失", "平均绝对偏差",
            "中位绝对偏差", "系统性偏差", "相关系数", "秩相关",
            "差值平均绝对偏差", "差值相关系数", "差值符号一致率", "耗时秒"]
    df = df[cols].sort_values("差值平均绝对偏差")
    df.to_csv(datd / "tune_grid.csv", index=False)

    print(f"\n{'='*76}\n按差值偏差排序（这才是决定贡献值准不准的量）")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    bestrow = df.iloc[0]
    meta = {"数据集": a.dataset, "目标": a.target, "候选池": p,
            "真实答案组合数": len(truth), "成对组合数": len(pairs),
            "最优配置": {"网络": bestrow["网络"], "轮次上限": int(bestrow["轮次上限"])},
            "总耗时秒": round(time.time() - t_start, 1)}
    (datd / "config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 验证损失和真正关心的校准偏差之间是不是一回事
    from scipy.stats import spearmanr
    rho = spearmanr(df["验证损失"], df["差值平均绝对偏差"]).statistic
    print(f"\n验证损失与差值偏差的秩相关 = {rho:.3f}")
    print("（如果这个数不高，说明按验证损失选参数是选错东西——"
          "必须按校准偏差选，这正是本脚本存在的理由）")
    print(f"\n最优：{bestrow['网络']}，{int(bestrow['轮次上限'])} 轮上限")
    print(f"总耗时 {meta['总耗时秒']/60:.1f} 分钟 → {base}")


if __name__ == "__main__":
    main()
