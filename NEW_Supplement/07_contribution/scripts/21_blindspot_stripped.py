"""剥离之后，"门控把能单独泄露目标的字段判为零"这件事还成不成立？

为什么必须查这一条
------------------
前面的统计（独立能力 ≥0.5 的字段里 53.8% 被分解式门控判为零）是在**未剥离**的
候选池上做的。本目录默认不剥离，因为验证需要把已知关系留在池子里当答案键。

但正式流程是**先剥离再门控**：第一层按数据定义剔掉公式可算的字段，
第二层逐层剥离到目标无法被线性还原（相对残差 ≥ 0.10）。
如果剥离之后那些高独立能力的字段都被剔走了，那"门控有盲区"这个论点
就只适用于不剥离的情形，适用范围要相应收窄。

这一条不查就报那个 53.8%，是拿一个对自己有利的口径说话。

两个口径都跑，并排报
--------------------
  未剥离  46 个公开字段全上
  已剥离  按正式流程剥离后的池子，**直接读 02_class1_identity 已经跑好的产物**
          （`strip_summary.csv` 的"公式剔除字段"和"逐层剔除字段"两列），
          不自己重算——自己重算漏过一次第一层，见下方注释

量法上的一个改动
----------------
"独立能力"这次**不用代理模型，改成逐字段真的训练一个普通网络**。
理由：代理模型要同时应付所有字段组合，在难目标上会明显低估
（PJM 上实测差过 1.5 个 R²，已在别处记录）。这里的结论不该建立在
代理模型的精度上，所以宁可慢一点，逐个真训。

代价是 12 个目标 × 40 多个字段 = 五百多次训练，但每次只有一个输入维度，
几秒钟就完事。
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
from src import baselines as bl         # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_src"))

GATE_THR = 0.01      # 与 gates.TrainConfig.dgate_threshold 一致

# 正式流程的剥离结果直接读现成的产物，不自己重算。
# 自己重算漏过一次：只实现了第二层（按残差逐层剥离），
# 漏掉第一层（按数据定义剔除公式可算字段，**不看残差**），
# 而 12 个目标里有 8 个恰恰只在第一层掉字段，于是"已剥离"池子和未剥离一模一样。
STRIP_FILES = {
    "rts_v2": ROOT / "04_outputs" / "rts_gmlc_2020_v2" /
              "02_class1_identity" / "data" / "strip_summary.csv",
}


def load_strip(dataset: str) -> dict[str, dict]:
    """读正式流程已经跑好的剥离结果，取每个目标被剔掉的字段。"""
    f = STRIP_FILES.get(dataset)
    if f is None or not f.exists():
        raise SystemExit(f"没找到 {dataset} 的剥离产物：{f}")
    d = pd.read_csv(f)
    out = {}
    for _, r in d.iterrows():
        def split(v):
            return [x for x in str(v).split("|") if x and x != "nan"]
        out[r["target"]] = dict(
            公式剔除=split(r.get("公式剔除字段", "")),
            逐层剔除=split(r.get("逐层剔除字段", "")),
            剩余候选=int(r["剩余候选"]))
    return out


def solo_by_retrain(X, y, pool, cfg, eval_idx) -> np.ndarray:
    """逐个字段单独训练一个普通网络，返回各自的测试 R²。"""
    out = np.zeros(len(pool))
    for j in range(len(pool)):
        out[j] = sg.retrain_reference(X, y, [j], cfg, eval_idx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--split", default="temporal", choices=["temporal", "random"])
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--solo-levels", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    base, figd, datd = ds.run_dir(a.dataset, "blindspot_stripped")
    print(f"输出目录 {base}")
    print(f"划分口径 {a.split}，独立能力用逐字段真训的普通网络量（不经代理模型）\n")

    strip = load_strip(a.dataset)
    print(f"剥离结果读自正式流程的产物：{STRIP_FILES[a.dataset]}\n")
    rows, detail, traces = [], [], []
    for ti, tgt in enumerate(a.targets or d.targets):
        pool_raw = d.pool(tgt)
        X_all = d.df[pool_raw].to_numpy(float)
        y = d.df[tgt].to_numpy(float)
        cfg = sg.SurrogateConfig(seed=42, split=a.split)
        _, _, i_te = sg._split_idx(len(X_all), cfg)
        ev = np.sort(np.random.default_rng(0).choice(
            i_te, size=min(a.n_eval, len(i_te)), replace=False))

        info = strip.get(tgt)
        if info is None:
            print(f"[{ti+1}] {d.label(tgt)}：剥离产物里没有这个目标，跳过")
            continue
        drop = set(info["公式剔除"]) | set(info["逐层剔除"])
        pool_str = [c for c in pool_raw if c not in drop]
        traces.append(dict(目标=d.label(tgt), 未剥离=len(pool_raw),
                           公式剔除=len(info["公式剔除"]),
                           逐层剔除=len(info["逐层剔除"]),
                           已剥离=len(pool_str),
                           产物记录的剩余=info["剩余候选"],
                           公式剔除字段="|".join(info["公式剔除"]),
                           逐层剔除字段="|".join(info["逐层剔除"])))
        print(f"[{ti+1}] {d.label(tgt)}：未剥离 {len(pool_raw)} → "
              f"公式剔除 {len(info['公式剔除'])} + 逐层剔除 {len(info['逐层剔除'])}"
              f" → 已剥离 {len(pool_str)}（产物记录 {info['剩余候选']}）", flush=True)

        for name, pool in (("未剥离", pool_raw), ("已剥离", pool_str)):
            X = d.df[pool].to_numpy(float)
            t1 = time.time()
            solo = solo_by_retrain(X, y, pool, cfg, ev)
            gate = np.asarray(bl.dgating_scores(X, y, seed=42)["score"], float)
            zero = np.abs(gate) < GATE_THR
            rec = dict(目标=d.label(tgt), 口径=name, 候选数=len(pool),
                       门控判零数=int(zero.sum()),
                       独立能力最高=float(solo.max()),
                       被判零字段的最高独立能力=float(solo[zero].max())
                       if zero.any() else np.nan)
            for lv in a.solo_levels:
                hi = solo >= lv
                rec[f"独立能力≥{lv}的字段数"] = int(hi.sum())
                rec[f"其中被判零"] = None
                rec[f"独立能力≥{lv}且被判零"] = int((hi & zero).sum())
                rec[f"占比≥{lv}"] = float((hi & zero).sum() / max(hi.sum(), 1))
            rec.pop("其中被判零", None)
            rows.append(rec)
            for f, s, g in zip(pool, solo, gate):
                detail.append(dict(目标=d.label(tgt), 口径=name, 字段=f,
                                   独立能力=s, 分解式门控=g,
                                   门控判零=bool(abs(g) < GATE_THR)))
            print(f"     {name}：门控判零 {rec['门控判零数']}/{len(pool)}，"
                  f"独立能力≥0.5 的 {rec['独立能力≥0.5的字段数']} 个里"
                  f"被判零 {rec['独立能力≥0.5且被判零']} 个"
                  f"（{rec['占比≥0.5']:.1%}），被判零字段的最高独立能力 "
                  f"{rec['被判零字段的最高独立能力']:.4f}  ({time.time()-t1:.0f}s)",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(datd / "blindspot_stripped.csv", index=False)
    pd.DataFrame(detail).to_csv(datd / "blindspot_detail.csv", index=False)
    pd.DataFrame(traces).to_csv(datd / "strip_trace.csv", index=False)

    print(f"\n{'='*88}\n两个口径并排")
    for name, g in df.groupby("口径", sort=False):
        n_pairs = int(g["候选数"].sum())
        print(f"\n【{name}】共 {len(g)} 个目标，{n_pairs} 个（目标, 字段）对")
        print(f"  被门控判零：{int(g['门控判零数'].sum())}"
              f"（{g['门控判零数'].sum()/n_pairs:.1%}）")
        for lv in a.solo_levels:
            tot = int(g[f"独立能力≥{lv}的字段数"].sum())
            hit = int(g[f"独立能力≥{lv}且被判零"].sum())
            print(f"  独立能力≥{lv} 的共 {tot} 个，其中被判零 {hit} 个"
                  f"（{hit/max(tot,1):.1%}）")
        print(f"  被判零字段的最高独立能力（跨目标最大）："
              f"{g['被判零字段的最高独立能力'].max():.4f}")

    print(f"\n{'='*88}")
    print("读法：如果'已剥离'那一栏的比例仍然可观，说明门控的盲区不是靠剥离能补上的，")
    print("      前面那个统计在正式流程下同样成立；如果大幅缩水，")
    print("      那这条论点只适用于不剥离的情形，适用范围要照实收窄。")

    (datd / "config.json").write_text(json.dumps(
        {"数据集": a.dataset, "划分": a.split,
         "剥离产物": str(STRIP_FILES[a.dataset]),
         "门控阈值": GATE_THR, "独立能力量法": "逐字段真训普通网络",
         "总耗时秒": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base}")


if __name__ == "__main__":
    main()
