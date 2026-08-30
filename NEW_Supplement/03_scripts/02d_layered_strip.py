"""第 2d 步：逐层剥离，把公式关系和近似关系一层一层挖到底。

上一步只剥了一层就停了，结果像总发电量、计量负荷、净实际交换这些目标，
删完之后线性 R² 还有 0.998 —— 说明底下还压着别的近似关系没被找出来。

这一步改成逐层挖：

  第 1 层  找出能把目标算出来的最小字段组，记下这条关系和它的误差
           只删掉其中"贡献最大"的一个字段（不是整组删掉）
  第 2 层  在剩下的字段里重新找，看还能不能算出来
  ...      一直挖到再也找不到误差小于门槛的关系为止

每层只删一个字段，是为了"部分屏蔽"——把主力字段拿掉之后，看剩下的字段能不能
自己凑出一条新路径。整组删掉会把可能参与其它路径的字段也一并带走，挖不深。

门槛按目标字段自己的尺度算：相对残差 = 剩余误差 / 该字段标准差。
默认挖到相对残差 0.10（对应 R² 约 0.99）为止。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_src"))
import dataio  # noqa: E402
import identity as idt  # noqa: E402

VERSION = "main"
TOL_RANK, TOL_RESID = 1e-5, 1e-3
SEED, TRAIN_RATIO = 42, 0.8
STOP_TAU = 0.10        # 挖到相对残差超过这个值就停（R² 约 0.99）
MAX_LAYERS = 20

# 每层关系按误差量级归档
BANDS = [(1e-6, "精确公式"), (1e-3, "含舍入的公式"), (3e-2, "高精度近似"), (1.0, "近似关系")]

TARGETS = [
    "net_actual_interchange_mw", "gross_actual_interchange_mw",
    "net_sched_interchange_mw", "total_gen", "metered_load_mw", "total_losses",
    "congestion_price_da", "congestion_price_rt", "marginal_loss_price_da",
    "total_lmp_da", "da_as_total_mw_primary_reserve",
    "da_as_total_mw_thirty_minutes_reserve",
]
CN = {
    "net_actual_interchange_mw": "净实际交换功率", "gross_actual_interchange_mw": "总实际交换功率",
    "net_sched_interchange_mw": "净计划交换功率", "total_gen": "总发电量",
    "metered_load_mw": "计量负荷", "total_losses": "总网损",
    "congestion_price_da": "日前阻塞价", "congestion_price_rt": "实时阻塞价",
    "marginal_loss_price_da": "日前边际损耗价", "total_lmp_da": "日前总电价",
    "da_as_total_mw_primary_reserve": "日前主用备用总量",
    "da_as_total_mw_thirty_minutes_reserve": "日前30分钟备用总量",
}


def band(rr: float) -> str:
    for hi, name in BANDS:
        if rr < hi:
            return name
    return "无关系"


def test_r2(df: pd.DataFrame, target: str, pool: list[str]) -> float:
    """测试集上的线性 R²（随机划分，与原有实验一致）。"""
    if not pool:
        return float("nan")
    X, y = df[pool].to_numpy(float), df[target].to_numpy(float)
    n = len(df)
    k = int(n * TRAIN_RATIO)
    perm = np.random.default_rng(SEED).permutation(n)
    tr, te = perm[:k], perm[k:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    m = Ridge(alpha=1e-6).fit((X[tr] - mu) / sd, y[tr])
    return float(r2_score(y[te], m.predict((X[te] - mu) / sd)))


def layered_strip(df: pd.DataFrame, target: str, pool: list[str]):
    """逐层剥离。每层只删一个字段，记录该层找到的关系。

    找关系用的是"从全部字段往下删"的反向消元，不用从零开始逐个加字段的前向法。
    前向法在这里必然失败：像"边际损耗价 = 总电价 − 能量价格 − 阻塞价"这种，
    目标是两个大数相减得到的小数，只加进其中一个字段时残差反而变大，
    前向法走不到正确的组合上。反向消元从"全都留着"出发，每次删掉贡献最小的一个，
    不需要猜从哪里起步。
    """
    cur = list(pool)
    layers = []
    while len(layers) < MAX_LAYERS:
        rr_full = idt.exact_fit_ratio(df, target, cur)
        if not np.isfinite(rr_full) or rr_full >= STOP_TAU:
            break
        support = idt._prune(df, target, cur, STOP_TAU)
        if not support:
            break
        coefs, const, rr = idt._refit(df, target, support)
        if not np.isfinite(rr) or rr >= STOP_TAU:
            break
        contrib = {s: abs(coefs[s]) * df[s].std() for s in support}
        victim = max(contrib, key=contrib.get)
        cur.remove(victim)
        layers.append({
            "layer": len(layers) + 1,
            "band": band(rr),
            "residual_ratio": rr,
            "r2_of_relation": 1.0 - rr**2,
            "n_support": len(support),
            "support": "|".join(support),
            "relation": f"{target} = " + " ".join(
                f"{coefs[s]:+.4g}*{s}" for s in support)
            + (f" {const:+.4g}" if abs(const) > 1e-9 else ""),
            "removed": victim,
            "n_pool_after": len(cur),
            "test_r2_after": test_r2(df, target, cur),
        })
    return layers, cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out-version", default=None)
    ap.add_argument("--stage", default="02_class1_identity",
                    help="输出子目录名")
    ap.add_argument("--exclude-targets", action="store_true",
                    help="候选池里排除其余目标字段，只保留普通字段")
    args = ap.parse_args()
    year = args.year

    df = dataio.load_clean(year, VERSION)
    out = dataio.out_dir(year, args.stage, args.out_version)
    FIG, DAT = dataio.split_dirs(out)
    old = json.loads(Path("/tmp/old_excludes.json").read_text()) \
        if Path("/tmp/old_excludes.json").exists() else {}

    idents, _ = idt.extract(df, tol_rank=TOL_RANK, tol_resid=TOL_RESID)
    constants = sorted({it.lead for it in idents if it.kind == "constant"})
    groups: list[set[str]] = []
    for it in idents:
        if it.kind != "duplicate":
            continue
        pair = {it.lead, *it.support}
        hit = [g for g in groups if g & pair]
        groups = ([g for g in groups if g not in hit] +
                  [set().union(*hit, pair)]) if hit else groups + [pair]

    all_layers, summary = [], []
    print(f"pjm_{year}  {df.shape[0]} 行 × {df.shape[1]} 列  "
          f"（挖到相对残差 {STOP_TAU} 为止）\n")

    for t in TARGETS:
        if t not in df.columns:
            continue
        dup_drop: list[str] = []
        for g in groups:
            keep = t if t in g else sorted(g)[0]
            dup_drop += [m for m in sorted(g) if m != keep]
        gdrop = sorted((set(constants) | set(dup_drop)) - {t})
        pool0 = [c for c in df.columns if c != t and c not in gdrop]
        if args.exclude_targets:
            pool0 = [c for c in pool0 if c not in TARGETS]
        # 第一层：先按已知公式剔除，逐层剥离在剩下的池子上做。
        # 不这样做的话，第二层是在"公式字段还在"的池子上算的，
        # 与"先按公式剥离、再按残差剥离"的设计不符。
        fdrop, fhit = idt.formula_drop(t)
        fdrop = [c for c in fdrop if c in pool0]
        pool0 = [c for c in pool0 if c not in fdrop]

        r2_start = test_r2(df, t, pool0)
        layers, final_pool = layered_strip(df, t, pool0)
        removed = [ly["removed"] for ly in layers]
        r2_end = test_r2(df, t, final_pool)

        print(f"── {CN.get(t, t)}  ({t})")
        print(f"   起点 {len(pool0)} 字段，测试 R² {r2_start:.4f}")
        for ly in layers:
            print(f"     第{ly['layer']:2d}层 [{ly['band']:8s}] 残差比 {ly['residual_ratio']:.2e}  "
                  f"支撑 {ly['n_support']}  删 {ly['removed']:38s} → R² {ly['test_r2_after']:7.4f}")
        print(f"   终点 {len(final_pool)} 字段，测试 R² {r2_end:.4f}，共剥 {len(layers)} 层")

        o = old.get(t, {})
        oex = set(o.get("exclude") or [])
        if oex:
            r2_old = test_r2(df, t, [c for c in pool0 if c not in oex])
            hit = oex & set(removed)
            print(f"   原配置排除 {len(oex)} 个（按精度试出来的），测试 R² {r2_old:.4f}；"
                  f"与本方法重合 {len(hit)} 个")
        else:
            r2_old = float("nan")
            print("   原配置未排除任何字段")
        print()

        for ly in layers:
            all_layers.append({"target": t, "中文名": CN.get(t, t), **ly})
        summary.append({
            "target": t, "中文名": CN.get(t, t),
            "n_start": len(pool0), "r2_start": r2_start,
            "n_layers": len(layers), "removed": "|".join(removed),
            "n_final": len(final_pool), "r2_final": r2_end,
            "n_old_exclude": len(oex), "r2_old_exclude": r2_old,
            "overlap_with_old": len(oex & set(removed)),
            "bands": "|".join(f"{ly['band']}" for ly in layers),
        })

    pd.DataFrame(all_layers).to_csv(DAT / "layered_strip_layers.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(DAT / "layered_strip_summary.csv", index=False)

    print("=" * 96)
    print(f"{'目标字段':22s}{'起点R²':>9s}{'层数':>5s}{'终点R²':>9s}{'终点字段':>9s}"
          f"{'原配置R²':>10s}{'原排除数':>9s}{'重合':>6s}")
    for _, r in s.iterrows():
        print(f"{r['中文名']:22s}{r['r2_start']:9.4f}{int(r['n_layers']):5d}"
              f"{r['r2_final']:9.4f}{int(r['n_final']):9d}"
              f"{r['r2_old_exclude']:10.4f}{int(r['n_old_exclude']):9d}"
              f"{int(r['overlap_with_old']):6d}")
    print(f"\n写入 {out / 'layered_strip_layers.csv'} 与 layered_strip_summary.csv")


if __name__ == "__main__":
    main()
