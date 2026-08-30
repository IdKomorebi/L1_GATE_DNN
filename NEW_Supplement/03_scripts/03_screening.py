"""第 3 步：多指标初筛。

对每个目标字段，算全部候选字段的六类依赖指标，用分块置换定出的阈值筛掉
明显无关的字段。采用"任一指标过阈值即保留"的宽松规则——初筛只负责去掉
明显不相关的，宁可多留，真正的取舍交给后面的门控模型。

候选池只去掉目标自身和数据质量问题字段（恒定字段、重复副本），不涉及
"是否排除其余关注字段"的选择，所以两套后续实验共用这一步的结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_src"))
import dataio  # noqa: E402
import identity as idt  # noqa: E402
import plots  # noqa: E402
import screening as scr  # noqa: E402

VERSION = "main"
N_DRAWS, BLOCK = 120, 168
# 分位数取 0.90 而不是统计检验惯用的 0.95。那个惯例控制的是误报率，
# 而初筛要控制的是漏报率——方法的设计目标写明"宁可多留弱冗余字段，
# 也不在早期排除可能有互补作用的字段"。留错了后面的门控会关掉它，代价很小；
# 漏掉了就永远找不回来。实测 0.95 会误删净实际交换的三个关键负荷字段，
# 其中预估小时负荷单独加回可使 R² 从 0.9020 升到 0.9613。
QUANTILE = 0.90

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
# 原项目人工指定的阈值，用来对照
OLD_THRESHOLDS = {"nmi": 0.06, "spearman": 0.15, "pearson": 0.15,
                  "kendall": 0.12, "distance_corr": 0.20, "hsic": 0.25}


def quality_drop(df: pd.DataFrame, target: str) -> list[str]:
    idents, _ = idt.extract(df, tol_rank=1e-5, tol_resid=1e-3)
    const = {i.lead for i in idents if i.kind == "constant"}
    groups: list[set[str]] = []
    for i in idents:
        if i.kind != "duplicate":
            continue
        p = {i.lead, *i.support}
        h = [g for g in groups if g & p]
        groups = ([g for g in groups if g not in h] + [set().union(*h, p)]) if h else groups + [p]
    dup = []
    for g in groups:
        keep = target if target in g else sorted(g)[0]
        dup += [m for m in sorted(g) if m != keep]
    return sorted((const | set(dup)) - {target})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--stage", default="03_screening")
    ap.add_argument("--out-version", default=None)
    args = ap.parse_args()

    df = dataio.load_clean(args.year, VERSION)
    out = dataio.out_dir(args.year, args.stage, args.out_version)
    FIG, DAT = dataio.split_dirs(out)
    print(f"pjm_{args.year}  {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"置换 {N_DRAWS} 次，分块长度 {BLOCK} 小时（一周），阈值取零分布的 {QUANTILE:.0%} 分位\n")

    summary, all_obs, all_thr = [], [], {}
    for t in TARGETS:
        if t not in df.columns:
            continue
        drop = quality_drop(df, t)
        pool = [c for c in df.columns if c != t and c not in drop]

        null = scr.null_thresholds(df, t, pool, n_draws=N_DRAWS,
                                   block=BLOCK, quantile=QUANTILE)
        obs = scr.screen(df, t, pool, null.thresholds)

        kept = int(obs.kept.sum())
        print(f"── {CN[t]}  候选 {len(pool)} → 保留 {kept}，筛除 {len(pool) - kept}")
        print("   阈值：" + "  ".join(
            f"{scr.METRIC_CN[m]} {null.thresholds[m]:.3f}" for m in scr.METRICS))
        dropped = obs[obs.kept == 0].field.tolist()
        if dropped:
            print(f"   筛除的字段：{dropped[:6]}{' …' if len(dropped) > 6 else ''}")

        obs.to_csv(DAT / f"screen_{t}.csv", index=False)
        null.draws.to_csv(DAT / f"null_{t}.csv", index=False)
        all_thr[t] = null.thresholds
        o = obs.copy(); o["target"] = t
        all_obs.append(o)

        hist = obs.n_pass.value_counts().sort_index().to_dict()
        summary.append({"target": t, "中文名": CN[t], "n_pool": len(pool),
                        "n_kept": kept, "n_dropped": len(pool) - kept,
                        "dropped": "|".join(dropped),
                        "pass_hist": {int(k): int(v) for k, v in hist.items()},
                        **{f"thr_{m}": null.thresholds[m] for m in scr.METRICS}})

        plots.null_vs_observed_plot(
            null.draws, obs, null.thresholds, scr.METRICS, scr.METRIC_CN,
            FIG / f"fig_null_vs_obs_{t}.png",
            f"{CN[t]}：各指标的实测值与无关系时的取值分布")

    s = pd.DataFrame(summary)
    s.drop(columns=["pass_hist"]).to_csv(DAT / "screening_summary.csv", index=False)
    obs_all = pd.concat(all_obs, ignore_index=True)
    obs_all.to_csv(DAT / "screening_all.csv", index=False)
    (DAT / "thresholds.json").write_text(
        json.dumps(all_thr, ensure_ascii=False, indent=2), encoding="utf-8")

    plots.screening_overview_plot(s, FIG / "fig_screening_overview.png",
                                  f"多指标初筛总览（PJM {args.year}）")
    plots.metric_agreement_plot(obs_all, scr.METRICS, scr.METRIC_CN,
                                FIG / "fig_metric_agreement.png",
                                "六类指标对字段取舍的判断吻合度")

    print("\n" + "=" * 78)
    print(f"{'目标':18s}{'候选':>6s}{'保留':>6s}{'筛除':>6s}")
    for _, r in s.iterrows():
        print(f"{r['中文名']:18s}{int(r['n_pool']):6d}{int(r['n_kept']):6d}{int(r['n_dropped']):6d}")
    print(f"\n平均筛除 {s.n_dropped.mean():.1f} 个字段（占 {s.n_dropped.sum()/s.n_pool.sum():.1%}）")

    print("\n本方法定出的阈值 vs 原项目人工指定的阈值：")
    print(f"{'指标':14s}{'本方法(均值)':>14s}{'原项目':>10s}")
    for m in scr.METRICS:
        v = s[f"thr_{m}"].mean()
        print(f"{scr.METRIC_CN[m]:14s}{v:14.3f}{OLD_THRESHOLDS.get(m, float('nan')):10.3f}")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
