"""rule2：沿用原稿的老规则确定统一预算。

老规则的做法是：对一个目标字段，让所有方法按字段数从 1 开始逐个递增，
**任意一种方法**的测试 R² 首次达到全量 DNN 的 95%（或 97%）时，
把对应的字段数定为该目标的统一预算 n，再比较各方法在这个 n 下的表现。

这条规则有一处结构性偏向，实现时照原样保留、但在结果里如实标出：
**预算是由"最先达标的那个方法"决定的**，那个方法是在自己最舒服的字段数上被测量的，
其余方法则是被拖到这个点上。所以每个目标都会记录"是谁先达标的"。

保留这条老规则的意义在于对照：换一种确定预算的方式，方法之间的排名会不会变。
两种规则下结论一致，才说明排名不是被预算规则挑出来的。

为控制计算量，递增搜索阶段用 100 轮训练（只用来找达标点），
确定预算之后再用 200 轮重新评估一遍作为最终结果。
"""

from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_src"))
import dataio  # noqa: E402
import gates  # noqa: E402
import plots  # noqa: E402
import runlock  # noqa: E402

_s = ilu.spec_from_file_location("bc", Path(__file__).resolve().parent /
                                 "05_baseline_compare.py")
bc = ilu.module_from_spec(_s)
_s.loader.exec_module(bc)
sl = bc.sl

warnings.filterwarnings("ignore")
VERSION = "main"
K_GRID = list(range(1, 16)) + [18, 22, 26, 30]
LEVELS = [0.95, 0.97]


def search_budget(X, y, ranks: dict, full_r2: float, cfg_fast, n_pool: int):
    """按字段数递增，直到任一方法达到全量 DNN 的 95% / 97%。"""
    need = {lv: full_r2 * lv for lv in LEVELS}
    hit = {lv: None for lv in LEVELS}
    curves = {m: [] for m in ranks}
    ks_done = []
    for k in [k for k in K_GRID if k <= n_pool]:
        ks_done.append(k)
        for m, idx in ranks.items():
            curves[m].append(gates.retrain_subset(X, y, list(idx[:k]), cfg_fast))
        for lv in LEVELS:
            if hit[lv] is None:
                win = [m for m in ranks if curves[m][-1] >= need[lv]]
                if win:
                    hit[lv] = (k, max(win, key=lambda m: curves[m][-1]))
        if all(v is not None for v in hit.values()):
            break
    for lv in LEVELS:                       # 全程没达标就用最大的 k
        if hit[lv] is None:
            hit[lv] = (ks_done[-1], "无方法达标")
    return hit, curves, ks_done


def run_target(df, target, pool, cfg, cfg_fast, out: Path):
    X, y = df[pool].to_numpy(float), df[target].to_numpy(float)
    d = out / f"target_{target}"
    d.mkdir(parents=True, exist_ok=True)
    FIG, DAT = dataio.split_dirs(d)

    full_r2 = gates.train("DNN", X, y, cfg).best_test_r2
    ours = gates.train("DGatingDNN", X, y, cfg)

    ranks, scores = {}, {}
    for m in ["DGatingDNN"] + bc.BASELINES:
        try:
            sc = ours.final_gates if m == "DGatingDNN" else bc.RANKERS[m](X, y, cfg=cfg)
        except Exception as e:
            print(f"        [{m}] 失败：{type(e).__name__} {e}", flush=True)
            continue
        scores[m] = np.asarray(sc, float)
        ranks[m] = np.argsort(-scores[m])

    hit, curves, ks = search_budget(X, y, ranks, full_r2, cfg_fast, len(pool))
    pd.DataFrame(curves, index=ks).rename_axis("k").to_csv(DAT / "search_curves.csv")

    rows = []
    for lv in LEVELS:
        n, first = hit[lv]
        for m, idx in ranks.items():
            rows.append({"目标": target, "中文名": sl.CN[target], "达标水平": f"{lv:.0%}",
                         "预算n": n, "先达标的方法": first, "方法": m,
                         "测试R2": gates.retrain_subset(X, y, list(idx[:n]), cfg),
                         "全量R2": full_r2, "候选数": len(pool),
                         "选中字段": "|".join(pool[i] for i in idx[:n])})
        print(f"        {lv:.0%} 达标 → n={n}，先达标的是 {first}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(DAT / "comparison.csv", index=False)

    plots.rule2_search_plot(curves, ks, full_r2, hit, FIG / "fig_search.png",
                            f"{sl.CN[target]}　按字段数递增搜索达标点")
    for lv in LEVELS:
        sub = res[res.达标水平 == f"{lv:.0%}"]
        plots.baseline_bar_plot(
            sub.rename(columns={"方法": "方法", "测试R2": "n"}).assign(
                说明=[bc.METHOD_NOTE[m] for m in sub.方法]),
            full_r2, hit[lv][0], FIG / f"fig_compare_{int(lv * 100)}.png",
            f"{sl.CN[target]}　{lv:.0%} 达标预算 n={hit[lv][0]} 下各方法对比")

    L = [f"# rule2 老规则对比：{sl.CN[target]}", "",
         "## 一、规则回顾", "",
         "按字段数从 1 开始递增，任一方法的测试 R² 首次达到全量 DNN 的 95%（或 97%）时，"
         "把对应字段数定为统一预算，再比各方法在该预算下的表现。", "",
         f"- 候选字段 {len(pool)} 个，全量 DNN 测试 R² = {full_r2:.4f}",
         f"- 递增搜索用 100 轮训练，确定预算后用 200 轮重新评估", "",
         "## 二、达标点", "", "| 达标水平 | 需要的 R² | 统一预算 n | 先达标的方法 |",
         "|---|---|---|---|"]
    for lv in LEVELS:
        L.append(f"| {lv:.0%} | {full_r2 * lv:.4f} | {hit[lv][0]} | **{hit[lv][1]}** |")
    L += ["", "> 预算由最先达标的方法决定，那个方法是在自己最舒服的字段数上被测量的，"
          "其余方法是被拖到这个点上——这是老规则本身的偏向，如实标出。", ""]
    for lv in LEVELS:
        sub = res[res.达标水平 == f"{lv:.0%}"].sort_values("测试R2", ascending=False)
        L += [f"## 三、{lv:.0%} 预算（n={hit[lv][0]}）下的结果", "",
              "| 方法 | 说明 | 测试 R² |", "|---|---|---|"]
        for _, r in sub.iterrows():
            L.append(f"| {r['方法']} | {bc.METHOD_NOTE[r['方法']]} | {r['测试R2']:.4f} |")
        L.append("")
    L += ["## 四、图", "",
          "- `fig_search.png`　递增搜索过程：各方法的 R² 随字段数变化，"
          "两条横线是 95% 和 97% 达标线",
          "- `fig_compare_95.png` / `fig_compare_97.png`　两个预算下的横向对比", ""]
    (d / "总结.md").write_text("\n".join(L), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out-version", default=None)
    ap.add_argument("--targets", nargs="+", default=bc.sl.TARGETS)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--search-epochs", type=int, default=100)
    args = ap.parse_args()

    with runlock.single_instance("rule2"):
        tm = runlock.Timer("rule2 老规则同预算对比")
        df = dataio.load_clean(args.year, VERSION)
        cfg = gates.TrainConfig(epochs=args.epochs, lambda_stg=0.005, lambda_dgate=0.005)
        cfg_fast = gates.TrainConfig(epochs=args.search_epochs, lambda_stg=0.005, lambda_dgate=0.005)
        root = (dataio.OUTPUTS / (args.out_version or f"pjm_{args.year}")
                / "05_baseline_compare" / "rule2")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # rule2 只在"排除其余关注字段"这一支上做（算力所限），初筛两种都做
        for scr in [False, True]:
            lv2 = "with_screening" if scr else "no_screening"
            out = root / "excl_targets" / lv2 / f"run_{stamp}"
            out.mkdir(parents=True, exist_ok=True)
            OFIG, ODAT = dataio.split_dirs(out)
            print(f"\n===== excl_targets / {lv2} =====", flush=True)
            allres = []
            for t in args.targets:
                pool, note = sl.build_pool(df, t, True, scr, args.year, True,
                                           args.out_version)
                print(f"    ── {sl.CN[t]}　候选 {len(pool)} 个（{note}）", flush=True)
                allres.append(run_target(df, t, pool, cfg, cfg_fast, out))
            agg = pd.concat(allres, ignore_index=True)
            agg.to_csv(ODAT / "overall_comparison.csv", index=False)
            (ODAT / "config.json").write_text(json.dumps(
                {"rule": "rule2", "候选池": "excl_targets", "初筛": lv2,
                 "达标水平": LEVELS, "搜索轮次": args.search_epochs,
                 "最终评估轮次": args.epochs, "targets": args.targets},
                ensure_ascii=False, indent=2), encoding="utf-8")
            write_overall(out / "总结.md", agg, lv2)
            plots.rule2_overall_plot(agg, OFIG / "fig_overall.png",
                                     f"excl_targets / {lv2}　rule2 老规则下各方法表现")
            tm.mark(f"excl_targets / {lv2}")
        (root / "timing.md").write_text(
            "# rule2 执行耗时\n\n" + tm.report() + "\n", encoding="utf-8")
        print("\n" + tm.report())


def write_overall(path: Path, agg: pd.DataFrame, lv2: str) -> None:
    L = [f"# rule2 老规则总体比较（excl_targets / {lv2}）", "",
         "## 一、规则", "",
         "按字段数递增，任一方法首次达到全量 DNN 的 95%（或 97%）时的字段数作为统一预算。"
         "**预算由最先达标的方法决定**，这条规则天然对该方法有利，结果中如实标出是谁先达标。", ""]
    for lv in LEVELS:
        s = agg[agg.达标水平 == f"{lv:.0%}"]
        m = s.groupby("方法")["测试R2"].mean().sort_values(ascending=False)
        L += [f"## 二、{lv:.0%} 达标水平", "",
              f"平均预算 {s.drop_duplicates('目标')['预算n'].mean():.2f} 个字段", "",
              "| 排名 | 方法 | 平均测试 R² |", "|---|---|---|"]
        for i, (k, v) in enumerate(m.items(), 1):
            L.append(f"| {i} | {'**' + k + '**' if i == 1 else k} | {v:.4f} |")
        w = s.drop_duplicates("目标")["先达标的方法"].value_counts()
        L += ["", "先达标的方法分布（这决定了预算，对该方法有利）：", ""]
        L += [f"- {k}：{v} 次" for k, v in w.items()]
        piv = s.pivot_table(index="方法", columns="中文名", values="测试R2")
        L += ["", "逐目标：", "", "| 方法 | " + " | ".join(piv.columns) + " |",
              "|---" * (len(piv.columns) + 1) + "|"]
        for k in m.index:
            L.append(f"| {k} | " + " | ".join(f"{piv.loc[k, c]:.4f}" for c in piv.columns) + " |")
        L.append("")
    # 逐目标把各方法到底选了哪些字段列出来。只给平均 R² 看不出方法之间的差别
    # 到底差在哪——同样给 6 个字段，谁挑的是负荷谁挑的是备用，这才是实质。
    for lv in LEVELS:
        sub = agg[agg.达标水平 == f"{lv:.0%}"]
        L += [f"## {lv:.0%} 达标水平下，各目标选中的字段", ""]
        for t in sub.目标.unique():
            g = sub[sub.目标 == t]
            r0 = g.iloc[0]
            L += [f"### {r0['中文名']}", "",
                  f"候选 {int(r0.候选数)} 个，统一预算 **{int(r0.预算n)}** 个，"
                  f"先达标的是 **{r0['先达标的方法']}**，全量 DNN R² = {r0.全量R2:.4f}", "",
                  "| 方法 | 测试 R² | 选中的字段 |", "|---|---|---|"]
            for _, r in g.sort_values("测试R2", ascending=False).iterrows():
                fs = "、".join(sl.cn(c) for c in str(r.选中字段).split("|") if c)
                nm = f"**{r['方法']}**" if r["方法"] == "DGatingDNN" else r["方法"]
                L.append(f"| {nm} | {r.测试R2:.4f} | {fs} |")
            L.append("")
    L += ["## 图", "", "- `fig_overall.png`　两个达标水平下各方法的平均表现",
          "- 各目标目录下 `fig_search.png` 是递增搜索的过程曲线", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
