"""新增发布字段时的增量评估——验证实验。

## 要验证什么

现实场景：电网今年多公布一个字段，要不要把整套评估从头跑一遍？

本方法给出的答案是两级的：
  快速得分  只算这一个新字段的六类依赖指标，与置换零分布比，得到超阈比。
            明显够不着门槛的直接判为不改变结论，连训练都不用。
  热启动增量 够得着的，把上一次训练好的门控模型取出来当初值，
            新字段的门控因子从 1 出发（与全新训练时每个字段的起点一致），
            只训练很少的轮次。

要验证的命题不是「增量能得到好结果」，而是：

    增量得到的结论，与把新字段加进去从头重跑一遍得到的结论一致，且代价显著更低。

所以必须三方对照，缺一不可：

    G_full   池子里有该字段，从头训练 200 轮        ← 标准答案
    G_minus  池子里没有该字段，从头训练 200 轮      ← 增量的起点
    G_inc    从 G_minus 出发，加回该字段，训练 30 轮

只比 G_inc 和 G_minus 说明不了问题，必须比 G_inc 和 G_full。

## 还要一个对照组

G_cold：同样只训练 30 轮，但初值随机，不沿用 G_minus。

如果 G_inc 和 G_cold 表现相同，那"热启动"就没有价值，收益只来自"少训练几轮"。
这个对照把两者分开。

## 为什么按门控强度分层挑字段

随便挑字段做留一验证会得到没有说服力的结果：候选池里绝大多数字段是无关的，
去掉再加回来当然没有变化。有区分度的是这三类：

    强   门控值最高的字段。检验增量认不认得出该选的字段，这是核心
    中   门控断崖附近的字段。最容易出分歧，最能暴露问题
    弱   门控值为 0 的字段。检验快速得分这条快速通道会不会误杀

只在净实际交换功率这一个目标上做——目的是找到能说明机制的例子，
不是统计意义上的全面评估。
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
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_src"))
import dataio  # noqa: E402
import gates  # noqa: E402
import plots  # noqa: E402
import runlock  # noqa: E402
import screening as scr  # noqa: E402

warnings.filterwarnings("ignore")


def _load(name: str, path: Path):
    saved = sys.argv
    try:
        sys.argv = [name]
        spec = ilu.spec_from_file_location(name, path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = saved


sl = _load("sl", SCRIPTS / "04_source_location.py")

TARGET = "net_actual_interchange_mw"
OUT_VERSION = "pjm_2025_v2"
YEAR = 2025


def fast_score(out_version: str, target: str) -> pd.Series:
    """快速得分：六类指标里超出置换阈值最多的那个的倍数。

    直接用第 3 步已经算好的结果——对一个真的新增字段，这一步就是
    只对它算六个指标再跟零分布比，成本极低，不需要碰门控模型。
    """
    d = dataio.OUTPUTS / out_version / "03_screening" / "data"
    obs = pd.read_csv(d / f"screen_{target}.csv")
    summ = pd.read_csv(d / "screening_summary.csv").set_index("target")
    thr = {m: summ.loc[target, f"thr_{m}"] for m in scr.METRICS}
    s = obs.set_index("field")[list(scr.METRICS)].apply(
        lambda r: max(r[m] / thr[m] for m in scr.METRICS), axis=1)
    return s.rename("超阈比")


def pick_fields(gate: pd.Series, n_each: int) -> list[tuple[str, str]]:
    """按门控强度分三层挑字段。"""
    g = gate.sort_values(ascending=False)
    active = g[g >= 0.01]
    picks = []
    picks += [(f, "强") for f in active.index[:n_each]]
    # 中层：断崖两侧各取——最后一个入选的，和第一个落选的
    edge = list(active.index[-1:]) + list(g[g < 0.01].index[:1])
    picks += [(f, "中") for f in edge[:n_each]]
    zero = g[g <= 1e-12]
    picks += [(f, "弱") for f in zero.index[:n_each]]
    seen, out = set(), []
    for f, lv in picks:
        if f not in seen:
            seen.add(f)
            out.append((f, lv))
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200, help="从头训练的轮次")
    ap.add_argument("--inc-epochs", type=int, default=30, help="增量训练的轮次")
    ap.add_argument("--n-each", type=int, default=2, help="每层挑几个字段")
    ap.add_argument("--reset-gates", action="store_true",
                    help="热启动只沿用网络权重，门控全部重置为 1 重新竞争")
    args = ap.parse_args()

    with runlock.single_instance("incremental"):
        tm = runlock.Timer("新增字段的增量评估验证")
        cfg = gates.TrainConfig(epochs=args.epochs, lambda_dgate=0.005)
        df = dataio.load_clean(YEAR, "main")
        pool, note = sl.build_pool(df, TARGET, True, False, YEAR, True, OUT_VERSION)
        X = df[pool].to_numpy(float)
        y = df[TARGET].to_numpy(float)
        out = (dataio.OUTPUTS / OUT_VERSION / "06_incremental" /
               f"run_{datetime.now():%Y%m%d_%H%M%S}")
        out.mkdir(parents=True, exist_ok=True)
        FIG, DAT = dataio.split_dirs(out)
        print(f"目标 {sl.CN[TARGET]}　候选 {len(pool)} 个（{note}）", flush=True)

        # ---- 标准答案：全池从头训练 ----
        print(f"\n[1/2] 标准答案：全池 {len(pool)} 字段从头训练 {args.epochs} 轮", flush=True)
        full = gates.train_keep_model("DGatingDNN", X, y, cfg)
        gfull = pd.Series(full.final_gates, index=pool)
        sel_full = set(gfull[gfull >= cfg.dgate_threshold].index)
        r2_full = gates.retrain_subset(
            X, y, [pool.index(c) for c in gfull.sort_values(ascending=False).index
                   if c in sel_full], cfg)
        print(f"      选中 {len(sel_full)} 个，选中后 R² {r2_full:.4f}，"
              f"全量 R² {full.best_test_r2:.4f}", flush=True)
        tm.mark("标准答案")

        fs = fast_score(OUT_VERSION, TARGET)
        picks = pick_fields(gfull, args.n_each)
        print(f"\n挑出 {len(picks)} 个字段：" +
              "　".join(f"{sl.cn(f)}({lv},门控{gfull[f]:.4f})" for f, lv in picks),
              flush=True)

        rows, traj = [], {}
        print(f"\n[2/2] 逐字段：去掉→从头训 {args.epochs} 轮→加回训 {args.inc_epochs} 轮",
              flush=True)
        for f, lv in picks:
            pos = pool.index(f)
            keep = [i for i in range(len(pool)) if i != pos]
            pool_m = [pool[i] for i in keep]

            minus = gates.train_keep_model("DGatingDNN", X[:, keep], y, cfg)
            gm = pd.Series(minus.final_gates, index=pool_m)
            sel_m = set(gm[gm >= cfg.dgate_threshold].index)

            inc = gates.train_incremental(X, y, cfg, pos, minus.model, args.inc_epochs,
                                          reset_gates=args.reset_gates)
            gi = pd.Series(inc.final_gates, index=pool)
            sel_i = set(gi[gi >= cfg.dgate_threshold].index)

            cold = gates.train_incremental(X, y, cfg, pos, None, args.inc_epochs)
            gc = pd.Series(cold.final_gates, index=pool)
            sel_c = set(gc[gc >= cfg.dgate_threshold].index)

            r2_i = gates.retrain_subset(
                X, y, [pool.index(c) for c in gi.sort_values(ascending=False).index
                       if c in sel_i], cfg)
            r2_c = gates.retrain_subset(
                X, y, [pool.index(c) for c in gc.sort_values(ascending=False).index
                       if c in sel_c], cfg)

            traj[f] = inc.gate_history[:, pos]
            rows.append({
                "字段": f, "中文名": sl.cn(f), "层": lv,
                "快速得分(超阈比)": float(fs.get(f, np.nan)),
                "标准_门控值": float(gfull[f]),
                "标准_是否选中": int(f in sel_full),
                "标准_选中数": len(sel_full), "标准_R2": r2_full,
                "去掉后_选中数": len(sel_m), "去掉后_全量R2": minus.best_test_r2,
                "增量_门控值": float(gi[f]), "增量_是否选中": int(f in sel_i),
                "增量_选中数": len(sel_i), "增量_R2": r2_i,
                "增量_与标准重合度": jaccard(sel_i, sel_full),
                "对照_门控值": float(gc[f]), "对照_是否选中": int(f in sel_c),
                "对照_选中数": len(sel_c), "对照_R2": r2_c,
                "对照_与标准重合度": jaccard(sel_c, sel_full),
                "判定一致": int((f in sel_i) == (f in sel_full)),
            })
            r = rows[-1]
            print(f"  {sl.cn(f):14s}[{lv}] 快速得分 {r['快速得分(超阈比)']:.2f}　"
                  f"标准 门控{r['标准_门控值']:.4f}{'选中' if r['标准_是否选中'] else '落选'}　"
                  f"增量 门控{r['增量_门控值']:.4f}{'选中' if r['增量_是否选中'] else '落选'}"
                  f"{'　一致' if r['判定一致'] else '　✗不一致'}　"
                  f"重合度 {r['增量_与标准重合度']:.2f}（对照 {r['对照_与标准重合度']:.2f}）",
                  flush=True)
            tm.mark(f"{sl.cn(f)}({lv})")

        res = pd.DataFrame(rows)
        res.to_csv(DAT / "incremental.csv", index=False)
        pd.DataFrame(traj).to_csv(DAT / "gate_trajectory.csv", index=False)
        (DAT / "config.json").write_text(json.dumps(
            {"target": TARGET, "候选数": len(pool), "从头轮次": args.epochs,
             "增量轮次": args.inc_epochs, "活跃阈值": cfg.dgate_threshold,
             "lambda_dgate": cfg.lambda_dgate}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        write_doc(out / "说明.md", res, args, len(pool), r2_full, len(sel_full))
        plot_traj(traj, res, cfg.dgate_threshold, FIG / "fig_gate_trajectory.png")
        (out / "timing.md").write_text(
            "# 增量验证耗时\n\n" + tm.report() + "\n", encoding="utf-8")
        print("\n" + tm.report())
        print(f"\n写入 {out}")


def plot_traj(traj, res, thr, path):
    import matplotlib.pyplot as plt
    plots.use_cn_font() if hasattr(plots, "use_cn_font") else None
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cn = dict(zip(res.字段, res.中文名))
    lv = dict(zip(res.字段, res.层))
    color = {"强": "#c0392b", "中": "#e67e22", "弱": "#7f8c8d"}
    for f, v in traj.items():
        ax.plot(range(len(v)), v, label=f"{cn[f]}（{lv[f]}）",
                color=color.get(lv[f], "#333"), lw=1.8,
                ls={"强": "-", "中": "--", "弱": ":"}.get(lv[f], "-"))
    ax.axhline(thr, color="k", ls="-.", lw=1, alpha=.6)
    ax.text(len(next(iter(traj.values()))) * .98, thr * 1.15, f"活跃阈值 {thr}",
            ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("增量训练轮次")
    ax.set_ylabel("新字段的门控值（对数刻度）")
    ax.set_title("加回一个字段之后，它的门控值怎么变")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_doc(path, res, args, n_pool, r2_full, n_sel):
    L = ["# 新增发布字段的增量评估", "",
         f"目标字段：净实际交换功率　候选 {n_pool} 个　"
         f"标准答案选中 {n_sel} 个，选中后 R² {r2_full:.4f}", "",
         "## 一、验证的是什么", "",
         "现实场景是电网新增一个发布字段，问要不要把整套评估从头跑一遍。"
         "本方法的做法是两级的：先只对这一个字段算六类依赖指标、与置换零分布比，"
         "得到快速得分；够不着门槛的直接判为不改变结论；够得着的才把上次训练好的"
         "门控模型取出来当初值，只训练很少的轮次。", "",
         "要验证的命题不是「增量能得到好结果」，而是**增量得到的结论与从头重跑一致，"
         "且代价显著更低**。所以三方对照：", "",
         "| 记号 | 含义 |", "|---|---|",
         f"| 标准 | 池子里有该字段，从头训练 {args.epochs} 轮（标准答案）|",
         f"| 去掉后 | 池子里没有该字段，从头训练 {args.epochs} 轮（增量的起点）|",
         f"| 增量 | 从「去掉后」出发加回该字段，训练 {args.inc_epochs} 轮 |",
         f"| 对照 | 同样只训练 {args.inc_epochs} 轮，但初值随机 |", "",
         "对照组是为了分清收益来自「沿用旧解」还是仅仅来自「少训练几轮」——"
         "如果两者表现相同，热启动就没有价值。", "",
         "## 二、结果", "",
         "| 字段 | 层 | 快速得分 | 标准门控值 | 标准判定 | 增量门控值 | 增量判定 | 判定一致 | 增量重合度 | 对照重合度 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r['中文名']} | {r['层']} | {r['快速得分(超阈比)']:.2f} | "
                 f"{r['标准_门控值']:.4f} | {'选中' if r['标准_是否选中'] else '落选'} | "
                 f"{r['增量_门控值']:.4f} | {'选中' if r['增量_是否选中'] else '落选'} | "
                 f"{'一致' if r['判定一致'] else '**不一致**'} | "
                 f"{r['增量_与标准重合度']:.2f} | {r['对照_与标准重合度']:.2f} |")
    ok = int(res.判定一致.sum())
    L += ["", f"判定一致 {ok}/{len(res)}；"
          f"增量与标准的选中集合重合度平均 {res.增量_与标准重合度.mean():.2f}，"
          f"随机初值的对照平均 {res.对照_与标准重合度.mean():.2f}。", "",
          f"训练代价：增量 {args.inc_epochs} 轮，从头重跑 {args.epochs} 轮，"
          f"比值 {args.inc_epochs / args.epochs:.0%}。", "",
          "## 三、图", "",
          "- `fig_gate_trajectory.png`　加回字段之后它的门控值随轮次的变化，"
          "纵轴对数刻度。强层字段应当稳在阈值之上，弱层字段应当很快被压下去。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
