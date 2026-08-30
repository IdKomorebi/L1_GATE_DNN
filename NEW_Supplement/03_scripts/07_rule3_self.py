"""rule3：各方法用自己的判据决定选多少个字段，从字段数和精度两方面一起评价。

前两条规则都强行统一了字段数，比的是"同样多字段下谁挑得准"。
但实际使用时没人会告诉你该选几个——**"选多少个"本来就是方法要自己回答的问题**。
rule3 就是让每个方法按自己的判据决定，然后同时看两件事：选了几个、推得多准。

各方法的判据：

  DGatingDNN   门控值超过阈值。门控值分布有断崖，阈值在五个数量级之间取任何值
               结果都一样，所以这个数不算人为设定
  STG          门控值超过 0.5。它的门控值同样饱和在 0 或 1，阈值也不敏感
  LassoNet     沿 λ 路径，在验证损失距最优不超过 5% 的点里挑最稀疏的那个
  Lasso        交叉验证按 1 倍标准误规则选 alpha，系数非零的字段即为结果
  XGBoost      **没有自带的停止判据**，只能外部给一个固定阈值
  Pearson      同上

最后两个是这条规则要暴露的问题：它们的重要性得分是连续的，方法本身回答不了
"选多少个"。按用户要求，给它们一个**统一的固定阈值**（重要性归一化到最大值为 1 之后卡阈值），
阈值经过调优确定，但**12 个目标共用同一个值，不做逐目标调整**。
这样它们在简单目标上会选太多、在难目标上会选太少，选中字段数的离散程度会明显大于
能自主定数的方法——这正是固定阈值这条路的缺陷所在。

评价不合成为单一分数。合成需要人为定权重，反而引入新的主观性，与本文
"减少人为设定"的主张相冲突。改用两个维度画散点，并给出帕累托前沿：
若不存在"字段更少且精度更高"的其他方法，该方法就在前沿上——这个判定不需要权重。
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
SELF_METHODS = ["DGatingDNN", "STG", "LassoNet", "Lasso"]
FIXED_METHODS = ["XGBoost", "Pearson"]
TAU_CANDIDATES = [0.02, 0.05, 0.08, 0.12, 0.20, 0.30]


def select_self(m: str, X, y, cfg, scores=None, extra=None):
    """各方法按自己的判据给出选中字段的下标。"""
    if m == "DGatingDNN":
        return np.where(scores >= cfg.dgate_threshold)[0]
    if m == "STG":
        return np.where(scores >= cfg.stg_threshold)[0]
    if m == "Lasso":
        return np.where(scores > 1e-10)[0]          # LassoCV 的非零系数
    if m == "LassoNet":
        path = extra
        if path:
            # 直接取验证损失最小的点会选中全部字段——路径上损失通常在正则最弱处最低。
            # 改成"在最优损失 5% 以内的点里挑最稀疏的那个"，这是稀疏模型选点的惯例做法。
            best = min(p.val_loss for p in path)
            ok = [p for p in path if p.val_loss <= best * 1.05 + 1e-12]
            pick = min(ok, key=lambda p: int(np.asarray(p.selected).sum()))
            return np.where(np.asarray(pick.selected))[0]
        return np.where(scores > 0)[0]
    raise ValueError(m)


def tune_tau(df, targets, pools, cfg, want: float):
    """为 XGBoost / Pearson 选一个全局统一的阈值。

    目标是让这两种方法的平均选中字段数与能自主定数的方法接近，
    这样比较才不至于因为字段数差太多而失真。选定后对全部目标一视同仁。
    """
    recs = {m: {} for m in FIXED_METHODS}
    for t in targets:
        pool = pools[t]
        X, y = df[pool].to_numpy(float), df[t].to_numpy(float)
        for m in FIXED_METHODS:
            sc = np.asarray(bc.RANKERS[m](X, y, cfg=cfg), float)
            sc = sc / sc.max() if sc.max() > 0 else sc
            recs[m][t] = sc
    best = {}
    for m in FIXED_METHODS:
        rows = []
        for tau in TAU_CANDIDATES:
            ns = [int((recs[m][t] >= tau).sum()) for t in targets]
            rows.append({"tau": tau, "平均选中数": float(np.mean(ns)),
                         "最少": int(np.min(ns)), "最多": int(np.max(ns)),
                         "与目标字段数之差": abs(float(np.mean(ns)) - want)})
        tab = pd.DataFrame(rows).sort_values("与目标字段数之差")
        best[m] = (float(tab.iloc[0]["tau"]), tab)
    return best, recs


def pareto(points):
    """(字段数, R²) 的帕累托前沿：不存在字段更少且精度更高的点。"""
    out = []
    for i, (n, r) in enumerate(points):
        if not any((n2 <= n and r2 >= r and (n2 < n or r2 > r))
                   for j, (n2, r2) in enumerate(points) if j != i):
            out.append(i)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out-version", default=None)
    ap.add_argument("--pools", default="excl", choices=["excl", "both"])
    ap.add_argument("--targets", nargs="+", default=bc.sl.TARGETS)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    with runlock.single_instance("rule3"):
        tm = runlock.Timer("rule3 各方法自主定数的双指标对比")
        df = dataio.load_clean(args.year, VERSION)
        cfg = gates.TrainConfig(epochs=args.epochs, lambda_stg=0.005, lambda_dgate=0.005)
        root = (dataio.OUTPUTS / (args.out_version or f"pjm_{args.year}")
                / "05_baseline_compare" / "rule3")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for excl in ([True] if args.pools == "excl" else [False, True]):
            lv1 = "excl_targets" if excl else "incl_targets"
            for scr in [False, True]:
                lv2 = "with_screening" if scr else "no_screening"
                out = root / lv1 / lv2 / f"run_{stamp}"
                out.mkdir(parents=True, exist_ok=True)
                FIG, DAT = dataio.split_dirs(out)
                print(f"\n===== {lv1} / {lv2} =====", flush=True)

                pools = {t: sl.build_pool(df, t, excl, scr, args.year, True,
                                          args.out_version)[0]
                         for t in args.targets}

                # 第一遍：先把能自主定数的方法跑完，记录它们各自选了几个
                self_sel = {}
                for t in args.targets:
                    pool = pools[t]
                    X, y = df[pool].to_numpy(float), df[t].to_numpy(float)
                    self_sel[t] = {}
                    for m in SELF_METHODS:
                        try:
                            extra = None
                            if m == "DGatingDNN":
                                sc = gates.train("DGatingDNN", X, y, cfg).final_gates
                            else:
                                sc = np.asarray(bc.RANKERS[m](X, y, cfg=cfg), float)
                                if m == "LassoNet":
                                    extra = getattr(bc.rank_lassonet, "last", None)
                            self_sel[t][m] = select_self(m, X, y, cfg, sc, extra)
                        except Exception as e:
                            print(f"        [{m}] 失败：{type(e).__name__} {e}", flush=True)

                # 固定阈值要对齐的字段数，取全部自主定数方法的平均，不单看本文方法，
                # 否则等于把对比方法调成跟我们一样多，比较就失去意义了
                want = float(np.mean([len(v) for d in self_sel.values()
                                      for v in d.values()]))
                taus, importances = tune_tau(df, args.targets, pools, cfg, want)
                print(f"    固定阈值调优：目标字段数约 {want:.1f}，"
                      + "，".join(f"{m} τ={taus[m][0]}" for m in FIXED_METHODS), flush=True)
                for m in FIXED_METHODS:
                    taus[m][1].to_csv(DAT / f"tau_tuning_{m}.csv", index=False)

                rows = []
                for t in args.targets:
                    pool = pools[t]
                    X, y = df[pool].to_numpy(float), df[t].to_numpy(float)
                    full = gates.train("DNN", X, y, cfg).best_test_r2
                    print(f"    ── {sl.CN[t]}　候选 {len(pool)}　全量 R²={full:.4f}",
                          flush=True)
                    for m in SELF_METHODS + FIXED_METHODS:
                        if m in SELF_METHODS and m not in self_sel.get(t, {}):
                            continue
                        try:
                            if m in FIXED_METHODS:
                                sc = importances[m][t]
                                idx = np.where(sc >= taus[m][0])[0]
                                if idx.size == 0:
                                    idx = np.array([int(np.argmax(sc))])
                                rule = f"固定阈值 τ={taus[m][0]}（12 个目标共用）"
                            else:
                                idx = self_sel[t][m]         # 复用第一遍的结果
                                rule = {"DGatingDNN": f"门控值 ≥ {cfg.dgate_threshold}（断崖，阈值不敏感）",
                                        "STG": f"门控值 ≥ {cfg.stg_threshold}",
                                        "LassoNet": "λ 路径上，验证损失在最优 5% 以内里最稀疏的点",
                                        "Lasso": "交叉验证按 1 倍标准误规则选 alpha 后的非零系数"}[m]
                            if idx.size == 0:
                                idx = np.array([0])
                            r2 = gates.retrain_subset(X, y, list(idx), cfg)
                            rows.append({"目标": t, "中文名": sl.CN[t], "方法": m,
                                         "定数规则": rule, "选中数": int(idx.size),
                                         "占候选比": idx.size / len(pool),
                                         "测试R2": r2, "全量R2": full,
                                         "候选数": len(pool),
                                         "选中字段": "|".join(pool[i] for i in idx)})
                            print(f"        {m:12s} 选 {idx.size:2d} 个 → R²={r2:.4f}",
                                  flush=True)
                        except Exception as e:
                            print(f"        [{m}] 失败：{type(e).__name__} {e}", flush=True)

                agg = pd.DataFrame(rows)
                agg.to_csv(DAT / "overall_comparison.csv", index=False)
                (DAT / "config.json").write_text(json.dumps(
                    {"rule": "rule3", "候选池": lv1, "初筛": lv2,
                     "固定阈值": {m: taus[m][0] for m in FIXED_METHODS},
                     "targets": args.targets, **cfg.to_dict()},
                    ensure_ascii=False, indent=2), encoding="utf-8")

                plots.rule3_scatter_plot(agg, FIG / "fig_scatter.png",
                                         f"{lv1} / {lv2}　各方法自主定数的结果")
                plots.rule3_spread_plot(agg, FIG / "fig_count_spread.png",
                                        "各方法在 12 个目标上选中字段数的离散程度")
                write_overall(out / "总结.md", agg, lv1, lv2, taus)
                tm.mark(f"{lv1} / {lv2}")

        (root / "timing.md").write_text(
            "# rule3 执行耗时\n\n" + tm.report() + "\n", encoding="utf-8")
        print("\n" + tm.report())


def write_overall(path: Path, agg: pd.DataFrame, lv1: str, lv2: str, taus) -> None:
    g = agg.groupby("方法").agg(平均选中数=("选中数", "mean"),
                               选中数标准差=("选中数", "std"),
                               最少=("选中数", "min"), 最多=("选中数", "max"),
                               平均R2=("测试R2", "mean")).round(4)
    pts = list(zip(g["平均选中数"], g["平均R2"]))
    front = set(g.index[i] for i in pareto(pts))
    L = [f"# rule3 各方法自主定数（{lv1} / {lv2}）", "",
         "## 一、规则", "",
         "不再统一字段数，让每个方法按自己的判据决定选多少个，"
         "然后同时看两件事：**选了几个**和**推得多准**。", "",
         "| 方法 | 定数规则 |", "|---|---|"]
    for m, r in agg.drop_duplicates("方法").set_index("方法")["定数规则"].items():
        L.append(f"| {m} | {r} |")
    L += ["", "> XGBoost 和 Pearson 的重要性得分是连续的，方法本身回答不了"
          "「选多少个」，只能外部给一个固定阈值。这里给的阈值经过调优，"
          "但 12 个目标共用同一个值、不做逐目标调整——"
          "这正是这条路的缺陷，下面的离散程度一栏可以看出来。", "",
          "## 二、总体结果", "",
          "| 方法 | 平均选中数 | 选中数标准差 | 最少–最多 | 平均测试 R² | 帕累托前沿 |",
          "|---|---|---|---|---|---|"]
    for m, r in g.sort_values("平均R2", ascending=False).iterrows():
        L.append(f"| {'**' + m + '**' if m in front else m} | {r['平均选中数']:.1f} | "
                 f"{r['选中数标准差']:.1f} | {int(r['最少'])}–{int(r['最多'])} | "
                 f"{r['平均R2']:.4f} | {'✓' if m in front else ''} |")
    L += ["", "帕累托前沿的含义：不存在另一个方法「字段更少且精度更高」。"
          "这个判定不需要给两个指标定权重，所以不引入新的人为设定。", "",
          "## 三、固定阈值方法的问题", "",
          "| 方法 | 阈值 | 选中数标准差 | 最少–最多 |", "|---|---|---|---|"]
    for m in FIXED_METHODS:
        if m in g.index:
            L.append(f"| {m} | τ={taus[m][0]} | {g.loc[m, '选中数标准差']:.1f} | "
                     f"{int(g.loc[m, '最少'])}–{int(g.loc[m, '最多'])} |")
    self_std = g.loc[[m for m in SELF_METHODS if m in g.index], "选中数标准差"].mean()
    fix_std = g.loc[[m for m in FIXED_METHODS if m in g.index], "选中数标准差"].mean()
    L += ["", f"能自主定数的方法，选中字段数的标准差平均 {self_std:.1f}；"
          f"用固定阈值的方法平均 {fix_std:.1f}。", "",
          "差别的来源是：能自主定数的方法会随目标的难易自动调整选几个，"
          "固定阈值只能一刀切，在容易推的目标上选太多、在难推的目标上选太少。", "",
          "## 四、逐目标明细", "",
          "| 目标 | 方法 | 选中数 | 占候选比 | 测试 R² | 全量 R² | 选中的字段 |",
          "|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        fs = "、".join(sl.cn(c) for c in str(r["选中字段"]).split("|") if c)
        if len(fs) > 300:                      # 有的方法选了几十个，全列会撑爆表格
            keep = fs[:300].rsplit("、", 1)[0]
            fs = keep + f"　…（共 {int(r['选中数'])} 个，完整列表见 data/overall_comparison.csv）"
        L.append(f"| {r['中文名']} | {r['方法']} | {int(r['选中数'])} | "
                 f"{r['占候选比']:.1%} | {r['测试R2']:.4f} | {r['全量R2']:.4f} | {fs} |")
    L += ["", "## 五、图", "",
          "- `fig_scatter.png`　横轴选中字段数、纵轴测试 R²，越靠左上越好，"
          "连线标出帕累托前沿",
          "- `fig_count_spread.png`　各方法在 12 个目标上选中字段数的分布，"
          "看得出固定阈值方法的波动更大", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
