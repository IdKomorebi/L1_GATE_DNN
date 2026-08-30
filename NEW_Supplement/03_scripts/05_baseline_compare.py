"""第 5 步：与其他特征选择方法的同预算对比。

所有运行都已剥离第一类关系（公式与近似关系）——这是方法的必需步骤，
不作为可选项：不剥离的话，任何特征选择方法都会被同一条公式捷径吸引过去，
选出来的是公式分量而不是隐性推断源，方法之间也就没有分辨力。

对比规则 rule1：**预算由本文方法定**。
先用本文的 D-Gating 门控选出 n 个字段（n 由门控值的断崖自然切出来，
阈值在 1e-6 到 0.01 之间取任何值都是同一个 n），
然后让每个对比方法按自己的排序取前 n 个，统一用同一个普通 DNN 训练测试。
比较的是"在同样多的字段下谁挑得更准"。

用本文方法的 n 作为统一预算是合理的，因为这个 n 不是人为设定的——
D-Gating 的门控值分布有断崖，被淘汰的字段直接落到机器零，
只要阈值取得足够小就必然得到同一个 n。其他方法的门控值/重要性是连续分布的，
它们自己定不出一个不依赖人工阈值的 n。

参与对比的方法：

  DGatingDNN   本文方法，可微稀疏门控（NeurIPS 2025）
  STG          随机门控，用高斯松弛逼近 L0（ICML 2020）
  LassoNet     神经网络旁挂线性通路，特征级稀疏（JMLR 2021）
  XGBoost      树模型特征重要性（2016）
  Lasso        线性稀疏（1996）
  Pearson      相关系数排序，作为"只看两两关系"的参照

目录组织：
  05_baseline_compare/rule1/<候选池>/<是否初筛>/run_<时间戳>/
      overall_comparison.md         全部目标字段的总体比较
      target_<字段>/comparison.md   单个目标各方法的比较
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_src"))
import dataio  # noqa: E402
import gates  # noqa: E402
import plots  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "sl", Path(__file__).resolve().parent / "04_source_location.py")
sl = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(sl)

warnings.filterwarnings("ignore")
VERSION = "main"
METHOD_NOTE = {
    "DGatingDNN": "本文方法：可微稀疏门控（NeurIPS 2025）",
    "STG": "随机门控，高斯松弛逼近 L0（ICML 2020）",
    "LassoNet": "神经网络旁挂线性通路的特征级稀疏（JMLR 2021）",
    "XGBoost": "树模型特征重要性（2016）",
    "Lasso": "线性稀疏（1996）",
    "Pearson": "相关系数排序（只看两两关系）",
}
BASELINES = ["STG", "LassoNet", "XGBoost", "Lasso", "Pearson"]


# --------------------------------------------------------------------------
# 各方法的字段排序
# --------------------------------------------------------------------------

def rank_pearson(X, y, **_):
    s = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if X[:, j].std() > 0 else 0.0
                  for j in range(X.shape[1])])
    return np.nan_to_num(s)


def rank_lasso(X, y, **_):
    """交叉验证选 alpha。用 1 倍标准误规则：在交叉验证误差不显著变差的前提下
    取正则最强（也就是最稀疏）的那个 alpha，这是 Lasso 选点的惯例做法。
    直接取误差最小的 alpha 会明显欠正则、留下过多字段。"""
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    cv = LassoCV(cv=5, random_state=42, max_iter=5000).fit(Xs, y)
    mse = cv.mse_path_.mean(axis=1)
    se = cv.mse_path_.std(axis=1) / np.sqrt(cv.mse_path_.shape[1])
    i = int(np.argmin(mse))
    ok = np.where(mse <= mse[i] + se[i])[0]
    alpha = float(cv.alphas_[ok.min()]) if ok.size else float(cv.alpha_)
    m = Lasso(alpha=alpha, max_iter=5000).fit(Xs, y)
    rank_lasso.alpha = alpha
    return np.abs(m.coef_)


def rank_xgboost(X, y, **_):
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.08,
                     random_state=42, verbosity=0).fit(X, y)
    return np.asarray(m.feature_importances_, dtype=float)


def rank_stg(X, y, cfg=None, **_):
    r = gates.train("STG", X, y, cfg)
    rank_stg.last = r          # 留住训练过程，后面画门控演化图
    return r.final_gates


def rank_lassonet(X, y, **_):
    from lassonet import LassoNetRegressor
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    ys = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
    m = LassoNetRegressor(hidden_dims=(64, 32), n_iters=(80, 20),
                          verbose=0, random_state=42)
    rank_lassonet.last = m.path(Xs, ys)     # λ 路径，用来画筛选过程
    return np.asarray(m.feature_importances_, dtype=float)


RANKERS = {"Pearson": rank_pearson, "Lasso": rank_lasso, "XGBoost": rank_xgboost,
           "STG": rank_stg, "LassoNet": rank_lassonet}


# --------------------------------------------------------------------------

def k_grid(n: int, m: int) -> list[int]:
    """取前 k 个字段的 k 取值网格。围绕统一预算 n 铺开，两端各留几档。"""
    cand = {1, 3, max(2, n // 2), n, min(m, n + 5), min(m, max(n * 2, n + 12)), m}
    return sorted(k for k in cand if 1 <= k <= m)


def compare_one_target(df, target, pool, cfg, out: Path):
    """先用本文方法定预算 n，再让各方法在同预算下比，并画出各自的筛选过程。"""
    X, y = df[pool].to_numpy(float), df[target].to_numpy(float)
    d = out / f"target_{target}"
    d.mkdir(parents=True, exist_ok=True)
    FIG, DAT = dataio.split_dirs(d)

    full_r2 = gates.train("DNN", X, y, cfg).best_test_r2
    ours = gates.train("DGatingDNN", X, y, cfg)
    g = ours.final_gates
    n = max(int((g >= cfg.dgate_threshold).sum()), 1)
    ks = k_grid(n, len(pool))

    rows, curves, sel, procs = [], {}, {}, {}
    for m in ["DGatingDNN"] + BASELINES:
        try:
            score = g if m == "DGatingDNN" else RANKERS[m](X, y, cfg=cfg)
        except Exception as e:
            print(f"        [{m}] 失败：{type(e).__name__} {e}", flush=True)
            continue
        idx = np.argsort(-np.asarray(score, float))
        (DAT / m).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"field": [pool[i] for i in idx],
                      "中文名": [sl.cn(pool[i]) for i in idx],
                      "score": np.asarray(score, float)[idx]}).to_csv(
            (DAT / m) / "ranking.csv", index=False)

        curve = [gates.retrain_subset(X, y, list(idx[:k]), cfg) for k in ks]
        curves[m] = curve
        sel[m] = [pool[i] for i in idx[:n]]
        rows.append({"方法": m, "说明": METHOD_NOTE[m],
                     "n": curve[ks.index(n)], "前n个字段": "|".join(sel[m])})
        print(f"        {m:12s} n={n} → R²={curve[ks.index(n)]:.4f}", flush=True)

        # 有训练过程的方法，把过程存下来并画图
        r = None
        if m == "DGatingDNN":
            r = ours
        elif m == "STG":
            r = getattr(rank_stg, "last", None)
        if r is not None and r.gate_history is not None:
            pd.DataFrame(r.gate_history, columns=pool).to_csv(
                (DAT / m) / "gate_history.csv", index=False)
            thr = cfg.dgate_threshold if m == "DGatingDNN" else cfg.stg_threshold
            (FIG / m).mkdir(parents=True, exist_ok=True)
            plots.gate_evolution_plot(
                r, thr, [sl.cn(c) for c in pool], (FIG / m) / "fig_process.png",
                f"{sl.CN[target]}　{m} 的筛选过程（门控值随训练轮次变化）")
            procs[m] = {"最终活跃数": int((r.final_gates >= thr).sum()),
                        "门控模型 R²": r.best_test_r2, "阈值": thr}
        if m == "LassoNet":
            pth = getattr(rank_lassonet, "last", None)
            if pth:
                pd.DataFrame({"lambda": [p.lambda_ for p in pth],
                              "n_selected": [int(p.selected.sum()) for p in pth],
                              "val_loss": [p.val_loss for p in pth]}).to_csv(
                    (DAT / m) / "lambda_path.csv", index=False)
                procs[m] = {"λ 路径长度": len(pth),
                            "字段数范围": f"{min(int(p.selected.sum()) for p in pth)}"
                                          f"–{max(int(p.selected.sum()) for p in pth)}"}

    res = pd.DataFrame(rows)
    res.to_csv(DAT / "comparison.csv", index=False)
    pd.DataFrame(curves, index=ks).rename_axis("k").to_csv(DAT / "topk_curves.csv")

    plots.baseline_bar_plot(res, full_r2, n, FIG / "fig_comparison.png",
                            f"{sl.CN[target]}　同预算（{n} 个字段）下各方法对比")
    plots.topk_curve_plot(curves, ks, full_r2, n, FIG / "fig_topk_curve.png",
                          f"{sl.CN[target]}　各方法取前 k 个字段的表现")
    plots.selection_matrix_plot(sel, pool, FIG / "fig_selection_matrix.png",
                                f"{sl.CN[target]}　各方法在预算 n={n} 下选中了哪些字段")

    best = res.loc[res["n"].idxmax()]
    ours_set = set(sel.get("DGatingDNN", []))
    L = [f"# 各方法对比：{sl.CN[target]}", "",
         "## 一、这次比的是什么", "",
         f"- 目标字段：`{target}`（{sl.CN[target]}）",
         f"- 候选字段：{len(pool)} 个",
         f"- **统一预算 n = {n}**，由本文方法的门控断崖自然切出来",
         f"- 全部 {len(pool)} 个字段的普通 DNN：R² = {full_r2:.4f}（能力上限参照）", "",
         "每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。", "",
         "## 二、同预算下的结果", "", "| 方法 | 说明 | 测试 R² |", "|---|---|---|"]
    for _, r in res.sort_values("n", ascending=False).iterrows():
        star = " ⭐" if r["方法"] == best["方法"] else ""
        L.append(f"| {r['方法']}{star} | {r['说明']} | **{r['n']:.4f}** |")
    L += ["", f"> 同预算下最好的是 **{best['方法']}**（R² = {best['n']:.4f}）。", "",
          "### 各方法在这个预算下选中的字段", "",
          "只看 R² 看不出方法差在哪。同样给几个字段，谁挑的是负荷、谁挑的是备用，"
          "这才是实质差别，也是判断结果合不合物理常识的依据。", "",
          "| 方法 | 选中的字段（按各自的排序） |", "|---|---|"]
    for _, r in res.iterrows():
        fs = "、".join(sl.cn(c) for c in str(r["前n个字段"]).split("|") if c)
        nm = f"**{r['方法']}**" if r["方法"] == "DGatingDNN" else r["方法"]
        L.append(f"| {nm} | {fs} |")
    L += ["",
          "## 三、不同字段数下的完整曲线", "",
          "只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现"
          "（对应图 `fig_topk_curve.png`）：", "",
          "| 方法 | " + " | ".join(f"k={k}" for k in ks) + " |",
          "|---" * (len(ks) + 1) + "|"]
    for m, v in sorted(curves.items(), key=lambda kv: -kv[1][ks.index(n)]):
        L.append(f"| {m} | " + " | ".join(f"{x:.4f}" for x in v) + " |")

    L += ["", "## 四、各方法的筛选过程", ""]
    for m in ["DGatingDNN", "STG"]:
        if m in procs:
            p = procs[m]
            L += [f"### {m}", "",
                  f"- 门控模型自身的测试 R²：{p['门控模型 R²']:.4f}",
                  f"- 活跃阈值 {p['阈值']}，最终活跃字段 {p['最终活跃数']} 个",
                  f"- 过程图 `{m}/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，"
                  "左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低",
                  f"- 逐轮门控值记录在 `{m}/gate_history.csv`", ""]
    if "LassoNet" in procs:
        p = procs["LassoNet"]
        L += ["### LassoNet", "",
              f"- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除",
              f"- 路径共 {p['λ 路径长度']} 个点，字段数从 {p['字段数范围']} 变化",
              "- 记录在 `LassoNet/lambda_path.csv`", ""]
    L += ["### XGBoost / Lasso / Pearson", "",
          "这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，"
          "选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。", "",
          "## 五、各方法选中的字段重合情况", "",
          "| 方法 | 与本文方法重合 | 重合率 |", "|---|---|---|"]
    for m, fs in sel.items():
        if m == "DGatingDNN":
            continue
        L.append(f"| {m} | {len(ours_set & set(fs))} / {n} | "
                 f"{len(ours_set & set(fs)) / n:.0%} |")
    L += ["", "详见图 `fig_selection_matrix.png`。", "",
          "## 六、本文方法选出的字段", "", "| 排名 | 字段 | 业务含义 |", "|---|---|---|"]
    for i, f in enumerate(sel.get("DGatingDNN", []), 1):
        L.append(f"| {i} | `{f}` | {sl.cn(f)} |")
    L += ["", "## 七、本目录下的图", "",
          "- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）",
          "- `fig_comparison.png`　统一预算下的横向对比",
          "- `fig_selection_matrix.png`　各方法选中了哪些字段",
          "- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）", ""]
    (d / "总结.md").write_text("\n".join(L), encoding="utf-8")

    res["target"] = target
    res["中文名"] = sl.CN[target]
    res["budget"] = n
    res["full_r2"] = full_r2
    res["n_pool"] = len(pool)
    return res, sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out-version", default=None)
    ap.add_argument("--pools", default="excl", choices=["excl", "both"])
    ap.add_argument("--rule", default="rule1")
    ap.add_argument("--targets", nargs="+",
                    default=["net_actual_interchange_mw", "metered_load_mw",
                             "congestion_price_rt"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lambda-stg", type=float, default=0.005)
    args = ap.parse_args()

    df = dataio.load_clean(args.year, VERSION)
    cfg = gates.TrainConfig(epochs=args.epochs, lambda_stg=args.lambda_stg,
                            lambda_dgate=0.005)
    root = (dataio.OUTPUTS / (args.out_version or f"pjm_{args.year}")
            / "05_baseline_compare" / args.rule)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for excl in ([True] if args.pools == "excl" else [False, True]):
        lv1 = "excl_targets" if excl else "incl_targets"
        if True:                       # 第一类关系一律剥离，不作为对照维度
            strip = True
            for scr in [False, True]:
                lv2 = "with_screening" if scr else "no_screening"
                out = root / lv1 / lv2 / f"run_{stamp}"
                out.mkdir(parents=True, exist_ok=True)
                OFIG, ODAT = dataio.split_dirs(out)
                print(f"\n===== {lv1} / {lv2} =====", flush=True)

                allres, sel_all = [], {}
                for t in args.targets:
                    pool, note = sl.build_pool(df, t, excl, scr, args.year, strip,
                                               args.out_version)
                    print(f"    ── {sl.CN[t]}　候选 {len(pool)} 个（{note}）", flush=True)
                    r, sel = compare_one_target(df, t, pool, cfg, out)
                    allres.append(r); sel_all[sl.CN[t]] = sel

                agg = pd.concat(allres, ignore_index=True)
                agg.to_csv(ODAT / "overall_comparison.csv", index=False)
                (ODAT / "config.json").write_text(json.dumps(
                    {"rule": args.rule, "候选池": lv1, "筛选设定": lv2,
                     "targets": args.targets, **cfg.to_dict()},
                    ensure_ascii=False, indent=2), encoding="utf-8")
                write_overall(out / "总结.md", agg, lv1, lv2, args.rule)
                plots.baseline_overall_plot(agg, OFIG / "fig_overall.png",
                                            f"{lv1} / {lv2}　各方法在统一预算下的表现")
                plots.method_agreement_plot(
                    sel_all, OFIG / "fig_method_agreement.png",
                    "各方法选中字段的重合度（各目标平均）")
    print(f"\n完成，输出根目录 {root}")


def write_overall(path: Path, agg: pd.DataFrame, lv1: str, lv2: str, rule: str) -> None:
    piv = agg.pivot_table(index="方法", columns="中文名", values="n")
    mean = agg.groupby("方法")["n"].mean().sort_values(ascending=False)
    L = [f"# 各方法总体比较（{rule}）", "",
         f"候选池：{'排除其余关注字段' if lv1 == 'excl_targets' else '含其余关注字段'}　"
         f"｜筛选设定：{lv2}", "",
         "## 一、对比规则", "",
         "每个目标字段的**统一预算 n 由本文方法定**——D-Gating 的门控值分布有断崖，"
         "被淘汰的字段直接落到机器零，阈值在五个数量级之间取任何值都得到同一个 n，"
         "所以这个 n 不依赖人工设定。其余方法按自己的排序取前 n 个，"
         "统一用同一个普通 DNN 训练测试。", "",
         "## 二、平均表现（各目标测试 R² 的均值）", "",
         "| 排名 | 方法 | 说明 | 平均 R² |", "|---|---|---|---|"]
    note = agg.drop_duplicates("方法").set_index("方法")["说明"].to_dict()
    for i, (m, v) in enumerate(mean.items(), 1):
        L.append(f"| {i} | {'**' + m + '**' if i == 1 else m} | {note.get(m, '')} | {v:.4f} |")
    L += ["", "## 三、逐个目标字段", "",
          "| 方法 | " + " | ".join(piv.columns) + " |",
          "|---" * (len(piv.columns) + 1) + "|"]
    for m in mean.index:
        L.append(f"| {m} | " + " | ".join(
            f"{piv.loc[m, c]:.4f}" if pd.notna(piv.loc[m, c]) else "—"
            for c in piv.columns) + " |")
    b = agg.drop_duplicates("中文名").set_index("中文名")
    L += ["", "## 四、各目标的预算与上限", "",
          "| 目标 | 候选字段数 | 统一预算 n | 全量 DNN R² |", "|---|---|---|---|"]
    for c in piv.columns:
        L.append(f"| {c} | {int(b.loc[c, 'n_pool'])} | {int(b.loc[c, 'budget'])} | "
                 f"{b.loc[c, 'full_r2']:.4f} |")
    L += ["", "## 五、图", "",
          "- `fig_overall.png`　各方法在各目标上的测试 R²",
          "- `fig_method_agreement.png`　各方法选中字段的重合度",
          "- 各目标目录下还有 `fig_topk_curve.png`（不同字段数下的完整曲线）"
          "和各方法的筛选过程图", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
