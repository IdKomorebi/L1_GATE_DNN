"""门控值由六类相关系数生成——能不能直接用在 D-Gating 上。

## 想法

标准 D-Gating 给每个字段一组自由的门控因子，参数个数随字段数增长，所以
来了一个没参与训练的新字段，没有办法给它算门控值，只能重训。

改进门控的思路是让门控值由字段自身的统计量算出来：

    γ[d, j] = c[j] · W[d] + b[d]          d = 1 … depth-1
    门控值   = |∏_d γ[d, j]|

c[j] 是第 j 个字段与目标之间的六个相关系数（按列标准化）。可学参数只有
(depth-1) × (6+1) 个，**与字段数无关**。于是新字段只要算出六个相关系数，
代入同一组 W、b 就能直接得到门控值，与阈值比较即可，**不需要重新训练**。

W 初始化为 0、b 初始化为 1，训练起点每个字段的门控值都是 1，
与标准 D-Gating 一致，两者才好比较。

## 这个脚本回答三个问题

1. **加上这层参数化，整体性能会不会变差**
   在同一个候选池上把标准 D-Gating 和改进版跑一遍，比测试 R²、
   选中字段数、选中子集重训后的 R²，以及门控值分布还有没有断崖。

2. **对没参与训练的字段，估出来的门控值准不准**
   拿掉若干个字段，用剩下的训练，再用学到的 W 给拿掉的字段算门控值，
   与它们参与训练时得到的门控值比较。

3. **拿掉的字段要覆盖不同强度**
   全挑无关字段的话，估准了也说明不了问题——池子里 48 个字段有 35 个
   门控值恰好是 0，随便挑几乎必然挑到这些。所以按标准 D-Gating 的门控值
   分强、中、弱三层各挑。
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
import screening as scr  # noqa: E402
import runlock  # noqa: E402

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
METRICS = list(scr.METRICS)


def corr_matrix(pool: list[str]) -> np.ndarray:
    """每个字段与目标之间的六个相关系数，直接用第 3 步已经算好的。"""
    d = dataio.OUTPUTS / OUT_VERSION / "03_screening" / "data"
    obs = pd.read_csv(d / f"screen_{TARGET}.csv").set_index("field")
    miss = [c for c in pool if c not in obs.index]
    if miss:
        raise KeyError(f"初筛结果里缺少字段：{miss[:5]}")
    return obs.loc[pool, METRICS].to_numpy(float)


def subset_r2(X, y, gate: np.ndarray, cfg, k: int | None, thr: float) -> tuple[float, int]:
    """按门控值取字段重训一个普通 DNN。k 给定就取前 k 个，否则按阈值取。"""
    order = np.argsort(-gate)
    idx = list(order[:k]) if k else [i for i in order if gate[i] >= thr]
    if not idx:
        idx = [int(order[0])]
    return gates.retrain_subset(X, y, idx, cfg), len(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--n-hold", type=int, default=5, help="一共拿掉几个字段")
    args = ap.parse_args()

    with runlock.single_instance("improved_gate"):
        tm = runlock.Timer("门控值由相关系数生成")
        cfg = gates.TrainConfig(epochs=args.epochs, lambda_dgate=0.005)
        df = dataio.load_clean(YEAR, "main")
        pool, note = sl.build_pool(df, TARGET, True, False, YEAR, True, OUT_VERSION)
        X = df[pool].to_numpy(float)
        y = df[TARGET].to_numpy(float)
        C = corr_matrix(pool)
        thr = cfg.dgate_threshold
        out = (dataio.OUTPUTS / OUT_VERSION / "06_improved_gate" /
               f"run_{datetime.now():%Y%m%d_%H%M%S}")
        out.mkdir(parents=True, exist_ok=True)
        FIG, DAT = dataio.split_dirs(out)
        print(f"目标 {sl.CN[TARGET]}　候选 {len(pool)} 个（{note}）", flush=True)

        # ---------- 问题 1：整体性能会不会变差 ----------
        print(f"\n[1/2] 全池对比：标准 D-Gating vs 门控由相关系数生成", flush=True)
        std = gates.train_keep_model("DGatingDNN", X, y, cfg)
        imp = gates.train_keep_model("ImprovedDGating", X, y, cfg, corr=C)
        gs, gi = std.final_gates, imp.final_gates
        n_std = int((gs >= thr).sum())
        r2_std, _ = subset_r2(X, y, gs, cfg, None, thr)
        r2_imp_same, _ = subset_r2(X, y, gi, cfg, n_std, thr)   # 同字段数，公平比
        r2_imp_own, n_imp = subset_r2(X, y, gi, cfg, None, thr)
        cmp = pd.DataFrame([
            {"方法": "标准 D-Gating", "门控模型R2": std.best_test_r2,
             "按阈值选中数": n_std, "选中子集R2": r2_std,
             "门控值最大": gs.max(), "门控值中位": float(np.median(gs)),
             "恰好为0的个数": int((gs <= 1e-12).sum())},
            {"方法": "改进门控（相关系数生成）", "门控模型R2": imp.best_test_r2,
             "按阈值选中数": n_imp, "选中子集R2": r2_imp_own,
             "门控值最大": gi.max(), "门控值中位": float(np.median(gi)),
             "恰好为0的个数": int((gi <= 1e-12).sum())},
        ])
        print(cmp.round(4).to_string(index=False), flush=True)
        print(f"  取同样 {n_std} 个字段时：标准 {r2_std:.4f}　改进 {r2_imp_same:.4f}",
              flush=True)
        cmp["同字段数下R2"] = [r2_std, r2_imp_same]
        cmp.to_csv(DAT / "overall.csv", index=False)
        pd.DataFrame({"field": pool, "中文名": [sl.cn(c) for c in pool],
                      "标准门控": gs, "改进门控": gi}).to_csv(
            DAT / "gates_compare.csv", index=False)
        tm.mark("全池对比")

        # ---------- 问题 2：对没参与训练的字段估得准不准 ----------
        # 按标准 D-Gating 的门控值分三层挑，务必覆盖不同强度：
        # 全挑无关字段的话，估准了也说明不了问题——48 个里有 35 个门控值恰好是 0。
        gser = pd.Series(gs, index=pool).sort_values(ascending=False)
        act, ina = gser[gser >= thr], gser[gser < thr]
        picked = ([(f, "强") for f in act.index[:2]] +          # 门控值最高的两个
                  [(f, "中") for f in list(act.index[-1:]) +
                   list(ina.index[:1])] +                       # 断崖两侧各一个
                  [(f, "弱") for f in ina.index[-1:]])          # 门控值最低的一个
        seen, hold, lvl = set(), [], {}
        for f, v in picked:
            if f not in seen and len(hold) < args.n_hold:
                seen.add(f); hold.append(f); lvl[f] = v
        print(f"\n[2/2] 拿掉 {len(hold)} 个字段：" +
              "　".join(f"{sl.cn(f)}({lvl[f]},{gser[f]:.4f})" for f in hold), flush=True)

        pos = [pool.index(f) for f in hold]
        keep = [i for i in range(len(pool)) if i not in pos]
        sub = gates.train_keep_model("ImprovedDGating", X[:, keep], y, cfg,
                                     corr=C[keep])
        pred = sub.model.gate_from_corr(C[pos], C[keep])
        rows = []
        for f, p_, gp in zip(hold, pos, pred):
            rows.append({
                "字段": f, "中文名": sl.cn(f), "层": lvl[f],
                "标准DGating门控": float(gs[p_]),
                "改进门控_参与训练": float(gi[p_]),
                "改进门控_外推估计": float(gp),
                "外推误差": float(gp - gi[p_]),
                "标准判定": int(gs[p_] >= thr),
                "外推判定": int(gp >= thr),
            })
        res = pd.DataFrame(rows)
        res["判定一致"] = (res.改进门控_参与训练 >= thr).astype(int).eq(
            res.外推判定).astype(int)
        print(res[["中文名", "层", "标准DGating门控", "改进门控_参与训练",
                   "改进门控_外推估计", "外推误差", "判定一致"]].round(4).to_string(index=False),
              flush=True)
        res.to_csv(DAT / "holdout.csv", index=False)
        tm.mark("留出字段外推")

        (DAT / "config.json").write_text(json.dumps(
            {"target": TARGET, "候选数": len(pool), "轮次": args.epochs,
             "活跃阈值": thr, "指标": METRICS, "留出字段": hold},
            ensure_ascii=False, indent=2), encoding="utf-8")
        write_doc(out / "说明.md", cmp, res, pool, gs, gi, thr, n_std, r2_std,
                  r2_imp_same)
        plot(FIG, gs, gi, res, thr, pool)
        (out / "timing.md").write_text("# 耗时\n\n" + tm.report() + "\n",
                                       encoding="utf-8")
        print("\n" + tm.report())
        print(f"\n写入 {out}")


def plot(FIG, gs, gi, res, thr, pool):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    o = np.argsort(-gs)
    ax[0].semilogy(np.maximum(gs[o], 1e-8), "o-", ms=3, lw=1.2, label="标准 D-Gating")
    ax[0].semilogy(np.maximum(gi[o], 1e-8), "s-", ms=3, lw=1.2,
                   label="改进门控（相关系数生成）")
    ax[0].axhline(thr, color="k", ls="-.", lw=1, alpha=.6)
    ax[0].text(len(gs) * .98, thr * 1.3, f"阈值 {thr}", ha="right", fontsize=9)
    ax[0].set_xlabel("字段（按标准门控值从大到小排）")
    ax[0].set_ylabel("门控值（对数刻度）")
    ax[0].set_title("门控值分布：断崖还在不在")
    ax[0].legend(fontsize=9)
    c = {"强": "#c0392b", "中": "#e67e22", "弱": "#7f8c8d"}
    for _, r in res.iterrows():
        ax[1].scatter(r.改进门控_参与训练, r.改进门控_外推估计, s=70,
                      color=c.get(r["层"], "#333"), zorder=3)
        ax[1].annotate(r["中文名"], (r.改进门控_参与训练, r.改进门控_外推估计),
                       fontsize=8, xytext=(4, 4), textcoords="offset points")
    lo = min(res.改进门控_参与训练.min(), res.改进门控_外推估计.min()) * .8
    hi = max(res.改进门控_参与训练.max(), res.改进门控_外推估计.max()) * 1.2
    ax[1].plot([lo, hi], [lo, hi], "k--", lw=1, alpha=.5)
    ax[1].set_xlabel("参与训练时得到的门控值")
    ax[1].set_ylabel("没参与训练、由相关系数外推的门控值")
    ax[1].set_title("外推估计准不准（点越靠近虚线越准）")
    fig.tight_layout()
    fig.savefig(FIG / "fig_improved_gate.png", dpi=150)
    plt.close(fig)


def md(d) -> str:
    """把表格写成 markdown，不引第三方依赖。"""
    cols = list(d.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
            for c in cols) + " |")
    return "\n".join(out)


def write_doc(path, cmp, res, pool, gs, gi, thr, n_std, r2_std, r2_imp_same):
    L = ["# 门控值由六类相关系数生成", "",
         f"目标字段：净实际交换功率　候选 {len(pool)} 个", "",
         "## 一、做法", "",
         "标准 D-Gating 给每个字段一组自由的门控因子，参数个数随字段数增长，"
         "所以没参与训练的字段没法算门控值。改进的做法是让因子由该字段与目标之间的"
         "六个相关系数算出来：", "",
         "```", "γ[d, j] = c[j] · W[d] + b[d]      d = 1 … depth-1",
         "门控值   = |∏_d γ[d, j]|", "```", "",
         f"可学参数只有 (depth-1)×(6+1) = {(4 - 1) * 7} 个，与字段数无关。"
         "新字段只要算出六个相关系数代进去就有门控值，不必重训。", "",
         "W 初始化为 0、b 初始化为 1，训练起点所有字段门控值都是 1，"
         "与标准 D-Gating 一致。", "",
         "## 二、整体性能有没有受影响", "",
         md(cmp.round(4)), "",
         f"取同样 {n_std} 个字段时：标准 {r2_std:.4f}，改进 {r2_imp_same:.4f}。", "",
         "## 三、对没参与训练的字段，外推估计准不准", "",
         md(res[["中文名", "层", "标准DGating门控", "改进门控_参与训练",
                 "改进门控_外推估计", "外推误差", "判定一致"]].round(4)),
         "",
         "## 四、图", "",
         "- `fig_improved_gate.png` 左：两种门控值分布，看断崖还在不在；"
         "右：外推估计与参与训练所得的对比，点越靠近虚线越准", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
