"""处置实验的时序划分补充：更严评估口径下的残余可推断性。

15_mitigation.py 的全部结果采用随机划分（对应"外部已掌握同批次部分配对样本"的
部分披露场景）。本脚本补一组按时间顺序划分的结果：前 80% 时段训练、后 20% 时段
测试，对应"外部用历史数据训练模型、推断新发布数据"的持续发布场景。

两轮实验的差别只有划分方式一处：

    处置实现、处置随机种子、候选池、选中字段、网络结构、训练轮数全部一致；
    处置函数直接从 15_mitigation.py 导入，未复制代码；
    gates.split 仅在训练期间被临时替换为时间顺序版本，结束后恢复。

每个目标记录（处置方式 × 划分方式）：

    处置后·选中   只处置选中字段，用处置后的选中字段推断
    处置后·全量   只处置选中字段、其余照常发布，用全部字段推断
    处置前·选中/全量   未处置数据在时序划分下的参照（随机划分的值已有，直接读取）

随机划分的处置结果从 15_mitigation.py 的既有 run 目录读取，不重跑。

结果写入既有 run 目录下的 supplement_timesplit/（若已存在则加时间戳后缀，不覆盖）。
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


# 处置实现与处置方式清单直接沿用 15，保证两轮对"处置"的定义逐位一致
M15 = _load("m15", SCRIPTS / "15_mitigation.py")
apply_treatment = M15.apply_treatment
TREATMENTS = M15.TREATMENTS

# 既有（随机划分）处置结果的 run 目录
RUN_DIRS = {
    "pjm_2025_v2": ROOT / "04_outputs" / "pjm_2025_v2" / "06_mitigation" / "run_20260815_010219",
    "rts_gmlc_2020_v2": ROOT / "04_outputs" / "rts_gmlc_2020_v2" / "06_mitigation" / "run_20260815_012955",
}

_orig_split = gates.split


def temporal_split(n: int, cfg):
    """按时间顺序划分：前 80% 时段训练、后 20% 时段测试。"""
    k = int(n * cfg.train_ratio)
    return np.arange(k), np.arange(k, n)


def build(ds: str):
    """与 15_mitigation.run_dataset 相同的数据与候选池构建。"""
    if ds == "pjm":
        sl = _load("sl", SCRIPTS / "04_source_location.py")
        ver, year = "pjm_2025_v2", 2025
        df = dataio.load_clean(year, "main")
        summ = pd.read_csv(dataio.OUTPUTS / ver / "source_location_summary.csv")
        summ = summ[summ.初筛 == "否"]
        cn = sl.CN

        def pool_of(t):
            return sl.build_pool(df, t, True, False, year, True, ver)[0]
    else:
        rts = _load("rts", SCRIPTS / "10_rts_pipeline.py")
        ver = "rts_gmlc_2020_v2"
        rts.set_dataset(ver, ver)
        df, pub, _, _ = rts.step1_preprocess("base")
        summ = pd.read_csv(dataio.OUTPUTS / ver / "source_location_summary.csv")
        summ = summ[summ.初筛 == "否"]
        cn = rts.CN

        def pool_of(t):
            p, _, _, _ = rts.build_pool(df, t, pub)
            return rts.layered_strip(df, t, p)[1]
    return ver, df, summ, cn, pool_of


def run_dataset(ds: str, cfg, tm, smoke: bool = False) -> pd.DataFrame:
    ver, df, summ, cn, pool_of = build(ds)

    # 随机划分的处置结果（不重跑，仅用于对照与合并）
    prev = pd.read_csv(RUN_DIRS[ver] / "data" / "mitigation.csv")
    pidx = {(r["target"], r["处置方式"]): r for _, r in prev.iterrows()}

    if smoke:
        summ = summ.head(2)
        treat_list = TREATMENTS[-1:]
        out = Path("/tmp") / "mitigation_timesplit_smoke" / ver
        if out.exists():
            import shutil
            shutil.rmtree(out)
        out.mkdir(parents=True)
    else:
        treat_list = TREATMENTS
        out = RUN_DIRS[ver] / "supplement_timesplit"
        if out.exists():
            out = RUN_DIRS[ver] / f"supplement_timesplit_{datetime.now():%Y%m%d_%H%M%S}"
        out.mkdir(parents=True)
    FIG, DAT = dataio.split_dirs(out)

    rows = []
    print(f"\n{'=' * 72}\n数据集 {ver}（时序划分补充）", flush=True)

    gates.split = temporal_split
    try:
        for _, r in summ.iterrows():
            t = r["target"]
            pool = pool_of(t)
            sel = [c for c in str(r["选中字段"]).split("|") if c]
            idx_sel = [pool.index(c) for c in sel if c in pool]
            if len(idx_sel) < 2:
                print(f"  {cn[t]:18s} 选中字段不足，跳过", flush=True)
                continue
            X = df[pool].to_numpy(float)
            y = df[t].to_numpy(float)

            # 未处置数据的时序划分参照（随机划分的值从既有结果读取）
            pre_sel_t = gates.retrain_subset(X, y, idx_sel, cfg)
            pre_full_t = gates.retrain_subset(X, y, list(range(len(pool))), cfg)
            b0 = prev.drop_duplicates("target").set_index("target").loc[t]
            print(f"  ── {cn[t]:18s} 候选 {len(pool)} 选中 {len(idx_sel)}　"
                  f"未处置时序：选中 {pre_sel_t:.4f}　全量 {pre_full_t:.4f}", flush=True)

            for k in treat_list:
                rng = np.random.default_rng(2025)
                Xt = apply_treatment(X, idx_sel, k, rng)
                a = gates.retrain_subset(Xt, y, idx_sel, cfg)                 # 处置后·选中
                b = gates.retrain_subset(Xt, y, list(range(len(pool))), cfg)  # 处置后·全量
                p = pidx[(t, k)]
                rows.append({
                    "数据集": ver, "target": t, "中文名": cn[t],
                    "候选数": len(pool), "选中数": len(idx_sel), "处置方式": k,
                    "处置前_选中R2_随机": float(b0["选中R2"]),
                    "处置前_全量R2_随机": float(b0["全量R2"]),
                    "处置前_选中R2_时序": pre_sel_t,
                    "处置前_全量R2_时序": pre_full_t,
                    "处置后_选中R2_随机": float(p["处置后_选中R2"]),
                    "处置后_全量R2_随机": float(p["处置后_全量R2"]),
                    "处置后_选中R2_时序": a,
                    "处置后_全量R2_时序": b,
                    "选中降幅_随机": float(p["处置后_选中R2"]) - float(p["选中R2"]),
                    "选中降幅_时序": a - pre_sel_t,
                })
                print(f"       {k:14s} 处置后选中：随机 {p['处置后_选中R2']:.4f}"
                      f" → 时序 {a:.4f}（较未处置时序降 {pre_sel_t - a:+.4f}）", flush=True)
            tm.mark(f"{ver}[时序] {cn[t]}")
    finally:
        gates.split = _orig_split

    res = pd.DataFrame(rows)
    res.to_csv(DAT / "mitigation_timesplit.csv", index=False)
    (DAT / "config.json").write_text(json.dumps({
        "数据集": ver, "处置方式": list(treat_list), "轮次": cfg.epochs,
        "划分": "时间顺序 8:2（前 80% 时段训练、后 20% 时段测试）",
        "说明": "处置实现与种子沿用 15_mitigation.py；仅划分方式不同",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if not smoke:
        write_doc(out / "说明.md", res, ver)
        plot(FIG / "fig_mitigation_timesplit.png", res, ver)
    return res


def plot(path, res: pd.DataFrame, ver: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ch = [f for f in ["Arial Unicode MS", "STHeiti", "Songti SC"]
          if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ch + ["DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 8})

    g = res.groupby("处置方式")[
        ["处置后_选中R2_随机", "处置后_选中R2_时序",
         "处置后_全量R2_随机", "处置后_全量R2_时序"]].mean().reindex(TREATMENTS)
    b = res.drop_duplicates("target")
    x = np.arange(len(g))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    for ax, col, ref_r, ref_t, title in (
        (axes[0], "选中", "处置前_选中R2_随机", "处置前_选中R2_时序", "仅处置后的选中字段"),
        (axes[1], "全量", "处置前_全量R2_随机", "处置前_全量R2_时序", "其余字段照常发布"),
    ):
        ax.axhline(b[ref_r].mean(), color="#7f8c8d", ls=":", lw=1)
        ax.axhline(b[ref_t].mean(), color="#8e44ad", ls="--", lw=1)
        ax.bar(x - w / 2, g[f"处置后_{col}R2_随机"], w, label="处置后·随机划分",
               color="#c9d6e5", edgecolor="k", lw=.4)
        ax.bar(x + w / 2, g[f"处置后_{col}R2_时序"], w, label="处置后·时序划分",
               color="#b23b3b", edgecolor="k", lw=.4)
        ax.text(len(g) - .5, b[ref_r].mean(), f" 未处置·随机 {b[ref_r].mean():.3f}",
                fontsize=6, va="bottom", ha="right")
        ax.text(len(g) - .5, b[ref_t].mean(), f" 未处置·时序 {b[ref_t].mean():.3f}",
                fontsize=6, va="top", ha="right", color="#8e44ad")
        ax.set_xticks(x)
        ax.set_xticklabels(g.index, rotation=18, ha="right", fontsize=6.5)
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("平均测试 R²")
        ax.grid(axis="y", alpha=.25, ls=":")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout(pad=.3)
    fig.savefig(path, dpi=600)
    plt.close()


def write_doc(path, res: pd.DataFrame, ver: str):
    b = res.drop_duplicates("target")
    L = [f"# 处置实验的时序划分补充（{ver}）", "",
         "15_mitigation.py 的处置结果采用随机划分；本目录补充按时间顺序划分"
         "（前 80% 时段训练、后 20% 时段测试）的结果，对应持续发布场景：外部用历史数据"
         "训练模型、推断新发布数据。处置实现与随机种子沿用原脚本，仅划分方式不同；"
         "随机划分数值直接取自原 run，未重跑。", "",
         "## 平均结果", "",
         "| 处置方式 | 处置后·选中（随机） | 处置后·选中（时序） | 处置后·全量（随机） | 处置后·全量（时序） |",
         "|---|---|---|---|---|"]
    g = res.groupby("处置方式").mean(numeric_only=True).reindex(TREATMENTS)
    for k in TREATMENTS:
        L.append(f"| {k} | {g.loc[k, '处置后_选中R2_随机']:.4f} | "
                 f"{g.loc[k, '处置后_选中R2_时序']:.4f} | "
                 f"{g.loc[k, '处置后_全量R2_随机']:.4f} | "
                 f"{g.loc[k, '处置后_全量R2_时序']:.4f} |")
    L += ["", "未处置基准：", "",
          "| 口径 | 随机划分 | 时序划分 |", "|---|---|---|",
          f"| 仅选中字段 | {b['处置前_选中R2_随机'].mean():.4f} | {b['处置前_选中R2_时序'].mean():.4f} |",
          f"| 全部字段 | {b['处置前_全量R2_随机'].mean():.4f} | {b['处置前_全量R2_时序'].mean():.4f} |",
          "", "## 逐目标明细（综合处置）", "",
          "| 目标 | 选中数 | 未处置选中（随机/时序） | 处置后选中（随机） | 处置后选中（时序） | 时序口径下降幅 |",
          "|---|---|---|---|---|---|"]
    for _, r in res[res.处置方式 == "综合处置"].iterrows():
        L.append(f"| {r['中文名']} | {int(r['选中数'])} | "
                 f"{r['处置前_选中R2_随机']:.4f} / {r['处置前_选中R2_时序']:.4f} | "
                 f"{r['处置后_选中R2_随机']:.4f} | {r['处置后_选中R2_时序']:.4f} | "
                 f"{r['处置前_选中R2_时序'] - r['处置后_选中R2_时序']:+.4f} |")
    L += ["", "## 图", "",
          "- `figures/fig_mitigation_timesplit.png`　左：仅处置后的选中字段；"
          "右：其余字段照常发布。柱为各处置方式的平均残余可推断性"
          "（浅=随机划分，深=时序划分），虚线为未处置基准。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--datasets", nargs="+", default=["pjm", "rts"])
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟测试：2 个目标、仅综合处置、输出到 /tmp，不写入正式目录")
    args = ap.parse_args()
    with runlock.single_instance("mitigation_timesplit"):
        tm = runlock.Timer("处置实验·时序划分补充")
        cfg = gates.TrainConfig(epochs=args.epochs, lambda_dgate=0.005)
        alls = [run_dataset(ds, cfg, tm, smoke=args.smoke) for ds in args.datasets]
        agg = pd.concat(alls, ignore_index=True)
        print("\n" + "=" * 72)
        for ver in agg.数据集.unique():
            s = agg[agg.数据集 == ver]
            b = s.drop_duplicates("target")
            print(f"\n{ver}　未处置时序基准：选中 {b['处置前_选中R2_时序'].mean():.4f}"
                  f"　全量 {b['处置前_全量R2_时序'].mean():.4f}")
            for k in TREATMENTS:
                x = s[s.处置方式 == k]
                if not len(x):
                    continue
                print(f"  {k:14s} 处置后选中：随机 {x['处置后_选中R2_随机'].mean():.4f}"
                      f" → 时序 {x['处置后_选中R2_时序'].mean():.4f}"
                      f"（较未处置时序降 {x['选中降幅_时序'].mean():+.4f}）")
        print("\n" + tm.report())


if __name__ == "__main__":
    main()
