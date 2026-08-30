"""对定位出的隐性推断源实施处置，量化残余可推断性。

对应方法章“风险度量与处置指引”一节：定位只是中间环节，发布决策真正关心的是
“处置之后还能被推到什么程度”。本脚本给出一个简单示范。

每个敏感目标记录四个值：

    1  全量字段 R²          不做任何处置，用全部候选字段推断
    2  选中推断源 R²        不做处置，只用门控选出的字段推断
    3  处置后 选中字段 R²   对选出的字段实施处置后，再用这些字段推断
    4  处置后 全量字段 R²   只处置选出的字段、其余字段照常发布，攻击方用全部字段推断

第 3 个值回答“处置对推断源本身是否有效”；第 4 个值对应更接近实际的发布形态——
其余字段仍然公开，因此它同时反映了字段之间的冗余程度。

处置方式取三种常见手段，强度都设得比较粗，目的是说明机制而非调参：

    加噪      叠加标准差为该字段自身标准差一定比例的高斯噪声
    降精度    量化到若干个等宽区间，模拟发布时降低有效数字
    时间聚合  按若干小时取均值后回填，模拟降低发布的时间分辨率
    综合处置  上述三者叠加

选中字段直接读取已有的定位结果，不重新训练门控，以保证与论文正文报告的数值一致。
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


# ---------------- 处置手段 ----------------

def add_noise(X: np.ndarray, cols: list[int], ratio: float, rng) -> np.ndarray:
    """叠加高斯噪声，标准差为该字段自身标准差的 ratio 倍。"""
    Z = X.copy()
    for j in cols:
        s = Z[:, j].std()
        if s > 0:
            Z[:, j] = Z[:, j] + rng.normal(0.0, ratio * s, size=len(Z))
    return Z


def quantize(X: np.ndarray, cols: list[int], n_bin: int) -> np.ndarray:
    """量化到 n_bin 个等宽区间，取区间中点，模拟降低发布精度。"""
    Z = X.copy()
    for j in cols:
        lo, hi = Z[:, j].min(), Z[:, j].max()
        if hi <= lo:
            continue
        w = (hi - lo) / n_bin
        Z[:, j] = lo + (np.floor((Z[:, j] - lo) / w).clip(0, n_bin - 1) + 0.5) * w
    return Z


def aggregate(X: np.ndarray, cols: list[int], hours: int) -> np.ndarray:
    """按 hours 小时取均值后回填，模拟降低发布的时间分辨率。"""
    Z = X.copy()
    n = len(Z)
    for j in cols:
        v = Z[:, j]
        m = (n // hours) * hours
        blk = v[:m].reshape(-1, hours).mean(axis=1)
        Z[:m, j] = np.repeat(blk, hours)
        if m < n:
            Z[m:, j] = v[m:].mean()
    return Z


def apply_treatment(X, cols, kind, rng):
    if kind == "加噪 20%":
        return add_noise(X, cols, 0.20, rng)
    if kind == "降精度 10 档":
        return quantize(X, cols, 10)
    if kind == "时间聚合 6 小时":
        return aggregate(X, cols, 6)
    if kind == "综合处置":
        Z = add_noise(X, cols, 0.20, rng)
        Z = quantize(Z, cols, 10)
        return aggregate(Z, cols, 6)
    raise ValueError(kind)


TREATMENTS = ["加噪 20%", "降精度 10 档", "时间聚合 6 小时", "综合处置"]


# ---------------- 主流程 ----------------

def run_dataset(ds: str, cfg, tm) -> pd.DataFrame:
    """ds 取 pjm 或 rts。"""
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
        rts.set_dataset("rts_gmlc_2020_v2", ver)
        df, pub, _, _ = rts.step1_preprocess("base")
        summ = pd.read_csv(dataio.OUTPUTS / ver / "source_location_summary.csv")
        summ = summ[summ.初筛 == "否"]
        cn = rts.CN
        def pool_of(t):
            p, _, _, _ = rts.build_pool(df, t, pub)
            return rts.layered_strip(df, t, p)[1]

    out = (dataio.OUTPUTS / ver / "06_mitigation" /
           f"run_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    FIG, DAT = dataio.split_dirs(out)
    rows = []
    print(f"\n{'=' * 72}\n数据集 {ver}", flush=True)

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
        r2_full, r2_sel = float(r["全量R2"]), float(r["选中再训练R2"])
        print(f"  ── {cn[t]:18s} 候选 {len(pool)} 选中 {len(idx_sel)}　"
              f"全量 {r2_full:.4f}　选中 {r2_sel:.4f}", flush=True)

        for k in TREATMENTS:
            rng = np.random.default_rng(2025)
            Xt = apply_treatment(X, idx_sel, k, rng)
            a = gates.retrain_subset(Xt, y, idx_sel, cfg)              # 只用处置后的选中字段
            b = gates.retrain_subset(Xt, y, list(range(len(pool))), cfg)  # 其余字段照常发布
            rows.append({
                "数据集": ver, "target": t, "中文名": cn[t],
                "候选数": len(pool), "选中数": len(idx_sel),
                "全量R2": r2_full, "选中R2": r2_sel, "处置方式": k,
                "处置后_选中R2": a, "处置后_全量R2": b,
                "选中降幅": r2_sel - a, "全量降幅": r2_full - b,
            })
            print(f"       {k:14s} 选中 {a:.4f}（降 {r2_sel - a:+.4f}）　"
                  f"全量 {b:.4f}（降 {r2_full - b:+.4f}）", flush=True)
        tm.mark(f"{ver} {cn[t]}")

    res = pd.DataFrame(rows)
    res.to_csv(DAT / "mitigation.csv", index=False)
    (DAT / "config.json").write_text(json.dumps(
        {"数据集": ver, "处置方式": TREATMENTS, "轮次": cfg.epochs,
         "说明": "选中字段取自已有定位结果，未重训门控"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    write_doc(out / "说明.md", res, ver)
    plot(FIG / "fig_mitigation.png", res, ver)
    return res


def plot(path, res, ver):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ch = [f for f in ["Arial Unicode MS", "STHeiti", "Songti SC"]
          if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ch + ["DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 8})
    g = res.groupby("处置方式")[["处置后_选中R2", "处置后_全量R2"]].mean()
    g = g.reindex(TREATMENTS)
    base_full = res.drop_duplicates("target").全量R2.mean()
    base_sel = res.drop_duplicates("target").选中R2.mean()
    x = np.arange(len(g)); w = 0.36
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.axhline(base_full, color="#7f8c8d", ls=":", lw=1)
    ax.axhline(base_sel, color="#b23b3b", ls="--", lw=1)
    ax.bar(x - w / 2, g.处置后_选中R2, w, label="处置后·仅选中字段",
           color="#b23b3b", edgecolor="k", lw=.4)
    ax.bar(x + w / 2, g.处置后_全量R2, w, label="处置后·全部字段",
           color="#c9d6e5", edgecolor="k", lw=.4, hatch="///")
    ax.text(len(g) - .5, base_full, f" 未处置全量 {base_full:.3f}",
            fontsize=6, va="bottom", ha="right")
    ax.text(len(g) - .5, base_sel, f" 未处置选中 {base_sel:.3f}",
            fontsize=6, va="top", ha="right", color="#b23b3b")
    ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=18, ha="right", fontsize=6.5)
    ax.set_ylabel("平均测试 R²"); ax.legend(fontsize=6, loc="lower left")
    ax.grid(axis="y", alpha=.25, ls=":")
    fig.tight_layout(pad=.3); fig.savefig(path, dpi=600); plt.close(fig)


def write_doc(path, res, ver):
    L = [f"# 对隐性推断源的处置与残余风险（{ver}）", "",
         "定位只是中间环节，发布决策关心的是处置之后还能被推到什么程度。"
         "本节对定位出的推断源实施处置，记录四个值：", "",
         "| 记号 | 含义 |", "|---|---|",
         "| 全量 R² | 不处置，用全部候选字段推断 |",
         "| 选中 R² | 不处置，只用门控选出的字段推断 |",
         "| 处置后·选中 | 处置这些字段后，再用它们推断 |",
         "| 处置后·全量 | 只处置选出的字段、其余照常发布，用全部字段推断 |", "",
         "处置强度设得较粗，目的是说明机制而非调参。", "",
         "## 各处置方式的平均结果", "",
         "| 处置方式 | 处置后·选中 | 相对未处置选中降幅 | 处置后·全量 | 相对未处置全量降幅 |",
         "|---|---|---|---|---|"]
    for k in TREATMENTS:
        s = res[res.处置方式 == k]
        if not len(s):
            continue
        L.append(f"| {k} | {s.处置后_选中R2.mean():.4f} | {s.选中降幅.mean():+.4f} | "
                 f"{s.处置后_全量R2.mean():.4f} | {s.全量降幅.mean():+.4f} |")
    b = res.drop_duplicates("target")
    L += ["", f"未处置基准：全量 {b.全量R2.mean():.4f}，选中 {b.选中R2.mean():.4f}"
          f"（{len(b)} 个目标）", "",
          "## 逐目标明细（综合处置）", "",
          "| 目标 | 候选 | 选中 | 全量 R² | 选中 R² | 处置后·选中 | 处置后·全量 |",
          "|---|---|---|---|---|---|---|"]
    for _, r in res[res.处置方式 == "综合处置"].iterrows():
        L.append(f"| {r['中文名']} | {int(r.候选数)} | {int(r.选中数)} | {r.全量R2:.4f} | "
                 f"{r.选中R2:.4f} | {r.处置后_选中R2:.4f} | {r.处置后_全量R2:.4f} |")
    L += ["", "## 图", "", "- `fig_mitigation.png`　各处置方式下的平均残余可推断性，"
          "两条参考线分别为未处置时的全量与选中精度", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--datasets", nargs="+", default=["pjm", "rts"])
    args = ap.parse_args()
    with runlock.single_instance("mitigation"):
        tm = runlock.Timer("推断源处置与残余风险")
        cfg = gates.TrainConfig(epochs=args.epochs, lambda_dgate=0.005)
        alls = [run_dataset(ds, cfg, tm) for ds in args.datasets]
        agg = pd.concat(alls, ignore_index=True)
        print("\n" + "=" * 72)
        for ver in agg.数据集.unique():
            s = agg[agg.数据集 == ver]
            b = s.drop_duplicates("target")
            print(f"\n{ver}　未处置：全量 {b.全量R2.mean():.4f}　选中 {b.选中R2.mean():.4f}")
            for k in TREATMENTS:
                x = s[s.处置方式 == k]
                print(f"  {k:14s} 处置后选中 {x.处置后_选中R2.mean():.4f}"
                      f"（降 {x.选中降幅.mean():+.4f}）　"
                      f"处置后全量 {x.处置后_全量R2.mean():.4f}"
                      f"（降 {x.全量降幅.mean():+.4f}）")
        print("\n" + tm.report())


if __name__ == "__main__":
    main()
