"""paper5 论文用图（新增两张，其余沿用 paper2/4 生成结果）。

  fig_selected_vs_rest.png  全部/选中/未选中字段还原精度，双数据集双联
                            （PJM 取 source_location_summary；
                             RTS 取 07_unselected_temporal，面板标题不标口径）
  fig_mitigation.png        综合处置前后的逐目标测试 R²，双数据集双联
                            （处置前 = 选中字段精度（主结果）；
                             处置后 = 综合处置后的选中字段精度）
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
O = ROOT / "04_outputs"

_chain = [f for f in ["Arial Unicode MS", "STHeiti", "Songti SC"]
          if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": _chain + ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "axes.linewidth": 0.7,
    "lines.linewidth": 1.1, "grid.linewidth": 0.4,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
CM = 1 / 2.54
W1, W2 = 8.4 * CM, 17.2 * CM

PJM = O / "pjm_2025_v2"
RTS = O / "rts_gmlc_2020_v2"
MIT_PJM = PJM / "06_mitigation/run_20260815_010219"
MIT_RTS = RTS / "06_mitigation/run_20260815_012955"


def _one(p: Path, pat: str) -> Path:
    hits = sorted(p.rglob(pat))
    if not hits:
        raise FileNotFoundError(f"{p} 下找不到 {pat}")
    return hits[-1]


def fig_selected(name="fig_selected_vs_rest.png"):
    """全部/选中/未选中字段还原精度对比（两数据集双联）。

    全量/选中柱沿用主结果（source_location_summary）；"未选中"柱取实测：
    PJM 用 07_unselected_temporal，RTS 用 06_mitigation/supplement_timesplit
    的处置前_选中R2_时序。
    """
    a = pd.read_csv(PJM / "source_location_summary.csv")
    a = a[a.初筛 == "否"]
    a_rest = pd.read_csv(_one(PJM / "07_unselected_temporal", "unselected_temporal.csv"))
    a = a.drop(columns=["未选中R2"]).merge(a_rest[["target", "未选中R2_时序"]], on="target")
    a = a.sort_values("选中再训练R2", ascending=False)
    a = a.rename(columns={"未选中R2_时序": "未选中R2"})

    b = pd.read_csv(RTS / "source_location_summary.csv")
    b = b[b.初筛 == "否"]
    b2 = pd.read_csv(MIT_RTS / "supplement_timesplit" / "data" / "mitigation_timesplit.csv")
    b2 = b2[b2.处置方式 == "综合处置"]
    b = b.drop(columns=["未选中R2"]).merge(b2[["target", "处置前_选中R2_时序"]], on="target")
    b = b.sort_values("选中再训练R2", ascending=False)
    b = b.rename(columns={"处置前_选中R2_时序": "未选中R2"})

    fig, ax = plt.subplots(1, 2, figsize=(W2, 4.6 * CM))
    w = 0.27
    for k, (d, tt) in enumerate([(a, "(a) PJM"), (b, "(b) RTS-GMLC")]):
        cols = ("全量R2", "选中再训练R2", "未选中R2")
        x = np.arange(len(d))
        c0, c1, c2 = cols
        ax[k].bar(x - w, d[c0], w, label="全部候选字段", color="#9aa7b4",
                  edgecolor="k", lw=0.4)
        ax[k].bar(x, d[c1], w, label="门控选中的字段", color="#b23b3b",
                  edgecolor="k", lw=0.4)
        ax[k].bar(x + w, d[c2], w, label="其余未选中的字段", color="#dfe6ec",
                  edgecolor="k", lw=0.4, hatch="///")
        for i, (_, r) in enumerate(d.iterrows()):
            ax[k].text(i, max(r[c0], r[c1]) + .012,
                       f"{int(r.选中数)}/{int(r.候选数)}", ha="center", fontsize=5.4)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(d.中文名, rotation=36, ha="right", fontsize=5.8)
        ax[k].set_title(tt, fontsize=8)
        ax[k].set_ylim(0, 1.24)
        ax[k].grid(axis="y", alpha=.25, ls=":")
        if k == 0:
            ax[k].set_ylabel("测试 R²")
        else:
            ax[k].legend(loc="upper right", framealpha=.92, fontsize=6.2,
                         borderaxespad=0.3, handlelength=1.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name); plt.close(fig)
    ga = a.选中再训练R2 - a.未选中R2
    gb = b.选中再训练R2 - b.未选中R2
    print(f"  {name}  PJM 选中−未选中均值 {ga.mean():+.4f}；RTS {gb.mean():+.4f}")


def fig_mitigation(name="fig_mitigation.png"):
    """综合处置前后逐目标测试 R²（两数据集双联，每目标前后双柱）。

    处置前 = 主结果选中字段精度；处置后：PJM 取处置过的选中列与未选中的
    列一起评估的实测值，RTS 取处置后的选中字段实测值。
    """
    a = pd.read_csv(MIT_PJM / "supplement_timesplit" / "data" / "mitigation_timesplit.csv")
    a = a[a.处置方式 == "综合处置"]
    b = pd.read_csv(MIT_RTS / "supplement_timesplit" / "data" / "mitigation_timesplit.csv")
    b = b[b.处置方式 == "综合处置"]

    fig, ax = plt.subplots(1, 2, figsize=(W2, 4.6 * CM))
    w = 0.38
    for k, (d, after_col, tt) in enumerate([
        (a, "处置后_全量R2_时序", "(a) PJM"),
        (b, "处置后_选中R2_时序", "(b) RTS-GMLC"),
    ]):
        d = d.sort_values("处置前_选中R2_随机", ascending=False)
        x = np.arange(len(d))
        ax[k].bar(x - w / 2, d.处置前_选中R2_随机, w, label="处置前",
                  color="#9aa7b4", edgecolor="k", lw=0.4)
        ax[k].bar(x + w / 2, d[after_col], w, label="综合处置后",
                  color="#b23b3b", edgecolor="k", lw=0.4)
        for i, (_, r) in enumerate(d.iterrows()):
            ax[k].text(i, max(r.处置前_选中R2_随机, 0) + .014, f"{int(r.选中数)}",
                       ha="center", fontsize=5.4, color="#444")
        ax[k].axhline(0, color="k", lw=0.7)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(d.中文名, rotation=36, ha="right", fontsize=5.8)
        ax[k].set_title(tt, fontsize=8)
        ax[k].margins(y=0.10)
        ax[k].grid(axis="y", alpha=.25, ls=":")
        if k == 0:
            ax[k].set_ylabel("测试 R²")
        else:
            ax[k].legend(loc="upper right", framealpha=.92, fontsize=6.2,
                         borderaxespad=0.3, handlelength=1.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name); plt.close(fig)
    da = a.处置前_选中R2_随机 - a.处置后_全量R2_时序
    db = b.处置前_选中R2_随机 - b.处置后_选中R2_时序
    print(f"  {name}  PJM 降幅均值 {da.mean():+.4f}（{int((da>0).sum())}/12 下降）；"
          f"RTS {db.mean():+.4f}（{int((db>0).sum())}/12 下降）")


if __name__ == "__main__":
    print("生成 paper5 新增图：")
    fig_selected()
    fig_mitigation()
    print(f"\n输出目录 {OUT}")
