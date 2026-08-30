"""paper4 论文用图。

在 paper2 绘图规范基础上新增三张图：

  fig_selected_vs_rest.png   选中/未选中/全部字段对比，两数据集双联
                             （PJM 用随机划分；RTS 用时序划分，因随机划分下
                              未选中字段与选中字段精度持平，区分不开）
  fig_mitigation_random.png  处置示范·随机划分口径：处置前 vs 处置后（综合处置）
  fig_mitigation_temporal.png 处置示范·时序划分口径：处置前 vs 处置后（综合处置）

其余图（门控演化、方法对比、鲁棒性）沿用 paper2 生成结果，直接复制于 figures/。
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
    """定位结果验证双联图。

    (a) PJM（随机划分）：全部/选中/未选中三联柱——选中与未选中差距明显，
        直接证明门控选到了承载推断能力的字段。
    (b) RTS-GMLC（时序划分）：全部/选中/处置后三联柱——RTS 公开字段为
        多级汇总口径、冗余极高（任意大子集精度相当，未选中对比无区分度，
        见 07_unselected_temporal 归档），故该数据集上改用"少量字段达到
        全量精度 + 综合处置后显著下降"验证定位结果。
    """
    a = pd.read_csv(PJM / "source_location_summary.csv")
    a = a[a.初筛 == "否"].sort_values("选中再训练R2", ascending=False)
    b = pd.read_csv(MIT_RTS / "supplement_timesplit" / "data" / "mitigation_timesplit.csv")
    b = b[b.处置方式 == "综合处置"].sort_values("处置前_选中R2_时序", ascending=False)

    fig, ax = plt.subplots(1, 2, figsize=(W2, 4.2 * CM))
    w = 0.27
    for k, (d, cols, tt) in enumerate([
        (a, ("全量R2", "选中再训练R2", "未选中R2"), "(a) PJM（随机划分）"),
        (b, ("处置前_全量R2_时序", "处置前_选中R2_时序", "处置后_选中R2_时序"),
         "(b) RTS-GMLC（时序划分）"),
    ]):
        x = np.arange(len(d))
        c0, c1, c2 = cols
        labels = (["全部候选字段", "门控选中的字段", "其余未选中的字段"] if k == 0
                  else ["全部候选字段", "门控选中的字段", "选中字段综合处置后"])
        ax[k].bar(x - w, d[c0], w, label=labels[0], color="#9aa7b4",
                  edgecolor="k", lw=0.4)
        ax[k].bar(x, d[c1], w, label=labels[1], color="#b23b3b",
                  edgecolor="k", lw=0.4)
        ax[k].bar(x + w, d[c2], w, label=labels[2], color="#dfe6ec",
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
    db = b.处置前_选中R2_时序 - b.处置后_选中R2_时序
    print(f"  {name}  PJM 选中−未选中均值 {ga.mean():+.4f}；"
          f"RTS 处置前后降幅均值 {db.mean():+.4f}")


def _mitigation_panel(ax, d, before, after, tt):
    d = d.sort_values(before, ascending=False)
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d[before], w, label="处置前（仅选中字段）",
           color="#9aa7b4", edgecolor="k", lw=0.4)
    ax.bar(x + w / 2, d[after], w, label="处置后（综合处置）",
           color="#b23b3b", edgecolor="k", lw=0.4)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(i, max(r[before], 0) + .014, f"{int(r.选中数)}",
                ha="center", fontsize=5.4, color="#444")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(d.中文名, rotation=36, ha="right", fontsize=5.8)
    ax.set_title(tt, fontsize=8)
    ax.margins(y=0.10)
    ax.grid(axis="y", alpha=.25, ls=":")


def fig_mitigation_random(name="fig_mitigation_random.png"):
    """处置示范·随机划分：处置前选中字段精度 vs 综合处置后精度（两数据集）。"""
    a = pd.read_csv(MIT_PJM / "data" / "mitigation.csv")
    a = a[a.处置方式 == "综合处置"]
    b = pd.read_csv(MIT_RTS / "data" / "mitigation.csv")
    b = b[b.处置方式 == "综合处置"]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 4.2 * CM))
    _mitigation_panel(ax[0], a, "选中R2", "处置后_选中R2", "(a) PJM（随机划分）")
    _mitigation_panel(ax[1], b, "选中R2", "处置后_选中R2", "(b) RTS-GMLC（随机划分）")
    ax[0].set_ylabel("测试 R²")
    ax[0].legend(loc="upper right", framealpha=.92, fontsize=6.2,
                 borderaxespad=0.3, handlelength=1.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name); plt.close(fig)
    print(f"  {name}  PJM 降幅 {((a.选中R2 - a.处置后_选中R2)).mean():+.4f}；"
          f"RTS {((b.选中R2 - b.处置后_选中R2)).mean():+.4f}")


def fig_mitigation_temporal(name="fig_mitigation_temporal.png"):
    """处置示范·时序划分：处置前选中字段精度 vs 综合处置后精度（两数据集）。"""
    a = pd.read_csv(MIT_PJM / "supplement_timesplit" / "data" / "mitigation_timesplit.csv")
    a = a[a.处置方式 == "综合处置"]
    b = pd.read_csv(MIT_RTS / "supplement_timesplit" / "data" / "mitigation_timesplit.csv")
    b = b[b.处置方式 == "综合处置"]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 4.2 * CM))
    _mitigation_panel(ax[0], a, "处置前_选中R2_时序", "处置后_选中R2_时序",
                       "(a) PJM（时序划分）")
    _mitigation_panel(ax[1], b, "处置前_选中R2_时序", "处置后_选中R2_时序",
                       "(b) RTS-GMLC（时序划分）")
    ax[0].set_ylabel("测试 R²")
    ax[0].legend(loc="upper right", framealpha=.92, fontsize=6.2,
                 borderaxespad=0.3, handlelength=1.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name); plt.close(fig)
    da = a.处置前_选中R2_时序 - a.处置后_选中R2_时序
    db = b.处置前_选中R2_时序 - b.处置后_选中R2_时序
    print(f"  {name}  PJM 降幅 {da.mean():+.4f}；RTS {db.mean():+.4f}")


if __name__ == "__main__":
    print("生成 paper4 新增图：")
    fig_mitigation_random()
    fig_mitigation_temporal()
    fig_selected()
    print(f"\n输出目录 {OUT}")
