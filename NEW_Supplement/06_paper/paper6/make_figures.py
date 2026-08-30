"""paper6 论文用图。

与 paper5 的区别：

1. 图 3 / 图 5 的数值来自同目录 figure_data.json（由 make_figure_data.py
   从实验输出生成），本脚本不依赖 04_outputs，重画图无需重跑实验。
2. 多联图全部拆分为独立子图文件：fig3a/fig3b、fig5a/fig5b、fig4a/fig4b、
   fig2a/fig2b、fig6a/fig6b/fig6c。子图内不画 (a)(b) 标题，由 LaTeX 排版。

沿用 paper2 的绘图规范（8pt 字号、600 dpi、线型/标记/颜色三重区分、
坐标轴用 Unicode 上标 R² 而非 $R^2$）。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
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
W1, W2 = 8.4 * CM, 8.4 * CM     # 子图统一按通栏一半（8.4cm）出图
W3 = 5.6 * CM                    # 三联图的每联

PJM = O / "pjm_2025_v2"
RTS = O / "rts_gmlc_2020_v2"
TGT = "net_actual_interchange_mw"

DATA = json.loads((HERE / "figure_data.json").read_text(encoding="utf-8"))


def _one(p: Path, pat: str) -> Path:
    hits = sorted(p.rglob(pat))
    if not hits:
        raise FileNotFoundError(f"{p} 下找不到 {pat}")
    return hits[-1]


# ---------------- 图 3：全部/选中/未选中（每数据集一张子图） ----------------
def fig3(d: list[dict], name: str):
    d = sorted(d, key=lambda x: -x["sel"])
    x = np.arange(len(d))
    w = 0.27
    fig, ax = plt.subplots(figsize=(W1, 4.4 * CM))
    ax.bar(x - w, [r["full"] for r in d], w, label="全部候选字段",
           color="#9aa7b4", edgecolor="k", lw=0.4)
    ax.bar(x, [r["sel"] for r in d], w, label="门控选中的字段",
           color="#b23b3b", edgecolor="k", lw=0.4)
    ax.bar(x + w, [r["rest"] for r in d], w, label="其余未选中的字段",
           color="#dfe6ec", edgecolor="k", lw=0.4, hatch="///")
    for i, r in enumerate(d):
        ax.text(i, max(r["full"], r["sel"]) + .012,
                f"{r['nsel']}/{r['ncand']}", ha="center", fontsize=5.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in d], rotation=36, ha="right", fontsize=6.2)
    ax.set_ylim(0, 1.24)
    ax.grid(axis="y", alpha=.25, ls=":")
    ax.set_ylabel("测试 R²")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name); plt.close(fig)


# ---------------- 图 5：综合处置前后（每数据集一张子图） ----------------
def fig5(d: list[dict], name: str):
    d = sorted(d, key=lambda x: -x["before"])
    x = np.arange(len(d))
    w = 0.38
    fig, ax = plt.subplots(figsize=(W1, 4.4 * CM))
    ax.bar(x - w / 2, [r["before"] for r in d], w, label="处置前（仅选中字段）",
           color="#9aa7b4", edgecolor="k", lw=0.4)
    ax.bar(x + w / 2, [r["after"] for r in d], w, label="综合处置后",
           color="#b23b3b", edgecolor="k", lw=0.4)
    for i, r in enumerate(d):
        ax.text(i, max(r["before"], 0) + .014, f"{r['nsel']}",
                ha="center", fontsize=5.4, color="#444")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in d], rotation=36, ha="right", fontsize=6.2)
    ax.margins(y=0.10)
    ax.grid(axis="y", alpha=.25, ls=":")
    ax.set_ylabel("测试 R²")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name); plt.close(fig)


# ---------------- 图 2：门控演化 (a) 与阈值响应 (b) 拆分 ----------------
def fig2a(name="fig2a_gate_evolution.png"):
    gh = pd.read_csv(_one(PJM / "05_baseline_compare/rule1/excl_targets/no_screening",
                          f"target_{TGT}/data/DGatingDNN/gate_history.csv"))
    H = gh.to_numpy(float)
    fin = H[-1]
    keep = np.argsort(-fin)[:11]
    fig, ax = plt.subplots(figsize=(W1, 4.2 * CM))
    for j in range(H.shape[1]):
        act = j in keep
        ax.plot(np.arange(H.shape[0]), np.maximum(H[:, j], 1e-7),
                color="#b23b3b" if act else "#9aa0a6",
                lw=1.0 if act else 0.5, alpha=1.0 if act else 0.55,
                zorder=3 if act else 1)
    ax.axhline(0.01, color="k", ls="--", lw=0.8, zorder=4)
    ax.text(H.shape[0] * 0.99, 0.016, "活跃阈值 0.01", ha="right", va="bottom", fontsize=6.5)
    ax.set_yscale("log"); ax.set_ylim(1e-7, 2)
    ax.set_xlabel("训练轮次"); ax.set_ylabel("门控值")
    ax.grid(alpha=.25, ls=":")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name); plt.close(fig)


def fig2b(name="fig2b_threshold.png"):
    gates = pd.read_csv(_one(PJM / "04b_source_location_excl_targets/stripped_no_screening",
                             f"target_{TGT}/DGatingDNN/*/data/gates.csv"))
    g = np.sort(gates.gate.to_numpy(float))[::-1]
    nz = int((g > 0).sum())
    th = np.logspace(-13, -0.7, 400)
    n = np.array([(g >= t).sum() for t in th])
    fig, ax = plt.subplots(figsize=(W1, 3.4 * CM))
    ax.semilogx(th, n, color="#1f4e79", lw=1.4)
    ax.axvline(0.01, color="k", ls="--", lw=0.8)
    ax.annotate("本文取 0.01", xy=(0.01, nz), xytext=(3e-4, max(n) + 2.2),
                fontsize=6.5, ha="center", arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("活跃阈值"); ax.set_ylabel("选中字段数")
    ax.set_ylim(0, max(n) + 5); ax.grid(alpha=.25, ls=":")
    flat = th[n == nz]
    if len(flat):
        ax.axvspan(flat.min(), flat.max(), color="#f0c419", alpha=.30, lw=0)
        ax.annotate(f"{nz} 个非零字段的平台，阈值跨\n"
                    f"{np.log10(flat.max()/flat.min()):.0f} 个数量级结果不变\n"
                    f"（其余 {len(g) - nz} 个精确为 0）",
                    xy=(1e-8, nz), xytext=(1e-8, nz * 0.38),
                    ha="center", fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.7))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name); plt.close(fig)


# ---------------- 图 4：同预算方法对比（每数据集一张子图） ----------------
def fig4(csv: Path, name: str):
    d = pd.read_csv(csv)
    order = ["DGatingDNN", "STG", "LassoNet", "XGBoost", "Lasso", "Pearson"]
    lab = ["本文方法", "STG", "LassoNet", "XGBoost", "Lasso", "相关系数排序"]
    mk = ["o", "s", "^", "D", "v", "x"]
    ls = ["-", "--", "-.", ":", "--", "-"]
    col = ["#b23b3b", "#1f4e79", "#2e7d32", "#e08a1e", "#7b5aa6", "#7f8c8d"]
    piv = d.pivot_table(index="中文名", columns="方法", values="n")
    piv = piv.loc[piv["DGatingDNN"].sort_values(ascending=False).index]
    x = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(W1, 4.6 * CM))
    for m, mm, l, c in zip(order, mk, ls, col):
        ax.plot(x, piv[m], marker=mm, ls=l, color=c, ms=3.2, lw=1.0, label=l,
                zorder=4 if m == "DGatingDNN" else 2,
                markerfacecolor="none" if m != "DGatingDNN" else c)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=90, fontsize=5.6)
    ax.grid(alpha=.25, ls=":")
    ax.set_ylabel("测试 R²")
    ax.legend(loc="lower left", ncol=2, framealpha=.92, handlelength=2.2, fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name); plt.close(fig)


# ---------------- 图 6：鲁棒性（每目标一张子图） ----------------
def fig6(name_prefix="fig6"):
    d = pd.read_csv(_one(PJM / "07_robustness", "data/robustness.csv"))
    tg = list(d.中文名.unique())
    for k, t in enumerate(tg):
        s = d[d.中文名 == t]
        b = s[s.是否基准 == 1].iloc[0]
        r = s[s.是否基准 == 0]
        x = np.arange(len(r))
        fig, ax = plt.subplots(figsize=(W3, 4.4 * CM))
        ax.plot(x, r.全量R2, "o-", color="#1f4e79", ms=3.5, label="全部候选字段")
        ax.plot(x, r.子集R2, "s--", color="#b23b3b", ms=3.5, label="本文选中字段")
        ax.axhline(b.全量R2, color="#7f8c8d", ls=":", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}\n{c}" for a, c in zip(r.条件, r.水平)],
                           rotation=90, fontsize=5.8)
        ax.margins(y=0.12)
        ax.grid(alpha=.25, ls=":")
        ax.set_ylabel("测试 R²")
        if k == 0:
            ax.legend(loc="lower left", framealpha=.9, fontsize=6)
        fig.tight_layout(pad=0.3)
        fig.savefig(OUT / f"{name_prefix}{chr(97 + k)}_robustness.png"); plt.close(fig)


if __name__ == "__main__":
    print("生成 paper6 图（子图独立文件，标题由 LaTeX 排版）：")
    fig3(DATA["fig3"]["pjm"], "fig3a_selected_pjm.png")
    print("  fig3a_selected_pjm.png")
    fig3(DATA["fig3"]["rts"], "fig3b_selected_rts.png")
    print("  fig3b_selected_rts.png")
    fig5(DATA["fig5"]["pjm"], "fig5a_mitigation_pjm.png")
    print("  fig5a_mitigation_pjm.png")
    fig5(DATA["fig5"]["rts"], "fig5b_mitigation_rts.png")
    print("  fig5b_mitigation_rts.png")
    fig2a(); print("  fig2a_gate_evolution.png")
    fig2b(); print("  fig2b_threshold.png")
    fig4(_one(PJM / "05_baseline_compare/rule1/excl_targets/no_screening",
              "data/overall_comparison.csv"), "fig4a_compare_pjm.png")
    print("  fig4a_compare_pjm.png")
    fig4(_one(RTS / "05_baseline_compare/rule1/no_screening",
              "data/overall_comparison.csv"), "fig4b_compare_rts.png")
    print("  fig4b_compare_rts.png")
    fig6(); print("  fig6a/b/c_robustness.png")
    print(f"\n输出目录 {OUT}")
