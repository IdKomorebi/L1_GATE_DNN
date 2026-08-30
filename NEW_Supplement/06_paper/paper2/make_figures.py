"""生成论文用图。

期刊排版的要求与实验用图不同，所以不复用 02_src/plots.py：

  - 图内不写标题（题目由 LaTeX 的 caption 给）
  - 单栏宽约 8.5 cm，双栏通栏约 17.5 cm，按最终尺寸出图，不靠缩放
  - 字号统一 8 pt 左右，保证缩到栏宽后仍清晰
  - 黑白打印也要能分辨，所以线型、标记形状与颜色三者同时区分
"""

from __future__ import annotations

import sys
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

# 图里同时要出现中文、减号和上标。实测各字体的字形覆盖：
#   Songti SC / Hiragino Sans GB  缺 U+2212，对数刻度指数里的负号会变成方块
#   DejaVu Sans                   有减号但没有中文
#   Arial Unicode MS              中文、U+2212、上标全都有 —— 用它
# 另外中文与 $...$ 混在同一个字符串里会让整串走数学字体，所以标签里一律
# 用 Unicode 上标（R²）而不是 $R^2$。
_chain = [f for f in ["Arial Unicode MS", "STHeiti", "Songti SC"]
          if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": _chain + ["DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": _chain[0] if _chain else "DejaVu Sans",
    "mathtext.it": _chain[0] if _chain else "DejaVu Sans",
    "mathtext.bf": _chain[0] if _chain else "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.4,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
CM = 1 / 2.54
W1, W2 = 8.4 * CM, 17.2 * CM          # 单栏 / 通栏宽度

PJM = O / "pjm_2025_v2"
RTS2 = O / "rts_gmlc_2020_v2"
TGT = "net_actual_interchange_mw"


def _one(p: Path, pat: str) -> Path:
    hits = sorted(p.rglob(pat))
    if not hits:
        raise FileNotFoundError(f"{p} 下找不到 {pat}")
    return hits[-1]


# --------------------------------------------------------------------------
def fig_gate(name="fig_gate_evolution.png"):
    """门控值演化 + 活跃字段数随阈值的变化（通栏两联图）。

    左图说明断崖是怎么形成的，右图说明形成之后阈值取在哪里都一样——
    这两件事合起来才回应了"阈值主观性"的质疑。
    """
    gh = pd.read_csv(_one(PJM / "05_baseline_compare/rule1/excl_targets/no_screening",
                          f"target_{TGT}/data/DGatingDNN/gate_history.csv"))
    gates = pd.read_csv(_one(PJM / "04b_source_location_excl_targets/stripped_no_screening",
                             f"target_{TGT}/DGatingDNN/*/data/gates.csv"))
    fig, ax = plt.subplots(2, 1, figsize=(W1, 8.4 * CM))

    H = gh.to_numpy(float)
    fin = H[-1]
    keep = np.argsort(-fin)[:11]
    for j in range(H.shape[1]):
        act = j in keep
        ax[0].plot(np.arange(H.shape[0]), np.maximum(H[:, j], 1e-7),
                   color="#b23b3b" if act else "#9aa0a6",
                   lw=1.0 if act else 0.5, alpha=1.0 if act else 0.55, zorder=3 if act else 1)
    ax[0].axhline(0.01, color="k", ls="--", lw=0.8, zorder=4)
    ax[0].text(H.shape[0] * 0.99, 0.016, "活跃阈值 0.01", ha="right", va="bottom", fontsize=6.5)
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-7, 2)
    ax[0].set_xlabel("训练轮次"); ax[0].set_ylabel("门控值")
    ax[0].grid(alpha=.25, ls=":")
    ax[0].text(0.02, 0.04, "(a)", transform=ax[0].transAxes, fontsize=8, weight="bold")

    g = np.sort(gates.gate.to_numpy(float))[::-1]
    nz = int((g > 0).sum())
    th = np.logspace(-13, -0.7, 400)
    n = np.array([(g >= t).sum() for t in th])
    ax[1].semilogx(th, n, color="#1f4e79", lw=1.4)
    ax[1].axvline(0.01, color="k", ls="--", lw=0.8)
    ax[1].annotate("本文取 0.01", xy=(0.01, nz), xytext=(3e-4, max(n) + 2.2),
                   fontsize=6.5, ha="center",
                   arrowprops=dict(arrowstyle="->", lw=0.7))
    ax[1].set_xlabel("活跃阈值"); ax[1].set_ylabel("选中字段数")
    ax[1].set_ylim(0, max(n) + 5); ax[1].grid(alpha=.25, ls=":")
    # 真正的平台在"门控值非零"这一刀上：多数字段被精确压到 0，
    # 阈值在机器零到最小非零值之间任取，得到的都是同一批字段。
    flat = th[n == nz]
    if len(flat):
        ax[1].axvspan(flat.min(), flat.max(), color="#f0c419", alpha=.30, lw=0)
        ax[1].annotate(f"{nz} 个非零字段的平台，阈值跨\n{np.log10(flat.max()/flat.min()):.0f} 个数量级结果不变\n"
                       f"（其余 {len(g) - nz} 个精确为 0）",
                       xy=(1e-8, nz), xytext=(1e-8, nz * 0.38),
                       ha="center", fontsize=6,
                       arrowprops=dict(arrowstyle="->", lw=0.7))
    ax[1].text(0.02, 0.04, "(b)", transform=ax[1].transAxes, fontsize=8, weight="bold")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / name); plt.close(fig)
    print(f"  {name}  非零 {nz}/{len(g)}，平台跨 {np.log10(flat.max()/flat.min()):.1f} 个数量级")


def fig_compare(name="fig_method_compare.png"):
    """逐目标的方法对比（通栏两联图）。

    只给 12 个目标的平均值看不出差异发生在哪里——有的目标各方法都差不多，
    有的目标差距很大。逐目标画出来，读者能看到差距集中在难推断的目标上。
    """
    a = pd.read_csv(_one(PJM / "05_baseline_compare/rule1/excl_targets/no_screening",
                         "data/overall_comparison.csv"))
    b = pd.read_csv(_one(RTS2 / "05_baseline_compare/rule1/no_screening",
                         "data/overall_comparison.csv"))
    order = ["DGatingDNN", "STG", "LassoNet", "XGBoost", "Lasso", "Pearson"]
    lab = ["本文方法", "STG", "LassoNet", "XGBoost", "Lasso", "相关系数排序"]
    mk = ["o", "s", "^", "D", "v", "x"]
    ls = ["-", "--", "-.", ":", "--", "-"]
    col = ["#b23b3b", "#1f4e79", "#2e7d32", "#e08a1e", "#7b5aa6", "#7f8c8d"]
    fig, ax = plt.subplots(1, 2, figsize=(W2, 5.2 * CM))
    for k, (d, tt) in enumerate([(a, "(a) PJM"), (b, "(b) RTS-GMLC")]):
        piv = d.pivot_table(index="中文名", columns="方法", values="n")
        piv = piv.loc[piv["DGatingDNN"].sort_values(ascending=False).index]
        x = np.arange(len(piv))
        for m, mm, l, c in zip(order, mk, ls, col):
            ax[k].plot(x, piv[m], marker=mm, ls=l, color=c, ms=3.2,
                       lw=1.0, label=lab[order.index(m)],
                       zorder=4 if m == "DGatingDNN" else 2,
                       markerfacecolor="none" if m != "DGatingDNN" else c)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(piv.index, rotation=90, fontsize=5.6)
        ax[k].set_title(tt, fontsize=8)
        ax[k].grid(alpha=.25, ls=":")
        ax[k].set_ylabel("测试 R²" if k == 0 else "")
    ax[0].legend(loc="lower left", ncol=2, framealpha=.92, handlelength=2.2)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / name); plt.close(fig)
    print(f"  {name}  PJM 本文均值 {a[a.方法=='DGatingDNN'].n.mean():.4f}，"
          f"RTS {b[b.方法=='DGatingDNN'].n.mean():.4f}")


def fig_selected(name="fig_selected_vs_rest.png"):
    """选中字段、未选中字段与全部字段的还原精度对比（PJM 12 个目标）。

    这张图是"门控确实选到了承载推断能力的字段"的直接证据：
    若未选中字段单独也能达到同样精度，选择就没有意义。
    """
    d = pd.read_csv(PJM / "source_location_summary.csv")
    d = d[d.初筛 == "否"].sort_values("选中再训练R2", ascending=False)
    x = np.arange(len(d)); w = 0.27
    fig, ax = plt.subplots(figsize=(W2, 4.6 * CM))
    ax.bar(x - w, d.全量R2, w, label="全部候选字段", color="#9aa7b4",
           edgecolor="k", lw=0.4)
    ax.bar(x, d.选中再训练R2, w, label="门控选中的字段", color="#b23b3b",
           edgecolor="k", lw=0.4)
    ax.bar(x + w, d.未选中R2, w, label="其余未选中的字段", color="#dfe6ec",
           edgecolor="k", lw=0.4, hatch="///")
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(i, max(r.全量R2, r.选中再训练R2) + .012,
                f"{int(r.选中数)}/{int(r.候选数)}", ha="center", fontsize=5.6)
    ax.set_xticks(x); ax.set_xticklabels(d.中文名, rotation=30, ha="right", fontsize=6.5)
    ax.set_ylabel("测试 R²"); ax.set_ylim(0, 1.30)
    ax.legend(loc="upper center", ncol=3, framealpha=.92, fontsize=6.5,
              bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
    ax.grid(axis="y", alpha=.25, ls=":")
    fig.tight_layout(pad=0.3); fig.savefig(OUT / name); plt.close(fig)
    gap = (d.选中再训练R2 - d.未选中R2)
    print(f"  {name}  选中−未选中：均值 {gap.mean():+.4f}，最大 {gap.max():+.4f}")


def fig_risk(name="fig_risk_map.png"):
    """风险分级图：横轴还原所需字段占比（集中度），纵轴还原精度（强度）。

    逐点标名字会挤成一团（24 个目标里有一半落在左上角很窄的区域），
    所以只用象限说明，具体数值放正文表格。
    """
    a = pd.read_csv(PJM / "source_location_summary.csv")
    a = a[a.初筛 == "否"]
    b = pd.read_csv(RTS2 / "source_location_summary.csv")
    b = b[b.初筛 == "否"]
    fig, ax = plt.subplots(figsize=(W1, 4.9 * CM))
    ax.axhspan(0.9, 1.05, xmin=0, xmax=0.25 / 0.5, color="#f5d0d0", alpha=.45, lw=0)
    for d, mk, c, lb in [(a, "o", "#1f4e79", "PJM"), (b, "^", "#b23b3b", "RTS-GMLC")]:
        ax.scatter(d.选中数 / d.候选数, d.选中再训练R2, marker=mk, s=24,
                   facecolor=c, edgecolor="k", lw=0.4, label=lb, zorder=3)
    ax.axhline(0.9, color="k", ls=":", lw=0.7)
    ax.axvline(0.25, color="k", ls=":", lw=0.7)
    ax.text(0.008, 1.028, "高风险区：少数字段即可高精度还原",
            fontsize=6.5, va="top", color="#8b2020")
    ax.text(0.487, 0.905, "风险分散", fontsize=6.5, ha="right", va="bottom", color="#555")
    ax.text(0.487, 0.885, "风险较低", fontsize=6.5, ha="right", va="top", color="#555")
    ax.set_xlabel("还原所需字段数占候选比例")
    ax.set_ylabel("选中字段重训后的测试 R²")
    ax.set_xlim(0, 0.5); ax.set_ylim(0.60, 1.05)
    ax.legend(loc="lower left", framealpha=.9, handletextpad=0.4)
    ax.grid(alpha=.22, ls=":")
    fig.tight_layout(pad=0.3); fig.savefig(OUT / name); plt.close(fig)
    n_hi = int(((pd.concat([a, b]).选中再训练R2 > 0.9) &
                (pd.concat([a, b]).选中数 / pd.concat([a, b]).候选数 < 0.25)).sum())
    print(f"  {name}  共 {len(a) + len(b)} 个目标，落在高风险区 {n_hi} 个")


def fig_robust(name="fig_robustness.png"):
    """鲁棒性：各降级条件下，全量模型与选中字段子集的精度对比。"""
    d = pd.read_csv(_one(PJM / "07_robustness", "data/robustness.csv"))
    tg = list(d.中文名.unique())
    conds = [c for c in d.条件.unique() if "基准" not in c]
    fig, ax = plt.subplots(1, len(tg), figsize=(W2, 4.7 * CM))
    for k, t in enumerate(tg):
        s = d[(d.中文名 == t)]
        b = s[s.是否基准 == 1].iloc[0]
        r = s[s.是否基准 == 0]
        x = np.arange(len(r))
        ax[k].plot(x, r.全量R2, "o-", color="#1f4e79", ms=3.5, label="全部候选字段")
        ax[k].plot(x, r.子集R2, "s--", color="#b23b3b", ms=3.5, label="本文选中字段")
        ax[k].axhline(b.全量R2, color="#7f8c8d", ls=":", lw=0.9)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels([f"{a}\n{c}" for a, c in zip(r.条件, r.水平)],
                              rotation=90, fontsize=5.8)
        ax[k].set_title(f"({chr(97 + k)}) {t}", fontsize=7.5)
        ax[k].margins(y=0.12)          # 不留边距的话最低点会被坐标轴切掉
        ax[k].grid(alpha=.25, ls=":")
        if k == 0:
            ax[k].set_ylabel("测试 R²"); ax[k].legend(loc="lower left", framealpha=.9)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / name); plt.close(fig)
    gap = (d[d.是否基准 == 0].子集R2 - d[d.是否基准 == 0].全量R2)
    print(f"  {name}  子集−全量：{gap.min():+.4f} ~ {gap.max():+.4f}，均值 {gap.mean():+.4f}")


if __name__ == "__main__":
    print("生成论文图：")
    fig_gate(); fig_compare(); fig_selected(); fig_risk(); fig_robust()
    print(f"\n输出目录 {OUT}")
