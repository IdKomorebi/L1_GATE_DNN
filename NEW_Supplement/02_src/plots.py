"""论文用图。中文字体在 macOS 上优先用系统宋体/黑体，缺失时回退。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, ticker  # noqa: E402

_available = {f.name for f in font_manager.fontManager.ttflist}
_chain = [f for f in ["Songti SC", "Arial Unicode MS", "Hiragino Sans GB", "STHeiti",
                      "PingFang SC", "Noto Sans CJK SC", "SimSun"] if f in _available]
# 末尾接一个西文字体兜底：中文字体常常缺少数学减号等字形
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = _chain + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# 数学排版单独用西文字体，中文字体缺少减号、除号等字形
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["font.size"] = 10


def _log_ticks(ax) -> None:
    """对数刻度用纯文本标注。中文字体普遍缺少数学减号字形，
    走数学排版会渲染成方框或问号，所以这里绕开它。"""
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}" if v > 0 else "")
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def threshold_scan_plot(scan: pd.DataFrame, path: Path, title: str = "") -> None:
    """门槛扫描图：横轴是判定门槛，纵轴是找到的关系条数。
    出现"怎么调门槛条数都不变"的平台，说明门槛不敏感。"""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x, y = scan["tol"].to_numpy(), scan["n_relations"].to_numpy()
    ax.step(x, y, where="post", color="#2b6cb0", lw=1.8)
    ax.scatter(x, y, s=22, color="#2b6cb0", zorder=3)

    # 标出平台段
    start = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start]:
            if i - start >= 2:
                ax.axvspan(x[start], x[i - 1], color="#2b6cb0", alpha=0.10, lw=0)
                ax.annotate(f"{y[start]} 条", xy=(np.sqrt(x[start] * x[i - 1]), y[start]),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=9, color="#2b6cb0")
            start = i
    ax.set_xscale("log")
    _log_ticks(ax)
    ax.invert_xaxis()
    ax.set_xlabel("判定门槛（归一化奇异值比）")
    ax.set_ylabel("检出的精确关系条数")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, ls=":")
    ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def residual_band_plot(res: pd.DataFrame, path: Path, title: str = "") -> None:
    """残差比分布图：每个字段用其余全部字段做最小二乘拟合后的相对残差。

    公式型关系落在 1e-15~1e-4 这一带（误差只来自数值精度和发布舍入），
    统计关系落在 1e-2 以上，中间是一片空白。判定门槛放在空白里，
    所以它不是拍脑袋挑的数字。
    """
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    v = res["residual_ratio"].to_numpy(float)
    pos = v[v > 0]
    v = np.where(v <= 0, pos.min() / 10 if len(pos) else 1e-17, v)

    # 找出分布里最大的两个空隙，作为分段依据（由数据决定，不是事先设定）
    sv = np.sort(pos)
    lg = np.log10(sv)
    gaps = np.diff(lg)
    top = sorted(np.argsort(gaps)[-2:])
    bounds = [(sv[i], sv[i + 1], gaps[i]) for i in top]

    kinds = res["group"].to_numpy()
    colors = {"公式型": "#c53030", "统计型": "#2b6cb0"}
    rng = np.random.default_rng(0)
    for g in ["公式型", "统计型"]:
        m = kinds == g
        if m.sum():
            ax.scatter(v[m], rng.uniform(-0.28, 0.28, m.sum()),
                       s=26, alpha=0.8, label=f"{g}（{m.sum()} 个字段）",
                       color=colors[g], edgecolors="none")
    for lo, hi, g in bounds:
        ax.axvspan(lo, hi, color="0.82", alpha=0.7, lw=0)
        ax.annotate(f"空隙\n{g:.1f} 个数量级", xy=(np.sqrt(lo * hi), 0.45),
                    ha="center", va="center", fontsize=8.5, color="0.3")
    ax.set_xscale("log")
    _log_ticks(ax)
    ax.set_yticks([])
    ax.set_ylim(-0.75, 0.62)
    ax.set_xlabel("相对残差 = 用其余全部字段拟合该字段的剩余误差 / 该字段标准差")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
    ax.grid(axis="x", alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def null_vs_observed_plot(
    draws: pd.DataFrame, obs: pd.DataFrame, thresholds: dict,
    metrics: list[str], names: dict, path: Path, title: str = ""
) -> None:
    """六个指标各画一格：灰色是"无真实关系"时的取值分布（分块置换得到），
    蓝色是各候选字段的实测值，竖线是由零分布定出的阈值。"""
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4))
    for ax, m in zip(axes.ravel(), metrics):
        lo = min(draws[m].min(), obs[m].min())
        hi = max(draws[m].max(), obs[m].max())
        bins = np.linspace(lo, hi + 1e-9, 26)
        ax.hist(draws[m], bins=bins, color="0.6", alpha=0.85, label="无关系时")
        ax.hist(obs[m], bins=bins, color="#2b6cb0", alpha=0.7, label="实测")
        ax.axvline(thresholds[m], color="#c53030", lw=1.6, ls="--")
        ax.annotate(f"阈值 {thresholds[m]:.3f}", xy=(thresholds[m], ax.get_ylim()[1] * 0.92),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=8, color="#c53030", va="top")
        ax.set_title(names.get(m, m), fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_yticks([])
    axes.ravel()[0].legend(frameon=False, fontsize=8, loc="upper right")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if title else 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def screening_overview_plot(summary: pd.DataFrame, path: Path, title: str = "") -> None:
    """左：每个目标筛前筛后的字段数；右：候选字段通过了几个指标的分布。"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(summary))
    a1.barh(y, summary["n_pool"], color="0.85", label="初筛前")
    a1.barh(y, summary["n_kept"], color="#2b6cb0", label="初筛后保留")
    a1.set_yticks(y)
    a1.set_yticklabels(summary["中文名"], fontsize=9)
    a1.invert_yaxis()
    a1.set_xlabel("候选字段数")
    a1.legend(frameon=False, fontsize=9, loc="lower right")
    for i, r in enumerate(summary.itertuples()):
        a1.annotate(f"筛除 {r.n_pool - r.n_kept}", xy=(r.n_pool, i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color="0.35")
    a1.set_title("各目标的初筛结果", fontsize=10)

    counts = summary["pass_hist"].apply(pd.Series).fillna(0).sum(axis=0)
    a2.bar(counts.index.astype(int), counts.values, color="#2b6cb0", alpha=0.85)
    a2.set_xlabel("该字段通过了几个指标")
    a2.set_ylabel("字段计数（12 个目标合计）")
    a2.set_title("通过指标数的分布", fontsize=10)
    a2.grid(axis="y", alpha=0.25, ls=":")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def metric_agreement_plot(obs_all: pd.DataFrame, metrics: list[str],
                          names: dict, path: Path, title: str = "") -> None:
    """六个指标之间的一致性：两两之间对"该保留哪些字段"的判断有多吻合。"""
    n = len(metrics)
    M = np.zeros((n, n))
    for i, a in enumerate(metrics):
        for j, b in enumerate(metrics):
            pa, pb = obs_all[f"{a}_pass"] > 0, obs_all[f"{b}_pass"] > 0
            u = (pa | pb).sum()
            M[i, j] = (pa & pb).sum() / u if u else 1.0
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    lab = [names.get(m, m) for m in metrics]
    ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=8.5)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.55 else "0.2")
    fig.colorbar(im, ax=ax, shrink=0.82, label="判断吻合度")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def training_curve_plot(results: dict, path: Path, title: str = "") -> None:
    """各模型的训练/测试 R² 随轮次变化，以及活跃字段数的收缩过程。"""
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(3.4 * (n + 1), 3.2))
    colors = {"DNN": "#718096", "L1GateDNN": "#2b6cb0", "DGatingDNN": "#c53030"}
    for ax, (name, r) in zip(axes[:n], results.items()):
        h = r.history
        ax.plot(h["epoch"], h["train_r2"], color=colors.get(name, "0.4"),
                ls="--", lw=1.2, label="训练集")
        ax.plot(h["epoch"], h["test_r2"], color=colors.get(name, "0.4"),
                lw=1.6, label="测试集")
        ax.scatter([r.best_epoch], [r.best_test_r2], s=30, zorder=3,
                   color=colors.get(name, "0.4"))
        ax.annotate(f"最优 {r.best_test_r2:.4f}\n第 {r.best_epoch} 轮",
                    xy=(r.best_epoch, r.best_test_r2), xytext=(-6, -30),
                    textcoords="offset points", fontsize=8, ha="right")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("训练轮次")
        lo = min(h["test_r2"][5:]) if len(h["test_r2"]) > 6 else 0
        ax.set_ylim(max(-0.15, lo - 0.05), 1.02)
        ax.grid(alpha=0.25, ls=":")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel("R²")

    ax = axes[n]
    for name, r in results.items():
        if r.gate_history is not None:
            ax.plot(r.history["epoch"], r.history["n_active"],
                    color=colors.get(name, "0.4"), lw=1.6, label=name)
    ax.set_xlabel("训练轮次"); ax.set_ylabel("活跃字段数")
    ax.set_title("活跃字段数的收缩", fontsize=10)
    ax.grid(alpha=0.25, ls=":"); ax.legend(frameon=False, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if title else 1))
    fig.savefig(path, dpi=300); plt.close(fig)


def gate_evolution_plot(r, threshold: float, names: list[str],
                        path: Path, title: str = "", top: int = 14) -> None:
    """门控值随训练轮次的变化。所有字段都从 1 出发，越往下掉说明被压得越狠。

    最终仍在阈值之上的字段画成彩色并标出名字，被淘汰的画成灰色。
    分化过程一目了然：一批字段迅速坠到零，另一批稳住在高位。
    """
    if r.gate_history is None:
        return
    G = r.gate_history
    final = G[-1]
    keep = [i for i in np.argsort(-final)[:top] if final[i] >= threshold]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1, 1]})
    ep = range(0, len(G))       # 第 0 个点是训练前的起点
    cmap = plt.get_cmap("tab20")
    for a, logy in [(ax, False), (ax2, True)]:
        for i in range(G.shape[1]):
            if i not in keep:
                a.plot(ep, np.clip(G[:, i], 1e-9, None) if logy else G[:, i],
                       color="0.82", lw=0.7, zorder=1)
        for k, i in enumerate(keep):
            v = np.clip(G[:, i], 1e-9, None) if logy else G[:, i]
            a.plot(ep, v, lw=1.7, color=cmap(k % 20), zorder=3,
                   label=f"{names[i]} ({final[i]:.3f})")
        a.axhline(threshold, color="#c53030", ls="--", lw=1.3)
        a.set_xlabel("训练轮次")
        a.grid(alpha=0.22, ls=":")
        if logy:
            a.set_yscale("log")
            a.set_ylabel("门控值（对数刻度）")
            a.set_title("对数刻度：看被淘汰字段掉到多低", fontsize=9.5)
        else:
            a.set_ylabel("门控值（起点为 1）")
            a.set_title("线性刻度：看保留字段的分化", fontsize=9.5)
    if keep:
        ax.legend(frameon=False, fontsize=7.2, loc="upper right", ncol=2)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if title else 1))
    fig.savefig(path, dpi=300); plt.close(fig)


def gate_bar_plot(gates: np.ndarray, names: list[str], threshold: float,
                  path: Path, title: str = "", top: int = 25) -> None:
    """最终门控值排名，红线是活跃阈值。"""
    idx = np.argsort(-gates)[:top]
    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.27 * len(idx))))
    y = np.arange(len(idx))
    cols = ["#2b6cb0" if gates[i] >= threshold else "0.75" for i in idx]
    ax.barh(y, gates[idx], color=cols)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in idx], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(threshold, color="#c53030", ls="--", lw=1.4)
    ax.set_xlabel("最终门控值")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25, ls=":")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def topn_plot(ns, r2s, full_r2: float, unsel_r2: float,
              path: Path, title: str = "") -> None:
    """按门控值排名逐步增加输入字段数，看测试 R² 怎么变。"""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(ns, r2s, "o-", color="#2b6cb0", lw=1.8, ms=5, label="Top-n 字段再训练")
    ax.axhline(full_r2, color="0.45", ls="--", lw=1.4, label=f"全部字段 {full_r2:.4f}")
    if np.isfinite(unsel_r2):
        ax.axhline(unsel_r2, color="#c53030", ls=":", lw=1.4,
                   label=f"未选中字段 {unsel_r2:.4f}")
    best = int(np.argmax(r2s))
    ax.annotate(f"{r2s[best]:.4f}", xy=(ns[best], r2s[best]), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9, color="#2b6cb0")
    ax.set_xlabel("输入字段数"); ax.set_ylabel("测试 R²")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, ls=":"); ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def gate_distribution_plot(gates_by_method: dict, threshold: float,
                           path: Path, title: str = "") -> None:
    """门控值按大小排开（纵轴取对数）。

    D-Gating 会把没用的字段直接压到机器零，图上表现为一道垂直断崖：
    断崖之上是选中的字段，之下的值小到无论阈值取 1e-7 还是 1e-1 都不会改变判定。
    L1 门控则是连续衰减，没有断崖，阈值取多少直接决定选几个。
    """
    n = len(gates_by_method)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.8), squeeze=False)
    for ax, (name, g) in zip(axes[0], gates_by_method.items()):
        s = np.clip(np.sort(g)[::-1], 1e-9, None)
        ax.semilogy(range(1, len(s) + 1), s, "o-", ms=3.6, lw=1.2, color="#2b6cb0")
        ax.axhline(threshold, color="#c53030", ls="--", lw=1.4)
        n_act = int((g >= threshold).sum())
        ax.annotate(f"阈值 {threshold}\n选中 {n_act} 个",
                    xy=(len(s) * 0.55, threshold), xytext=(0, 8),
                    textcoords="offset points", fontsize=8.5, color="#c53030")
        above = s[s >= threshold]
        below = s[s < threshold]
        if len(above) and len(below) and below.max() > 0:
            ratio = above.min() / below.max()
            ax.annotate(f"断崖落差 {ratio:.0f} 倍", xy=(0.03, 0.06),
                        xycoords="axes fraction", fontsize=8.5, color="0.35")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("字段排名（按门控值从大到小）")
        ax.grid(alpha=0.25, ls=":")
    axes[0][0].set_ylabel("门控值（对数刻度）")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if title else 1))
    fig.savefig(path, dpi=300); plt.close(fig)


def baseline_bar_plot(res: pd.DataFrame, full_r2: float, n: int,
                      path: Path, title: str = "") -> None:
    """同预算下各方法的测试 R²。虚线是用全部字段的普通 DNN，作为能力上限参照。"""
    d = res.sort_values("n", ascending=True)
    fig, ax = plt.subplots(figsize=(7.4, max(2.6, 0.5 * len(d))))
    y = np.arange(len(d))
    cols = ["#c53030" if m == "DGatingDNN" else "#2b6cb0" for m in d["方法"]]
    ax.barh(y, d["n"], color=cols)
    ax.set_yticks(y); ax.set_yticklabels(d["方法"], fontsize=9)
    for i, v in enumerate(d["n"]):
        if np.isfinite(v):
            ax.annotate(f"{v:.4f}", xy=(v, i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8.5)
    ax.axvline(full_r2, color="0.35", ls="--", lw=1.4)
    ax.annotate(f"全部字段 {full_r2:.4f}", xy=(full_r2, len(d) - 0.4),
                xytext=(-4, 0), textcoords="offset points",
                ha="right", fontsize=8.5, color="0.35")
    ax.set_xlabel(f"测试 R²（每个方法各取自己排序的前 {n} 个字段）")
    lo = min(0.0, float(np.nanmin(d["n"])) - 0.05)
    ax.set_xlim(lo, max(1.02, full_r2 + 0.08))
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25, ls=":")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def baseline_overall_plot(agg: pd.DataFrame, path: Path, title: str = "") -> None:
    """各方法在每个目标上的测试 R²，分组柱状图。"""
    piv = agg.pivot_table(index="中文名", columns="方法", values="n")
    order = piv.mean().sort_values(ascending=False).index.tolist()
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(piv)), 4.2))
    w = 0.8 / len(order)
    x = np.arange(len(piv))
    cmap = plt.get_cmap("tab10")
    for k, m in enumerate(order):
        c = "#c53030" if m == "DGatingDNN" else cmap(k % 10)
        ax.bar(x + k * w - 0.4 + w / 2, piv[m].values, width=w, label=m, color=c)
    ax.set_xticks(x); ax.set_xticklabels(piv.index, fontsize=9)
    ax.set_ylabel("测试 R²")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.25, ls=":")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def topk_curve_plot(curves: dict, ks: list, full_r2: float, budget: int,
                    path: Path, title: str = "") -> None:
    """各方法的「取前 k 个字段」曲线。比单点柱状图信息量大得多——
    能看出某个方法是全程领先，还是只在某个字段数上碰巧占优。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, (m, v) in enumerate(curves.items()):
        c = "#c53030" if m == "DGatingDNN" else cmap(i % 10)
        lw = 2.2 if m == "DGatingDNN" else 1.5
        ax.plot(ks, v, "o-", color=c, lw=lw, ms=4.5, label=m,
                zorder=5 if m == "DGatingDNN" else 2)
    ax.axhline(full_r2, color="0.4", ls="--", lw=1.3)
    ax.annotate(f"全部字段 {full_r2:.4f}", xy=(ks[-1], full_r2), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=8.5, color="0.4")
    ax.axvline(budget, color="0.55", ls=":", lw=1.3)
    ax.annotate(f"统一预算 n={budget}", xy=(budget, ax.get_ylim()[0]),
                xytext=(4, 8), textcoords="offset points", fontsize=8.5, color="0.4")
    ax.set_xlabel("取前几个字段"); ax.set_ylabel("测试 R²")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
    ax.grid(alpha=0.25, ls=":")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def selection_matrix_plot(sel: dict, names: list, path: Path,
                          title: str = "", top: int = 30) -> None:
    """字段 × 方法的选中矩阵：一眼看出各方法挑的是不是同一批字段。"""
    methods = list(sel)
    score = {f: sum(f in sel[m] for m in methods) for f in names}
    fields = sorted([f for f in names if score[f] > 0], key=lambda f: -score[f])[:top]
    if not fields:
        return
    M = np.array([[1.0 if f in sel[m] else 0.0 for m in methods] for f in fields])
    fig, ax = plt.subplots(figsize=(1.15 * len(methods) + 3.6,
                                    max(3.0, 0.30 * len(fields))))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1.4, aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(fields))); ax.set_yticklabels(fields, fontsize=8)
    for i in range(len(fields)):
        for j in range(len(methods)):
            if M[i, j]:
                ax.text(j, i, "✓", ha="center", va="center", fontsize=9, color="white")
    ax.set_xticks(np.arange(-.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(fields), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def method_agreement_plot(sel_by_target: dict, path: Path, title: str = "") -> None:
    """方法之间的选择重合度（各目标平均）。"""
    ms = sorted({m for d in sel_by_target.values() for m in d})
    if len(ms) < 2:
        return
    M = np.zeros((len(ms), len(ms)))
    for i, a in enumerate(ms):
        for j, b in enumerate(ms):
            vals = []
            for d in sel_by_target.values():
                if a in d and b in d:
                    u = len(set(d[a]) | set(d[b]))
                    vals.append(len(set(d[a]) & set(d[b])) / u if u else 1.0)
            M[i, j] = float(np.mean(vals)) if vals else np.nan
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=8.5)
    for i in range(len(ms)):
        for j in range(len(ms)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.55 else "0.2")
    fig.colorbar(im, ax=ax, shrink=0.8, label="选中字段的重合度")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def rule2_search_plot(curves: dict, ks: list, full_r2: float, hit: dict,
                      path: Path, title: str = "") -> None:
    """按字段数递增搜索的过程：各方法的 R² 曲线，以及 95%/97% 达标线。"""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    cmap = plt.get_cmap("tab10")
    for i, (m, v) in enumerate(curves.items()):
        c = "#c53030" if m == "DGatingDNN" else cmap(i % 10)
        ax.plot(ks[:len(v)], v, "o-", ms=4, lw=2.1 if m == "DGatingDNN" else 1.4,
                color=c, label=m, zorder=5 if m == "DGatingDNN" else 2)
    for lv, col in [(0.95, "#dd6b20"), (0.97, "#2f855a")]:
        ax.axhline(full_r2 * lv, color=col, ls="--", lw=1.3)
        ax.annotate(f"{lv:.0%} 线 {full_r2 * lv:.4f}", xy=(ks[-1], full_r2 * lv),
                    xytext=(-4, 3), textcoords="offset points",
                    ha="right", fontsize=8, color=col)
        if lv in hit and hit[lv][0]:
            ax.axvline(hit[lv][0], color=col, ls=":", lw=1.2)
            ax.annotate(f"n={hit[lv][0]}", xy=(hit[lv][0], ax.get_ylim()[0]),
                        xytext=(3, 6), textcoords="offset points",
                        fontsize=8, color=col)
    ax.axhline(full_r2, color="0.4", ls="-.", lw=1.1)
    ax.set_xlabel("取前几个字段"); ax.set_ylabel("测试 R²")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.25, ls=":")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def rule2_overall_plot(agg: pd.DataFrame, path: Path, title: str = "") -> None:
    lv = sorted(agg["达标水平"].unique())
    fig, axes = plt.subplots(1, len(lv), figsize=(5.4 * len(lv), 4.0), squeeze=False)
    for ax, l in zip(axes[0], lv):
        s = agg[agg.达标水平 == l].groupby("方法")["测试R2"].mean().sort_values()
        cols = ["#c53030" if m == "DGatingDNN" else "#2b6cb0" for m in s.index]
        ax.barh(np.arange(len(s)), s.values, color=cols)
        ax.set_yticks(np.arange(len(s))); ax.set_yticklabels(s.index, fontsize=9)
        for i, v in enumerate(s.values):
            ax.annotate(f"{v:.4f}", xy=(v, i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8.5)
        ax.set_title(f"{l} 达标预算", fontsize=10)
        ax.set_xlabel("各目标平均测试 R²")
        ax.set_xlim(min(s.values) - 0.05, max(s.values) + 0.06)
        ax.grid(axis="x", alpha=0.25, ls=":")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93 if title else 1))
    fig.savefig(path, dpi=300); plt.close(fig)


def rule3_scatter_plot(agg: pd.DataFrame, path: Path, title: str = "") -> None:
    """横轴选中字段数、纵轴测试 R²，越靠左上越好。虚线连出帕累托前沿。"""
    g = agg.groupby("方法").agg(n=("选中数", "mean"), r=("测试R2", "mean"))
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    cmap = plt.get_cmap("tab10")
    for i, (m, r) in enumerate(g.iterrows()):
        c = "#c53030" if m == "DGatingDNN" else cmap(i % 10)
        sub = agg[agg.方法 == m]
        ax.scatter(sub["选中数"], sub["测试R2"], s=16, alpha=0.28, color=c)
        ax.scatter([r["n"]], [r["r"]], s=150, color=c, edgecolors="white",
                   linewidths=1.4, zorder=5)
        ax.annotate(m, xy=(r["n"], r["r"]), xytext=(7, 5),
                    textcoords="offset points", fontsize=9, color=c, weight="bold")
    pts = list(zip(g["n"], g["r"]))
    front = [i for i, (n, v) in enumerate(pts)
             if not any(n2 <= n and v2 >= v and (n2 < n or v2 > v)
                        for j, (n2, v2) in enumerate(pts) if j != i)]
    if len(front) > 1:
        f = sorted([pts[i] for i in front])
        ax.plot([p[0] for p in f], [p[1] for p in f], "--", color="0.4", lw=1.3,
                zorder=1, label="帕累托前沿")
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_xlabel("选中字段数（小点是各目标，大点是均值）")
    ax.set_ylabel("测试 R²")
    ax.grid(alpha=0.25, ls=":")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def rule3_spread_plot(agg: pd.DataFrame, path: Path, title: str = "") -> None:
    """各方法在各目标上选中字段数的分布。固定阈值的方法波动会明显更大。"""
    ms = agg.groupby("方法")["选中数"].std().sort_values().index.tolist()
    data = [agg[agg.方法 == m]["选中数"].values for m in ms]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.55)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor("#c53030" if ms[i] == "DGatingDNN" else "#2b6cb0")
        b.set_alpha(0.65)
    for i, d in enumerate(data, 1):
        ax.scatter(np.random.default_rng(0).normal(i, 0.055, len(d)), d,
                   s=14, color="0.25", alpha=0.6, zorder=3)
    ax.set_xticks(range(1, len(ms) + 1)); ax.set_xticklabels(ms, fontsize=9, rotation=15)
    ax.set_ylabel("选中字段数")
    ax.grid(axis="y", alpha=0.25, ls=":")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
