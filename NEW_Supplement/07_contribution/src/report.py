"""出图与出表。

画图口径沿用 NEW_Supplement 既有约定（这些都是踩过坑总结出来的）：
- 图内不写标题，题目交给 LaTeX 的 caption 出，避免两处不一致
- 字体用 Arial Unicode MS。Songti SC 缺 U+2212（数学减号），
  对数刻度的负号会渲染成方块
- 中文和 $...$ 不能出现在同一个字符串里，否则整串会走数学字体、中文全部丢失。
  要上标就用 Unicode 字符，比如 R²
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

# 三类字段的固定配色，全项目统一
COLOR = {
    "替身型": "#4E79A7",
    "协同型": "#E15759",
    "独立可加": "#59A14F",
    "无贡献": "#BAB0AC",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def contribution_bar(tab: pd.DataFrame, path: Path, top: int = 25,
                     label_col: str = "字段", value_col: str = "贡献值",
                     class_col: str | None = "类型", zero_tol: float | None = None):
    """贡献值排序条形图，按类型着色。"""
    t = tab.sort_values(value_col, ascending=False).head(top).iloc[::-1]
    colors = [COLOR.get(c, "#888888") for c in t[class_col]] if class_col in t \
        else "#4E79A7"
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.26 * len(t))))
    ax.barh(np.arange(len(t)), t[value_col], color=colors)
    ax.set_yticks(np.arange(len(t)))
    ax.set_yticklabels(t[label_col], fontsize=8)
    ax.set_xlabel("贡献值")
    if zero_tol is not None:
        ax.axvline(zero_tol, color="#666666", ls="--", lw=0.9)
        ax.text(zero_tol, len(t) - 0.5, " 判零门槛", fontsize=7, color="#666666",
                va="top")
    if class_col in t:
        seen = [c for c in ["替身型", "协同型", "独立可加", "无贡献"]
                if c in set(t[class_col])]
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR[c]) for c in seen],
                  labels=seen, fontsize=8, loc="lower right")
    _save(fig, path)


def risk_map(tab: pd.DataFrame, path: Path, zero_tol: float | None = None,
             inter_tol: float = 0.05, annotate_top: int = 8):
    """风险坐标图：横轴贡献份额，纵轴协同–冗余指数。

    这张图是整套方法的主图。四个区域的读法：
      右下  份额大、指数负 → 有替身的高风险字段。删掉它没用，别人顶上
      右上  份额大、指数正 → 协同型高风险字段。单看两两相关性完全看不出来
      右中  份额大、指数≈0 → 独立可加，删掉确实能降风险
      左侧  份额小 → 无论哪一类都不是当前的风险来源
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for cls, g in tab.groupby("类型"):
        ax.scatter(g["贡献值"], g["协同冗余指数"], s=38, alpha=0.8,
                   color=COLOR.get(cls, "#888888"), label=cls,
                   edgecolors="white", linewidths=0.7)
    ax.axhline(0, color="#444444", lw=0.9)
    ax.axhspan(-inter_tol, inter_tol, color="#000000", alpha=0.04)
    if zero_tol is not None:
        ax.axvline(zero_tol, color="#666666", ls="--", lw=0.9)
        ax.axvspan(-zero_tol, zero_tol, color="#000000", alpha=0.04)
    ax.set_xlabel("贡献值（该字段分到的还原能力份额）")
    ax.set_ylabel("协同–冗余指数（负=有替身，正=靠配合）")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)

    # 只给真正有份额的字段加标注。判零门槛以下的字段挤在原点附近，
    # 全标出来会叠成一团糊，什么也读不出来。
    t = tab.copy()
    if zero_tol is not None:
        t = t[t["贡献值"].abs() > zero_tol]
    t = t.sort_values("贡献值", ascending=False).head(annotate_top)
    _annotate_spread(ax, t["贡献值"], t["协同冗余指数"], t["字段"])
    _save(fig, path)


def _annotate_spread(ax, xs, ys, labels, fontsize: float = 6.8):
    """加标注并躲开彼此。

    重复字段的贡献值按定义就该相等，于是它们的点会**完全重合**，
    标签直接盖在一起谁也看不见——而"两个点重合"恰恰是这张图最想让人看到的事。
    所以重合的点用一条细引线把标签拉开，并在标签上标出重合了几个。
    """
    import numpy as _np
    MAXLEN = 26
    pts = []
    for x, y, lab in zip(_np.asarray(xs, float), _np.asarray(ys, float),
                         list(labels)):
        s = str(lab)
        pts.append((x, y, s if len(s) <= MAXLEN else s[:MAXLEN - 1] + "…"))
    xr = (ax.get_xlim()[1] - ax.get_xlim()[0]) or 1.0
    yr = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0

    # 判重要按**标签占的地方**算，不能只看两个点离多远：
    # 字段名很长（area_1_fuel_nuclear_mw 这种），两个点在横轴上离得不近，
    # 标签照样会压在一起。所以横向占位按字符数估，纵向按行高逐级往上让。
    # 每让一级挪多少：先按字号定一个点数，再换算成数据单位，两边必须一致。
    # 之前这里把两种单位混着写，结果每级只挪不到一个点，等于没让开。
    ax_pts = float(ax.figure.get_size_inches()[1] * 72 *
                   ax.get_position().height) or 300.0
    step_pts = fontsize * 1.6
    step_data = step_pts / ax_pts * yr

    boxes: list[tuple[float, float, float]] = []   # (左, 右, 标签实际落点)
    for x, y, lab in sorted(pts, key=lambda t: (-t[1], -t[0])):
        w = 0.0115 * len(lab) * xr        # 每个字符约占横轴 1.15%
        lvl = 0
        while lvl <= 8:
            ytry = y + lvl * step_data
            clash = any(not (x + w < bl or x > br)
                        and abs(ytry - by) < 0.9 * step_data
                        for bl, br, by in boxes)
            if not clash:
                break
            lvl += 1
        ax.annotate(lab, (x, y), fontsize=fontsize,
                    xytext=(5, 4 + lvl * step_pts),
                    textcoords="offset points", alpha=0.9,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999",
                                    shrinkA=0, shrinkB=1) if lvl else None)
        boxes.append((x, x + w, y + lvl * step_data))


def solo_vs_marginal(tab: pd.DataFrame, path: Path, annotate_top: int = 8):
    """独立能力 vs 不可替代性。落在对角线上的是可加型，偏离越远互动越强。"""
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lo = float(min(tab["独立能力"].min(), tab["不可替代性"].min(), 0))
    hi = float(max(tab["独立能力"].max(), tab["不可替代性"].max()))
    pad = 0.05 * max(hi - lo, 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#444444",
            lw=0.9, ls="--")
    for cls, g in tab.groupby("类型"):
        ax.scatter(g["独立能力"], g["不可替代性"], s=34, alpha=0.85,
                   color=COLOR.get(cls, "#888888"), label=cls,
                   edgecolors="white", linewidths=0.6)
    ax.set_xlabel("独立能力  v({j}) − v(∅)")
    ax.set_ylabel("不可替代性  v(F) − v(F∖{j})")
    ax.legend(fontsize=8)
    t = tab.sort_values("贡献值", ascending=False).head(annotate_top)
    _annotate_spread(ax, t["独立能力"], t["不可替代性"], t["字段"])
    _save(fig, path)


def calibration(df: pd.DataFrame, path: Path, x="真实重训R2", y="代理模型v(S)"):
    """校准图：代理模型给的 v(S) 对不对得上真刀真枪重训出来的 R²。

    这张图是整套方法的地基。点越贴近对角线，代理模型越可信；
    系统性偏离对角线就说明贡献值不能用。
    """
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lo = float(min(df[x].min(), df[y].min()))
    hi = float(max(df[x].max(), df[y].max()))
    pad = 0.05 * max(hi - lo, 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#E15759", lw=1.2)
    sizes = df["字段数"] if "字段数" in df else 30
    sc = ax.scatter(df[x], df[y], c=sizes, cmap="viridis", s=32, alpha=0.9,
                    edgecolors="white", linewidths=0.5)
    if "字段数" in df:
        plt.colorbar(sc, ax=ax, label="组合里的字段数")
    err = float(np.mean(np.abs(df[y] - df[x])))
    r = float(np.corrcoef(df[x], df[y])[0, 1])
    ax.set_xlabel("真实值：只用这些字段重新训练一个普通网络得到的 R²")
    ax.set_ylabel("代理模型给出的 v(S)")
    ax.text(0.03, 0.97, f"平均绝对偏差 {err:.4f}\n相关系数 {r:.4f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
    _save(fig, path)


def topk_curve(curves: dict[str, tuple[list[int], list[float]]], path: Path,
               full_r2: float | None = None, ylabel: str = "重新训练后的 R²"):
    """按各方法给出的排序取前 k 个字段、独立重训的 R²–k 曲线。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, (ks, r2s) in curves.items():
        ax.plot(ks, r2s, marker="o", ms=3.4, lw=1.4, label=name)
    if full_r2 is not None:
        ax.axhline(full_r2, color="#444444", ls="--", lw=0.9)
        ax.text(ax.get_xlim()[1], full_r2, " 全部发布", fontsize=7,
                va="bottom", ha="right", color="#444444")
    ax.set_xlabel("保留的字段数 k")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    _save(fig, path)


def convergence(curve: dict, path: Path):
    """采样量收敛曲线：抽多少个字段组合，贡献值才稳定下来。"""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    b = curve["budget"]
    axes[0].plot(b, curve["spearman_vs_prev"], marker="o", ms=4)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("抽取的字段组合数")
    axes[0].set_ylabel("与上一档的秩相关")
    axes[1].plot(b, curve["maxdiff_vs_prev"], marker="s", ms=4, color="#E15759")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("抽取的字段组合数")
    axes[1].set_ylabel("与上一档的最大绝对差")
    _save(fig, path)


def training_curve(hist: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(hist["epoch"], hist["train_loss"], lw=1.3, label="训练损失")
    ax.plot(hist["epoch"], hist["val_loss"], lw=1.3, label="验证损失")
    ax.set_yscale("log")
    ax.set_xlabel("轮次")
    ax.set_ylabel("损失（对数刻度）")
    ax.legend(fontsize=8)
    _save(fig, path)


def stability_compare(df: pd.DataFrame, path: Path):
    """换种子后的一致程度：本方法与分解式门控并排比。

    两件事必须画进去，否则这张图会读出相反的结论：
    1. 比的是**同类量**——本方法的贡献值排序 对 门控的门控值排序，
       不能拿本方法的两个指标并排冒充"新旧口径对比"；
    2. 门控只选一两个字段时，"集合重合"和"秩相关"都会白得满分，
       这类退化情形要在图上标出来，不能混在平均里。
    """
    need = {"目标", "本方法_贡献值秩相关", "门控_门控值秩相关"}
    if not need <= set(df.columns):          # 兼容只有本方法的旧表
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        x = np.arange(len(df))
        w = 0.38
        ax.bar(x - w / 2, df["集合重合度"], width=w,
               label="本方法·选中集合的重合度", color="#BAB0AC")
        ax.bar(x + w / 2, df["贡献值秩相关"], width=w,
               label="本方法·贡献值排序的秩相关", color="#4E79A7")
        ax.set_xticks(x)
        ax.set_xticklabels(df["目标"], rotation=38, ha="right", fontsize=7.5)
        ax.set_ylabel("换随机种子后的一致程度")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
        _save(fig, path)
        return

    degen = df["门控_集合大小"].astype(str).str.contains(r"\[1, 1, 1\]") \
        if "门控_集合大小" in df.columns else pd.Series(False, index=df.index)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w / 2, df["本方法_贡献值秩相关"], width=w,
           label="本方法·贡献值排序", color="#4E79A7")
    ax.bar(x + w / 2, df["门控_门控值秩相关"], width=w,
           label="分解式门控·门控值排序", color="#BAB0AC")
    for i in np.where(degen)[0]:
        ax.text(i + w / 2, df["门控_门控值秩相关"].iloc[i] + 0.015, "只选1个",
                ha="center", fontsize=7, color="#E15759")
    ax.set_xticks(x)
    ax.set_xticklabels(df["目标"], rotation=38, ha="right", fontsize=7.5)
    ax.set_ylabel("换随机种子后排序的一致程度")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def heatmap(mat: np.ndarray, rows: list[str], cols: list[str], path: Path,
            cbar_label: str = "贡献值", top_rows: int | None = 30):
    """字段 × 目标的贡献矩阵热力图。"""
    m = np.asarray(mat, dtype=float)
    if top_rows and m.shape[0] > top_rows:
        order = np.argsort(-np.nansum(np.abs(m), axis=1))[:top_rows]
        m, rows = m[order], [rows[i] for i in order]
    fig, ax = plt.subplots(figsize=(0.52 * len(cols) + 4.2,
                                    0.22 * len(rows) + 1.8))
    im = ax.imshow(m, aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(m)), vmax=np.nanmax(np.abs(m)))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=42, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.grid(False)
    plt.colorbar(im, ax=ax, label=cbar_label)
    _save(fig, path)


def build_table(fields: list[str], dec: dict, labels: dict[str, str] | None = None,
                zero_tol: float | None = None, inter_tol: float = 0.05
                ) -> pd.DataFrame:
    """整理成统一的中文表头表格。

    zero_tol / inter_tol 两个门槛都应当由同池注入的噪声字段实测得到
    （见 attribution.zero_band / interaction_band），不要用默认值出正式结果。
    """
    """把分解结果整理成统一的中文表头表格，全项目共用这一个格式。"""
    from . import attribution as att
    labels = labels or {}
    cls = att.classify(dec, zero_tol if zero_tol is not None else -np.inf, inter_tol)
    t = pd.DataFrame({
        "字段": fields,
        "中文名": [labels.get(f, "") for f in fields],
        "贡献值": np.asarray(dec["phi"]).ravel(),
        "归一化份额": np.asarray(dec["share"]).ravel(),
        "独立能力": np.asarray(dec["solo"]).ravel(),
        "不可替代性": np.asarray(dec["marginal"]).ravel(),
        "协同冗余指数": np.asarray(dec["interaction"]).ravel(),
        "类型": cls,
    })
    return t.sort_values("贡献值", ascending=False).reset_index(drop=True)
