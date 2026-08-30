"""第 2 步：第一类（公式型）关系的检测。

产出：
  threshold_scan.csv      门槛扫描表，用来说明门槛落在平台里、不敏感
  identities.csv          全部精确关系清单（含分类、系数整洁性、跨年复验）
  known_formulas.csv      已有公式的验算结果（原样 vs 口径标定）
  derived_relations.csv   比值型与乘积型关系
  strip_closure.csv       各目标字段的反复剥离过程
  summary.md              汇总
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_src"))
import dataio  # noqa: E402
import identity as idt  # noqa: E402
import plots  # noqa: E402

VERSION = "main"
TOLS = [1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
# 求秩门槛放宽一些，宁可多给几个候选方向（负责不漏）；
# 是否成立、属于哪一档，交给残差门槛、跨年复验和系数整洁性去判（负责不错）。
TOL_RANK = 1e-5
TOL_RESID = 1e-3

FUELS = ["coal", "gas", "hydro", "multiple_fuels", "nuclear", "oil",
         "other_renewables", "solar", "storage", "wind"]

STRIP_TARGETS = [
    "net_actual_interchange_mw",
    "total_lmp_da",
    "congestion_price_da",
    "metered_load_mw",
    "total_gen",
]


def known_formulas(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("total_lmp_da", "日前电价三分量分解",
         lambda d: [d["system_energy_price_da"], d["congestion_price_da"],
                    d["marginal_loss_price_da"]]),
        ("total_lmp_rt", "实时电价三分量分解",
         lambda d: [d["system_energy_price_rt"], d["congestion_price_rt"],
                    d["marginal_loss_price_rt"]]),
        ("net_actual_interchange_mw", "净交换功率账目关系",
         lambda d: [d["net_sched_interchange_mw"], -d["net_inadv_interchange_mw"]]),
        ("gross_actual_interchange_mw", "总交换功率同式（对照）",
         lambda d: [d["gross_sched_interchange_mw"], -d["gross_inadv_interchange_mw"]]),
        ("total_gen", "分燃料出力求和",
         lambda d: [d[f"gen_fuel_{f}_mw"] for f in FUELS]),
        ("total_gen", "功率平衡（教科书写法）",
         lambda d: [d["metered_load_mw"], d["total_losses"],
                    d["net_actual_interchange_mw"]]),
        ("total_gen", "功率平衡（符号修正）",
         lambda d: [d["metered_load_mw"], d["total_losses"],
                    -d["net_actual_interchange_mw"]]),
    ]
    rows = []
    for target, name, expr in specs:
        for cal in (False, True):
            r = idt.check_formula(df, target, expr, calibrate=cal)
            r["name"] = name
            rows.append(r)
    out = pd.DataFrame(rows)[["name", "target", "calibrated", "residual_ratio", "coef"]]
    return out


def derived_relations(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """比值型与乘积型：这两类在原空间是非线性的，线性方法抓不到。"""
    rows = []

    def rr(y, pred):
        y, pred = np.asarray(y, float), np.asarray(pred, float)
        m = np.isfinite(y) & np.isfinite(pred)
        if m.sum() < 10 or np.std(y[m]) == 0:
            return float("nan")
        return float(np.sqrt(np.mean((y[m] - pred[m]) ** 2)) / np.std(y[m]))

    s = df[[f"gen_fuel_{f}_mw" for f in FUELS]].sum(axis=1)
    s_o = other[[f"gen_fuel_{f}_mw" for f in FUELS]].sum(axis=1)
    for f in FUELS:
        rows.append({
            "kind": "比值",
            "relation": f"gen_fuel_{f}_pct = gen_fuel_{f}_mw / Σ gen_fuel_mw",
            "residual_ratio": rr(df[f"gen_fuel_{f}_pct"], df[f"gen_fuel_{f}_mw"] / s),
            "cross_year": rr(other[f"gen_fuel_{f}_pct"], other[f"gen_fuel_{f}_mw"] / s_o),
        })

    prices = ["rmccp", "rmpcp"]
    credits = ["total_pjm_rmccp_cr", "total_pjm_rmpcp_cr"]
    qtys = ["total_pjm_reg_purchases", "total_pjm_assigned_reg",
            "total_pjm_self_sched_reg", "total_pjm_rt_load_mwh"]
    for p, c in zip(prices, credits):
        for q in qtys:
            v = rr(df[c], df[p] * df[q])
            if np.isfinite(v) and v < 0.3:
                rows.append({
                    "kind": "乘积",
                    "relation": f"{c} = {p} × {q}",
                    "residual_ratio": v,
                    "cross_year": rr(other[c], other[p] * other[q]),
                })
    return pd.DataFrame(rows).sort_values("residual_ratio")


def tier(it) -> str:
    """按残差量级、跨年是否退化、系数是否整齐，把关系分成三档。

    恒等式    —— 账目或定义决定，系数是 ±1 这种整数，换一年照样成立
    规则型    —— 由市场规则决定（例如"主用备用需求 = 1.5 倍同步备用需求"），
                 系数是规则参数，规则一改跨年就退化
    近似关系  —— 发布精度或口径造成的近似，系数不整齐，残差在 1e-4 量级
    """
    if it.kind in ("constant", "duplicate"):
        return it.kind
    r, cy = it.residual_ratio, it.cross_year
    degrade = (cy / r) if (np.isfinite(cy) and np.isfinite(r) and r > 0) else np.inf
    if r < 1e-4 and degrade < 10:
        return "恒等式"
    if degrade > 100:
        return "规则型(跨年退化)"
    if not it.clean_coefs or r > 1e-4:
        return "近似关系"
    return "恒等式"


def residual_bands(df: pd.DataFrame, idents) -> pd.DataFrame:
    """每个字段用其余全部字段拟合后的相对残差，用来画"公式型与统计型之间有空白带"。"""
    formula_fields = {c for it in idents for c in [it.lead] + it.support}
    rows = []
    for c in df.columns:
        pool = [x for x in df.columns if x != c]
        rr = idt.exact_fit_ratio(df, c, pool)
        if not np.isfinite(rr):
            rr = 0.0
        rows.append({"field": c, "residual_ratio": rr,
                     "group": "公式型" if c in formula_fields else "统计型"})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--other", type=int, default=2024)
    ap.add_argument("--out-version", default=None)
    args = ap.parse_args()
    global YEAR, OTHER
    YEAR, OTHER = args.year, args.other
    df = dataio.load_clean(YEAR, VERSION)
    other = dataio.load_clean(OTHER, VERSION)
    out = dataio.out_dir(YEAR, "02_class1_identity", args.out_version)
    FIG, DAT = dataio.split_dirs(out)
    print(f"pjm_{YEAR} [{VERSION}] {df.shape[0]} 行 × {df.shape[1]} 列\n")

    # --- 1. 门槛扫描 ---
    scan = idt.threshold_scan(df, TOLS)
    scan.to_csv(DAT / "threshold_scan.csv", index=False)
    plots.threshold_scan_plot(scan, FIG / "fig_threshold_scan.png",
                              f"判定门槛的敏感性（PJM {YEAR}）")
    print("门槛扫描：")
    for _, r in scan.iterrows():
        mark = "  ← 平台" if r["plateau"] else ""
        print(f"  {r['tol']:.0e}  ->  {int(r['n_relations']):3d} 条{mark}")

    # --- 2. 提取 + 分类 + 跨年复验 ---
    idents, stats = idt.extract(df, tol_rank=TOL_RANK, tol_resid=TOL_RESID)
    for it in idents:
        it.cross_year = idt.verify(it, other)
    print(f"\n求秩门槛 {TOL_RANK:.0e} 给出 {stats['n_directions']} 条；"
          f"写出成立 {stats['n_valid']} 条"
          f"（化简得到 {stats['n_from_rref']}，逐字段扫描补回 {stats['n_recovered']}）")
    rows = []
    for it in idents:
        print(f"  [{tier(it):14s}] 残差比={it.residual_ratio:.2e} "
              f"跨年={it.cross_year:.2e}  {it.text()}")
        rows.append({
            "space": it.space, "kind": it.kind, "tier": tier(it), "lead": it.lead,
            "support": "|".join(it.support), "n_support": len(it.support),
            "residual_ratio": it.residual_ratio, "cross_year": it.cross_year,
            "clean_coefs": it.clean_coefs, "text": it.text(),
        })

    # --- 3. 对数空间 ---
    log_idents, log_stats = idt.extract_log(df, tol_rank=TOL_RANK, tol_resid=TOL_RESID)
    for it in log_idents:
        it.cross_year = idt.verify(it, other)
        rows.append({
            "space": it.space, "kind": it.kind, "tier": tier(it), "lead": it.lead,
            "support": "|".join(it.support), "n_support": len(it.support),
            "residual_ratio": it.residual_ratio, "cross_year": it.cross_year,
            "clean_coefs": it.clean_coefs, "text": it.text(),
        })
    print(f"\n对数空间：求秩给出 {log_stats['n_directions']} 条，写出成立 {log_stats['n_valid']} 条")
    for it in log_idents:
        print(f"  [{it.kind:9s}] 残差比={it.residual_ratio:.2e}  {it.text()}")
    pd.DataFrame(rows).to_csv(DAT / "identities.csv", index=False)

    # --- 4. 已有公式验算 ---
    kf = known_formulas(df)
    kf.to_csv(DAT / "known_formulas.csv", index=False)
    print("\n已有公式验算（残差比越小越接近精确成立）：")
    for _, r in kf.iterrows():
        tag = "标定后" if r["calibrated"] else "原样  "
        print(f"  {tag} {r['name']:22s} {r['residual_ratio']:.4e}")

    # --- 5. 比值型与乘积型 ---
    dr = derived_relations(df, other)
    dr.to_csv(DAT / "derived_relations.csv", index=False)
    print("\n比值型与乘积型关系：")
    for _, r in dr.iterrows():
        print(f"  [{r['kind']}] 残差比={r['residual_ratio']:.3e} "
              f"跨年={r['cross_year']:.3e}  {r['relation']}")

    # --- 6. 反复剥离 ---
    strip_rows = []
    print("\n反复剥离（删到目标字段无法被精确算出为止）：")
    for t in STRIP_TARGETS:
        if t not in df.columns:
            continue
        removed, trace = idt.strip_closure(df, t, tol=1e-6)
        print(f"  {t:30s} 删除 {len(removed)} 个：{removed}")
        for step in trace:
            strip_rows.append({"target": t, **step})
    pd.DataFrame(strip_rows).to_csv(DAT / "strip_closure.csv", index=False)

    # --- 6b. 残差比分布图 ---
    bands = residual_bands(df, idents)
    bands.to_csv(DAT / "residual_bands.csv", index=False)
    plots.residual_band_plot(bands, FIG / "fig_residual_bands.png",
                             f"公式型关系与统计型关系的残差分布（PJM {YEAR}）")

    # --- 7. 汇总 ---
    kinds = pd.Series([r["tier"] for r in rows if r["space"] == "linear"]).value_counts()
    md = [
        f"# 第一类关系检测汇总 — pjm_{YEAR}（{VERSION} 版本）", "",
        f"数据规模：{df.shape[0]} 行 × {df.shape[1]} 列，跨年复验数据 pjm_{OTHER}", "",
        "## 门槛扫描", "",
        "| 门槛 | 关系条数 | 是否平台 |", "|---|---|---|",
    ]
    md += [f"| {r['tol']:.0e} | {int(r['n_relations'])} | {'是' if r['plateau'] else ''} |"
           for _, r in scan.iterrows()]
    md += ["", f"采用求秩门槛 {TOL_RANK:.0e}、残差门槛 {TOL_RESID:.0e}。", "", "## 关系分类", ""]
    md += [f"- {k}：{v} 条" for k, v in kinds.items()]
    md += ["", "## 全部关系", "",
           "| 空间 | 档次 | 残差比 | 跨年复验 | 系数整齐 | 关系 |",
           "|---|---|---|---|---|---|"]
    md += [f"| {r['space']} | {r['tier']} | {r['residual_ratio']:.2e} | "
           f"{r['cross_year']:.2e} | {'是' if r['clean_coefs'] else '否'} | "
           f"`{r['text']}` |" for r in rows]
    (out / "说明.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
