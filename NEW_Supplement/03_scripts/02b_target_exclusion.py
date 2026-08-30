"""第 2b 步：为每个目标字段确定要删掉哪些输入字段。

删除分三层，理由各不相同：

  全局层  恒定字段（整年不变，没有任何信息）
          重复字段的多余副本（同一份数据发布两遍，留一个即可）
          —— 这两类对所有目标都该删，属于数据质量问题

  目标层  与该目标存在公式关系的字段
          —— 只对特定目标删，删到该目标无法再被精确算出为止

诊断用线性拟合的 R² 做快速筛查（同一份数据内拟合，是可学到程度的上界）。
真正的评估在第 3 步用神经网络做。
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

VERSION = "main"
TOL_RANK, TOL_RESID = 1e-5, 1e-3
STRIP_TOL = 1e-3

TARGETS = [
    "net_actual_interchange_mw",
    "gross_actual_interchange_mw",
    "net_sched_interchange_mw",
    "total_gen",
    "metered_load_mw",
    "total_losses",
    "congestion_price_da",
    "congestion_price_rt",
    "marginal_loss_price_da",
    "total_lmp_da",
    "da_as_total_mw_primary_reserve",
    "da_as_total_mw_thirty_minutes_reserve",
]


def linear_r2(df: pd.DataFrame, target: str, pool: list[str]) -> float:
    rr = idt.exact_fit_ratio(df, target, pool)
    return float("nan") if not np.isfinite(rr) else float(1.0 - rr**2)


def duplicate_groups(idents) -> list[set[str]]:
    """把两两相同的字段并成组。"""
    groups: list[set[str]] = []
    for it in idents:
        if it.kind != "duplicate":
            continue
        pair = {it.lead, *it.support}
        hit = [g for g in groups if g & pair]
        if hit:
            merged = set().union(*hit, pair)
            groups = [g for g in groups if g not in hit] + [merged]
        else:
            groups.append(pair)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    args = ap.parse_args()
    year = args.year

    df = dataio.load_clean(year, VERSION)
    out = dataio.out_dir(year, "02_class1_identity")
    idents, _ = idt.extract(df, tol_rank=TOL_RANK, tol_resid=TOL_RESID)

    constants = sorted({it.lead for it in idents if it.kind == "constant"})
    groups = duplicate_groups(idents)

    print(f"pjm_{year}  {df.shape[0]} 行 × {df.shape[1]} 列\n")
    print(f"恒定字段 {len(constants)} 个（对所有目标都删）：")
    for c in constants:
        print(f"    {c}")
    print(f"\n重复字段 {len(groups)} 组（每组保留一个）：")
    for g in groups:
        print(f"    {{{', '.join(sorted(g))}}}")

    rows = []
    print("\n" + "=" * 74)
    for t in TARGETS:
        if t not in df.columns:
            print(f"{t}：不在字段表里，跳过")
            continue

        # 全局层：恒定字段 + 重复组里除目标外的其余成员
        dup_drop: list[str] = []
        for g in groups:
            members = sorted(g)
            keep = t if t in g else members[0]
            dup_drop += [m for m in members if m != keep]
        global_drop = sorted(set(constants) | set(dup_drop) - {t})

        pool_all = [c for c in df.columns if c != t]
        pool_g = [c for c in pool_all if c not in global_drop]

        # 目标层：反复剥离
        sub = df[[t] + pool_g]
        removed, _ = idt.strip_closure(sub, t, tol=STRIP_TOL)
        pool_f = [c for c in pool_g if c not in removed]

        r2_all = linear_r2(df, t, pool_all)
        r2_g = linear_r2(df, t, pool_g)
        r2_f = linear_r2(df, t, pool_f)

        print(f"\n{t}")
        print(f"  全局删除 {len(global_drop):2d} 个 → 线性 R² {r2_all:.6f} → {r2_g:.6f}")
        print(f"  公式删除 {len(removed):2d} 个：{removed if removed else '无'}")
        print(f"  最终输入 {len(pool_f):2d} 个 → 线性 R² {r2_f:.6f}")

        rows.append({
            "target": t,
            "n_global_drop": len(global_drop),
            "global_drop": "|".join(global_drop),
            "n_formula_drop": len(removed),
            "formula_drop": "|".join(removed),
            "n_input_final": len(pool_f),
            "r2_all_fields": r2_all,
            "r2_after_global": r2_g,
            "r2_after_formula": r2_f,
            "r2_drop": r2_g - r2_f,
        })

    res = pd.DataFrame(rows)
    res.to_csv(out / "target_exclusion.csv", index=False)

    # --- 剥离力度扫描 ---
    # 精确公式和"精度高到实际等价的近似公式"之间没有绝对界线。
    # 把剥离门槛从严到松扫一遍，看每个目标要删多少字段、剩下多少可推断性。
    sweeps = [(1e-3, "只删精确公式"), (1e-2, "含高精度近似"),
              (3e-2, "含 R²>0.999 的关系"), (1e-1, "含 R²>0.99 的关系")]
    sw_rows = []
    print("\n" + "=" * 74)
    print("剥离力度扫描：门槛越松删得越多，剩下的才是真正的统计推断\n")
    hdr = f"{'目标字段':38s}" + "".join(f"{lab[:10]:>16s}" for _, lab in sweeps)
    print(hdr)
    for t in TARGETS:
        if t not in df.columns:
            continue
        gd = res.loc[res.target == t, "global_drop"].iloc[0]
        gd = gd.split("|") if gd else []
        pool_g = [c for c in df.columns if c != t and c not in gd]
        line, cells = f"{t:38s}", []
        for tau, _ in sweeps:
            sub = df[[t] + pool_g]
            rm, _ = idt.strip_closure(sub, t, tol=tau)
            pool = [c for c in pool_g if c not in rm]
            r2 = linear_r2(df, t, pool)
            cells.append(f"{len(rm):2d}删/R²{r2:6.4f}")
            sw_rows.append({"target": t, "tau": tau, "n_drop": len(rm),
                            "dropped": "|".join(rm), "n_input": len(pool), "r2": r2})
        print(line + "".join(f"{c:>16s}" for c in cells))
    pd.DataFrame(sw_rows).to_csv(out / "target_exclusion_sweep.csv", index=False)

    print("\n" + "=" * 74)
    print("汇总（线性 R²，同数据内拟合，是可学到程度的上界）\n")
    print(f"{'目标字段':38s} {'原始':>8s} {'去公式后':>9s} {'降幅':>8s} {'输入数':>6s}")
    for _, r in res.iterrows():
        print(f"{r['target']:38s} {r['r2_all_fields']:8.4f} "
              f"{r['r2_after_formula']:9.4f} {r['r2_drop']:8.4f} "
              f"{int(r['n_input_final']):6d}")
    print(f"\n写入 {out / 'target_exclusion.csv'}")


if __name__ == "__main__":
    main()
