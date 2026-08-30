"""从存好的表重新生成全部图。

出图代码改过之后（比如把重叠的标注拉开），已经跑完的那些运行的图还是旧的。
重跑实验只为出图太浪费——所有画图需要的数据都在 data/ 下的 CSV 里，
从表重出即可，几秒钟的事。

顺带保证一件事：图和表永远是同一份数据出来的。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import report as rp            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def zero_tol_of(run: Path, label: str | None = None) -> float | None:
    """从 summary.csv 里取该目标的判零门槛。"""
    f = run / "data" / "summary.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f)
    if label is not None and "目标" in s.columns:
        m = s[s["目标"] == label]
        if len(m) and "判零门槛" in m.columns:
            return float(m["判零门槛"].iloc[0])
    return float(s["判零门槛"].median()) if "判零门槛" in s.columns else None


def redo_contrib_table(run: Path, csv: str, key: str, tag: str) -> int:
    """对一张"每行一个字段"的表，按分组逐组重出三张图。"""
    f = run / "data" / csv
    if not f.exists():
        return 0
    df = pd.read_csv(f)
    if key not in df.columns:
        return 0
    figd = run / "figures"
    figd.mkdir(exist_ok=True)
    n = 0
    for label, g in df.groupby(key, sort=False):
        if not {"贡献值", "协同冗余指数", "类型"} <= set(g.columns):
            continue
        zt = zero_tol_of(run, str(label))
        safe = str(label).replace("/", "_")
        rp.contribution_bar(g, figd / f"fig_{tag}贡献值_{safe}.png", top=28,
                            zero_tol=zt)
        rp.risk_map(g, figd / f"fig_{tag}风险坐标_{safe}.png", zero_tol=zt)
        if "独立能力" in g.columns:
            rp.solo_vs_marginal(g, figd / f"fig_{tag}独立vs不可替代_{safe}.png")
        n += 3
    return n


def redo_matrix(run: Path) -> int:
    f = run / "data" / "contribution_matrix.csv"
    if not f.exists():
        return 0
    m = pd.read_csv(f, index_col=0)
    rp.heatmap(m.to_numpy(), list(m.index), list(m.columns),
               run / "figures" / "fig_贡献矩阵.png", top_rows=32)
    return 1


def redo_stability(run: Path) -> int:
    f = run / "data" / "stability.csv"
    if not f.exists():
        return 0
    df = pd.read_csv(f)
    if "本方法_集合重合度" not in df.columns:
        return 0
    rp.stability_compare(df, run / "figures" / "fig_稳定性对照.png")
    return 1


def redo_calibration(run: Path) -> int:
    f = run / "data" / "calibration_probes.csv"
    if not f.exists():
        return 0
    df = pd.read_csv(f)
    if not {"真实重训R2", "代理模型v(S)"} <= set(df.columns):
        return 0
    rp.calibration(df, run / "figures" / "fig_校准_全部.png")
    n = 1
    if "目标" in df.columns:
        for lab, g in df.groupby("目标"):
            safe = str(lab).replace("/", "_")
            rp.calibration(g, run / "figures" / f"fig_校准_{safe}.png")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只处理路径里含该字串的运行")
    a = ap.parse_args()

    runs = sorted(p for p in OUT.rglob("run_*")
                  if p.is_dir() and "_archive" not in p.parts)
    total = 0
    for run in runs:
        if a.only and a.only not in str(run):
            continue
        n = 0
        n += redo_contrib_table(run, "contributions_all.csv", "目标", "")
        n += redo_contrib_table(run, "synthetic_contributions.csv", "题目", "")
        n += redo_contrib_table(run, "L1_contributions.csv", "目标", "L1")
        n += redo_contrib_table(run, "L2_contributions.csv", "伪目标", "L2")
        n += redo_contrib_table(run, "joint_contributions.csv", None, "") \
            if False else 0
        # 联合贡献表没有分组列，单独处理
        jf = run / "data" / "joint_contributions.csv"
        if jf.exists():
            g = pd.read_csv(jf)
            if {"贡献值", "协同冗余指数", "类型"} <= set(g.columns):
                rp.contribution_bar(g, run / "figures" / "fig_联合贡献值.png",
                                    top=28)
                rp.risk_map(g, run / "figures" / "fig_联合风险坐标.png")
                n += 2
        n += redo_matrix(run)
        n += redo_stability(run)
        n += redo_calibration(run)
        if n:
            total += n
            print(f"  {run.relative_to(OUT)} → 重出 {n} 张")
    print(f"\n合计重出 {total} 张图。")


if __name__ == "__main__":
    main()
