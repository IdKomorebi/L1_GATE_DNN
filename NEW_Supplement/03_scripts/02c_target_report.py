"""第 2c 步：12 个目标字段的逐个详细报告。

回答四个问题：
  1. 这个目标有没有公式能精确算出来？公式是什么？公式本身有多准？
  2. 用全部字段能推到多准？
  3. 把公式相关的字段删掉之后，还能推到多准？
  4. 上面这些数字是怎么算出来的？

输入设置分四档，逐档变严：
  S0  全部字段（除目标自己）
  S1  再删掉恒定字段和重复字段的多余副本      —— 数据质量问题，所有目标通用
  S2  再删掉与该目标存在精确公式关系的字段     —— 残差门槛 1e-3
  S3  再删掉与该目标存在高精度近似关系的字段   —— 残差门槛 1e-2

每档都用两种模型评估，都在测试集上报告：
  线性     最小二乘，能反映"线性可推"的程度
  GBDT     梯度提升树，能抓非线性关系，更接近真实可推断上限

划分方式也报两种：
  随机划分   与原有实验一致，便于和旧结果对比
  时序划分   前 80% 训练、后 20% 测试，更贴近"用历史推断未来"的真实场景
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_src"))
import dataio  # noqa: E402
import identity as idt  # noqa: E402

VERSION = "main"
TOL_RANK, TOL_RESID = 1e-5, 1e-3
TAU_EXACT, TAU_APPROX = 1e-3, 1e-2
SEED, TRAIN_RATIO = 42, 0.8

TARGETS = [
    "net_actual_interchange_mw", "gross_actual_interchange_mw",
    "net_sched_interchange_mw", "total_gen", "metered_load_mw", "total_losses",
    "congestion_price_da", "congestion_price_rt", "marginal_loss_price_da",
    "total_lmp_da", "da_as_total_mw_primary_reserve",
    "da_as_total_mw_thirty_minutes_reserve",
]

CN = {
    "net_actual_interchange_mw": "净实际交换功率",
    "gross_actual_interchange_mw": "总实际交换功率",
    "net_sched_interchange_mw": "净计划交换功率",
    "total_gen": "总发电量",
    "metered_load_mw": "计量负荷",
    "total_losses": "总网损",
    "congestion_price_da": "日前阻塞价",
    "congestion_price_rt": "实时阻塞价",
    "marginal_loss_price_da": "日前边际损耗价",
    "total_lmp_da": "日前总电价",
    "da_as_total_mw_primary_reserve": "日前主用备用总量",
    "da_as_total_mw_thirty_minutes_reserve": "日前30分钟备用总量",
}


def splits(n: int):
    k = int(n * TRAIN_RATIO)
    perm = np.random.default_rng(SEED).permutation(n)
    return {"随机划分": (perm[:k], perm[k:]),
            "时序划分": (np.arange(k), np.arange(k, n))}


def fit_r2(df: pd.DataFrame, target: str, pool: list[str]) -> dict:
    """两种模型 × 两种划分，全部在测试集上评估。"""
    if not pool:
        return {}
    X = df[pool].to_numpy(float)
    y = df[target].to_numpy(float)
    out = {}
    for sname, (tr, te) in splits(len(df)).items():
        if np.std(y[te]) == 0:
            out[f"线性/{sname}"] = out[f"GBDT/{sname}"] = float("nan")
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        lin = Ridge(alpha=1e-6).fit((X[tr] - mu) / sd, y[tr])
        out[f"线性/{sname}"] = float(r2_score(y[te], lin.predict((X[te] - mu) / sd)))
        gb = HistGradientBoostingRegressor(
            max_iter=200, random_state=SEED).fit(X[tr], y[tr])
        out[f"GBDT/{sname}"] = float(r2_score(y[te], gb.predict(X[te])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--stage", default="02_class1_identity",
                    help="输出子目录名")
    ap.add_argument("--exclude-targets", action="store_true",
                    help="候选池里排除其余目标字段，只保留普通字段")
    args = ap.parse_args()
    year = args.year

    df = dataio.load_clean(year, VERSION)
    out = dataio.out_dir(year, args.stage)
    idents, _ = idt.extract(df, tol_rank=TOL_RANK, tol_resid=TOL_RESID)

    constants = sorted({it.lead for it in idents if it.kind == "constant"})
    groups: list[set[str]] = []
    for it in idents:
        if it.kind != "duplicate":
            continue
        pair = {it.lead, *it.support}
        hit = [g for g in groups if g & pair]
        groups = ([g for g in groups if g not in hit] +
                  [set().union(*hit, pair)]) if hit else groups + [pair]
    formulas = [it for it in idents if it.kind == "formula"]

    rows, detail = [], []
    print(f"pjm_{year}  {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"恒定字段 {len(constants)} 个，重复字段 {len(groups)} 组，公式关系 {len(formulas)} 条\n")

    for t in TARGETS:
        if t not in df.columns:
            continue
        # --- 该目标直接参与的精确公式 ---
        mine = [it for it in formulas if t == it.lead or t in it.support]

        dup_drop: list[str] = []
        for g in groups:
            keep = t if t in g else sorted(g)[0]
            dup_drop += [m for m in sorted(g) if m != keep]
        global_drop = sorted((set(constants) | set(dup_drop)) - {t})

        s0 = [c for c in df.columns if c != t]
        if args.exclude_targets:
            s0 = [c for c in s0 if c not in TARGETS]
        s1 = [c for c in s0 if c not in global_drop]
        rm_exact, _ = idt.strip_closure(df[[t] + s1], t, tol=TAU_EXACT)
        s2 = [c for c in s1 if c not in rm_exact]
        rm_approx, _ = idt.strip_closure(df[[t] + s2], t, tol=TAU_APPROX)
        s3 = [c for c in s2 if c not in rm_approx]

        r = {"target": t, "中文名": CN.get(t, t),
             "参与的精确公式": " ; ".join(it.text() for it in mine) or "无",
             "公式残差比": " ; ".join(f"{it.residual_ratio:.1e}" for it in mine) or "",
             "n_S0": len(s0), "n_S1": len(s1), "n_S2": len(s2), "n_S3": len(s3),
             "删_精确公式": "|".join(rm_exact), "删_近似关系": "|".join(rm_approx)}
        for tag, pool in [("S0", s0), ("S1", s1), ("S2", s2), ("S3", s3)]:
            for k, v in fit_r2(df, t, pool).items():
                r[f"{tag} {k}"] = v
        rows.append(r)

        print(f"── {CN.get(t, t)}  {t}")
        print(f"   精确公式：{r['参与的精确公式']}"
              + (f"   （公式残差比 {r['公式残差比']}）" if mine else ""))
        if rm_exact:
            print(f"   删精确公式相关字段 {len(rm_exact)} 个：{rm_exact}")
        if rm_approx:
            print(f"   再删近似关系字段 {len(rm_approx)} 个：{rm_approx}")
        for tag, lab, pool in [("S0", "全部字段", s0), ("S1", "去质量问题", s1),
                               ("S2", "去精确公式", s2), ("S3", "去近似关系", s3)]:
            print(f"   {lab:6s}({len(pool):2d}字段)  "
                  f"线性 随机{r.get(f'{tag} 线性/随机划分', float('nan')):7.4f} "
                  f"时序{r.get(f'{tag} 线性/时序划分', float('nan')):8.4f}   "
                  f"GBDT 随机{r.get(f'{tag} GBDT/随机划分', float('nan')):7.4f} "
                  f"时序{r.get(f'{tag} GBDT/时序划分', float('nan')):8.4f}")
        print()

    res = pd.DataFrame(rows)
    res.to_csv(out / "target_report.csv", index=False)
    print(f"写入 {out / 'target_report.csv'}")


if __name__ == "__main__":
    main()
