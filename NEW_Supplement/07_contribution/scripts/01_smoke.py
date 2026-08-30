"""冒烟：在 RTS v2 的一个目标上把整条链路跑通。

只验流程通不通，不验数值对不对——参数一律取小，结论不能用。
选的目标是 121 号核电机组出力，因为它在数据里有一个已知的确切答案：
公开池里的"全系统核电出力"和"1 区核电出力"两个字段与它**完全相同**
（三列数值一模一样）。也就是说这两个公开字段互为完美替身，各自单独就能
把这个敏感目标完全还原。

正确的贡献值必须给这两个字段**相等**的份额（各约一半），
而稀疏门控只会留下其中一个、把另一个判为零——这是一眼就能看出差别的地方。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import datasets as ds          # noqa: E402
from src import surrogate as sg         # noqa: E402
from src import attribution as att      # noqa: E402

TARGET = "gen_121_nuclear_1_pg_mw"


def main() -> None:
    t0 = time.time()
    d = ds.load("rts_v2")
    print(f"数据集 {d.name}：{d.df.shape[0]} 行 × {d.df.shape[1]} 列")
    print(f"敏感目标 {len(d.targets)} 个，恒定字段 {len(d.constants)} 个")
    for g in d.dup_groups:
        print(f"完全重复的字段组：{g}")

    pool = d.pool(TARGET)
    print(f"\n目标：{d.label(TARGET)}（{TARGET}）")
    print(f"候选池 {len(pool)} 个字段")

    X = d.df[pool].to_numpy(float)
    y = d.df[TARGET].to_numpy(float)

    cfg = sg.SurrogateConfig(hidden=(64, 48), epochs=60, min_epochs=20,
                             patience=15, batch_size=256, seed=42)
    print(f"\n训练代理模型（冒烟参数：{cfg.epochs} 轮上限）…")
    res = sg.fit(X, y, pool, cfg, verbose=True)
    print(f"最佳轮次 {res.best_epoch}，实跑 {len(res.history['epoch'])} 轮")

    vf = sg.ValueFunction(res, X, y, n_eval=1500, seed=0)
    v_full, v_empty = float(vf.full()), float(vf.empty())
    print(f"\nv(全部发布) = {v_full:.4f}")
    print(f"v(什么都不发布) = {v_empty:.4f}   ← 应当接近 0")

    print("\n算贡献值（冒烟：1024 个组合）…")
    r = att.kernel_shap(vf, len(pool), n_coalitions=1024, seed=0)
    dec = att.decompose(r)

    tab = pd.DataFrame({
        "字段": pool,
        "贡献值": dec["phi"],
        "独立能力": dec["solo"],
        "不可替代性": dec["marginal"],
        "协同度": dec["synergy"],
        "可替代度": dec["substitutability"],
    }).sort_values("贡献值", ascending=False)

    print(f"\n加和性偏差 = {float(r.efficiency_gap):.2e}（应为 0）")
    print(f"v(全部)-v(空) = {v_full - v_empty:.4f}，Σ贡献值 = {dec['phi'].sum():.4f}")
    print("\n贡献值前 10：")
    print(tab.head(10).to_string(index=False,
                                 float_format=lambda x: f"{x:8.4f}"))

    # ---- 已知答案的核对 ----
    twins = ["gen_fuel_nuclear_mw", "area_1_fuel_nuclear_mw"]
    print("\n" + "-" * 60)
    print("已知答案核对：这两个公开字段与目标完全相同，互为完美替身")
    sub = tab[tab["字段"].isin(twins)]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    if len(sub) == 2:
        a, b = sub["贡献值"].to_numpy()
        rank = [tab["字段"].tolist().index(t) + 1 for t in twins]
        print(f"\n两者贡献值 {a:.4f} vs {b:.4f}，相对差 {abs(a-b)/max(abs(a),1e-9):.1%}")
        print(f"两者排名 {rank}（应当并列在最前）")
        print(f"两者贡献值之和 {a+b:.4f}，占 v(全部) 的 {(a+b)/max(v_full,1e-9):.1%}")
        print(f"两者的不可替代性 {sub['不可替代性'].to_numpy()}"
              f" ← 应当都接近 0（抽掉一个，另一个顶上）")

    print(f"\n总耗时 {time.time()-t0:.1f} 秒")
    print("冒烟通过：链路打通。数值不作数，参数是刻意调小的。")


if __name__ == "__main__":
    main()
