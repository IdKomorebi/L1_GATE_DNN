"""把"稀疏门控判为零贡献、但它自己就能推出目标"的字段找出来数一数。

这是整套工作最直接的一条证据
----------------------------
主实验的贡献值表里反复出现同一种字段：

  独立能力 0.6~0.8   —— 只发布它一个，敏感目标就能还原六到八成
  不可替代性 0.00x   —— 但从完整发布集里抽掉它，几乎没有损失
  分解式门控 0.0000  —— 于是它被判为零贡献

三行放在一起，说的是同一件事：**这个字段完全有能力独立泄露目标，
只是因为别的字段也能做到同样的事，所以"抽掉它"看不出损失。**
稀疏门控的目标函数是"用最少的字段达到足够精度"，它当然会把这类字段扔掉——
扔掉它精度确实不掉。但发布前审查问的不是"最少要几个字段"，
而是"这个字段能不能发"。按门控的输出做白名单，这类字段会被自由发布。

这个脚本把这类字段在全部目标上数一遍，给出：
  - 有多少字段被门控判为精确零、而它们的独立能力却很高
  - 这些字段里独立能力最高的是哪个、有多高
  - 本方法给这些字段的贡献值排在第几

判据里的两个数（独立能力门槛、门控零值判定）都取自既有口径，不是新定的：
门控阈值沿用 02_src/gates.py 的 dgate_threshold=0.01，
独立能力门槛给出多档扫描而不是钉死一个。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import datasets as ds          # noqa: E402

GATE_THR = 0.01          # 沿用 02_src/gates.py 的 dgate_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="主实验的 contributions_all.csv 路径")
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--solo-levels", type=float, nargs="*",
                    default=[0.3, 0.5, 0.7])
    a = ap.parse_args()

    df = pd.read_csv(a.run)
    out = Path(a.run).parent
    d = ds.load(a.dataset)

    need = {"目标", "字段", "贡献值", "独立能力", "不可替代性", "分解式门控"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"表里缺少列：{miss}")

    print("=" * 84)
    print("被稀疏门控判为零贡献、但独立能力很高的字段")
    print("=" * 84)

    rows, cases = [], []
    for tgt, g in df.groupby("目标", sort=False):
        g = g.copy()
        g["门控零"] = g["分解式门控"].abs() < GATE_THR
        g["贡献值排名"] = g["贡献值"].rank(ascending=False).astype(int)
        rec = dict(目标=tgt, 候选数=len(g), 门控判零字段数=int(g["门控零"].sum()))
        for lv in a.solo_levels:
            hit = g[g["门控零"] & (g["独立能力"] >= lv)]
            rec[f"独立能力≥{lv}且被门控判零"] = len(hit)
        blind = g[g["门控零"]]
        rec["被判零字段的最高独立能力"] = float(blind["独立能力"].max()) \
            if len(blind) else np.nan
        if len(blind):
            w = blind.loc[blind["独立能力"].idxmax()]
            rec["最严重的那个字段"] = w["字段"]
            rec["它的本方法贡献值排名"] = int(w["贡献值排名"])
            cases.append(dict(目标=tgt, 字段=w["字段"],
                              独立能力=float(w["独立能力"]),
                              不可替代性=float(w["不可替代性"]),
                              分解式门控=float(w["分解式门控"]),
                              本方法贡献值=float(w["贡献值"]),
                              本方法排名=int(w["贡献值排名"])))
        rows.append(rec)

    res = pd.DataFrame(rows)
    res.to_csv(out / "gate_blindspot.csv", index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    print("\n" + "=" * 84)
    print("每个目标上最严重的一例（门控判零、但独立能力最高的那个字段）")
    print("=" * 84)
    cdf = pd.DataFrame(cases)
    if len(cdf):
        cdf = cdf.sort_values("独立能力", ascending=False)
        cdf.to_csv(out / "gate_blindspot_cases.csv", index=False)
        print(cdf.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

        print("\n读法：以第一行为例——这个公开字段**单独发布**就能把敏感目标还原到")
        print(f"      R² {cdf.iloc[0]['独立能力']:.3f}，但分解式门控给它的门控值是"
              f" {cdf.iloc[0]['分解式门控']:.4f}（判为零贡献），")
        print(f"      而本方法把它排在第 {int(cdf.iloc[0]['本方法排名'])} 位。")
        print("      按门控的输出做发布白名单，这个字段会被放行。")

    # 全局汇总
    print("\n" + "=" * 84)
    tot = len(df)
    zero = int((df["分解式门控"].abs() < GATE_THR).sum())
    print(f"全部 {df['目标'].nunique()} 个目标合计 {tot} 个（目标, 字段）对：")
    print(f"  被分解式门控判为零贡献的：{zero}（{zero/tot:.1%}）")
    # 比例一律连分母一起报，并且分母太小的时候明确说"别报比例"。
    # 按阈值分层之后样本会迅速变少：这一档剩几个字段，决定了那个百分比
    # 是个结论还是个噪声。实测剥离后 ≥0.7 那一档只剩 2 个字段，
    # "比例从 55.6% 降到 0%"字面为真，但 0/2 支撑不起一般性断言。
    MIN_N = 10
    for lv in a.solo_levels:
        n = int(((df["分解式门控"].abs() < GATE_THR) &
                 (df["独立能力"] >= lv)).sum())
        n_all = int((df["独立能力"] >= lv).sum())
        line = (f"  其中独立能力≥{lv} 的：{n}/{n_all}"
                f"（{n/max(n_all,1):.1%}）")
        if n_all < MIN_N:
            line += f"  ← 分母只有 {n_all}，别报成比例，改用上界陈述"
        print(line)

    # 同一件事换个问法：不可替代性和独立能力差多少
    zeroed = df[df["分解式门控"].abs() < GATE_THR]
    if len(zeroed):
        print(f"\n上界陈述（不依赖分档，最稳）：被门控判为零贡献的字段中，"
              f"独立还原能力最高者为 **{zeroed['独立能力'].max():.4f}**")

    print(f"\n所有（目标, 字段）对上：")
    print(f"  独立能力中位数   {df['独立能力'].median():.4f}")
    print(f"  不可替代性中位数 {df['不可替代性'].median():.4f}")
    hi = df[df["独立能力"] >= 0.5]
    if len(hi):
        print(f"  在独立能力≥0.5 的 {len(hi)} 个字段上，"
              f"不可替代性中位数只有 {hi['不可替代性'].median():.4f}")
        print("  → 这就是'单独发布足以泄露、但抽掉它毫无损失'的量化表述，"
              "也是只处置被选中字段无效的原因。")

    (out / "gate_blindspot_summary.json").write_text(json.dumps(
        {"门控阈值": GATE_THR, "独立能力档": a.solo_levels,
         "合计对数": tot, "被判零对数": zero,
         "独立能力≥0.5且被判零": int(((df["分解式门控"].abs() < GATE_THR) &
                                (df["独立能力"] >= 0.5)).sum())},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物写入 {out}")


if __name__ == "__main__":
    main()
