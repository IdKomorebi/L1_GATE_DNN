"""L4 重训认证：把各目标的 R²–k 曲线汇总成一张表和一张图。

判据沿用仓库既有口径：按某个方法给出的排序取前 k 个字段，
用**无门控的普通网络独立重训**，看能达到多少 R²。
排序好的方法，同样的 k 下应当更高。

要一并报的两件事
----------------
1. **最低的 k 个**那条曲线。只报"最高的 k 个"是不够的——一个方法可以靠
   把所有信息量都堆在前几名而显得好看。真正说明排序对不对的是两端：
   最高的 k 个要涨得快，**最低的 k 个要涨得慢**。
2. **这不是本方法的主场**。分解式门控的目标函数就是"用最少的字段达到足够精度"，
   在这一项上它本来就该强。本方法赢的是别的地方（份额、分类、补集刻画）。
   如实报，不挑口径。
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import report as rp            # noqa: E402


def parse(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:                                        # noqa: BLE001
            return None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="主实验运行目录")
    a = ap.parse_args()

    run = Path(a.run)
    s = pd.read_csv(run / "data" / "summary.csv")
    curve_cols = [c for c in s.columns
                  if c.startswith("曲线_") and c != "曲线_k"]
    if not curve_cols:
        raise SystemExit("summary.csv 里没有曲线列（跑主实验时可能加了 --skip-topk）")

    ks = parse(s["曲线_k"].iloc[0])
    full = s["v_全部"].to_numpy(float)
    rows = []
    for c in curve_cols:
        name = c.replace("曲线_", "")
        vals = np.array([parse(v) for v in s[c] if parse(v) is not None],
                        dtype=float)
        if vals.ndim != 2:
            continue
        rec = {"方法": name}
        for i, k in enumerate(ks):
            rec[f"k={k}"] = float(vals[:, i].mean())
        # 每个目标各自算"达到该目标全量 R² 九成需要几个字段"，再取中位数
        need = []
        for row, fr in zip(vals, full):
            hit = [k for k, v in zip(ks, row) if v >= 0.9 * fr]
            need.append(hit[0] if hit else np.nan)
        rec["达到全量九成所需的k"] = float(np.nanmedian(need))
        rows.append(rec)

    df = pd.DataFrame(rows)
    order = ["本方法贡献值", "通道A门控值", "分解式门控", "随机门控",
             "两两相关性", "置换重要性", "Lasso系数", "本方法·最低的k个"]
    df["_o"] = df["方法"].apply(lambda m: order.index(m) if m in order else 99)
    df = df.sort_values("_o").drop(columns=["_o"])
    df.to_csv(run / "data" / "topk_summary.csv", index=False)

    print("=" * 96)
    print(f"重训认证：按各方法排序取前 k 个字段独立重训的 R²（{len(s)} 个目标平均）")
    print("=" * 96)
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.4f}"))
    print(f"\n全量 R² 平均 {full.mean():.4f}")

    top = df[df["方法"] != "本方法·最低的k个"]
    for k in [1, 3, 8]:
        col = f"k={k}"
        if col in top.columns:
            best = top.loc[top[col].idxmax(), "方法"]
            ours = float(top[top["方法"] == "本方法贡献值"][col].iloc[0])
            print(f"  k={k:<3} 最好的是 {best}（{top[col].max():.4f}），"
                  f"本方法 {ours:.4f}")

    bot = df[df["方法"] == "本方法·最低的k个"]
    if len(bot):
        print("\n最低的 k 个（这条要**低**才对）：")
        print("  " + "  ".join(f"k={k}:{bot[f'k={k}'].iloc[0]:.3f}"
                               for k in ks if f"k={k}" in bot.columns))
        ourstop = df[df["方法"] == "本方法贡献值"]
        gaps = [float(ourstop[f"k={k}"].iloc[0] - bot[f"k={k}"].iloc[0])
                for k in ks if f"k={k}" in bot.columns]
        print(f"  与'最高的 k 个'的差：最大 {max(gaps):.3f}"
              f"（k={ks[int(np.argmax(gaps))]}）")
        print("  差距越大，说明贡献值的排序两端都站得住。")

    curves = {}
    for _, r in df.iterrows():
        xs = [k for k in ks if f"k={k}" in r.index]
        curves[r["方法"]] = (xs, [float(r[f"k={k}"]) for k in xs])
    rp.topk_curve(curves, run / "figures" / "fig_重训认证_全目标平均.png",
                  full_r2=float(full.mean()))

    (run / "data" / "topk_summary.json").write_text(json.dumps(
        {"目标数": len(s), "k档": ks, "全量R2均值": float(full.mean())},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物写入 {run/'data'}")


if __name__ == "__main__":
    main()
