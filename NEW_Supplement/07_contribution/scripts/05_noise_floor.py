"""参照答案自己有多不稳：校准偏差的噪声下限在哪。

要回答的问题
------------
上一步量出代理模型的校准偏差大约 0.05~0.07。这个数是大是小？
光看数字没法判断，因为**参照答案本身就是有噪声的**：
同一个字段组合，换一个随机种子重新训练一个普通网络，得到的 R² 并不相同
（初始化不同、批次顺序不同、早停停在不同轮次）。

如果参照答案自己换种子就能差 0.05，那代理模型 0.06 的偏差就已经贴到地板了，
再怎么调参也压不下去——那 0.05 不是代理模型的错，是"用一个网络的测试 R²
去代表某个字段组合的还原能力"这件事本身的不确定度。

反过来，如果参照答案很稳（比如换种子只差 0.005），那 0.06 就是代理模型的真实亏空，
必须继续改进。

这个对照不做，"偏差 0.06 算不算好"这句话就没法回答。
仓库既有的做法里也是这个思路——鲁棒性实验先跑 5 个种子量出天然波动，
再拿退化后的结果去和这条线比，而不是和 1.0 比。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import datasets as ds          # noqa: E402
from src import surrogate as sg         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--target", default="branch_ab1_loading_pct")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-sets", type=int, default=3)
    ap.add_argument("--n-eval", type=int, default=1500)
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    base, figd, datd = ds.run_dir(a.dataset, "L0_noise_floor")
    print(f"输出目录 {base}")

    pool = d.pool(a.target)
    X = d.df[pool].to_numpy(float)
    y = d.df[a.target].to_numpy(float)
    p = len(pool)
    print(f"目标 {d.label(a.target)}，候选池 {p} 个字段")

    cfg = sg.SurrogateConfig(seed=42)
    _, _, i_te = sg._split_idx(len(X), cfg)
    eval_idx = np.sort(np.random.default_rng(0).choice(
        i_te, size=min(a.n_eval, len(i_te)), replace=False))

    rng = np.random.default_rng(11)
    bands = sorted({b for b in [1, 2, 3, 5, 8, 12, 18, 26, 34, p] if 1 <= b <= p})
    probes = []
    for k in bands:
        for _ in range(a.n_sets):
            probes.append(sorted(rng.choice(p, size=k, replace=False).tolist()))

    print(f"\n{len(probes)} 个字段组合 × {len(a.seeds)} 个种子 = "
          f"{len(probes)*len(a.seeds)} 次重训")

    rows = []
    for i, s in enumerate(probes):
        vals = [sg.retrain_reference(X, y, s, cfg, eval_idx, seed=sd)
                for sd in a.seeds]
        v = np.array(vals)
        rows.append(dict(字段数=len(s), 均值=v.mean(), 标准差=v.std(),
                         极差=v.max() - v.min(),
                         各种子=json.dumps([round(x, 4) for x in vals])))
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(probes)}  ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(datd / "noise_floor.csv", index=False)

    print(f"\n{'='*70}")
    print("参照答案换种子后的波动（按字段数分档）")
    g = df.groupby("字段数").agg(组合数=("均值", "size"), 平均值=("均值", "mean"),
                                种子间标准差=("标准差", "mean"),
                                种子间极差=("极差", "mean"))
    print(g.to_string(float_format=lambda x: f"{x:9.4f}"))

    # 把"换种子的平均绝对偏差"折算成和校准偏差可比的量：
    # 任取两个种子，它们之间的期望绝对差 ≈ 1.128 × 标准差（正态假设）
    mad = float((df["标准差"] * 1.128).mean())
    print(f"\n参照答案两两之间的期望绝对差 ≈ {mad:.4f}")
    print(f"其中小组合（≤3 字段）≈ "
          f"{float((df[df['字段数']<=3]['标准差']*1.128).mean()):.4f}")
    print(f"     大组合（≥18 字段）≈ "
          f"{float((df[df['字段数']>=18]['标准差']*1.128).mean()):.4f}")
    print("\n读法：代理模型的校准偏差如果和这个数是一个量级，"
          "说明它已经贴到'参照答案本身能有多准'的下限，"
          "剩下的差距不是模型没训好，是这个参照量自己就有这么大的不确定度。")

    meta = dict(数据集=a.dataset, 目标=a.target, 种子=a.seeds,
                组合数=len(probes), 评估行数=a.n_eval,
                期望绝对差=mad, 总耗时秒=round(time.time() - t0, 1))
    (datd / "config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base}")


if __name__ == "__main__":
    main()
