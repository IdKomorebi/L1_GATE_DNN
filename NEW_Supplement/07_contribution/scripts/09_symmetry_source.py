"""对称性误差到底出在哪一环：是贡献值算法，还是代理模型自己？

背景
----
候选池里"全系统核电出力"和"1 区核电出力"这两列数字**逐个数完全相同**。
任何讲得通的分法都必须给它们完全相等的份额。实测差了 7%~20%。
加大抽样量到 8 倍，误差几乎没降——说明这不是抽样误差。

那问题出在哪？链条上有两个环节：

  环节一  代理模型给出的 v(S)：两列数字一样，但它们是网络的两个不同输入位置，
          网络学到的权重不一样，所以 v(只发布 a) 未必等于 v(只发布 b)。
          **被求解的那个函数自己就不对称。**
  环节二  从 v(S) 解出贡献值的算法：这一环在玩具例子上已经验证到 1e-10。

这个脚本直接量环节一：把两个重复字段互换位置，看 v(S) 变不变。
如果 v 自己就不对称，那贡献值不对称就是它的忠实反映，不是算错——
要改就得改代理模型（比如让它见过更多两者一起被屏蔽/单独出现的组合），
而不是加大抽样量。分不清这两环，就会往错误的方向调参数。

顺带量一个更基本的东西：**两个真的完全相同的字段，用同一份数据、
只换随机种子各训练一个普通网络，得到的 R² 本来差多少**。
这是"对称性能做到多准"的天然下限。
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
from src import attribution as att      # noqa: E402
from src import truthsets as ts         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--target", default="bus_215_va_deg")
    ap.add_argument("--hidden", default="384,256,192,128")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--n-probe", type=int, default=400)
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    base, figd, datd = ds.run_dir(a.dataset, "L0_symmetry_source")
    print(f"输出目录 {base}")

    pool = d.pool(a.target, drop_constants=False)
    dups = ts.duplicate_groups_in_pool(d.df, pool)
    assert dups, "这个目标的池子里没有完全重复的字段"
    g = dups[0]
    i, j = pool.index(g[0]), pool.index(g[1])
    print(f"目标 {d.label(a.target)}，候选池 {len(pool)}")
    print(f"完全重复的一对：{g[0]}（第 {i} 位） 与 {g[1]}（第 {j} 位）")
    col_i, col_j = d.df[g[0]].to_numpy(float), d.df[g[1]].to_numpy(float)
    print(f"两列数值最大绝对差 = {np.abs(col_i - col_j).max():.3e}（确认完全相同）")

    X = d.df[pool].to_numpy(float)
    y = d.df[a.target].to_numpy(float)
    p = len(pool)

    cfg = sg.SurrogateConfig(hidden=tuple(int(x) for x in a.hidden.split(",")),
                             epochs=a.epochs, min_epochs=150, patience=150, seed=42)
    print("\n训练代理模型…")
    res = sg.fit(X, y, pool, cfg)
    vf = sg.ValueFunction(res, X, y, n_eval=a.n_eval, seed=0)
    print(f"  完成（{time.time()-t0:.0f}s），v(全部)={float(vf.full()):.4f}")

    # ---------- 环节一：v(S) 本身对不对称 ----------
    print(f"\n{'='*70}")
    print("环节一：代理模型给出的 v(S)，把两个字段互换位置会不会变")

    eye = np.eye(p, dtype=np.float32)
    v_i, v_j = float(vf(eye[i:i+1])[0]), float(vf(eye[j:j+1])[0])
    print(f"  只发布 {g[0]}：v = {v_i:.5f}")
    print(f"  只发布 {g[1]}：v = {v_j:.5f}")
    print(f"  两者之差 = {abs(v_i-v_j):.5f}   ← 理论上应当**完全相等**")

    m_full = np.ones((2, p), dtype=np.float32)
    m_full[0, i] = 0.0
    m_full[1, j] = 0.0
    lo = vf(m_full)
    print(f"  全部发布但抽掉 {g[0]}：v = {lo[0]:.5f}")
    print(f"  全部发布但抽掉 {g[1]}：v = {lo[1]:.5f}")
    print(f"  两者之差 = {abs(lo[0]-lo[1]):.5f}")

    # 随机组合上做互换检验：S 里含 i 不含 j，换成含 j 不含 i
    rng = np.random.default_rng(5)
    others = [q for q in range(p) if q not in (i, j)]
    diffs = []
    A, B = [], []
    for _ in range(a.n_probe):
        k = int(rng.integers(0, len(others) + 1))
        sub = rng.choice(others, size=k, replace=False).tolist()
        A.append(sorted(sub + [i]))
        B.append(sorted(sub + [j]))
    va, vb = vf.of_sets(A), vf.of_sets(B)
    diffs = np.abs(va - vb)
    print(f"\n  在 {a.n_probe} 个随机组合上做互换检验（同一组其余字段，"
          f"只把 {g[0]} 换成 {g[1]}）：")
    print(f"    v 的绝对差：均值 {diffs.mean():.5f}，中位 {np.median(diffs):.5f}，"
          f"最大 {diffs.max():.5f}")
    print(f"    相对 v(全部)={float(vf.full()):.3f} 而言，平均相对差 "
          f"{diffs.mean()/max(abs(float(vf.full())),1e-9):.2%}")

    # ---------- 环节二：贡献值算法 ----------
    print(f"\n{'='*70}")
    print("环节二：从 v(S) 解贡献值这一步，自己引入了多少不对称")
    print("做法：把 v(S) 强行对称化（互换后取平均），再算一次贡献值。")
    print("如果对称化之后误差基本消失，说明问题全在环节一。")

    class Symmetrized:
        """把 v 强行对称化：v'(S) = (v(S) + v(S 里 i/j 互换)) / 2。"""
        def __init__(self, vf, i, j):
            self.vf, self.i, self.j = vf, i, j
            self.p = vf.p

        def __call__(self, masks, batch: int = 64):
            m = np.atleast_2d(np.asarray(masks, dtype=np.float32)).copy()
            m2 = m.copy()
            m2[:, [self.i, self.j]] = m2[:, [self.j, self.i]]
            return 0.5 * (self.vf(m, batch) + self.vf(m2, batch))

    rows = []
    for nb in (2048, 8192, 32768):
        r0 = att.kernel_shap(vf, p, n_coalitions=nb, seed=7)
        r1 = att.kernel_shap(Symmetrized(vf, i, j), p, n_coalitions=nb, seed=7)
        e0 = abs(float(r0.phi[i] - r0.phi[j]))
        e1 = abs(float(r1.phi[i] - r1.phi[j]))
        sc = float(np.abs(r0.phi).max())
        rows.append(dict(抽取组合数=nb, 原始_绝对误差=e0, 原始_相对误差=e0 / sc,
                         对称化后_绝对误差=e1, 对称化后_相对误差=e1 / sc))
        print(f"  {nb:>6} 组合：原始误差 {e0:.5f}（相对 {e0/sc:.2%}）  "
              f"→ 对称化后 {e1:.5f}（相对 {e1/sc:.2%}）", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(datd / "symmetry_source.csv", index=False)

    # ---------- 天然下限 ----------
    print(f"\n{'='*70}")
    print("参照：两个完全相同的字段，各自单独训练一个普通网络，R² 本来差多少")
    eval_idx = vf.idx
    for sd in (42, 43, 44):
        ri = sg.retrain_reference(X, y, [i], cfg, eval_idx, seed=sd)
        rj = sg.retrain_reference(X, y, [j], cfg, eval_idx, seed=sd)
        print(f"  种子 {sd}：只用 {g[0]} → R² {ri:.5f}；"
              f"只用 {g[1]} → R² {rj:.5f}；差 {abs(ri-rj):.5f}")
    print("  （两列数字完全相同，所以这个差完全来自训练本身的随机性。"
          "它给出'对称性最好能做到多准'的天然下限。）")

    meta = dict(数据集=a.dataset, 目标=a.target, 重复字段=g,
                v_只发布甲=v_i, v_只发布乙=v_j, v差=abs(v_i - v_j),
                互换检验平均绝对差=float(diffs.mean()),
                互换检验最大绝对差=float(diffs.max()),
                各档=rows, 总耗时秒=round(time.time() - t0, 1))
    (datd / "config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base}")


if __name__ == "__main__":
    main()
