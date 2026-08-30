"""L5 稳定性：换个随机种子，结论还是不是同一个结论。

已知的弱点
----------
现有方法的输出是一个**字段集合**。仓库实测：只换随机种子，两次选中的集合
两两 Jaccard 相似度（交集除以并集）只有 0.46~0.61，选中数在 8~12 之间跳，
而全量 R² 稳到 ±0.0012。也就是说模型很稳，"哪些字段是推断源"这个答案不稳。

这不是实现的毛病，是**定义的必然**：最小充分集合只要够用就行，
有多条路径都够用时挑哪一条本来就没有唯一答案，随机初始化落到哪条就是哪条。

要检验的主张
------------
**推断源集合不稳定，但每个字段的贡献份额稳定。**

贡献值是对所有字段组合取平均得到的，不需要在等价的路径之间做选择，
所以它应当对随机种子不敏感。如果这一点成立，就说明"哪些字段有风险"
这个问题换一种问法之后是有稳定答案的。

三个口径一起报
--------------
  集合重合度      按判零门槛切出来的集合，两两 Jaccard（对应旧口径）
  贡献值秩相关    完整排序的 Spearman 秩相关
  前 k 名重合度   只看份额最大的前 k 个字段的重合程度
                  （全字段秩相关会被一堆接近零的字段稀释——那些字段的排序
                   本来就是噪声，把它们算进去会低估真实的一致性）

同时报一个对照：**分解式门控在同样的种子上的集合重合度**，
这才是苹果对苹果。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import datasets as ds          # noqa: E402
from src import surrogate as sg         # noqa: E402
from src import attribution as att      # noqa: E402
from src import baselines as bl         # noqa: E402
from src import report as rp            # noqa: E402


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--n-coalitions", type=int, default=16384)
    ap.add_argument("--n-noise", type=int, default=6)
    ap.add_argument("--topk", type=int, default=10)
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    base_dir, figd, datd = ds.run_dir(a.dataset, "L5_stability")
    print(f"输出目录 {base_dir}")
    print(f"种子 {a.seeds}，抽取组合数 {a.n_coalitions}")

    targets = a.targets or d.targets
    rows, detail = [], []

    for ti, tgt in enumerate(targets):
        pool = d.pool(tgt)
        X = d.df[pool].to_numpy(float)
        y = d.df[tgt].to_numpy(float)
        p = len(pool)
        print(f"\n{'='*74}\n[{ti+1}/{len(targets)}] {d.label(tgt)}，候选池 {p}")

        phis, gates, ztols, vfulls = [], [], [], []
        for sd in a.seeds:
            cfg = sg.SurrogateConfig(seed=sd)
            t1 = time.time()
            res = sg.fit(X, y, pool, cfg)
            vf = sg.ValueFunction(res, X, y, n_eval=a.n_eval, seed=0)
            r = att.kernel_shap(vf, p, n_coalitions=a.n_coalitions, seed=sd)
            phis.append(np.asarray(r.phi))
            vfulls.append(float(np.ravel(r.v_full)[0]))

            # 判零门槛逐种子重算，不能借用别的种子的
            rng = np.random.default_rng(sd + 7)
            Xn = np.column_stack([X, rng.standard_normal((len(X), a.n_noise))])
            res_n = sg.fit(Xn, y, pool + [f"__n{i}" for i in range(a.n_noise)], cfg)
            vf_n = sg.ValueFunction(res_n, Xn, y, n_eval=a.n_eval, seed=0)
            r_n = att.kernel_shap(vf_n, p + a.n_noise,
                                  n_coalitions=a.n_coalitions, seed=sd)
            ztols.append(att.zero_band(np.asarray(r_n.phi)[p:], q=0.95))

            gates.append(bl.dgating_scores(X, y, seed=sd)["score"])
            print(f"  种子 {sd}：v(全部)={vfulls[-1]:.4f}  "
                  f"判零门槛={ztols[-1]:.5f}  ({time.time()-t1:.0f}s)", flush=True)

        # ---- 三个口径 ----
        ours_sets = [set(np.where(np.abs(ph) > zt)[0])
                     for ph, zt in zip(phis, ztols)]
        gate_sets = [set(np.where(g >= 0.01)[0]) for g in gates]
        pairs = list(itertools.combinations(range(len(a.seeds)), 2))

        j_ours = float(np.mean([jaccard(ours_sets[i], ours_sets[j])
                                for i, j in pairs]))
        j_gate = float(np.mean([jaccard(gate_sets[i], gate_sets[j])
                                for i, j in pairs]))
        rho_all = float(np.mean([spearmanr(phis[i], phis[j]).statistic
                                 for i, j in pairs]))
        topk_sets = [set(np.argsort(-np.abs(ph))[:a.topk]) for ph in phis]
        j_topk = float(np.mean([jaccard(topk_sets[i], topk_sets[j])
                                for i, j in pairs]))
        rho_gate = float(np.mean([spearmanr(gates[i], gates[j]).statistic
                                  for i, j in pairs]))
        # 份额本身的波动：每个字段跨种子的标准差，按最大份额归一
        P = np.vstack(phis)
        sd_norm = float((P.std(axis=0) / max(np.abs(P).max(), 1e-12)).mean())

        rec = dict(目标=d.label(tgt), 候选数=p,
                   本方法_集合重合度=j_ours, 本方法_贡献值秩相关=rho_all,
                   本方法_前k名重合度=j_topk, 本方法_份额相对波动=sd_norm,
                   门控_集合重合度=j_gate, 门控_门控值秩相关=rho_gate,
                   本方法_集合大小=[len(s) for s in ours_sets],
                   门控_集合大小=[len(s) for s in gate_sets],
                   v全部各种子=[round(v, 4) for v in vfulls])
        rows.append(rec)
        print(f"  本方法：集合重合度 {j_ours:.3f}，贡献值秩相关 {rho_all:.3f}，"
              f"前{a.topk}名重合度 {j_topk:.3f}，集合大小 {rec['本方法_集合大小']}")
        print(f"  分解式门控：集合重合度 {j_gate:.3f}，门控值秩相关 {rho_gate:.3f}，"
              f"集合大小 {rec['门控_集合大小']}")

        for sd, ph in zip(a.seeds, phis):
            for f, v in zip(pool, ph):
                detail.append(dict(目标=d.label(tgt), 种子=sd, 字段=f, 贡献值=v))

    df = pd.DataFrame(rows)
    df.to_csv(datd / "stability.csv", index=False)
    pd.DataFrame(detail).to_csv(datd / "stability_detail.csv", index=False)

    print(f"\n{'='*74}\n汇总")
    show = ["目标", "本方法_集合重合度", "本方法_贡献值秩相关",
            "本方法_前k名重合度", "本方法_份额相对波动",
            "门控_集合重合度", "门控_门控值秩相关"]
    print(df[show].to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"\n跨目标平均：")
    print(f"  本方法  集合重合度 {df['本方法_集合重合度'].mean():.3f}   "
          f"贡献值秩相关 {df['本方法_贡献值秩相关'].mean():.3f}   "
          f"前{a.topk}名重合度 {df['本方法_前k名重合度'].mean():.3f}")
    print(f"  门控    集合重合度 {df['门控_集合重合度'].mean():.3f}   "
          f"门控值秩相关 {df['门控_门控值秩相关'].mean():.3f}")
    print("\n（仓库既有实测：门控选中集合换种子的 Jaccard 是 0.46~0.61，"
          "可以和上面这一行对照）")

    rp.stability_compare(df, figd / "fig_稳定性对照.png")

    (datd / "config.json").write_text(json.dumps(
        {"数据集": a.dataset, "种子": a.seeds, "抽取组合数": a.n_coalitions,
         "评估行数": a.n_eval, "前k名的k": a.topk,
         "总耗时秒": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base_dir}")


if __name__ == "__main__":
    main()
