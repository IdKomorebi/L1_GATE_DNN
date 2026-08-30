"""第二阶段·多目标：一次训练同时回答十二个敏感目标，并找出枢纽字段。

要回应的意见
------------
审稿人二提过"隐性推断源的多目标耦合场景不够清晰"，这一条到现在完全没有回应。
它的实际含义是：真实的发布决策不是一次只保护一个字段，而是一批敏感目标同时
要保护。一个公开字段可能同时给好几个目标供料——这种字段的治理价值最高，
但逐个目标单独分析是看不出来的。

结构上的一个便宜
----------------
随机屏蔽这套框架天然支持多目标：主干共享，末端换成十二个输出头，
掩码机制一个字不用改。于是
  v(S) 从一个数变成一个十二维向量（每个目标各自的还原能力），
  贡献值也从一列变成一个 **字段 × 目标** 的矩阵。

关键在于**求解只做一次**：加权最小二乘的设计矩阵只和字段组合有关，
和目标是谁无关，所以十二个目标共用同一次分解，多出来的开销只是矩阵右端多几列。
换句话说，多目标版本的代价和单目标跑一次差不多，而不是十二倍。

三样产出
--------
1. 字段 × 目标的贡献矩阵，以及它的热力图
2. **枢纽字段**：按行求和，找出同时给多个目标供料的字段。
   同时报"广度"（在几个目标上超过判零门槛）和"深度"（合计贡献）——
   一个字段可能只对一个目标贡献极大，也可能对十个目标各贡献一点，
   两者的治理含义完全不同，不能混成一个数。
3. 与单目标逐个跑出来的结果对照。两者接近说明共享主干没有损害精度；
   有系统性差异也要如实报出来。
"""

from __future__ import annotations

import argparse
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
from src import report as rp            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--n-coalitions", type=int, default=32768)
    ap.add_argument("--n-noise", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--compare-run", default=None,
                    help="单目标主实验的 contributions_all.csv 路径，用于对照")
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    base_dir, figd, datd = ds.run_dir(a.dataset, "multitarget")
    print(f"输出目录 {base_dir}")

    targets = d.targets
    # 候选池：全部公开字段（去掉全部敏感目标和恒定字段）。
    # 多目标共用同一个池子，这样贡献矩阵的行才对得齐。
    pool = [c for c in d.df.columns
            if c not in targets and c not in d.constants]
    p = len(pool)
    print(f"候选池 {p} 个公开字段，同时保护 {len(targets)} 个敏感目标")

    X = d.df[pool].to_numpy(float)
    Y = d.df[targets].to_numpy(float)

    rng = np.random.default_rng(a.seed + 7)
    Xa = np.column_stack([X, rng.standard_normal((len(X), a.n_noise))])
    names_a = pool + [f"__噪声{i+1}" for i in range(a.n_noise)]

    cfg = sg.SurrogateConfig(epochs=a.epochs, seed=a.seed)
    print(f"\n训练多目标代理模型（主干 {cfg.hidden}，{len(targets)} 个输出头）…")
    res = sg.fit(Xa, Y, names_a, cfg, verbose=True)
    print(f"  完成，{time.time()-t0:.0f}s，最佳轮次 {res.best_epoch}")

    vf = sg.ValueFunction(res, Xa, Y, n_eval=a.n_eval, seed=0)
    v_full = np.atleast_1d(vf.full())
    v_empty = np.atleast_1d(vf.empty())
    print("\n各目标的全量还原能力：")
    for t, vf_, ve_ in zip(targets, v_full, v_empty):
        print(f"  {d.label(t):<22} v(全部)={vf_:7.4f}   v(空)={ve_:7.4f}")

    print(f"\n解贡献值（{a.n_coalitions} 个组合，"
          f"{len(targets)} 个目标共用同一次分解）…")
    t1 = time.time()
    r = att.kernel_shap(vf, len(names_a), n_coalitions=a.n_coalitions, seed=a.seed)
    dec = att.decompose(r)
    print(f"  完成，{time.time()-t1:.0f}s")
    gaps = np.atleast_1d(r.efficiency_gap)
    print(f"  加和性偏差最大 {np.abs(gaps).max():.2e}")

    PHI = np.asarray(dec["phi"])                 # (p+noise, k)
    phi_noise = PHI[p:]
    # 判零门槛逐目标各算一个——不同目标的份额尺度不一样
    ztol = np.array([att.zero_band(phi_noise[:, c], q=0.95)
                     for c in range(PHI.shape[1])])
    PHI_real = PHI[:p]
    print(f"  各目标判零门槛：{np.round(ztol, 4)}")

    mat = pd.DataFrame(PHI_real, index=pool,
                       columns=[d.label(t) for t in targets])
    mat.to_csv(datd / "contribution_matrix.csv")

    inter = np.asarray(dec["interaction"])[:p]
    pd.DataFrame(inter, index=pool,
                 columns=[d.label(t) for t in targets]).to_csv(
        datd / "interaction_matrix.csv")

    rp.heatmap(mat.to_numpy(), list(mat.index), list(mat.columns),
               figd / "fig_贡献矩阵.png", top_rows=32)

    # ---------------------------------------------- 枢纽字段
    over = np.abs(PHI_real) > ztol[None, :]
    hub = pd.DataFrame({
        "字段": pool,
        "广度_有贡献的目标数": over.sum(axis=1),
        "深度_跨目标贡献合计": np.abs(PHI_real).sum(axis=1),
        "最大单目标贡献": np.abs(PHI_real).max(axis=1),
        "最强目标": [mat.columns[i] for i in np.abs(PHI_real).argmax(axis=1)],
        "平均协同冗余指数": inter.mean(axis=1),
    }).sort_values(["广度_有贡献的目标数", "深度_跨目标贡献合计"],
                   ascending=False).reset_index(drop=True)
    hub.to_csv(datd / "hub_fields.csv", index=False)

    print(f"\n{'='*80}\n枢纽字段（同时给多个目标供料的公开字段）")
    print(hub.head(15).to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    n_all = int((hub["广度_有贡献的目标数"] == len(targets)).sum())
    print(f"\n  在**全部 {len(targets)} 个**敏感目标上都有贡献的字段：{n_all} 个")
    print(f"  只对 1 个目标有贡献的字段：{int((hub['广度_有贡献的目标数']==1).sum())} 个")
    print(f"  对任何目标都没有贡献的字段：{int((hub['广度_有贡献的目标数']==0).sum())} 个")
    print("\n  读法：广度大的字段治理价值最高——处理它一次，多个目标同时受益；"
          "\n  广度小但深度大的字段则要针对性处理。这两类不能混为一谈，"
          "\n  逐个目标单独分析时也看不出前一类。")

    # ---------------------------------------------- 联合可推断性
    # 把十二个目标的还原能力平均成一个数，再算一次贡献值：
    # 回答"这个字段对**整批**敏感目标的整体风险贡献多少"
    class JointVF:
        def __init__(self, vf):
            self.vf, self.p = vf, vf.p

        def __call__(self, masks, batch: int = 64):
            return np.asarray(self.vf(masks, batch)).mean(axis=1)

        def of_sets(self, sets):
            return np.asarray(self.vf.of_sets(sets)).mean(axis=1)

        def full(self):
            return self(np.ones((1, self.p), dtype=np.float32))[0]

        def empty(self):
            return self(np.zeros((1, self.p), dtype=np.float32))[0]

    print(f"\n{'='*80}\n联合可推断性（十二个目标的还原能力取平均）")
    rj = att.kernel_shap(JointVF(vf), len(names_a),
                         n_coalitions=a.n_coalitions, seed=a.seed)
    decj = att.decompose(rj)
    ztol_j = att.zero_band(np.asarray(decj["phi"])[p:], q=0.95)
    tabj = rp.build_table(pool, {k: np.asarray(v)[:p] for k, v in decj.items()},
                          labels=d.cn, zero_tol=ztol_j).drop(columns=["中文名"])
    tabj.to_csv(datd / "joint_contributions.csv", index=False)
    rp.contribution_bar(tabj, figd / "fig_联合贡献值.png", top=28, zero_tol=ztol_j)
    rp.risk_map(tabj, figd / "fig_联合风险坐标.png", zero_tol=ztol_j)
    print(f"  v(全部)={float(np.ravel(rj.v_full)[0]):.4f}，判零门槛 {ztol_j:.5f}")
    print(tabj.head(12).to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"\n  字段分类：{tabj['类型'].value_counts().to_dict()}")

    # ---------------------------------------------- 与单目标结果对照
    cmp_rows = []
    if a.compare_run and Path(a.compare_run).exists():
        single = pd.read_csv(a.compare_run)
        print(f"\n{'='*80}\n与单目标逐个跑出来的结果对照")
        for t in targets:
            lab = d.label(t)
            sub = single[single["目标"] == lab]
            if not len(sub):
                continue
            s1 = sub.set_index("字段")["贡献值"]
            s2 = mat[lab]
            common = [f for f in s2.index if f in s1.index]
            rho = float(spearmanr(s1.loc[common], s2.loc[common]).statistic)
            top10 = set(s1.loc[common].abs().sort_values(ascending=False)
                        .head(10).index)
            top10b = set(s2.loc[common].abs().sort_values(ascending=False)
                         .head(10).index)
            j = len(top10 & top10b) / len(top10 | top10b)
            cmp_rows.append(dict(目标=lab, 秩相关=rho, 前10名重合度=j,
                                 共同字段数=len(common)))
            print(f"  {lab:<22} 秩相关 {rho:.4f}   前10名重合度 {j:.3f}")
        if cmp_rows:
            cdf = pd.DataFrame(cmp_rows)
            cdf.to_csv(datd / "vs_single_target.csv", index=False)
            print(f"\n  平均秩相关 {cdf['秩相关'].mean():.4f}，"
                  f"平均前10名重合度 {cdf['前10名重合度'].mean():.3f}")
            print("  （高 ⇒ 共享主干没有损害精度，多目标版本可以直接替代"
                  "逐个跑，而且只要一次训练）")

    (datd / "config.json").write_text(json.dumps(
        {"数据集": a.dataset, "代理模型": cfg.to_dict(), "目标": targets,
         "候选池": p, "注入噪声字段数": a.n_noise,
         "抽取组合数": a.n_coalitions, "评估行数": a.n_eval,
         "各目标v全部": [float(x) for x in v_full],
         "各目标判零门槛": [float(x) for x in ztol],
         "总耗时秒": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base_dir}")


if __name__ == "__main__":
    main()
