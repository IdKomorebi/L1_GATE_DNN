"""主实验：一个数据集的全部敏感目标，贡献值 + 分解 + 对照 + 重训认证。

产出四样东西
------------
1. 每个目标一张贡献值表：份额、独立能力、不可替代性、协同–冗余指数、类型
2. 风险坐标图：横轴份额、纵轴指数。右下是"有替身的高风险字段"——
   删掉它们没用，别人顶上；右上是"协同型"——两两相关性完全看不出来
3. 重训认证曲线：按各方法的排序取前 k 个字段，用无门控的普通网络独立重训，
   看 R² 随 k 怎么涨。**这是沿用仓库既有口径的公平比较**
4. 两条通道的对照：
   通道 A（便宜）随机屏蔽下训练的分解式门控，读它的门控值
   通道 B（有理论保证）从 v(S) 解出的 Shapley 贡献值
   两者秩相关高 ⇒ 便宜的门控读数可以近似昂贵的贡献值，是个有用的结论；
   相关低 ⇒ 以 B 为准，A 降为消融。两种结果都能写，都不算失败

关于重训认证要说清楚的一点
--------------------------
本方法**不是为"用最少字段达到最高精度"设计的**，分解式门控才是。
所以在前 k 个字段的 R² 这一项上未必赢，这一点如实报告。
本方法赢的是别的地方：排序合理、补集也被正确刻画、
以及能说清楚每个字段属于哪一类。
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
from src import baselines as bl         # noqa: E402
from src import report as rp            # noqa: E402
from src import truthsets as ts         # noqa: E402


def topk_curve(X, y, order, ks, cfg, eval_idx):
    """按给定排序取前 k 个字段，独立重训一个普通网络，返回各 k 的 R²。"""
    return [sg.retrain_reference(X, y, list(order[:k]), cfg, eval_idx) for k in ks]


def run_target(d, tgt, cfg, a, figd, datd):
    pool = d.pool(tgt, drop_constants=not a.keep_constants)
    X = d.df[pool].to_numpy(float)
    y = d.df[tgt].to_numpy(float)
    p = len(pool)
    print(f"\n{'='*78}\n目标 {d.label(tgt)}（{tgt}），候选池 {p} 个字段")

    t0 = time.time()
    # ---- 通道 B：代理模型 + Shapley ----
    # 判零门槛要靠"与目标无关的字段实际拿到多少份额"来量。做法是把若干列
    # 随机数**一起放进候选池**，从头到尾和真实字段同等对待：同一个模型、
    # 同一次贡献值计算。这样量出来的门槛才真的代表"纯属估计误差能有多大"，
    # 而且省掉了单独再训一个模型的开销。
    rng = np.random.default_rng(a.seed + 7)
    n_noise = a.n_noise
    Xa = np.column_stack([X, rng.standard_normal((len(X), n_noise))])
    names_a = pool + [f"__噪声{i+1}" for i in range(n_noise)]

    res = sg.fit(Xa, y, names_a, cfg)
    vf = sg.ValueFunction(res, Xa, y, n_eval=a.n_eval, seed=0)
    v_full, v_empty = float(vf.full()), float(vf.empty())
    r = att.kernel_shap(vf, len(names_a), n_coalitions=a.n_coalitions,
                        seed=a.seed)
    dec_all = att.decompose(r)
    phi_noise = np.asarray(dec_all["phi"])[p:]
    inter_noise = np.asarray(dec_all["interaction"])[p:]
    ztol = att.zero_band(phi_noise, q=0.95)
    # "指数多接近 0 算独立可加"这个门槛也由噪声字段量出来，不用拍的数字。
    # 噪声字段的独立能力和不可替代性真值都是 0，所以指数真值也是 0，
    # 观测到的散布完全来自估计误差。
    itol = att.interaction_band(inter_noise, q=0.95)
    dec = {k: np.asarray(v)[:p] for k, v in dec_all.items()}
    print(f"  通道B 代理模型+贡献值：{time.time()-t0:.0f}s  "
          f"v(全部)={v_full:.4f}  v(空)={v_empty:.4f}  "
          f"加和性偏差={float(np.ravel(r.efficiency_gap)[0]):.1e}")
    print(f"  判零门槛 {ztol:.5f}，指数判零门槛 {itol:.5f}"
          f"（都由同池的 {n_noise} 个噪声字段量出；"
          f"它们的份额范围 {phi_noise.min():.4f}~{phi_noise.max():.4f}）")

    # ---- 摊销代价：代理模型说的"全部发布能推到多少"，和专门为这批字段
    # 训练一个普通网络得到的 R² 差多少。这一条必须查，而且必须报出来。
    # 代理模型要同时应付所有字段组合，在任何单个组合上都不如专训的模型；
    # 目标越难预测，这个亏空越大。实测 PJM 的交换功率类目标在时序划分下
    # 专训网络能到 0.34，而代理模型的 v(全部) 是 −1.21——差了 1.5 个 R²。
    # 不查这一条就直接分摊，会把"代理模型的能力上限"错当成"可推断性"。
    r_plain = sg.retrain_reference(X, y, list(range(p)), cfg, vf.idx)
    gap = r_plain - v_full
    print(f"  摊销代价：专训普通网络 R²={r_plain:.4f}，代理模型 v(全部)={v_full:.4f}，"
          f"差 {gap:+.4f}")
    if gap > 0.10:
        print(f"  [警告] 差距 {gap:.3f} 过大。代理模型在这个目标上没能逼近"
              f"专训模型的水平，下面的贡献值分摊的是一个偏弱的函数，"
              f"结论要打折扣，必须在报告里标出来。")

    tab = rp.build_table(pool, dec, labels=d.cn, zero_tol=ztol,
                         inter_tol=itol).drop(columns=["中文名"])

    # ---- 通道 A：随机屏蔽下的分解式门控 ----
    cfg_g = sg.SurrogateConfig(**{**cfg.to_dict(), "use_gate": True,
                                  "hidden": tuple(cfg.hidden)})
    t1 = time.time()
    res_g = sg.fit(Xa, y, names_a, cfg_g)          # 同一个增广池，下标才对得上
    gate_all = np.asarray(res_g.gates, dtype=float)
    gate = gate_all[:p]
    tab["通道A门控值"] = tab["字段"].map(pd.Series(gate, index=pool))
    rho_ab = float(spearmanr(tab["贡献值"], tab["通道A门控值"]).statistic)
    n_zero_a = int((gate == 0).sum())
    print(f"  通道A 屏蔽下门控：{time.time()-t1:.0f}s  "
          f"精确为零 {n_zero_a}/{p}  与贡献值秩相关 {rho_ab:.4f}  "
          f"（噪声字段的门控值 {np.round(gate_all[p:], 4)}）")

    # ---- 对照方法 ----
    t1 = time.time()
    base = bl.run_all(X, y, seed=cfg.seed)
    for m, sc in base.items():
        tab[m] = tab["字段"].map(pd.Series(sc, index=pool))
    print(f"  对照方法：{time.time()-t1:.0f}s")

    # ---- 重训认证曲线 ----
    curves = {}
    if not a.skip_topk:
        t1 = time.time()
        ks = [k for k in [1, 2, 3, 5, 8, 12, 18, 25, 35, p] if k <= p]
        rank_sources = {"本方法贡献值": tab.sort_values("贡献值", ascending=False)["字段"],
                        "通道A门控值": tab.reindex(
                            tab["通道A门控值"].abs().sort_values(
                                ascending=False).index)["字段"]}
        for m in base:
            rank_sources[m] = tab.reindex(
                tab[m].abs().sort_values(ascending=False).index)["字段"]
        for name, order in rank_sources.items():
            idx = [pool.index(f) for f in order]
            curves[name] = (ks, topk_curve(X, y, idx, ks, cfg, vf.idx))
        # 补集曲线：按贡献值**从小到大**取前 k
        idx_bot = [pool.index(f) for f in
                   tab.sort_values("贡献值")["字段"]]
        curves["本方法·最低的k个"] = (ks, topk_curve(X, y, idx_bot, ks, cfg, vf.idx))
        print(f"  重训认证曲线：{time.time()-t1:.0f}s，"
              f"{len(curves)} 条 × {len(ks)} 个点")
        rp.topk_curve(curves, figd / f"fig_重训认证_{tgt}.png", full_r2=v_full)

    # ---- 出图与存表 ----
    tab.insert(0, "目标", d.label(tgt))
    rp.contribution_bar(tab, figd / f"fig_贡献值_{tgt}.png", top=28, zero_tol=ztol)
    rp.risk_map(tab, figd / f"fig_风险坐标_{tgt}.png", zero_tol=ztol,
                inter_tol=itol)
    rp.solo_vs_marginal(tab, figd / f"fig_独立vs不可替代_{tgt}.png")
    rp.training_curve(res.history, figd / f"fig_训练曲线_{tgt}.png")

    cls = tab["类型"].value_counts().to_dict()
    print(f"  字段分类：{cls}")
    if v_full < 0.3:
        print(f"  [注意] v(全部)={v_full:.4f} 很低，这个目标在当前划分口径下"
              f"基本推不出来。贡献值分摊的是 v(全部)−v(空)={v_full-v_empty:.4f}，"
              f"数值仍自洽，但'谁是推断源'这个问题在这里意义不大，"
              f"结论里要单独标出。")
    top = tab.head(6)[["字段", "贡献值", "独立能力", "不可替代性",
                       "协同冗余指数", "类型"]]
    print(top.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    summary = dict(
        目标=d.label(tgt), 字段名=tgt, 候选数=p,
        v_全部=v_full, v_空=v_empty,
        加和性偏差=float(np.ravel(r.efficiency_gap)[0]),
        判零门槛=ztol, 指数判零门槛=itol,
        超过判零门槛的字段数=int((tab["贡献值"].abs() > ztol).sum()),
        替身型=cls.get("替身型", 0), 协同型=cls.get("协同型", 0),
        独立可加=cls.get("独立可加", 0), 无贡献=cls.get("无贡献", 0),
        通道A精确零点=n_zero_a, 通道AB秩相关=rho_ab,
        通道A噪声字段门控均值=float(np.mean(gate_all[p:])),
        噪声字段份额范围=[float(phi_noise.min()), float(phi_noise.max())],
        前3份额占比=float(tab.head(3)["贡献值"].sum() /
                      max(tab["贡献值"].sum(), 1e-12)),
        专训普通网络R2=r_plain, 摊销代价=gap,
        可分析=bool(v_full >= 0.3 and gap <= 0.10),
        耗时秒=round(time.time() - t0, 1))
    if curves:
        for name, (ks, v) in curves.items():
            summary[f"曲线_{name}"] = [round(x, 4) for x in v]
        summary["曲线_k"] = ks
    return tab, summary, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--n-coalitions", type=int, default=32768)
    ap.add_argument("--n-noise", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=1000)
    # 主口径是时序划分：用历史训练、推断新数据，对应"外部只有历史公开数据"。
    # 随机划分对应"外部已掌握同批次的部分配对样本"，是仓库既有主结果用的口径，
    # 两者都要报——同一批字段在两种攻击者假设下的风险是不一样的。
    ap.add_argument("--split", default="temporal", choices=["temporal", "random"])
    ap.add_argument("--skip-topk", action="store_true")
    ap.add_argument("--keep-constants", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    stage = "main" if a.split == "temporal" else f"main_{a.split}split"
    base_dir, figd, datd = ds.run_dir(a.dataset, stage)
    print(f"输出目录 {base_dir}")
    print(f"数据集 {d.name}：{d.df.shape}，敏感目标 {len(d.targets)} 个")

    cfg = sg.SurrogateConfig(epochs=a.epochs, seed=a.seed, split=a.split)
    print(f"代理模型配置：{cfg.hidden}，{cfg.epochs} 轮，划分方式 {cfg.split}，"
          f"权重衰减 {cfg.weight_decay}，掩码分布 {cfg.mask_dist}")
    print(f"抽取组合数 {a.n_coalitions}，评估行数 {a.n_eval}")

    targets = a.targets or d.targets
    tabs, summaries = [], []
    for i, tgt in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}]", end="")
        tab, summ, _ = run_target(d, tgt, cfg, a, figd, datd)
        tabs.append(tab)
        summaries.append(summ)
        # 每个目标跑完就落盘，长任务中途被打断也不至于全丢
        pd.concat(tabs, ignore_index=True).to_csv(
            datd / "contributions_all.csv", index=False)
        pd.DataFrame(summaries).to_csv(datd / "summary.csv", index=False)

    allt = pd.concat(tabs, ignore_index=True)
    summ = pd.DataFrame(summaries)

    print(f"\n{'='*78}\n汇总")
    show = ["目标", "候选数", "v_全部", "专训普通网络R2", "摊销代价", "可分析",
            "判零门槛", "超过判零门槛的字段数",
            "替身型", "协同型", "独立可加", "无贡献", "通道A精确零点",
            "通道AB秩相关", "前3份额占比"]
    print(summ[show].to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    bad = summ[~summ["可分析"]] if "可分析" in summ.columns else summ.iloc[:0]
    if len(bad):
        print(f"\n[警告] 有 {len(bad)} 个目标不适合直接引用结论"
              f"（v(全部) 太低或摊销代价 > 0.10）：")
        print(bad[["目标", "v_全部", "专训普通网络R2", "摊销代价"]].to_string(
            index=False, float_format=lambda x: f"{x:8.4f}"))

    print(f"\n通道 A 与通道 B 的秩相关：均值 {summ['通道AB秩相关'].mean():.4f}，"
          f"范围 {summ['通道AB秩相关'].min():.4f}~{summ['通道AB秩相关'].max():.4f}")

    # 字段 × 目标的贡献矩阵（枢纽字段的原料）
    piv = allt.pivot_table(index="字段", columns="目标", values="贡献值")
    piv.to_csv(datd / "contribution_matrix.csv")
    rp.heatmap(piv.to_numpy(), list(piv.index), list(piv.columns),
               figd / "fig_贡献矩阵.png", top_rows=30)
    hub = piv.abs().sum(axis=1).sort_values(ascending=False)
    print(f"\n跨目标贡献总和最高的 10 个字段（枢纽字段）：")
    for f, v in hub.head(10).items():
        n_hit = int((piv.loc[f].abs() > 0.01).sum())
        print(f"  {f:<34} 合计 {v:.4f}，在 {n_hit}/{len(piv.columns)} 个目标上有贡献")
    hub.to_frame("跨目标贡献合计").to_csv(datd / "hub_fields.csv")

    (datd / "config.json").write_text(json.dumps(
        {"数据集": a.dataset, "代理模型": cfg.to_dict(),
         "抽取组合数": a.n_coalitions, "评估行数": a.n_eval,
         "注入噪声字段数": a.n_noise, "目标": targets,
         "总耗时秒": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base_dir}")


if __name__ == "__main__":
    main()
