"""L3 人造真值：答案由构造决定，看方法能不能答对。

为什么要造题
------------
真实数据上没有标准答案——谁也不知道"哪几个公开字段才是真正的推断来源"。
所以先造几道**答案已知**的题：用真实的电力数据列拼出目标字段，
拼法自己定，于是每个字段应该拿多少份额是可以事先写下来的。
方法答对了，才有资格拿去跑没有答案的真实题。

用真实列而不是随机数造题，是因为电力数据有很强的时间自相关和彼此相关，
用独立同分布的随机数造题会把难度降得不真实。

五道题
------
1. 可加型   y = 3a + 2b + c      份额应当按方差贡献 9:4:1 分，协同–冗余指数≈0
2. 纯协同   y = a·b              a、b 份额应当相等且很大，但**各自单独毫无用处**
3. 纯替身   池子里放 a 和 a 的副本，y = a
                                 两者份额应当**完全相等**（各一半），
                                 而各自的不可替代性应当为 0（删一个另一个顶上）
4. 汇总替代 y = a+b，池子里同时放 a、b 和汇总量 s=a+b
                                 三者都应当拿到份额：{s} 够用，{a,b} 也够用
5. 纯噪声   若干与目标独立的列    份额应当为 0；它们的分布宽度给出"多小算 0"的标尺

每道题的候选池都压到 12~14 个字段，这样 2^12 = 4096 种组合可以**全部枚举**，
得到精确的 Shapley 值，连采样误差都排除掉。

对照
----
同一道题上跑现有方法（分解式门控、两两相关性、置换重要性等），
看它们答成什么样。第 2 题和第 3 题是分水岭：
- 第 2 题里协同的两个字段两两相关性接近 0，相关性方法必然漏掉；
- 第 3 题里两个完美替身，稀疏门控只会留一个、把另一个判为零，
  而"删掉被选中的那个就能阻断"是错的——副本立刻顶上。
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
from src import baselines as bl         # noqa: E402
from src import report as rp            # noqa: E402

# 造题用的真实底料（RTS v2 的公开字段），彼此有真实的相关结构
BASE_FIELDS = [
    "area_1_load_actual_mw", "area_2_load_actual_mw", "area_3_load_actual_mw",
    "gen_fuel_coal_mw", "gen_fuel_natural_gas_mw", "gen_fuel_wind_mw",
    "gen_fuel_hydro_mw", "reserve_reg_up_mw", "reserve_spin_r1_mw",
    "hour_of_day", "system_losses_mw",
]
N_NOISE = 3


def z(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-12)


def residualize(v: np.ndarray, F: np.ndarray) -> np.ndarray:
    """把 v 里能被填充字段 F 线性预测的部分减掉，只留下残差。

    为什么必须做这一步
    ------------------
    这些题的答案是"只有甲和乙能推出目标，其余字段都不能"。
    但真实的电力字段彼此高度相关——1 区负荷和 2 区负荷、和煤电出力、
    和系统网损，都是一起涨一起落的。直接拿真实列来造题，
    "其余字段都不能"这个前提根本不成立：抽掉甲，别的字段立刻能顶上大半。
    题目就不纯了，答对答错都说明不了问题。

    减掉线性可预测的部分之后，甲保留了真实数据的时间自相关和分布形状
    （所以题目的难度是真实的），但填充字段再也无法线性还原它。
    残留的非线性依赖有多少，run_case 里会实测并报出来。
    """
    A = np.column_stack([F, np.ones(len(F))])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return z(v - A @ coef)


def make_cases(df: pd.DataFrame, seed: int = 0) -> list[dict]:
    """造出五道题。每道题返回候选池矩阵、字段名、目标、以及应有的答案。

    构造纪律（前一版在这两点上翻过车，都记在这里）：
      1. 用作"甲乙丙"的列，绝不能同时又出现在填充字段里，否则会造出意外的重复；
      2. 用作"甲乙丙"的列必须先对填充字段做正交化，否则题目不纯。
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # 填充字段：题目里"不该有贡献"的那些。
    # 这里刻意不放 hour_of_day——它是周期量，和别的字段是非线性关系，
    # 线性正交化去不掉，会从后门漏一点贡献进来（实测漏了 0.065）。
    FILL = ["gen_fuel_coal_mw", "gen_fuel_natural_gas_mw", "gen_fuel_hydro_mw",
            "reserve_reg_up_mw", "reserve_spin_r1_mw", "area_3_load_actual_mw"]
    # 造题字段：从填充池之外取
    SRC = ["area_1_load_actual_mw", "gen_fuel_wind_mw", "system_losses_mw"]
    assert not (set(FILL) & set(SRC)), "造题字段和填充字段不能重叠"

    fill = {f: z(df[f].to_numpy()) for f in FILL}
    F = np.column_stack(list(fill.values()))

    # 两道正交化，缺一不可：
    # 1) 对填充字段正交化——否则填充字段能顶替造题字段，"只有甲乙能推"不成立；
    # 2) 造题字段**彼此**也要正交化——否则"份额按 9:4:1 分"这个理论答案本身就是错的。
    #    系统网损和 1 区负荷是强相关的，不做这一步，甲和丙互相抢份额，
    #    对不上答案说明不了任何问题（前一版就栽在这里）。
    src = []
    for f in SRC:
        v = residualize(df[f].to_numpy(float), F)
        for u in src:                       # 逐个减掉在已有造题字段上的投影
            v = v - float(v @ u) / float(u @ u) * u
        src.append(z(v))
    a, b, c = src

    def noise(k: int) -> dict:
        return {f"噪声{i+1}": z(rng.standard_normal(n)) for i in range(k)}

    cases = []

    # ---- 1 可加型 ----
    cases.append(dict(
        名称="可加型", 目标式="y = 3·甲 + 2·乙 + 丙",
        cols={"甲": a, "乙": b, "丙": c} | fill | noise(N_NOISE),
        y=3 * a + 2 * b + c,
        期望=dict(有份额=["甲", "乙", "丙"], 零份额=[f"噪声{i+1}" for i in range(N_NOISE)],
                 份额比={"甲": 9, "乙": 4, "丙": 1}, 指数方向="≈0"),
        看点="三者互相独立且各自与填充字段无关，份额应当按方差贡献 9:4:1 分；"
             "协同–冗余指数应当都接近 0。"))

    # ---- 2 纯协同 ----
    cases.append(dict(
        名称="纯协同", 目标式="y = 甲 × 乙",
        cols={"甲": a, "乙": b} | fill | noise(N_NOISE),
        y=z(a * b),
        期望=dict(有份额=["甲", "乙"], 零份额=[f"噪声{i+1}" for i in range(N_NOISE)],
                 指数方向=">0", 相等=["甲", "乙"]),
        看点="甲乙份额应当相等且最大，但各自单独的还原能力接近 0，"
             "而抽掉任一个损失惨重 ⇒ 协同–冗余指数为正。"
             "两两相关性方法在这里必然失效：乘积和各自因子的相关性接近 0。"))

    # ---- 3 纯替身 ----
    cases.append(dict(
        名称="纯替身", 目标式="y = 甲（池子里另有一个和甲完全相同的副本）",
        cols={"甲": a, "甲的副本": a.copy()} | fill | noise(N_NOISE),
        y=a,
        期望=dict(有份额=["甲", "甲的副本"], 零份额=[f"噪声{i+1}" for i in range(N_NOISE)],
                 相等=["甲", "甲的副本"], 指数方向="<0"),
        看点="两者份额必须完全相等，且各自的不可替代性应当为 0。"
             "稀疏门控只会留一个、把另一个判为零——这正是现有方法的核心缺陷。"))

    # ---- 4 汇总替代 ----
    cases.append(dict(
        名称="汇总替代", 目标式="y = 甲 + 乙，池子里同时有甲、乙和汇总量(甲+乙)",
        cols={"甲": a, "乙": b, "汇总量": z(a + b)} | fill | noise(N_NOISE),
        y=z(a + b),
        期望=dict(有份额=["甲", "乙", "汇总量"],
                 零份额=[f"噪声{i+1}" for i in range(N_NOISE)], 指数方向="<0"),
        看点="{汇总量} 单独够用，{甲,乙} 合起来也够用。三者都应当拿到份额，"
             "且都是替身型。这一题对应真实数据里的分量—汇总结构——"
             "一台具名机组同时被计入四个层级的汇总量。"))

    # ---- 5 纯噪声 ----
    cases.append(dict(
        名称="纯噪声", 目标式="y = 甲，其余 10 个字段全是与目标独立的随机数",
        cols={"甲": a} | noise(10), y=a,
        期望=dict(有份额=["甲"], 零份额=[f"噪声{i+1}" for i in range(10)],
                 指数方向="≈0"),
        看点="10 个噪声字段的贡献值分布宽度，就是判零门槛的客观标尺——"
             "多小算 0 是从数据里量出来的，不是人为定的。"))

    for cse in cases:
        names = list(cse["cols"].keys())
        assert len(names) == len(set(names)), f"{cse['名称']}：字段名重复"
        cse["fields"] = names
        cse["X"] = np.column_stack([cse["cols"][k] for k in names])
        cse["构造字段"] = [k for k in names
                        if k not in FILL and not k.startswith("噪声")]
        del cse["cols"]
    return cases


def run_case(cse: dict, cfg: sg.SurrogateConfig, n_eval: int,
             exact: bool, figd: Path) -> tuple[pd.DataFrame, dict]:
    X, y, names = cse["X"], cse["y"], cse["fields"]
    p = len(names)
    print(f"\n{'='*70}")
    print(f"【{cse['名称']}】{cse['目标式']}")
    print(f"候选池 {p} 个字段：{names}")
    print(f"看点：{cse['看点']}")

    t0 = time.time()
    res = sg.fit(X, y, names, cfg)
    vf = sg.ValueFunction(res, X, y, n_eval=n_eval, seed=0)
    v_full, v_empty = float(vf.full()), float(vf.empty())
    print(f"  代理模型训练完毕（{time.time()-t0:.0f}s，最佳轮次 {res.best_epoch}）"
          f"  v(全部)={v_full:.4f}  v(空)={v_empty:.4f}")

    # 题目纯不纯：只发布"不该有贡献"的那些字段，目标能被还原到什么程度？
    # 这个数必须接近 0，否则"只有甲乙能推出目标"这个前提就不成立，题目白造了。
    # 直接查 v(不该有贡献的字段全集) 就是最贴切的度量，而且不额外花钱。
    src_set = set(cse.get("构造字段", []))
    fill_idx = [i for i, nm in enumerate(names) if nm not in src_set]
    purity = {}
    if fill_idx:
        m = np.zeros((1, p), dtype=np.float32)
        m[0, fill_idx] = 1.0
        purity["非来源字段全集的还原能力"] = float(vf(m)[0])
        print(f"  题目纯度检验：只发布全部非来源字段（{len(fill_idx)} 个），"
              f"目标还原能力 v = {purity['非来源字段全集的还原能力']:.4f}（应当接近 0）")

    if exact and p <= 16:
        print(f"  穷举全部 2^{p} = {2**p} 种组合，算精确 Shapley 值…")
        phi = att.exact_shapley(vf, p)
        eye = np.eye(p, dtype=np.float32)
        solo = vf(eye) - v_empty
        marg = v_full - vf(1.0 - eye)
        r = att.ShapResult(phi=phi, v_full=v_full, v_empty=v_empty,
                           n_coalitions=2 ** p,
                           efficiency_gap=phi.sum() - (v_full - v_empty),
                           solo=solo, marginal=marg)
        mode = f"穷举精确（2^{p}）"
    else:
        r = att.kernel_shap(vf, p, n_coalitions=8192, seed=0)
        mode = "KernelSHAP 8192"
    dec = att.decompose(r)
    print(f"  贡献值算法：{mode}，加和性偏差 {float(np.ravel(r.efficiency_gap)[0]):.2e}")

    # 判零门槛：用本题里的噪声字段量出来
    noise_idx = [i for i, nm in enumerate(names) if nm.startswith("噪声")]
    ztol = att.zero_band(dec["phi"][noise_idx]) if noise_idx else 0.0

    tab = rp.build_table(names, dec, zero_tol=ztol)
    tab = tab.drop(columns=["中文名"])

    # 对照方法
    print("  跑对照方法…")
    base = bl.run_all(X, y, seed=cfg.seed)
    for mname, sc in base.items():
        s = pd.Series(sc, index=names)
        tab[mname] = tab["字段"].map(s)
        # 归一到 0~1 便于横向看排名
        mx = tab[mname].abs().max()
        tab[mname + "_归一"] = tab[mname].abs() / (mx if mx > 1e-12 else 1.0)

    print(f"\n  判零门槛（由 {len(noise_idx)} 个噪声字段量出）= {ztol:.4f}")
    show = ["字段", "贡献值", "归一化份额", "独立能力", "不可替代性",
            "协同冗余指数", "类型"] + \
           [c for c in tab.columns if c.endswith("_归一")]
    print(tab[show].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

    # ---- 逐题核对答案 ----
    verdict = {"名称": cse["名称"], "目标式": cse["目标式"], "v_full": v_full,
               "题目纯度": purity,
               "判零门槛": ztol, "贡献值算法": mode}
    exp = cse["期望"]
    idx = {nm: i for i, nm in enumerate(names)}
    phi = np.asarray(dec["phi"])
    inter = np.asarray(dec["interaction"])

    print("\n  ---- 核对 ----")
    if "零份额" in exp and exp["零份额"]:
        mx = max(abs(phi[idx[k]]) for k in exp["零份额"])
        mn = min(abs(phi[idx[k]]) for k in exp["有份额"])
        ok = mx < mn
        verdict["噪声最大份额"] = float(mx)
        verdict["真源最小份额"] = float(mn)
        verdict["噪声与真源完全分开"] = bool(ok)
        print(f"  {'✓' if ok else '✗'} 无关字段最大份额 {mx:.4f} "
              f"< 真实来源最小份额 {mn:.4f}")

    if "相等" in exp:
        vals = [phi[idx[k]] for k in exp["相等"]]
        rel = abs(vals[0] - vals[1]) / max(abs(vals[0]), 1e-12)
        verdict["替身份额相对差"] = float(rel)
        print(f"  {'✓' if rel < 0.15 else '✗'} 两个对称字段的份额 "
              f"{vals[0]:.4f} vs {vals[1]:.4f}，相对差 {rel:.1%}（应当很小）")
        mg = [float(dec['marginal'][idx[k]]) for k in exp["相等"]]
        verdict["替身不可替代性"] = mg
        print(f"      两者的不可替代性 {mg[0]:.4f} / {mg[1]:.4f}（应当接近 0）")

    if "份额比" in exp:
        ks = list(exp["份额比"])
        got = np.array([phi[idx[k]] for k in ks])
        want = np.array([exp["份额比"][k] for k in ks], dtype=float)
        got_n, want_n = got / got.sum(), want / want.sum()
        verdict["份额比_实际"] = got_n.round(4).tolist()
        verdict["份额比_理论"] = want_n.round(4).tolist()
        err = float(np.max(np.abs(got_n - want_n)))
        print(f"  {'✓' if err < 0.10 else '✗'} 份额比 实际 {got_n.round(3)} "
              f"vs 理论 {want_n.round(3)}，最大偏差 {err:.3f}")

    if exp.get("指数方向") in (">0", "<0"):
        keys = exp["有份额"]
        vals = [float(inter[idx[k]]) for k in keys]
        ok = all(v > 0 for v in vals) if exp["指数方向"] == ">0" \
            else all(v < 0 for v in vals)
        verdict["指数"] = dict(zip(keys, [round(v, 4) for v in vals]))
        print(f"  {'✓' if ok else '✗'} 协同–冗余指数 "
              f"{dict(zip(keys, [round(v,3) for v in vals]))} 应当 {exp['指数方向']}")

    # 对照方法在这道题上错在哪
    print("\n  ---- 对照方法的表现 ----")
    for mname in base:
        s = pd.Series(np.abs(base[mname]), index=names)
        rank = s.rank(ascending=False)
        true_src = exp["有份额"]
        pos = [int(rank[k]) for k in true_src]
        verdict[f"对照_{mname}_真源排名"] = pos
        print(f"  {mname:<10} 真实来源 {true_src} 的排名 {pos}"
              f"（理想是 {list(range(1, len(true_src)+1))}）")
        if "相等" in exp:
            v2 = [float(s[k]) for k in exp["相等"]]
            rel2 = abs(v2[0] - v2[1]) / max(abs(v2[0]), abs(v2[1]), 1e-12)
            verdict[f"对照_{mname}_替身相对差"] = float(rel2)
            print(f"             两个对称字段的分数 {v2[0]:.4f} vs {v2[1]:.4f}，"
                  f"相对差 {rel2:.1%}")

    rp.contribution_bar(tab, figd / f"fig_贡献值_{cse['名称']}.png",
                        top=len(tab), zero_tol=ztol)
    rp.risk_map(tab, figd / f"fig_风险坐标_{cse['名称']}.png", zero_tol=ztol)
    rp.solo_vs_marginal(tab, figd / f"fig_独立vs不可替代_{cse['名称']}.png")
    return tab, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--hidden", default="256,192,128")
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--no-exact", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    t_start = time.time()
    d = ds.load(a.dataset)
    base_dir, figd, datd = ds.run_dir(a.dataset, "L3_synthetic")
    print(f"输出目录 {base_dir}")

    cfg = sg.SurrogateConfig(
        hidden=tuple(int(x) for x in a.hidden.split(",")),
        epochs=a.epochs, min_epochs=min(150, a.epochs), patience=150, seed=42)
    print(f"代理模型配置：网络 {cfg.hidden}，轮次上限 {cfg.epochs}\n")

    cases = make_cases(d.df, seed=0)
    if a.only:
        cases = [c for c in cases if c["名称"] in a.only]

    tabs, verdicts = {}, []
    for cse in cases:
        tab, vd = run_case(cse, cfg, a.n_eval, not a.no_exact, figd)
        tab.insert(0, "题目", cse["名称"])
        tabs[cse["名称"]] = tab
        verdicts.append(vd)

    allt = pd.concat(tabs.values(), ignore_index=True)
    allt.to_csv(datd / "synthetic_contributions.csv", index=False)
    (datd / "verdicts.json").write_text(
        json.dumps(verdicts, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    (datd / "config.json").write_text(json.dumps(
        {"数据集": a.dataset, "代理模型": cfg.to_dict(),
         "评估行数": a.n_eval, "穷举精确解": not a.no_exact,
         "总耗时秒": round(time.time() - t_start, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*70}\n总耗时 {(time.time()-t_start)/60:.1f} 分钟 → {base_dir}")


if __name__ == "__main__":
    main()
