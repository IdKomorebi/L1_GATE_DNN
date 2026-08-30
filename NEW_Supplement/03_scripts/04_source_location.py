"""第 4 步：隐性推断源定位。

对一个目标字段，用门控模型找出哪几个字段承担主要推断作用，再把选中的字段
单独拿出来重新训练一个普通 DNN，确认它们本身确实携带足够信息。

目录组织：

  04_source_location/                     候选池含其余关注字段
  04b_source_location_excl_targets/       候选池排除其余关注字段
    ├── no_screening/                     未做初筛
    └── with_screening/                   经过初筛
        └── target_<字段名>/
            └── <方法名>/
                └── run_<时间戳>/
                    ├── config.json       本次运行的全部参数
                    ├── summary.md        给人看的总结
                    └── *.png / *.csv     图和明细
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_src"))
import dataio  # noqa: E402
import gates  # noqa: E402
import identity as idt  # noqa: E402
import plots  # noqa: E402

VERSION = "main"
TARGETS = [
    "net_actual_interchange_mw", "gross_actual_interchange_mw",
    "net_sched_interchange_mw", "total_gen", "metered_load_mw", "total_losses",
    "congestion_price_da", "congestion_price_rt", "marginal_loss_price_da",
    "total_lmp_da", "da_as_total_mw_primary_reserve",
    "da_as_total_mw_thirty_minutes_reserve",
]
CN = {
    "net_actual_interchange_mw": "净实际交换功率", "gross_actual_interchange_mw": "总实际交换功率",
    "net_sched_interchange_mw": "净计划交换功率", "total_gen": "总发电量",
    "metered_load_mw": "计量负荷", "total_losses": "总网损",
    "congestion_price_da": "日前阻塞价", "congestion_price_rt": "实时阻塞价",
    "marginal_loss_price_da": "日前边际损耗价", "total_lmp_da": "日前总电价",
    "da_as_total_mw_primary_reserve": "日前主用备用总量",
    "da_as_total_mw_thirty_minutes_reserve": "日前30分钟备用总量",
}
FIELD_CN = {
    "total_gen": "总发电量", "total_losses": "总网损", "metered_load_mw": "计量负荷",
    "prelim_load_avg_hourly": "预估小时负荷", "total_pjm_rt_load_mwh": "实时负荷",
    "forecast_load_mw_latest_available": "最新负荷预测",
    "wind_generation_mw": "风电出力", "solar_generation_mw": "光伏出力",
    "net_sched_interchange_mw": "净计划交换", "net_inadv_interchange_mw": "净非计划交换",
    "gross_actual_interchange_mw": "总实际交换", "gross_sched_interchange_mw": "总计划交换",
    "gross_inadv_interchange_mw": "总非计划交换", "net_actual_interchange_mw": "净实际交换",
    "total_lmp_da": "日前总电价", "total_lmp_rt": "实时总电价",
    "congestion_price_da": "日前阻塞价", "congestion_price_rt": "实时阻塞价",
    "system_energy_price_da": "日前能量价格", "system_energy_price_rt": "实时能量价格",
    "marginal_loss_price_da": "日前边际损耗价", "marginal_loss_price_rt": "实时边际损耗价",
    "da_as_nsr_mw_primary_reserve": "非同步备用",
    "da_as_as_req_mw_primary_reserve": "日前主用备用需求",
    "da_as_as_req_mw_synchronized_reserve": "日前同步备用需求",
    "da_as_as_req_mw_thirty_minutes_reserve": "日前30分钟备用需求",
    "da_as_as_mw_primary_reserve": "日前主用备用实际量",
    "da_as_as_mw_synchronized_reserve": "日前同步备用实际量",
    "da_as_as_mw_thirty_minutes_reserve": "日前30分钟备用实际量",
    "da_as_total_mw_primary_reserve": "日前主用备用总量",
    "da_as_total_mw_synchronized_reserve": "日前同步备用总量",
    "da_as_total_mw_thirty_minutes_reserve": "日前30分钟备用总量",
    "da_as_ss_mw_primary_reserve": "日前主用备用自调度",
    "da_as_ss_mw_synchronized_reserve": "日前同步备用自调度",
    "da_as_ss_mw_thirty_minutes_reserve": "日前30分钟备用自调度",
    "da_as_mcp_primary_reserve": "日前主用备用出清价",
    "da_as_mcp_synchronized_reserve": "日前同步备用出清价",
    "da_as_mcp_thirty_minutes_reserve": "日前30分钟备用出清价",
    "da_as_nsr_mw_primary_reserve": "非同步备用",
    "da_as_ircmwt2_primary_reserve": "日前主用备用IRC",
    "da_as_ircmwt2_synchronized_reserve": "日前同步备用IRC",
    "da_as_ircmwt2_thirty_minutes_reserve": "日前30分钟备用IRC",
    "rmccp": "调频容量出清价",
    "rmpcp": "调频性能出清价",
    "total_pjm_rt_load_mwh": "实时负荷电量",
    "total_pjm_loc_credit": "机会成本信用",
    "total_pjm_reg_purchases": "调频购买量",
    "total_pjm_self_sched_reg": "调频自调度量",
    "total_pjm_assigned_reg": "调频分配量",
    "total_pjm_rmccp_cr": "调频容量信用额",
    "total_pjm_rmpcp_cr": "调频性能信用额",
    "forecast_load_mw_day_ahead": "日前负荷预测",
}
for f in ["coal", "gas", "hydro", "nuclear", "oil", "solar", "wind",
          "storage", "multiple_fuels", "other_renewables"]:
    z = {"coal": "燃煤", "gas": "燃气", "hydro": "水电", "nuclear": "核电", "oil": "燃油",
         "solar": "光伏", "wind": "风电", "storage": "储能",
         "multiple_fuels": "多燃料", "other_renewables": "其他可再生"}[f]
    FIELD_CN[f"gen_fuel_{f}_mw"] = f"{z}出力"
    FIELD_CN[f"gen_fuel_{f}_pct"] = f"{z}占比"


def cn(f: str) -> str:
    return FIELD_CN.get(f, f)


def quality_drop(df, target):
    ids, _ = idt.extract(df, tol_rank=1e-5, tol_resid=1e-3)
    const = {i.lead for i in ids if i.kind == "constant"}
    groups: list[set[str]] = []
    for i in ids:
        if i.kind != "duplicate":
            continue
        p = {i.lead, *i.support}
        h = [g for g in groups if g & p]
        groups = ([g for g in groups if g not in h] + [set().union(*h, p)]) if h else groups + [p]
    dup = []
    for g in groups:
        keep = target if target in g else sorted(g)[0]
        dup += [m for m in sorted(g) if m != keep]
    return sorted((const | set(dup)) - {target})


def class1_layers(target: str, excl_targets: bool, year: int,
                  version: str | None = None) -> pd.DataFrame:
    """读第 2d 步逐层剥离的结果。候选池含不含其余关注字段，要对应到不同的产出目录。"""
    stage = ("02b_class1_identity_excl_targets" if excl_targets
             else "02_class1_identity")
    f = (dataio.OUTPUTS / (version or f"pjm_{year}") / stage /
         "data" / "layered_strip_layers.csv")
    if not f.exists():
        raise SystemExit(f"缺少第一类关系结果 {f}，请先运行 02d_layered_strip.py")
    d = pd.read_csv(f)
    return d[d.target == target].sort_values("layer")


def build_pool(df, target, excl_targets: bool, screening: bool, year: int,
               strip_class1: bool = False, version: str | None = None):
    pool = [c for c in df.columns if c != target and c not in quality_drop(df, target)]
    note = []
    if excl_targets:
        pool = [c for c in pool if c not in TARGETS]
        note.append("排除其余关注字段")
    if strip_class1:
        # 第一层：按已知公式剥离，不看残差。依据是官方文档与电力常识，不需要阈值。
        fdrop, fhit = idt.formula_drop(target)
        fdrop = [c for c in fdrop if c in pool]
        pool = [c for c in pool if c not in fdrop]
        # 第二层：按数据发现的关系逐层剥离，门槛 0.10
        lay = class1_layers(target, excl_targets, year, version)
        ldrop = [c for c in lay.removed.dropna().tolist() if c in pool]
        pool = [c for c in pool if c not in ldrop]
        note.append(f"剥离第一类关系（已知公式 {len(fdrop)} 个 + 数据发现 {len(ldrop)} 个）")
        build_pool.last_layers = {"formula": fdrop, "formula_hit": fhit,
                                  "layered": ldrop}
    if screening:
        p = (dataio.OUTPUTS / (version or f"pjm_{year}") / "03_screening" /
             "data" / f"screen_{target}.csv")
        if not p.exists():
            raise SystemExit(f"缺少初筛结果 {p}，请先运行 03_screening.py")
        keep = set(pd.read_csv(p).query("kept == 1").field)
        pool = [c for c in pool if c in keep]
        note.append("经过多指标初筛")
    return pool, "；".join(note) if note else "未做额外限制"


def write_class1_doc(path: Path, df, target: str, excl_targets: bool,
                     year: int, pool_before: list[str], pool_after: list[str],
                     version: str | None = None) -> None:
    """把该目标字段的公式/近似关系逐层写清楚：每层是什么关系、误差多大、
    删了哪个字段、删完之后还能被线性回归推到多准。"""
    lay = class1_layers(target, excl_targets, year, version)
    info = getattr(build_pool, "last_layers", {})
    L = [f"# 第一类关系剥离过程：{CN.get(target, target)}", "",
         f"候选池设定：{'排除其余关注字段' if excl_targets else '含其余关注字段'}", "",
         "剥离分两层：**第一层按已知公式，不看残差**；第二层按数据发现的关系逐层剥离。", "",
         "## 第一层：按已知公式剥离", ""]
    if info.get("formula"):
        L += [f"该目标命中了这几条已知公式：{'、'.join(info.get('formula_hit', []))}", "",
              "因此剔除以下字段（依据是官方文档与电力常识，不需要任何阈值）：", "",
              "| 字段 | 业务含义 |", "|---|---|"]
        L += [f"| `{c}` | {cn(c)} |" for c in info["formula"]]
        L += ["", "这一层堵的是这样一个漏洞：某些公式分量在当前候选池里的残差可能刚好"
              "高于按残差剥离的门槛而漏过去（例如日前总电价的 `总电价 ≈ 能量价格` "
              "残差 0.116，高于门槛 0.10），导致「剥离之后」仍能被一两个字段推得很准。", ""]
    else:
        L += ["该目标不在任何已知公式中，本层未剔除字段。", ""]
    L += ["## 第二层：按数据发现的关系逐层剥离", ""]
    if lay.empty:
        L += ["本层一层都挖不出来——用剩余字段整体拟合该目标，残差比已高于门槛 0.10。", "",
              "## 结论：第二层没有可剥离的关系", "",
              "逐层剥离一层都挖不出来——用剩余字段整体拟合该目标，"
              "残差比已经高于判定门槛 0.10（对应 R² 低于 0.99），"
              "说明它不存在「少数几个字段就能算出来」的捷径。", "",
              f"因此候选池不做任何剥离，仍为 {len(pool_before)} 个字段。", ""]
        path.write_text("\n".join(L), encoding="utf-8")
        return

    bands = lay.band.value_counts()
    L += [f"## 一、这个目标一共有 {len(lay)} 层关系", "",
          "按误差量级归档：", ""]
    L += [f"- {k}：{v} 层" for k, v in bands.items()]
    L += ["", "档次的划分标准：精确公式（相对残差 < 1e-6）、含舍入的公式（1e-6 ~ 1e-3）、",
          "高精度近似（1e-3 ~ 3e-2）、近似关系（3e-2 ~ 1e-1）。",
          "相对残差 = 拟合剩下的误差 ÷ 该目标字段自身的波动幅度。", "",
          "## 二、一层一层是怎么挖出来的", "",
          "每一层的做法：用当时还剩下的全部字段拟合目标，"
          "再从全字段出发逐个删掉贡献最小的，缩到最小的一组；",
          "然后只删掉这组里贡献最大的那**一个**字段，进入下一层看还能不能再找出关系。",
          "只删一个而不是整组删掉，是为了让剩下的字段有机会自己凑出新路径，这样才挖得深。", ""]
    for r in lay.itertuples():
        L += [f"### 第 {r.layer} 层　[{r.band}]　相对残差 {r.residual_ratio:.2e}"
              f"（相当于 R² = {1 - r.residual_ratio ** 2:.4f}）", "",
              "```", str(r.relation), "```", "",
              f"- 这条关系用到 {r.n_support} 个字段",
              f"- **删掉贡献最大的 `{r.removed}`（{cn(r.removed)}）**",
              f"- 删完之后候选池剩 {r.n_pool_after} 个字段，"
              f"目标仍能被线性回归推到 R² = **{r.test_r2_after:.4f}**", ""]

    dropped = lay.removed.dropna().tolist()
    L += ["## 三、最终从候选池里剥掉了这些字段", "",
          "| 层 | 字段 | 业务含义 | 剥离后线性 R² |", "|---|---|---|---|"]
    for r in lay.itertuples():
        L.append(f"| {r.layer} | `{r.removed}` | {cn(r.removed)} | {r.test_r2_after:.4f} |")
    kept = [c for c in dropped if c not in pool_before]
    L += ["", f"候选池：**{len(pool_before)} → {len(pool_after)} 个字段**"
          f"（实际剥掉 {len(pool_before) - len(pool_after)} 个"
          + (f"，另有 {len(kept)} 个本就不在池中" if kept else "") + "）。", "",
          "## 四、为什么必须先做这一步", "",
          "这些关系有公式可循，属于已知的、可以靠规则枚举禁止的风险，"
          "不是本文要找的隐性推断源。",
          "如果不剥离就直接跑门控模型，模型一定优先选中这些字段——因为它们推断得最准——",
          "真正没有公式可循的推断路径就被完全盖住了。", "",
          f"剥离之后目标仍能被推到 R² = "
          f"{lay.test_r2_after.iloc[-1]:.4f}，**这部分才是隐性推断的部分**。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def proxy_report(df, active: list[str], excluded: list[str],
                 tol: float = 0.10) -> list[dict]:
    """找出选中字段里，哪些其实是被排除字段的替身。

    候选池按字段名排除了受关注字段，但如果池子里还留着某个字段能几乎完整地
    还原被排除的那个，排除就等于没做——推断路径换了个入口又通了。
    这里逐一检验：用某个选中字段单独去拟合每个被排除字段，残差比很小就说明是替身。
    """
    out = []
    for a in active:
        for e in excluded:
            if e not in df.columns:
                continue
            _, _, rr = idt._refit(df, e, [a])
            if np.isfinite(rr) and rr < tol:
                out.append({"选中字段": a, "替身对象": e, "残差比": float(rr)})
    return sorted(out, key=lambda r: r["残差比"])


def write_summary(path: Path, ctx: dict) -> None:
    L = [f"# 推断源定位：{ctx['target_cn']}", "",
         f"运行时间 {ctx['stamp']}　方法 **{ctx['method']}**", "",
         "## 一、这次跑的是什么", "",
         f"- 目标字段：`{ctx['target']}`（{ctx['target_cn']}）",
         f"- 数据：pjm_{ctx['year']}，{ctx['n_rows']} 行",
         f"- 候选字段：**{ctx['n_pool']} 个**（{ctx['pool_note']}）",
         f"- 网络结构：{'-'.join(map(str, ctx['cfg']['hidden']))}，"
         f"训练 {ctx['cfg']['epochs']} 轮，批大小 {ctx['cfg']['batch_size']}，"
         f"学习率 {ctx['cfg']['lr']}",
         f"- 数据划分：随机 {ctx['cfg']['train_ratio']:.0%} 训练 / "
         f"{1 - ctx['cfg']['train_ratio']:.0%} 测试，随机种子 {ctx['cfg']['seed']}", ""]

    L += ["## 二、结论先看这里", "",
          f"- 从 {ctx['n_pool']} 个候选字段里选出 **{ctx['n_active']} 个**推断源"
          f"（占 {ctx['n_active'] / ctx['n_pool']:.1%}）",
          f"- 只用这 {ctx['n_active']} 个字段重新训练普通 DNN，测试 R² = "
          f"**{ctx['retrain_r2']:.4f}**",
          f"- 作为对照，用全部 {ctx['n_pool']} 个字段的普通 DNN 测试 R² = "
          f"{ctx['dnn_r2']:.4f}",
          f"- 没被选中的 {ctx['n_pool'] - ctx['n_active']} 个字段单独训练，"
          f"测试 R² = {ctx['unsel_r2']:.4f}", ""]
    gap = ctx["retrain_r2"] - ctx["dnn_r2"]
    if gap >= -0.01:
        L += [f"> 用 {ctx['n_active'] / ctx['n_pool']:.0%} 的字段达到了和全量相当的精度"
              f"（相差 {gap:+.4f}），说明这组字段确实承载了推断目标所需的主要信息。", ""]
    else:
        L += [f"> 压缩到 {ctx['n_active']} 个字段后精度下降 {-gap:.4f}，"
              f"说明还有信息分散在未选中的字段里。", ""]

    L += ["## 三、选出来的推断源", "",
          "| 排名 | 字段 | 业务含义 | 门控值 |", "|---|---|---|---|"]
    for i, (f, g) in enumerate(ctx["active_list"], 1):
        L.append(f"| {i} | `{f}` | {cn(f)} | {g:.4f} |")
    L += ["", f"活跃阈值 **{ctx['threshold']}**：门控值不低于它的字段判定为推断源。", "",
          "### 阈值取多少影响大不大", "",
          "把阈值在几个量级之间来回调，看选中的字段数变不变：", "",
          "| 阈值 | 选中字段数 |", "|---|---|"]
    for th, k in ctx["thr_sens"]:
        L.append(f"| {th:g} | {k} |")
    L += ["", ctx["thr_note"], ""]

    if ctx["below"]:
        L += ["紧挨在阈值下方、被判为非活跃的几个字段（供参考）：", "",
              "| 字段 | 业务含义 | 门控值 |", "|---|---|---|"]
        for f, g in ctx["below"]:
            L.append(f"| `{f}` | {cn(f)} | {g:.4f} |")
        L.append("")

    if ctx["proxies"]:
        L += ["### 注意：这些选中字段是被排除字段的替身", "",
              "候选池按字段名排除了受关注字段，但下面这些留在池子里的字段，"
              "单独一个就能几乎完整还原被排除的那个——"
              "也就是说排除规则被绕过了，推断路径换了个入口又通了。", "",
              "| 选中字段 | 能还原的被排除字段 | 残差比 |", "|---|---|---|"]
        for r in ctx["proxies"]:
            L.append(f"| `{r['选中字段']}`（{cn(r['选中字段'])}） | "
                     f"`{r['替身对象']}`（{cn(r['替身对象'])}） | {r['残差比']:.4f} |")
        L += ["", "> 残差比越小替身越像。这一现象本身是结果的一部分，不作剔除处理："
              "它说明按字段名逐个排除，在组合推断面前是失效的。", ""]

    L += ["## 四、逐步增加字段数的验证", "",
          "按门控值从高到低依次取前 n 个字段，各自单独训练普通 DNN：", "",
          "| 字段数 | 测试 R² |", "|---|---|"]
    for n, r in zip(ctx["topn_ns"], ctx["topn_r2s"]):
        L.append(f"| {n} | {r:.4f} |")
    L += ["", "## 五、三种模型的对比", "",
          "| 模型 | 最优测试 R² | 出现轮次 | 最终活跃字段数 |", "|---|---|---|---|"]
    for name, r in ctx["all_results"].items():
        na = r.history["n_active"][-1]
        L.append(f"| {name} | {r.best_test_r2:.4f} | 第 {r.best_epoch} 轮 | {na} |")

    L += ["", "## 六、图", "",
          "- `fig_training.png`　三种模型的训练曲线，以及活跃字段数如何一步步收缩",
          "- `fig_gate_evolution.png`　每个字段的门控值随训练轮次的变化轨迹",
          "- `fig_gate_bar.png`　最终门控值排名",
          "- `fig_topn.png`　逐步增加字段数时测试 R² 的变化",
          "- `fig_gate_distribution.png`　门控值从大到小排开（对数刻度），"
          "D-Gating 的断崖说明阈值不敏感", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def run_one(df, target, method, pool, pool_note, cfg, out_root, year,
            excl_targets=False, strip_class1=False, pool_before=None,
            out_version=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = out_root / f"target_{target}" / method / f"run_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    FIG, DAT = dataio.split_dirs(d)

    if strip_class1:
        write_class1_doc(d / "第一类关系剥离.md", df, target, excl_targets,
                         year, pool_before or pool, pool, out_version)

    X = df[pool].to_numpy(float)
    y = df[target].to_numpy(float)

    results = {}
    for m in ["DNN", "L1GateDNN", "DGatingDNN"]:
        results[m] = gates.train(m, X, y, cfg)
        print(f"      {m:12s} 最优测试 R² {results[m].best_test_r2:.4f}"
              f"（第 {results[m].best_epoch} 轮）", flush=True)

    r = results[method]
    thr = cfg.active_threshold if method == "L1GateDNN" else cfg.dgate_threshold
    g = r.final_gates
    order = np.argsort(-g)
    active = [i for i in order if g[i] >= thr]
    inactive = [i for i in order if g[i] < thr]

    excluded = [c for c in df.columns if c not in pool and c != target]
    proxies = proxy_report(df, [pool[i] for i in active], excluded)

    retrain = gates.retrain_subset(X, y, active, cfg)
    unsel = gates.retrain_subset(X, y, inactive, cfg)

    step = max(1, len(active) // 6)
    ns = sorted({*range(step, len(active) + 1, step), len(active)}) if active else []
    r2s = [gates.retrain_subset(X, y, list(order[:n]), cfg) for n in ns]

    pd.DataFrame({"field": [pool[i] for i in order],
                  "中文名": [cn(pool[i]) for i in order],
                  "gate": g[order],
                  "active": [int(g[i] >= thr) for i in order]}).to_csv(
        DAT / "gates.csv", index=False)
    pd.DataFrame({"n": ns, "test_r2": r2s}).to_csv(DAT / "topn.csv", index=False)
    for m, rr in results.items():
        pd.DataFrame(rr.history).to_csv(DAT / f"history_{m}.csv", index=False)

    plots.training_curve_plot(results, FIG / "fig_training.png",
                              f"{CN[target]}　三种模型的训练过程")
    plots.gate_evolution_plot(r, thr, [cn(c) for c in pool],
                              FIG / "fig_gate_evolution.png",
                              f"{CN[target]}　{method} 门控值的演化")
    plots.gate_bar_plot(g, [cn(c) for c in pool], thr, FIG / "fig_gate_bar.png",
                        f"{CN[target]}　{method} 最终门控值排名")
    if ns:
        plots.topn_plot(ns, r2s, results["DNN"].best_test_r2, unsel,
                        FIG / "fig_topn.png", f"{CN[target]}　逐步增加字段数")
    plots.gate_distribution_plot(
        {m: results[m].final_gates for m in ["L1GateDNN", "DGatingDNN"]},
        thr, FIG / "fig_gate_distribution.png",
        f"{CN[target]}　门控值分布：断崖越陡，阈值越无所谓")

    (DAT / "config.json").write_text(json.dumps({
        "target": target, "method": method, "year": year, "version": VERSION,
        "pool_note": pool_note, "n_pool": len(pool), "pool": pool,
        **cfg.to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")

    write_summary(d / "总结.md", {
        "target": target, "target_cn": CN[target], "method": method, "year": year,
        "stamp": stamp, "n_rows": len(df), "n_pool": len(pool),
        "pool_note": pool_note, "cfg": cfg.to_dict(), "threshold": thr,
        "n_active": len(active),
        "active_list": [(pool[i], float(g[i])) for i in active],
        "thr_sens": [(th, int((g >= th).sum()))
                     for th in [1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.1]],
        "thr_note": (
            "> 这几档阈值给出的字段数完全一致，说明阈值落在门控值分布的空档里，"
            "取多少都不影响结论——这正是用 D-Gating 替代普通 L1 门控的原因。"
            if len({int((g >= th).sum()) for th in [1e-4, 1e-3, 0.005, 0.01, 0.02]}) == 1
            else "> 不同阈值给出的字段数不一致，说明门控值分布没有明显空档，"
                 "结论对阈值敏感，需要谨慎解释。"),
        "below": [(pool[i], float(g[i])) for i in inactive[:5]],
        "retrain_r2": retrain, "unsel_r2": unsel,
        "dnn_r2": results["DNN"].best_test_r2,
        "topn_ns": ns, "topn_r2s": r2s, "all_results": results,
        "proxies": proxies})
    if proxies:
        pd.DataFrame(proxies).to_csv(DAT / "proxy_fields.csv", index=False)
    return {"n_active": len(active), "retrain_r2": retrain,
            "dnn_r2": results["DNN"].best_test_r2, "dir": d,
            "unsel_r2": unsel, "n_proxy": len(proxies),
            "active": [pool[i] for i in active], "proxies": proxies}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--out-version", default=None,
                    help="输出到哪个版本目录，例如 pjm_2025_v2；默认 pjm_<year>")
    ap.add_argument("--pools", default="excl", choices=["excl", "both"],
                    help="excl=只跑排除其余关注字段；both=含/排除都跑")
    ap.add_argument("--targets", nargs="+", default=["net_actual_interchange_mw"])
    ap.add_argument("--method", default="DGatingDNN")
    ap.add_argument("--lambda-l1", type=float, default=0.01)
    ap.add_argument("--active-threshold", type=float, default=0.05)
    ap.add_argument("--lambda-dgate", type=float, default=0.005)
    ap.add_argument("--dgate-depth", type=int, default=4)
    ap.add_argument("--dgate-threshold", type=float, default=0.01)
    ap.add_argument("--strip-modes", nargs="+", type=int, default=[1],
                    help="第一类关系一律剥离，默认只跑剥离版本；传 0 才会跑不剥离的对照")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    df = dataio.load_clean(args.year, VERSION)
    cfg = gates.TrainConfig(
        epochs=args.epochs, lambda_l1=args.lambda_l1,
        active_threshold=args.active_threshold,
        lambda_dgate=args.lambda_dgate, dgate_depth=args.dgate_depth,
        dgate_threshold=args.dgate_threshold)

    rows = []
    excl_list = [True] if args.pools == "excl" else [False, True]
    for target in args.targets:
        for excl in excl_list:
            root = (dataio.OUTPUTS / (args.out_version or f"pjm_{args.year}") /
                    ("04b_source_location_excl_targets" if excl
                     else "04_source_location"))
            for strip in args.strip_modes:
                for scr in [False, True]:
                    name = ("stripped_" if strip else "") + (
                        "with_screening" if scr else "no_screening")
                    sub = root / name
                    base, _ = build_pool(df, target, excl, False, args.year,
                                        False, args.out_version)
                    pool, note = build_pool(df, target, excl, scr, args.year,
                                            strip, args.out_version)
                    print(f"\n=== {CN[target]}"
                          f"｜{'排除其余关注字段' if excl else '含其余关注字段'}"
                          f"｜{'剥离第一类' if strip else '不剥离'}"
                          f"｜{'已初筛' if scr else '未初筛'}｜候选 {len(pool)} 个 ===",
                          flush=True)
                    res = run_one(df, target, args.method, pool, note, cfg, sub,
                                  args.year, excl_targets=excl,
                                  strip_class1=strip, pool_before=base,
                                  out_version=args.out_version)
                    print(f"      → 选中 {res['n_active']} 个，再训练 R² "
                          f"{res['retrain_r2']:.4f}（全量 {res['dnn_r2']:.4f}），"
                          f"替身字段 {res['n_proxy']} 个", flush=True)
                    rows.append({
                        "target": target, "中文名": CN[target],
                        "排除关注字段": "是" if excl else "否",
                        "剥离第一类": "是" if strip else "否",
                        "初筛": "是" if scr else "否",
                        "候选数": len(pool), "选中数": res["n_active"],
                        "全量R2": res["dnn_r2"], "选中再训练R2": res["retrain_r2"],
                        "未选中R2": res["unsel_r2"], "替身字段数": res["n_proxy"],
                        "选中字段": "|".join(res["active"]),
                        "run": str(res["dir"].relative_to(dataio.OUTPUTS))})

    agg = pd.DataFrame(rows)
    aout = dataio.OUTPUTS / (args.out_version or f"pjm_{args.year}")
    f = aout / "source_location_summary.csv"
    if f.exists():
        agg = pd.concat([pd.read_csv(f), agg], ignore_index=True)
    agg.to_csv(f, index=False)
    print(f"\n汇总写入 {aout / 'source_location_summary.csv'}")


if __name__ == "__main__":
    main()
