"""RTS-GMLC 2020 的完整流程：与 PJM 用同一套方法，参数保持一致。

这个数据集回答的是 PJM 回答不了的一件事：**用汇总量推细粒度运行量**。
候选发布字段是系统级/区域级/燃料级汇总（电网确实公开这类量），
敏感目标是具名节点电压相角、具名线路负载率、具名机组出力与状态
（电网确实不公开这类量）。边界是物理上真实存在的，不需要人为标密级。

流程与 PJM 完全一致，便于横向对照：

  1  数据清洗
  2  两层剥离：第一层按已知公式（不看残差），第二层按残差逐层剥离
  3  多指标初筛（分块置换定阈值，取零分布 90% 分位）
  4  推断源定位（D-Gating，λ=0.005，因子数 4，活跃阈值 0.01）
  5  三条对比规则

输出目录结构：每个阶段固定 `说明.md` + `figures/` + `data/`，
根目录只放 README 与配置。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_src"))
import gates  # noqa: E402
import identity as idt  # noqa: E402
import plots  # noqa: E402
import runlock  # noqa: E402
import screening as scr  # noqa: E402

warnings.filterwarnings("ignore")

# 默认跑扩展版（公开候选 58 个，其中 12 个全年恒定，可用 46 个）。
# 用 --dataset rts_gmlc_2020 可以切回原版 27 个候选字段做对照，
# set_dataset() 会把下面三个路径一起改掉。
RAW = ROOT / "01_data" / "raw" / "rts_gmlc_2020_v2"
CLEAN = ROOT / "01_data" / "clean" / "rts_2020_v2"
OUT = ROOT / "04_outputs" / "rts_gmlc_2020_v2"


def set_dataset(raw_name: str, out_name: str) -> None:
    """切换数据集版本。原版和扩展版的字段名完全兼容，只是扩展版多 31 个字段，
    所以公式表、中文名表都不用动——公式里引用不存在的字段会自动被忽略。"""
    global RAW, CLEAN, OUT
    RAW = ROOT / "01_data" / "raw" / raw_name
    CLEAN = ROOT / "01_data" / "clean" / raw_name.replace("rts_gmlc_", "rts_")
    OUT = ROOT / "04_outputs" / out_name

SCENARIOS = {"base": "rts_gmlc_hourly_2020_acpf_base.csv",
             "noise_1pct": "rts_gmlc_hourly_2020_acpf_noise_1pct.csv",
             "noise_3pct": "rts_gmlc_hourly_2020_acpf_noise_3pct.csv",
             "noise_5pct": "rts_gmlc_hourly_2020_acpf_noise_5pct.csv"}

# 12 个目标：3 个节点 + 5 条线路 + 4 台机组，覆盖三类物理对象
TARGETS = [
    "bus_115_va_deg", "bus_215_va_deg", "bus_315_va_deg",
    "branch_ab1_loading_pct", "branch_ab2_loading_pct", "branch_ab3_loading_pct",
    "branch_ca_1_loading_pct", "branch_cb_1_loading_pct",
    "gen_121_nuclear_1_pg_mw", "gen_218_cc_1_pg_mw",
    "gen_317_wind_1_pg_mw", "gen_321_cc_1_status",
]
CN = {
    "bus_115_va_deg": "节点115电压相角", "bus_215_va_deg": "节点215电压相角",
    "bus_315_va_deg": "节点315电压相角",
    "branch_ab1_loading_pct": "线路AB1负载率", "branch_ab2_loading_pct": "线路AB2负载率",
    "branch_ab3_loading_pct": "线路AB3负载率", "branch_ca_1_loading_pct": "线路CA-1负载率",
    "branch_cb_1_loading_pct": "线路CB-1负载率",
    "gen_121_nuclear_1_pg_mw": "机组121核电出力", "gen_218_cc_1_pg_mw": "机组218联合循环出力",
    "gen_317_wind_1_pg_mw": "机组317风电出力", "gen_321_cc_1_status": "机组321投入状态",
}
FIELD_CN = {
    "area_1_load_actual_mw": "区域1实时负荷", "area_2_load_actual_mw": "区域2实时负荷",
    "area_3_load_actual_mw": "区域3实时负荷", "system_load_actual_mw": "系统实时负荷",
    "area_1_load_day_ahead_mw": "区域1日前负荷", "area_2_load_day_ahead_mw": "区域2日前负荷",
    "area_3_load_day_ahead_mw": "区域3日前负荷", "system_load_day_ahead_mw": "系统日前负荷",
    "system_generation_mw": "系统发电量", "system_losses_mw": "系统网损",
    "area_1_net_export_mw": "区域1净送出", "area_2_net_export_mw": "区域2净送出",
    "area_3_net_export_mw": "区域3净送出",
    "reserve_reg_up_mw": "调频上备用", "reserve_reg_down_mw": "调频下备用",
    "reserve_spin_r1_mw": "区域1旋转备用", "reserve_spin_r2_mw": "区域2旋转备用",
    "reserve_spin_r3_mw": "区域3旋转备用",
    "gen_fuel_coal_mw": "燃煤出力", "gen_fuel_natural_gas_mw": "燃气出力",
    "gen_fuel_nuclear_mw": "核电出力", "gen_fuel_oil_mw": "燃油出力",
    "gen_fuel_hydro_mw": "水电出力", "gen_fuel_wind_mw": "风电出力",
    "gen_fuel_solar_mw": "光伏出力", "gen_fuel_storage_mw": "储能出力",
    "gen_sync_condenser_mw": "同步调相机",
    "hour_of_day": "小时", "day_of_week": "星期", "month_of_year": "月份",
    "is_weekend": "是否周末",
}
_FUEL_CN = {"coal": "燃煤", "natural_gas": "燃气", "nuclear": "核电", "oil": "燃油",
            "hydro": "水电", "wind": "风电", "solar": "光伏", "storage": "储能"}
for _a in (1, 2, 3):                      # 扩展版新增的 27 个分区级字段
    for _f, _c in _FUEL_CN.items():
        FIELD_CN[f"area_{_a}_fuel_{_f}_mw"] = f"区域{_a}{_c}出力"
    FIELD_CN[f"area_{_a}_generation_mw"] = f"区域{_a}发电总量"


def cn(f: str) -> str:
    return FIELD_CN.get(f, CN.get(f, f))


# 第一层剥离用的已知公式。全部经过实测验证（残差比列在 source 里）。
KNOWN_FORMULAS = [
    {"name": "系统负荷等于三区之和", "source": "官方定义，实测残差比 5.7e-10",
     "fields": ["system_load_actual_mw", "area_1_load_actual_mw",
                "area_2_load_actual_mw", "area_3_load_actual_mw"]},
    {"name": "系统日前负荷等于三区之和", "source": "官方定义，实测残差比 1.2e-10",
     "fields": ["system_load_day_ahead_mw", "area_1_load_day_ahead_mw",
                "area_2_load_day_ahead_mw", "area_3_load_day_ahead_mw"]},
    {"name": "系统功率平衡", "source": "潮流守恒，实测残差比 2.2e-09",
     "fields": ["system_generation_mw", "system_load_actual_mw", "system_losses_mw"]},
    {"name": "分燃料出力求和", "source": "按燃料汇总的定义",
     "fields": ["system_generation_mw"] + [
         f"gen_fuel_{f}_mw" for f in ["coal", "natural_gas", "nuclear", "oil",
                                      "hydro", "wind", "solar", "storage"]]},
    # 以下三条是扩展版新增字段带进来的关系，实测偏差都在 1.0e-06 以内
    #（1e-06 是 CSV 写出时的小数位舍入，不是计算误差）
    {"name": "系统发电等于三区发电之和", "source": "分区汇总的定义，实测偏差 1.0e-06",
     "fields": ["system_generation_mw"] + [f"area_{i}_generation_mw" for i in (1, 2, 3)]},
] + [
    {"name": f"区域{a}发电等于该区八种燃料之和", "source": "分区分燃料汇总的定义，实测偏差 1.0e-06",
     "fields": [f"area_{a}_generation_mw"] + [
         f"area_{a}_fuel_{f}_mw" for f in ["coal", "natural_gas", "nuclear", "oil",
                                           "hydro", "wind", "solar", "storage"]]}
    for a in (1, 2, 3)
]

# 上面八条都是公开字段互相之间的关系，目标字段不在其中，所以对目标不起作用。
# 真正会牵连到目标的是下面这一类：**目标本身就是某个公开汇总量的一个组成部分**。
# 这类关系写在数据定义里，不需要看数据就知道，属于第一层该剥掉的。
#
#   具名机组出力 → 该燃料类型的全系统出力汇总
#     实测：121 号核电机组出力与 gen_fuel_nuclear_mw 相关系数 1.0000、残差比 0.0000，
#     也就是说这台机组就是全系统核电出力的全部，公开汇总量等于直接公开该机组出力。
#   跨区联络线 → 相关区域的净送出
#     实测：三个区域净送出用五条联络线潮流拟合，残差比 5.6e-08，是精确的会计恒等式。
#     反过来三个汇总量推五条线路是欠定的（R² 0.23~0.88），推不精确，
#     但"分量—汇总"这层结构关系是文档写明的，按同一条规则一并剥掉。
FUEL_OF_UNIT = {"nuclear": "nuclear", "cc": "natural_gas", "wind": "wind",
                "ct": "natural_gas", "steam": "coal", "hydro": "hydro",
                "pv": "solar", "rtpv": "solar", "csp": "solar"}
AREA_EXPORTS = [f"area_{i}_net_export_mw" for i in (1, 2, 3)]


def _component_formulas(targets: list[str]) -> list[dict]:
    """按"目标是公开汇总量的分量"这条规则，逐个目标生成公式条目。

    规则要覆盖**所有层级**的汇总，不能只看最粗的那一层。用单个汇总字段去推目标
    机组，全年 8784 小时的残差比（残差比 = 拟合剩下的误差 ÷ 该字段标准差，
    0 表示完全算得出来）：

        目标机组          同燃料全系统   本区同燃料
        121 核电出力         0.0000       0.0000
        218 联合循环出力     0.7700       0.5651
        317 风电出力         0.4297       0.4837
        321 投入状态         0.6479       0.5241

    121 号核电机组是彻底泄露：它就是全系统核电出力的全部，公开核电汇总等于
    直接公开这台机组。

    218 和 321 说明**发布粒度细化会加重泄露**：口径从全系统收窄到分区，
    残差比分别从 0.7700 降到 0.5651、从 0.6479 降到 0.5241。218 号机组出力
    占区 2 燃气出力的比例中位数是 1.000、最小 0.273——超过半数小时它就是
    区 2 燃气出力的全部，只是全年不构成恒等关系。

    317 风电则相反，细化后反而略差（0.4297 → 0.4837），因为区 3 还有别的风电
    机组，分区汇总并没有把它单独凸显出来。所以"细化必然加重泄露"并不普遍成立，
    要逐个字段看。

    正因为事前看不出哪一层会泄露，规则就得覆盖全部层级：对一台具名机组，
    凡是它参与求和的公开量一律剔除，共四层——同燃料全系统汇总、本区同燃料汇总、
    本区发电总量、系统发电总量。是否真的泄露不影响该不该剔，规则要统一，
    不能看着数据好看就留下。各层实际的残差比记在说明文档里，
    读者自己能看出哪几层是真起作用的。
    """
    out = []
    for t in targets:
        if t.startswith("gen_"):
            fuel = next((FUEL_OF_UNIT[k] for k in FUEL_OF_UNIT
                         if f"_{k}_" in t), None)
            # RTS-GMLC 的母线编号首位就是分区号：1xx 属区 1，2xx 属区 2，3xx 属区 3
            m = re.match(r"gen_(\d)\d\d_", t)
            area = int(m.group(1)) if m else None
            if fuel:
                agg = [f"gen_fuel_{fuel}_mw"]
                if area:
                    agg += [f"area_{area}_fuel_{fuel}_mw",
                            f"area_{area}_generation_mw"]
                agg.append("system_generation_mw")
                out.append({
                    "name": f"{t} 参与 {len(agg)} 个公开汇总量的求和",
                    "source": "按燃料/分区汇总的定义（数据字典）",
                    "kind": "component",
                    "fields": [t] + agg})
        elif t.startswith("branch_") and t.endswith("_loading_pct"):
            out.append({
                "name": f"{t} 是跨区联络线，参与区域净送出的求和",
                "source": "净送出定义为跨区支路潮流之和，实测残差比 5.6e-08",
                "kind": "component",
                "fields": [t] + AREA_EXPORTS})
    return out


KNOWN_FORMULAS += _component_formulas(TARGETS)


def formula_expr(f: dict) -> str:
    """把公式条目写成能看懂的式子。

    两类语义不一样，必须分开表达，否则读者会误以为都是等式：
      求和型：第一个字段等于其余字段之和，写成 A = B + C + D
      分量型：第一个字段是其余每个汇总量的一个加数，写成 A ⊂ B、C
              这类不能写成等式——目标只是汇总量里的一项，不是全部。
    """
    if f.get("kind") == "component":
        head, aggs = f["fields"][0], f["fields"][1:]
        return f"{head} 是 " + "、".join(aggs) + " 各自求和中的一个加数"
    head, rest = f["fields"][0], f["fields"][1:]
    return f"{head} = " + " + ".join(rest)


def formula_drop(target: str) -> tuple[list[str], list[str]]:
    drop, hit = set(), []
    for f in KNOWN_FORMULAS:
        if target in f["fields"]:
            drop |= set(f["fields"]) - {target}
            hit.append(f["name"])
    return sorted(drop), hit


# --------------------------------------------------------------------------

def step1_preprocess(scenario: str = "base"):
    """读原始 CSV，分出候选字段与目标字段，去掉恒定字段。"""
    d = pd.read_csv(RAW / SCENARIOS[scenario])
    fd = pd.read_csv(RAW / "field_dictionary.csv")
    pub = [c for c in fd[fd.role == "published_candidate"].column_name if c in d.columns]
    sen = [c for c in fd[fd.role == "sensitive_target"].column_name if c in d.columns]
    const = [c for c in pub if d[c].nunique(dropna=True) <= 1]
    pub = [c for c in pub if c not in const]
    keep = pub + [t for t in TARGETS if t in d.columns]
    df = d[keep].apply(pd.to_numeric, errors="coerce").dropna()
    CLEAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN / f"{scenario}.csv", index=False)
    return df, pub, sen, const


def build_pool(df, target, pub, screen_keep=None):
    """候选池 = 候选发布字段 − 已知公式牵连的字段 −（可选）初筛剔除的字段。

    敏感目标之间不互相作为输入——它们都是不公开的量，
    用一个不公开的量去推另一个不公开的量属于循环论证。
    """
    pool = [c for c in pub if c != target]
    fdrop, fhit = formula_drop(target)
    fdrop = [c for c in fdrop if c in pool]
    pool = [c for c in pool if c not in fdrop]
    note = [f"剥离已知公式 {len(fdrop)} 个"]
    if screen_keep is not None:
        pool = [c for c in pool if c in screen_keep]
        note.append("经过多指标初筛")
    return pool, "；".join(note), fdrop, fhit


def layered_strip(df, target, pool, stop=0.10, max_layers=20):
    """第二层：在公式剥离之后的池子上逐层剥离。"""
    cur, layers = list(pool), []
    while len(layers) < max_layers:
        rr_full = idt.exact_fit_ratio(df, target, cur)
        if not np.isfinite(rr_full) or rr_full >= stop:
            break
        sup = idt._prune(df, target, cur, stop)
        if not sup:
            break
        coefs, const, rr = idt._refit(df, target, sup)
        if not np.isfinite(rr) or rr >= stop:
            break
        victim = max(sup, key=lambda s: abs(coefs[s]) * df[s].std())
        cur.remove(victim)
        layers.append({"layer": len(layers) + 1, "residual_ratio": rr,
                       "n_support": len(sup), "support": "|".join(sup),
                       "relation": f"{target} = " + " ".join(
                           f"{coefs[s]:+.4g}*{s}" for s in sup),
                       "removed": victim, "n_pool_after": len(cur)})
    return layers, cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--dataset", default="rts_gmlc_2020_v2",
                    help="原始数据目录名，默认扩展版；原版是 rts_gmlc_2020")
    ap.add_argument("--out", default=None,
                    help="输出目录名，默认与 --dataset 同名")
    args = ap.parse_args()
    set_dataset(args.dataset, args.out or args.dataset)

    with runlock.single_instance("rts_pipeline"):
        tm = runlock.Timer("RTS-GMLC 2020 完整流程")
        cfg = gates.TrainConfig(epochs=args.epochs, lambda_dgate=0.005,
                                lambda_stg=0.005)
        OUT.mkdir(parents=True, exist_ok=True)

        # ---------- 第 1 步 ----------
        print(f"######## 第1步 数据准备  {datetime.now():%F %T}", flush=True)
        df, pub, sen, const = step1_preprocess(args.scenario)
        d1 = OUT / "01_preprocess"; d1.mkdir(exist_ok=True)
        _, D1 = _dirs(d1)
        pd.DataFrame({"field": pub, "中文名": [cn(c) for c in pub]}).to_csv(
            D1 / "candidate_fields.csv", index=False)
        (d1 / "说明.md").write_text(
            f"# 数据准备\n\n- 场景：{args.scenario}\n- 行数：{len(df)}\n"
            f"- 候选发布字段：**{len(pub)} 个**（剔除 {len(const)} 个全年恒定："
            f"{'、'.join(cn(c) for c in const)}）\n"
            f"- 敏感目标字段：{len(sen)} 个，本轮选取其中 {len(TARGETS)} 个作为推断目标\n\n"
            f"敏感目标之间不互相作为输入——都是不公开的量，用一个不公开的量推另一个"
            f"属于循环论证。\n", encoding="utf-8")
        print(f"  候选 {len(pub)}（剔除恒定 {len(const)}），行数 {len(df)}", flush=True)
        tm.mark("第1步 数据准备")

        # ---------- 第 2 步 两层剥离 ----------
        print(f"######## 第2步 两层剥离  {datetime.now():%F %T}", flush=True)
        d2 = OUT / "02_class1_identity"; d2.mkdir(exist_ok=True)
        F2, D2 = _dirs(d2)
        strip_rows, layer_rows = [], []
        for t in TARGETS:
            pool, note, fdrop, fhit = build_pool(df, t, pub)
            lay, final = layered_strip(df, t, pool)
            strip_rows.append({"target": t, "中文名": CN[t], "候选起点": len(pub),
                               "公式剔除": len(fdrop), "命中公式": "|".join(fhit),
                               "逐层剔除": len(lay), "剩余候选": len(final),
                               "公式剔除字段": "|".join(fdrop),
                               "逐层剔除字段": "|".join(x["removed"] for x in lay)})
            for x in lay:
                layer_rows.append({"target": t, "中文名": CN[t], **x})
            print(f"  {CN[t]:18s} 公式剔除 {len(fdrop)}，逐层剔除 {len(lay)}，"
                  f"剩余 {len(final)}", flush=True)
        pd.DataFrame(strip_rows).to_csv(D2 / "strip_summary.csv", index=False)
        pd.DataFrame(layer_rows).to_csv(D2 / "layered_strip_layers.csv", index=False)
        _write_strip_doc(d2 / "说明.md", strip_rows, layer_rows, set(df.columns))
        tm.mark("第2步 两层剥离")

        # ---------- 第 3 步 初筛 ----------
        print(f"######## 第3步 初筛  {datetime.now():%F %T}", flush=True)
        d3 = OUT / "03_screening"; d3.mkdir(exist_ok=True)
        F3, D3 = _dirs(d3)
        keeps, srows = {}, []
        for t in TARGETS:
            pool, _, _, _ = build_pool(df, t, pub)
            null = scr.null_thresholds(df, t, pool, n_draws=120, block=168,
                                       quantile=0.90)
            obs = scr.screen(df, t, pool, null.thresholds)
            keeps[t] = set(obs.query("kept==1").field)
            obs.to_csv(D3 / f"screen_{t}.csv", index=False)
            plots.null_vs_observed_plot(
                null.draws, obs, null.thresholds, scr.METRICS, scr.METRIC_CN,
                F3 / f"fig_null_vs_obs_{t}.png",
                f"{CN[t]}：各指标实测值与无关系时的取值分布")
            srows.append({"target": t, "中文名": CN[t], "n_pool": len(pool),
                          "n_kept": len(keeps[t]), "n_dropped": len(pool) - len(keeps[t]),
                          "pass_hist": obs.n_pass.value_counts().sort_index().to_dict(),
                          **{f"thr_{m}": null.thresholds[m] for m in scr.METRICS}})
            print(f"  {CN[t]:18s} 候选 {len(pool)} → 保留 {len(keeps[t])}", flush=True)
        s3 = pd.DataFrame(srows)
        s3.drop(columns=["pass_hist"]).to_csv(D3 / "screening_summary.csv", index=False)
        plots.screening_overview_plot(s3, F3 / "fig_screening_overview.png",
                                      "多指标初筛总览（RTS-GMLC 2020）")
        (d3 / "说明.md").write_text(
            "# 多指标初筛\n\n阈值由分块置换（按周整段打乱目标）构造的零分布给出，"
            "取 **90% 分位**。分位数取 0.90 而不是统计检验惯用的 0.95，"
            "因为初筛控制的是漏报而非误报——留错了后面的门控会关掉，代价小；"
            "漏掉了就永远找不回来。\n\n"
            "| 目标 | 候选 | 保留 | 筛除 |\n|---|---|---|---|\n" +
            "\n".join(f"| {r['中文名']} | {int(r.n_pool)} | {int(r.n_kept)} | "
                      f"{int(r.n_dropped)} |" for _, r in s3.iterrows()) +
            f"\n\n平均筛除 {s3.n_dropped.mean():.1f} 个"
            f"（占 {s3.n_dropped.sum()/s3.n_pool.sum():.1%}）。\n", encoding="utf-8")
        tm.mark("第3步 初筛")

        # ---------- 第 4 步 推断源定位 ----------
        print(f"######## 第4步 推断源定位  {datetime.now():%F %T}", flush=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        allrows = []
        for scr_on in [False, True]:
            lv = "with_screening" if scr_on else "no_screening"
            base = OUT / "04_source_location" / lv / f"run_{stamp}"
            base.mkdir(parents=True, exist_ok=True)
            FB, DB = _dirs(base)
            print(f"  ===== {lv} =====", flush=True)
            rows = []
            for t in TARGETS:
                pool, note, fdrop, fhit = build_pool(
                    df, t, pub, keeps[t] if scr_on else None)
                lay, pool = layered_strip(df, t, pool)
                if len(pool) < 2:
                    print(f"    {CN[t]:18s} 候选不足，跳过", flush=True)
                    continue
                X, y = df[pool].to_numpy(float), df[t].to_numpy(float)
                full = gates.train("DNN", X, y, cfg).best_test_r2
                r = gates.train("DGatingDNN", X, y, cfg)
                g = r.final_gates
                order = np.argsort(-g)
                act = [i for i in order if g[i] >= cfg.dgate_threshold]
                retr = gates.retrain_subset(X, y, act, cfg) if act else float("nan")
                unsel = gates.retrain_subset(
                    X, y, [i for i in order if g[i] < cfg.dgate_threshold], cfg)
                td = FB / f"target_{t}"; td.mkdir(exist_ok=True)
                plots.gate_evolution_plot(r, cfg.dgate_threshold,
                                          [cn(c) for c in pool],
                                          td / "fig_gate_evolution.png",
                                          f"{CN[t]}　门控值演化")
                plots.gate_bar_plot(g, [cn(c) for c in pool], cfg.dgate_threshold,
                                    td / "fig_gate_bar.png", f"{CN[t]}　门控值排名")
                pd.DataFrame(r.gate_history, columns=pool).to_csv(
                    DB / f"gate_history_{t}.csv", index=False)
                rows.append({"target": t, "中文名": CN[t], "初筛": "是" if scr_on else "否",
                             "候选数": len(pool), "选中数": len(act),
                             "全量R2": full, "选中再训练R2": retr, "未选中R2": unsel,
                             "选中字段": "|".join(pool[i] for i in act)})
                print(f"    {CN[t]:18s} 候选 {len(pool):2d} 选中 {len(act):2d} "
                      f"全量 {full:.4f} 再训练 {retr:.4f}", flush=True)
            rr = pd.DataFrame(rows)
            rr.to_csv(DB / "summary.csv", index=False)
            _write_loc_doc(base / "总结.md", rr, lv, cfg)
            allrows.append(rr)
        pd.concat(allrows).to_csv(OUT / "source_location_summary.csv", index=False)
        tm.mark("第4步 推断源定位")

        (OUT / "timing.md").write_text("# 执行耗时\n\n" + tm.report() + "\n",
                                       encoding="utf-8")
        print(f"######## 全部完成  {datetime.now():%F %T}", flush=True)
        print(tm.report())


def _dirs(base: Path):
    f, d = base / "figures", base / "data"
    f.mkdir(parents=True, exist_ok=True); d.mkdir(parents=True, exist_ok=True)
    return f, d


def _write_strip_doc(path, strip_rows, layer_rows, available=None):
    s = pd.DataFrame(strip_rows)
    L = ["# 两层剥离", "",
         "第一层按已知公式剔除，**不看残差**——依据是官方定义与潮流守恒，不需要阈值。",
         "第二层在剩下的池子上按残差逐层剥离（门槛 0.10）。", "",
         "## 用到的已知公式", "",
         "求和型写成等式；分量型写成 `A 是 B 求和中的一个加数`——"
         "目标只是汇总量里的一项，不是全部，不能写成等式。", "",
         "| 公式 | 式子 | 依据 |", "|---|---|---|"]
    # 公式表是全局的，但两个数据集的字段不一样（原版没有分区分燃料那批）。
    # 只列出本数据集真实存在的公式，否则文档里会出现数据里根本没有的字段名。
    shown = [f for f in KNOWN_FORMULAS
             if available is None or sum(c in available for c in f["fields"]) >= 2]
    L += [f"| {f['name']} | `{formula_expr(f)}` | {f['source']} |" for f in shown]
    if available is not None and len(shown) < len(KNOWN_FORMULAS):
        L += ["", f"> 全局公式表共 {len(KNOWN_FORMULAS)} 条，本数据集里字段齐备的有 "
              f"{len(shown)} 条，其余涉及本数据集没有的字段（扩展版才有的分区分燃料汇总），"
              "已略去。"]
    L += ["", "## 各目标的剥离结果", "",
          "| 目标 | 候选起点 | 公式剔除 | 逐层剔除 | 剩余 | 命中的公式 |",
          "|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        L.append(f"| {r['中文名']} | {int(r.候选起点)} | {int(r.公式剔除)} | "
                 f"{int(r.逐层剔除)} | {int(r.剩余候选)} | {r.命中公式 or '—'} |")
    lay = pd.DataFrame(layer_rows)
    if len(lay):
        L += ["", "## 第二层逐层找到的关系", "",
              "| 目标 | 层 | 残差比 | 支撑字段数 | 删除的字段 |", "|---|---|---|---|---|"]
        for _, r in lay.iterrows():
            L.append(f"| {r['中文名']} | {int(r.layer)} | {r.residual_ratio:.2e} | "
                     f"{int(r.n_support)} | `{r.removed}` |")
    L += ["", "> 第一层的公式分两类。一类是候选池内部的关系（例如系统负荷等于三区之和），"
          "它不牵连目标，只是候选之间的冗余。另一类是**目标本身参与某个公开汇总量的求和**"
          "（例如具名机组出力被计入同燃料汇总、跨区联络线潮流被计入区域净送出），"
          "这一类才是真正会泄露目标的，必须剔除。", "",
          "> 判断该不该剔除只看关系是否文档写明、能否靠规则枚举，不看它解不解得精确。"
          "所以对一台具名机组，凡是它参与求和的公开量一律剔除，"
          "覆盖同燃料全系统、本区同燃料、本区发电总量、系统发电总量四个层级。", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


def _write_loc_doc(path, rr, lv, cfg):
    L = [f"# 推断源定位（{'经过初筛' if 'with' in lv else '未初筛'}）", "",
         "## 结论先看这里", "",
         f"- {len(rr)} 个目标，平均候选 {rr.候选数.mean():.1f} 个，"
         f"平均选中 **{rr.选中数.mean():.1f} 个**（占 {rr.选中数.sum()/rr.候选数.mean()/len(rr):.1%}）",
         f"- 选中字段再训练的平均测试 R² = **{rr.选中再训练R2.mean():.4f}**，"
         f"全量为 {rr.全量R2.mean():.4f}",
         f"- 未选中字段单独训练平均 {rr.未选中R2.mean():.4f}", "",
         f"参数：D-Gating 因子数 {cfg.dgate_depth}，λ={cfg.lambda_dgate}，"
         f"活跃阈值 {cfg.dgate_threshold}，训练 {cfg.epochs} 轮，随机划分 8:2。", "",
         "## 逐个目标", "",
         "| 目标 | 候选 | 选中 | 全量 R² | 选中再训练 R² | 未选中 R² |",
         "|---|---|---|---|---|---|"]
    for _, r in rr.iterrows():
        L.append(f"| {r['中文名']} | {int(r.候选数)} | {int(r.选中数)} | {r.全量R2:.4f} | "
                 f"**{r.选中再训练R2:.4f}** | {r.未选中R2:.4f} |")
    L += ["", "## 各目标选中的字段", ""]
    for _, r in rr.iterrows():
        L += [f"**{r['中文名']}**（{int(r.选中数)} 个）："
              + "、".join(cn(c) for c in str(r.选中字段).split("|") if c), ""]
    L += ["## 图", "",
          "- `figures/target_<字段>/fig_gate_evolution.png`　门控值随训练轮次的演化，"
          "左线性刻度看保留字段的分化、右对数刻度看被淘汰字段坠到多低",
          "- `figures/target_<字段>/fig_gate_bar.png`　最终门控值排名", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
