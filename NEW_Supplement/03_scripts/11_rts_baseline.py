"""RTS-GMLC 的第 5 步：三条对比规则。

10_rts_pipeline.py 只做到第 4 步推断源定位就结束了——它的 argparse 里留了个
`--skip-rules` 开关，但代码里根本没写第 5 步，所以跑起来看不出少了东西。
这个脚本把缺掉的补上，让 RTS 和 PJM 在方法对比这一层可以横向对照。

三条规则各自回答一个不同的问题：

  rule1  本文方法自己定出用几个字段，其余方法在同样的字段数下比精度。
         问的是：字段数相同时，谁挑得准。
  rule2  沿用原稿的老规则：字段数从 1 开始逐个加，任一方法首次达到全量 DNN
         的 95%（或 97%）时，把该字段数定为统一预算，再横向比。
         这条规则天生偏向"最先达标的那个方法"，结果里如实标出是谁先达标。
         保留它是为了对照——换一种定预算的方式，排名会不会变。
  rule3  各方法按自己的判据自主决定选几个，然后在"字段数 × 精度"两个维度上
         看帕累托前沿。问的是：不给统一预算，各方法自己的取舍合不合理。

候选池的构成与第 4 步完全一致：已知公式剔除 →（可选）多指标初筛 → 逐层剥离。
不一致的话对比就没有意义。

不设"排除其余关注字段"这一支。PJM 需要那一支是因为它的目标字段本身也在
候选池里（比如用日前总电价推实时阻塞价）；RTS 的敏感目标从来不进候选池，
用一个不公开的量去推另一个不公开的量属于循环论证，所以只有初筛开关这一个维度。
"""

from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_src"))
import dataio  # noqa: E402
import gates  # noqa: E402
import plots  # noqa: E402
import runlock  # noqa: E402

warnings.filterwarnings("ignore")


def _load(name: str, path: Path):
    """导入另一个脚本当模块用。

    被导入的脚本顶层会碰 sys.argv（比如 06 用 bc.sl.TARGETS 当 argparse 默认值），
    所以导入期间要把 argv 换成只有程序名的样子。关键是**导入完必须换回来**——
    这几个 _load 是在本模块顶层调用的，如果就地把 argv 改掉不还原，
    等 main() 再去 parse_args() 时真正的命令行参数已经没了，
    所有选项都会静默退回默认值。之前 --dataset 不生效就是这么来的：
    本该跑原版 27 字段，实际又跑了一遍扩展版，而且不报任何错。
    """
    saved = sys.argv
    try:
        sys.argv = [name]
        spec = ilu.spec_from_file_location(name, path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = saved


rts = _load("rts", SCRIPTS / "10_rts_pipeline.py")
bc = _load("bc", SCRIPTS / "05_baseline_compare.py")
r2m = _load("r2m", SCRIPTS / "06_rule2_budget.py")
r3m = _load("r3m", SCRIPTS / "07_rule3_self.py")

# 三个对比脚本里的字段中文名、目标中文名都是从 sl（PJM 的 04_source_location）
# 取的。这里把它们统统换成 RTS 的模块，其余逻辑一行都不用改。
bc.sl = rts
r2m.bc, r2m.sl = bc, rts
r3m.bc, r3m.sl = bc, rts

OUT = ROOT / "04_outputs" / "rts_gmlc_2020_v2"
ROOT_CMP = OUT / "05_baseline_compare"
DATASET_LABEL = "RTS-GMLC 2020 扩展版（候选 58）"


def set_dataset(dataset: str, out_name: str) -> None:
    """与 10_rts_pipeline.set_dataset 保持一致，好用同一套代码跑原版和扩展版。"""
    global OUT, ROOT_CMP, DATASET_LABEL
    rts.set_dataset(dataset, out_name)
    OUT = ROOT / "04_outputs" / out_name
    ROOT_CMP = OUT / "05_baseline_compare"
    DATASET_LABEL = ("RTS-GMLC 2020 扩展版（候选 58）" if dataset.endswith("_v2")
                     else "RTS-GMLC 2020 原版（候选 27）")


def load_screen_keeps() -> dict[str, list[str]]:
    """读回第 3 步初筛的结果，保证候选池和第 4 步用的完全一致。"""
    d = OUT / "03_screening" / "data"   # set_dataset 之后才是正确路径
    keeps = {}
    for t in rts.TARGETS:
        f = d / f"screen_{t}.csv"
        if not f.exists():
            raise FileNotFoundError(f"缺少初筛结果 {f}，请先跑 10_rts_pipeline.py")
        s = pd.read_csv(f)
        keeps[t] = s[s.kept == 1].field.tolist()
    return keeps


def build_pools(df, pub, keeps, screening: bool) -> dict[str, list[str]]:
    """与第 4 步同样的构池方式：公式剔除 →（可选）初筛 → 逐层剥离。"""
    pools = {}
    for t in rts.TARGETS:
        pool, _, _, _ = rts.build_pool(df, t, pub, keeps[t] if screening else None)
        _, pool = rts.layered_strip(df, t, pool)
        pools[t] = pool
    return pools


# --------------------------------------------------------------------------

def run_rule1(df, pools, cfg, stamp, lv2, tm):
    out = ROOT_CMP / "rule1" / lv2 / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    OFIG, ODAT = dataio.split_dirs(out)
    print(f"\n===== rule1 / {lv2} =====", flush=True)
    res = []
    for t in rts.TARGETS:
        pool = pools[t]
        if len(pool) < 2:
            print(f"    {rts.CN[t]} 候选不足，跳过", flush=True)
            continue
        print(f"    ── {rts.CN[t]}　候选 {len(pool)}", flush=True)
        r, _ = bc.compare_one_target(df, t, pool, cfg, out)
        r["目标"], r["中文名"] = t, rts.CN[t]
        res.append(r)
    agg = pd.concat(res, ignore_index=True)
    agg.to_csv(ODAT / "overall_comparison.csv", index=False)
    (ODAT / "config.json").write_text(json.dumps(
        {"rule": "rule1", "数据集": DATASET_LABEL, "初筛": lv2,
         "targets": rts.TARGETS, **cfg.to_dict()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    bc.write_overall(out / "总结.md", agg, "rts", lv2, "rule1")
    plots.baseline_overall_plot(agg, OFIG / "fig_overall.png",
                                f"RTS-GMLC / {lv2}　同预算下各方法对比")
    tm.mark(f"rule1 / {lv2}")
    return agg


def run_rule2(df, pools, cfg, cfg_fast, stamp, lv2, tm):
    out = ROOT_CMP / "rule2" / lv2 / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    OFIG, ODAT = dataio.split_dirs(out)
    print(f"\n===== rule2 / {lv2} =====", flush=True)
    res = []
    for t in rts.TARGETS:
        pool = pools[t]
        if len(pool) < 2:
            continue
        print(f"    ── {rts.CN[t]}　候选 {len(pool)}", flush=True)
        res.append(r2m.run_target(df, t, pool, cfg, cfg_fast, out))
    agg = pd.concat(res, ignore_index=True)
    agg.to_csv(ODAT / "overall_comparison.csv", index=False)
    (ODAT / "config.json").write_text(json.dumps(
        {"rule": "rule2", "数据集": DATASET_LABEL, "初筛": lv2,
         "达标水平": r2m.LEVELS, "搜索轮次": cfg_fast.epochs,
         "最终评估轮次": cfg.epochs, "targets": rts.TARGETS},
        ensure_ascii=False, indent=2), encoding="utf-8")
    r2m.write_overall(out / "总结.md", agg, lv2)
    plots.rule2_overall_plot(agg, OFIG / "fig_overall.png",
                             f"RTS-GMLC / {lv2}　rule2 老规则下各方法表现")
    tm.mark(f"rule2 / {lv2}")
    return agg


def run_rule3(df, pools, cfg, stamp, lv2, tm):
    """复现 07_rule3_self.py 的主循环，只把候选池换成 RTS 的。"""
    out = ROOT_CMP / "rule3" / lv2 / f"run_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    OFIG, ODAT = dataio.split_dirs(out)
    print(f"\n===== rule3 / {lv2} =====", flush=True)
    targets = [t for t in rts.TARGETS if len(pools[t]) >= 2]

    # 第一遍：能自主定数的方法各自选，记下它们选了几个
    self_sel = {}
    for t in targets:
        pool = pools[t]
        X, y = df[pool].to_numpy(float), df[t].to_numpy(float)
        self_sel[t] = {}
        for m in r3m.SELF_METHODS:
            try:
                extra = None
                if m == "DGatingDNN":
                    sc = gates.train("DGatingDNN", X, y, cfg).final_gates
                else:
                    sc = np.asarray(bc.RANKERS[m](X, y, cfg=cfg), float)
                    if m == "LassoNet":
                        extra = getattr(bc.rank_lassonet, "last", None)
                self_sel[t][m] = r3m.select_self(m, X, y, cfg, sc, extra)
            except Exception as e:
                print(f"        [{m}] 失败：{type(e).__name__} {e}", flush=True)

    # 固定阈值方法要对齐的字段数：取全部自主定数方法的平均，不单看本文方法，
    # 否则等于把对比方法调成跟我们一样多，比较就失去意义了
    want = float(np.mean([len(v) for d in self_sel.values() for v in d.values()]))
    taus, importances = r3m.tune_tau(df, targets, pools, cfg, want)
    print(f"    固定阈值调优：目标字段数约 {want:.1f}，"
          + "，".join(f"{m} τ={taus[m][0]}" for m in r3m.FIXED_METHODS), flush=True)
    for m in r3m.FIXED_METHODS:
        taus[m][1].to_csv(ODAT / f"tau_tuning_{m}.csv", index=False)

    rows = []
    for t in targets:
        pool = pools[t]
        X, y = df[pool].to_numpy(float), df[t].to_numpy(float)
        full = gates.train("DNN", X, y, cfg).best_test_r2
        print(f"    ── {rts.CN[t]}　候选 {len(pool)}　全量 R²={full:.4f}", flush=True)
        for m in r3m.SELF_METHODS + r3m.FIXED_METHODS:
            if m in r3m.SELF_METHODS and m not in self_sel.get(t, {}):
                continue
            try:
                if m in r3m.FIXED_METHODS:
                    sc = importances[m][t]
                    idx = np.where(sc >= taus[m][0])[0]
                    if idx.size == 0:
                        idx = np.array([int(np.argmax(sc))])
                    rule = f"固定阈值 τ={taus[m][0]}（{len(targets)} 个目标共用）"
                else:
                    idx = self_sel[t][m]
                    rule = {"DGatingDNN": f"门控值 ≥ {cfg.dgate_threshold}（断崖，阈值不敏感）",
                            "STG": f"门控值 ≥ {cfg.stg_threshold}",
                            "LassoNet": "λ 路径上，验证损失在最优 5% 以内里最稀疏的点",
                            "Lasso": "交叉验证按 1 倍标准误规则选 alpha 后的非零系数"}[m]
                if idx.size == 0:
                    idx = np.array([0])
                r2v = gates.retrain_subset(X, y, list(idx), cfg)
                rows.append({"目标": t, "中文名": rts.CN[t], "方法": m,
                             "定数规则": rule, "选中数": int(idx.size),
                             "占候选比": idx.size / len(pool),
                             "测试R2": r2v, "全量R2": full, "候选数": len(pool),
                             "选中字段": "|".join(pool[i] for i in idx)})
                print(f"        {m:12s} 选 {idx.size:2d} 个 → R²={r2v:.4f}", flush=True)
            except Exception as e:
                print(f"        [{m}] 失败：{type(e).__name__} {e}", flush=True)

    agg = pd.DataFrame(rows)
    agg.to_csv(ODAT / "overall_comparison.csv", index=False)
    (ODAT / "config.json").write_text(json.dumps(
        {"rule": "rule3", "数据集": DATASET_LABEL, "初筛": lv2,
         "固定阈值": {m: taus[m][0] for m in r3m.FIXED_METHODS},
         "targets": targets, **cfg.to_dict()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    plots.rule3_scatter_plot(agg, OFIG / "fig_scatter.png",
                             f"RTS-GMLC / {lv2}　各方法自主定数的结果")
    plots.rule3_spread_plot(agg, OFIG / "fig_count_spread.png",
                            f"各方法在 {len(targets)} 个目标上选中字段数的离散程度")
    r3m.write_overall(out / "总结.md", agg, "rts", lv2, taus)
    tm.mark(f"rule3 / {lv2}")
    return agg


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="base")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--search-epochs", type=int, default=100)
    ap.add_argument("--rules", nargs="+", default=["rule1", "rule2", "rule3"])
    ap.add_argument("--dataset", default="rts_gmlc_2020_v2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    set_dataset(args.dataset, args.out or args.dataset)

    with runlock.single_instance("rts_baseline"):
        tm = runlock.Timer("RTS-GMLC 三条对比规则")
        cfg = gates.TrainConfig(epochs=args.epochs, lambda_dgate=0.005,
                                lambda_stg=0.005)
        cfg_fast = gates.TrainConfig(epochs=args.search_epochs,
                                     lambda_dgate=0.005, lambda_stg=0.005)
        df, pub, _, const = rts.step1_preprocess(args.scenario)
        keeps = load_screen_keeps()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"{DATASET_LABEL}　{len(df)} 行，候选 {len(pub)} 个"
              f"（已剔除 {len(const)} 个全年恒定）", flush=True)

        for screening in [False, True]:
            lv2 = "with_screening" if screening else "no_screening"
            pools = build_pools(df, pub, keeps, screening)
            print(f"\n######## 候选池（{lv2}）：" +
                  "，".join(f"{rts.CN[t]} {len(pools[t])}" for t in rts.TARGETS),
                  flush=True)
            if "rule1" in args.rules:
                run_rule1(df, pools, cfg, stamp, lv2, tm)
            if "rule2" in args.rules:
                run_rule2(df, pools, cfg, cfg_fast, stamp, lv2, tm)
            if "rule3" in args.rules:
                run_rule3(df, pools, cfg, stamp, lv2, tm)

        ROOT_CMP.mkdir(parents=True, exist_ok=True)
        (ROOT_CMP / "timing.md").write_text(
            "# RTS 三条对比规则执行耗时\n\n" + tm.report() + "\n", encoding="utf-8")
        print("\n" + tm.report())


if __name__ == "__main__":
    main()
