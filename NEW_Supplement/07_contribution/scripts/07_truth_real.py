"""L1 + L2：在真实数据上验证，用的是数据自己带的答案。

L1 公理自检（零成本，答案确切）
------------------------------
不需要任何领域知识，光靠数据本身就能查三件事：
  对称性  池子里取值**逐个数完全相同**的字段，份额必须相等
  零贡献  全期恒定的字段（信息量为零），份额必须为 0
  加和性  所有份额之和必须等于"全部发布"相比"什么都不发布"多出来的还原能力

第一条是最狠的：两列数字一模一样，任何讲得通的方法都必须给它们同样的分数。
稀疏门控在这里必然失败——它的目标是"用最少的字段达到精度"，
留一个、扔一个正是它想要的最优解。

L2 已知关系真值（RTS-GMLC，答案确切）
------------------------------------
RTS-GMLC 的数据由交流潮流仿真生成，各级汇总量是加出来的，
所以"系统总发电 = 六个分燃料之和"这类关系是数据的定义而不是拟合结果
（实测残差比 1e-9~1e-10，系数精确为 1.0000）。

挑出来的伪目标里，系统总发电有**四条互不相同的精确路径**，
每条单独就够。这是判别力最强的一道真实题：
正确的答案是把份额分给全部四条路径，稀疏门控只会留一条。

这一步刻意**不做第一类公式剥离**——正式流程里那些关系是要剥掉的，
但这里正需要它们留在池子里当答案键。剥掉了就没有答案可对了。
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
from src import truthsets as ts         # noqa: E402
from src import report as rp            # noqa: E402


def analyse(X, y, fields, cfg, n_eval, n_coal, seed, want_baselines=True):
    """训练代理模型 → 算贡献值 → 跑对照方法。返回统一格式的表。"""
    res = sg.fit(X, y, fields, cfg)
    vf = sg.ValueFunction(res, X, y, n_eval=n_eval, seed=0)
    r = att.kernel_shap(vf, len(fields), n_coalitions=n_coal, seed=seed)
    dec = att.decompose(r)
    tab = rp.build_table(fields, dec, zero_tol=None).drop(columns=["中文名"])
    if want_baselines:
        for mname, sc in bl.run_all(X, y, seed=cfg.seed).items():
            tab[mname] = tab["字段"].map(pd.Series(sc, index=fields))
    return tab, r, dec, vf, res


# ================================================================ L1

def run_l1(d, cfg, a, figd, datd):
    print(f"\n{'#'*74}\n# L1 公理自检：{d.name}\n{'#'*74}")
    rows, tabs = [], []

    for tgt in (a.targets or d.targets):
        pool = d.pool(tgt, drop_constants=False)     # 恒定字段要留着才能查零贡献
        dups = ts.duplicate_groups_in_pool(d.df, pool)
        consts = [c for c in pool if c in d.constants]
        twins = ts.exact_twins(d.df, tgt, pool)
        if not dups and not consts:
            continue

        print(f"\n--- {d.label(tgt)}（{tgt}）候选池 {len(pool)} ---")
        if dups:
            print(f"  池内完全重复的字段组 {len(dups)} 组：")
            for g in dups:
                print(f"    {g}")
        if consts:
            print(f"  全期恒定字段 {len(consts)} 个")
        if twins:
            print(f"  与目标完全相同的公开字段：{twins}  ← 完美替身")

        X = d.df[pool].to_numpy(float)
        y = d.df[tgt].to_numpy(float)
        t0 = time.time()
        tab, r, dec, vf, _ = analyse(X, y, pool, cfg, a.n_eval, a.n_coalitions,
                                     a.seed)
        tab.insert(0, "目标", d.label(tgt))
        tabs.append(tab)
        s = tab.set_index("字段")
        print(f"  （{time.time()-t0:.0f}s） v(全部)={float(np.ravel(r.v_full)[0]):.4f}  "
              f"加和性偏差={float(np.ravel(r.efficiency_gap)[0]):.2e}")

        rec = dict(目标=d.label(tgt), 候选数=len(pool),
                   v_全部=float(np.ravel(r.v_full)[0]),
                   v_空=float(np.ravel(r.v_empty)[0]),
                   加和性偏差=float(np.ravel(r.efficiency_gap)[0]))

        # --- 对称性 ---
        worst, worst_g, worst_base = 0.0, None, {}
        for g in dups:
            v = np.array([s.loc[f, "贡献值"] for f in g])
            spread = float(v.max() - v.min())
            rel = spread / max(np.abs(v).max(), 1e-12)
            if rel > worst:
                worst, worst_g = rel, g
                worst_base = {m: [float(s.loc[f, m]) for f in g]
                              for m in bl.ALL if m in s.columns}
            print(f"    对称性 {g[0]}… ({len(g)} 个)：贡献值 "
                  f"{np.round(v, 5)}  相对离散 {rel:.2%}")
        if dups:
            rec["对称性最大相对离散"] = worst
            rec["对称性最差字段组"] = worst_g
            for m, vals in worst_base.items():
                mv = np.array(vals)
                rec[f"对照_{m}_同组相对离散"] = float(
                    (mv.max() - mv.min()) / max(np.abs(mv).max(), 1e-12))
            print(f"    → 本方法同组最大相对离散 {worst:.2%}")
            for m, val in rec.items():
                if m.startswith("对照_") and m.endswith("同组相对离散"):
                    print(f"      {m.replace('对照_','').replace('_同组相对离散','')}"
                          f" 同组相对离散 {val:.2%}")

        # --- 零贡献 ---
        if consts:
            cv = np.array([abs(s.loc[c, "贡献值"]) for c in consts])
            rec["恒定字段最大绝对份额"] = float(cv.max())
            live = [f for f in pool if f not in consts]
            lv = np.array([abs(s.loc[f, "贡献值"]) for f in live])
            rec["非恒定字段份额中位数"] = float(np.median(lv))
            print(f"    零贡献：{len(consts)} 个恒定字段最大绝对份额 {cv.max():.5f}"
                  f"（非恒定字段份额中位数 {np.median(lv):.5f}）")

        # --- 与目标完全相同的公开字段 ---
        if len(twins) >= 2:
            tv = np.array([s.loc[f, "贡献值"] for f in twins])
            mg = np.array([s.loc[f, "不可替代性"] for f in twins])
            rec["完美替身份额"] = tv.round(5).tolist()
            rec["完美替身不可替代性"] = mg.round(5).tolist()
            rec["完美替身相对差"] = float(
                (tv.max() - tv.min()) / max(np.abs(tv).max(), 1e-12))
            print(f"    完美替身 {twins}：份额 {np.round(tv,4)}（相对差 "
                  f"{rec['完美替身相对差']:.2%}），不可替代性 {np.round(mg,4)}")
            for m in bl.ALL:
                if m in s.columns:
                    bv = np.array([abs(float(s.loc[f, m])) for f in twins])
                    rel = (bv.max() - bv.min()) / max(bv.max(), 1e-12)
                    rec[f"对照_{m}_完美替身相对差"] = float(rel)
                    print(f"      {m}：{np.round(bv,4)}  相对差 {rel:.1%}")

        rows.append(rec)
        rp.contribution_bar(tab, figd / f"fig_L1贡献值_{tgt}.png", top=25)
        rp.risk_map(tab, figd / f"fig_L1风险坐标_{tgt}.png")

    if tabs:
        pd.concat(tabs, ignore_index=True).to_csv(
            datd / "L1_contributions.csv", index=False)
    return rows


# ================================================================ L2

def run_l2(d, cfg, a, figd, datd):
    print(f"\n{'#'*74}\n# L2 已知关系真值：{d.name}\n{'#'*74}")

    ver = ts.verify_routes(d.df, ts.RTS_ROUTES)
    ver.to_csv(datd / "L2_route_verification.csv", index=False)
    print("\n先核对每条路径确实成立（这一步每次都要重做，写死的关系最容易腐烂）：")
    print(ver.to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    bad = ver[~ver["成立"]]
    if len(bad):
        print(f"\n[警告] 有 {len(bad)} 条路径不成立，将跳过对应的伪目标")

    rows, tabs = [], []
    for tgt, info in ts.RTS_ROUTES.items():
        ok = ver[(ver["伪目标"] == tgt) & ver["成立"]]
        if tgt not in d.df.columns or len(ok) == 0:
            continue
        good_routes = {r: info["路径"][r] for r in ok["路径"]}

        # 池子：全部公开字段（去掉敏感目标和伪目标自己），不做公式剥离
        pool = [c for c in d.df.columns
                if c != tgt and c not in d.targets and c not in d.constants]
        print(f"\n{'='*74}")
        print(f"【伪目标】{info['中文名']}（{tgt}），候选池 {len(pool)} 个字段")
        print(f"  {info['说明']}")
        for rname, fs in good_routes.items():
            rr = float(ok[ok['路径'] == rname]['残差比'].iloc[0])
            print(f"  路径《{rname}》{len(fs)} 个字段，实测残差比 {rr:.2e}")

        X = d.df[pool].to_numpy(float)
        y = d.df[tgt].to_numpy(float)
        t0 = time.time()
        tab, r, dec, vf, _ = analyse(X, y, pool, cfg, a.n_eval, a.n_coalitions,
                                     a.seed)
        tab.insert(0, "伪目标", info["中文名"])
        s = tab.set_index("字段")
        v_full = float(np.ravel(r.v_full)[0])
        print(f"  （{time.time()-t0:.0f}s）v(全部)={v_full:.4f}")

        route_all = ts.route_fields({"路径": good_routes}) & set(pool)
        others = [f for f in pool if f not in route_all]

        # 路径字段是否都拿到了份额
        rec = dict(伪目标=info["中文名"], 候选数=len(pool), 路径数=len(good_routes),
                   v_全部=v_full)
        print(f"\n  各条路径上的字段拿到了多少份额：")
        for rname, fs in good_routes.items():
            fs2 = [f for f in fs if f in s.index]
            v = np.array([s.loc[f, "贡献值"] for f in fs2])
            inter = np.array([s.loc[f, "协同冗余指数"] for f in fs2])
            rk = np.array([list(tab["字段"]).index(f) + 1 for f in fs2])
            rec[f"路径_{rname}_份额和"] = float(v.sum())
            rec[f"路径_{rname}_最小份额"] = float(v.min())
            rec[f"路径_{rname}_最差排名"] = int(rk.max())
            rec[f"路径_{rname}_指数均值"] = float(inter.mean())
            print(f"    《{rname}》份额和 {v.sum():.4f}，最小 {v.min():.4f}，"
                  f"最差排名 {rk.max()}/{len(pool)}，指数均值 {inter.mean():.3f}")

        ov = np.array([abs(s.loc[f, "贡献值"]) for f in others]) if others \
            else np.array([0.0])
        allroute = np.array([s.loc[f, "贡献值"] for f in route_all])
        rec["路径外字段最大份额"] = float(ov.max())
        rec["路径内字段最小份额"] = float(allroute.min())
        rec["路径内外完全分开"] = bool(allroute.min() > ov.max())
        print(f"\n  路径内最小份额 {allroute.min():.4f} vs "
              f"路径外最大份额 {ov.max():.4f} → "
              f"{'完全分开 ✓' if rec['路径内外完全分开'] else '有重叠 ✗'}")

        # 对照方法：它保留了几条路径
        print(f"\n  对照方法覆盖了几条路径（看每条路径上有多少字段被判为有贡献）：")
        for m in bl.ALL:
            if m not in s.columns:
                continue
            sc = s[m].abs()
            thr = sc.max() * 0.01                    # 相对最大值 1% 以上算"保留"
            cov = {}
            for rname, fs in good_routes.items():
                fs2 = [f for f in fs if f in s.index]
                cov[rname] = int(sum(sc[f] > thr for f in fs2))
            rec[f"对照_{m}_各路径保留数"] = cov
            tot = {rname: len([f for f in fs if f in s.index])
                   for rname, fs in good_routes.items()}
            txt = "  ".join(f"{rn}:{cov[rn]}/{tot[rn]}" for rn in cov)
            full_routes = sum(1 for rn in cov if cov[rn] == tot[rn])
            rec[f"对照_{m}_完整保留路径数"] = full_routes
            print(f"    {m:<10} {txt}   完整保留 {full_routes}/{len(cov)} 条")

        ours = {}
        thr_ours = float(np.abs(tab["贡献值"]).max()) * 0.01
        for rname, fs in good_routes.items():
            fs2 = [f for f in fs if f in s.index]
            ours[rname] = int(sum(abs(s.loc[f, "贡献值"]) > thr_ours for f in fs2))
        tot = {rn: len([f for f in fs if f in s.index])
               for rn, fs in good_routes.items()}
        rec["本方法_各路径保留数"] = ours
        rec["本方法_完整保留路径数"] = sum(1 for rn in ours if ours[rn] == tot[rn])
        txt = "  ".join(f"{rn}:{ours[rn]}/{tot[rn]}" for rn in ours)
        print(f"    {'本方法':<10} {txt}   完整保留 "
              f"{rec['本方法_完整保留路径数']}/{len(ours)} 条")

        rows.append(rec)
        tabs.append(tab)
        rp.contribution_bar(tab, figd / f"fig_L2贡献值_{tgt}.png", top=28)
        rp.risk_map(tab, figd / f"fig_L2风险坐标_{tgt}.png")
        rp.solo_vs_marginal(tab, figd / f"fig_L2独立vs不可替代_{tgt}.png")

    if tabs:
        pd.concat(tabs, ignore_index=True).to_csv(
            datd / "L2_contributions.csv", index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--hidden", default="384,256,192,128")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--n-coalitions", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-l1", action="store_true")
    ap.add_argument("--skip-l2", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    d = ds.load(a.dataset)
    base, figd, datd = ds.run_dir(a.dataset, "L1L2_truth")
    print(f"输出目录 {base}")
    print(f"数据集 {d.name}：{d.df.shape[0]} 行 × {d.df.shape[1]} 列，"
          f"敏感目标 {len(d.targets)} 个")

    cfg = sg.SurrogateConfig(hidden=tuple(int(x) for x in a.hidden.split(",")),
                             epochs=a.epochs, min_epochs=150, patience=150,
                             seed=a.seed)

    l1 = run_l1(d, cfg, a, figd, datd) if not a.skip_l1 else []
    l2 = run_l2(d, cfg, a, figd, datd) \
        if (not a.skip_l2 and a.dataset.startswith("rts")) else []

    (datd / "verdicts.json").write_text(json.dumps(
        {"L1": l1, "L2": l2}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    (datd / "config.json").write_text(json.dumps(
        {"数据集": a.dataset, "代理模型": cfg.to_dict(), "评估行数": a.n_eval,
         "抽取组合数": a.n_coalitions, "总耗时秒": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'='*74}\n总耗时 {(time.time()-t0)/60:.1f} 分钟 → {base}")


if __name__ == "__main__":
    main()
