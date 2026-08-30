"""把"保住了几条精确路径"这个指标读公平。

为什么要补这一步
----------------
L2 那一步统计的是"每个方法在几条已知的精确路径上把字段都保住了"。
本方法保住 6/7 条，分解式门控和置换重要性 0/7——看起来差距悬殊。

但这个指标对**不做区分的方法**过于宽容：一个把 45 个字段全判为"有贡献"的方法，
自然把每条路径都保住了，可它什么也没告诉你。实测随机门控和两两相关性
就是这种情况，它们 7/7 全中。

所以必须再补一列：**这个方法总共认为多少字段有贡献**。
两个数一起看才有意义——
  保住的路径多 + 判定有贡献的字段少  ⇒ 真的分得清
  保住的路径多 + 几乎全部字段都算有贡献 ⇒ 只是没排除任何东西
  保住的路径少 + 判定有贡献的字段少  ⇒ 分得清，但分错了

不补这一列就报 6/7 对 0/7，是在拿一个对自己有利的口径说话。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import datasets as ds          # noqa: E402
from src import truthsets as ts         # noqa: E402

METHODS = ["分解式门控", "随机门控", "两两相关性", "置换重要性", "Lasso系数"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="L1L2 运行目录")
    ap.add_argument("--dataset", default="rts_v2")
    ap.add_argument("--rel-thr", type=float, default=0.01,
                    help="分数超过该方法自身最大值的多少比例算'有贡献'")
    a = ap.parse_args()

    run = Path(a.run)
    f = run / "data" / "L2_contributions.csv"
    if not f.exists():
        raise SystemExit(f"没找到 {f}")
    df = pd.read_csv(f)
    d = ds.load(a.dataset)
    ver = pd.read_csv(run / "data" / "L2_route_verification.csv")

    cn2key = {v["中文名"]: k for k, v in ts.RTS_ROUTES.items()}

    rows = []
    for lab, g in df.groupby("伪目标", sort=False):
        key = cn2key.get(lab)
        if key is None:
            continue
        info = ts.RTS_ROUTES[key]
        ok = ver[(ver["伪目标"] == key) & (ver["成立"])]
        routes = {r: info["路径"][r] for r in ok["路径"]}
        s = g.set_index("字段")
        n_pool = len(s)

        rec = {"伪目标": lab, "候选数": n_pool, "路径数": len(routes)}

        def n_routes_kept(keep: set) -> int:
            return sum(1 for fs in routes.values()
                       if all(x in keep for x in fs if x in s.index))

        # ---- 口径一：各方法按"超过自身最大值 1%"算有贡献 ----
        for m, col in [("本方法", "贡献值")] + [(m, m) for m in METHODS]:
            if col not in s.columns:
                continue
            thr = float(np.abs(g[col]).max()) * a.rel_thr
            keep = set(s.index[np.abs(s[col]) > thr])
            rec[f"{m}_判定有贡献数"] = len(keep)
            rec[f"{m}_完整保留路径数"] = n_routes_kept(keep)

        # ---- 口径二：同预算（仓库既有的 rule1 口径）----
        # 让每个方法只保留和分解式门控一样多的字段，再看路径还在不在。
        # 这才是苹果对苹果：上一个口径下本方法也判了九成字段"有贡献"，
        # "保住路径多"于是变成白送的，说明不了任何问题。
        if "分解式门控" in s.columns:
            budget = int((np.abs(s["分解式门控"]) >= 0.01).sum())
            budget = max(budget, 1)
            rec["同预算_k"] = budget
            for m, col in [("本方法", "贡献值")] + [(m, m) for m in METHODS]:
                if col not in s.columns:
                    continue
                topk = set(s[col].abs().sort_values(ascending=False)
                           .head(budget).index)
                rec[f"{m}_同预算保留路径数"] = n_routes_kept(topk)
                # 路径字段进了前 k 的比例，比 0/1 的"完整保留"更有分辨率
                allf = [x for fs in routes.values() for x in fs if x in s.index]
                rec[f"{m}_同预算覆盖率"] = (
                    len(topk & set(allf)) / max(len(set(allf)), 1))
        # ---- 口径三：覆盖率随预算 k 变化 ----
        # 只报同预算那一个点是不完整的：分解式门控自己定的 k 只有 1~3，
        # 那么小的预算下谁都保不住整条路径。扫一遍 k 才看得出全貌，
        # 也才知道本方法是"一直不如"还是"只在极小预算处不如"。
        allf = sorted({x for fs in routes.values() for x in fs if x in s.index})
        for m, col in [("本方法", "贡献值")] + [(m, m) for m in METHODS]:
            if col not in s.columns:
                continue
            order = list(s[col].abs().sort_values(ascending=False).index)
            for k in [1, 2, 3, 5, 8, 12, 18, 25, 35, n_pool]:
                if k > n_pool:
                    continue
                topk = set(order[:k])
                rec[f"覆盖率_{m}_k{k}"] = len(topk & set(allf)) / max(len(allf), 1)
        rows.append(rec)

    res = pd.DataFrame(rows)
    res.to_csv(run / "data" / "L2_route_fairness.csv", index=False)

    print("=" * 92)
    print("口径一：各方法按自身最大值的 1% 算有贡献")
    print("（保住的路径数必须和判定有贡献的字段数一起看）")
    print("=" * 92)
    tot_routes = int(res["路径数"].sum())
    n_pool = int(res["候选数"].iloc[0])
    lines = []
    for m in ["本方法"] + METHODS:
        c = f"{m}_完整保留路径数"
        k = f"{m}_判定有贡献数"
        if c not in res.columns:
            continue
        lines.append(dict(方法=m,
                          完整保留路径数=int(res[c].sum()),
                          总路径数=tot_routes,
                          平均判定有贡献字段数=float(res[k].mean()),
                          占候选池比例=float(res[k].mean()) / n_pool))
    out = pd.DataFrame(lines).sort_values(
        ["完整保留路径数", "平均判定有贡献字段数"], ascending=[False, True])
    print(out.to_string(index=False, float_format=lambda x: f"{x:9.3f}"))

    print(f"\n候选池 {n_pool} 个字段。读法：")
    print("  · 保住路径多、判定有贡献的字段又少 ⇒ 真的分得清")
    print("  · 保住路径多、但几乎所有字段都算有贡献 ⇒ 只是没排除任何东西，")
    print("    这一条对它不构成优势")
    print("  · 保住路径少、判定有贡献的字段也少 ⇒ 分得清，但分错了")

    # ---------------- 口径二 ----------------
    if "同预算_k" in res.columns:
        print("\n" + "=" * 92)
        print("口径二·同预算：每个方法只保留和分解式门控一样多的字段")
        print("（沿用仓库既有的 rule1 口径，这才是苹果对苹果）")
        print("=" * 92)
        ks = res["同预算_k"].tolist()
        print(f"各伪目标的预算 k = {ks}（由分解式门控自己定的字段数）")
        lines2 = []
        for m in ["本方法"] + METHODS:
            c, cov = f"{m}_同预算保留路径数", f"{m}_同预算覆盖率"
            if c not in res.columns:
                continue
            lines2.append(dict(方法=m,
                               同预算完整保留路径数=int(res[c].sum()),
                               总路径数=tot_routes,
                               路径字段覆盖率=float(res[cov].mean())))
        out2 = pd.DataFrame(lines2).sort_values(
            ["同预算完整保留路径数", "路径字段覆盖率"], ascending=False)
        print(out2.to_string(index=False, float_format=lambda x: f"{x:9.3f}"))
        print("\n  覆盖率 = 全部路径字段里有多少进了前 k 名。"
              "比 0/1 的'完整保留'更有分辨率——")
        print("  预算很小时谁都保不住整条路径，但覆盖率仍能区分排序好坏。")
        out2.to_csv(run / "data" / "L2_route_fairness_budget.csv", index=False)

    # ---------------- 口径三 ----------------
    ks = [1, 2, 3, 5, 8, 12, 18, 25, 35, n_pool]
    cov_rows = []
    for m in ["本方法"] + METHODS:
        row = {"方法": m}
        for k in ks:
            c = f"覆盖率_{m}_k{k}"
            if c in res.columns:
                row[f"k={k}"] = float(res[c].mean())
        if len(row) > 1:
            cov_rows.append(row)
    if cov_rows:
        cov = pd.DataFrame(cov_rows)
        cov.to_csv(run / "data" / "L2_route_coverage_by_k.csv", index=False)
        print("\n" + "=" * 92)
        print("口径三：路径字段覆盖率随保留字段数 k 的变化")
        print("（只报同预算那一个点不完整——门控自己定的 k 只有 1~3，")
        print(" 那么小的预算下谁都保不住整条路径）")
        print("=" * 92)
        print(cov.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for _, r in cov.iterrows():
            xs = [k for k in ks if f"k={k}" in r.index]
            ys = [r[f"k={k}"] for k in xs]
            ours = r["方法"] == "本方法"
            ax.plot(xs, ys, marker="o", ms=3.5, label=r["方法"],
                    zorder=3 if ours else 2, lw=2.2 if ours else 1.3)
        ax.set_xlabel("保留的字段数 k")
        ax.set_ylabel("已知精确路径上的字段进入前 k 的比例")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        (run / "figures").mkdir(exist_ok=True)
        fig.savefig(run / "figures" / "fig_路径覆盖率随k.png", bbox_inches="tight")
        plt.close(fig)

    (run / "data" / "L2_route_fairness.json").write_text(json.dumps(
        {"相对门槛": a.rel_thr, "候选池": n_pool, "总路径数": tot_routes,
         "各方法": lines}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物写入 {run/'data'}")


if __name__ == "__main__":
    main()
