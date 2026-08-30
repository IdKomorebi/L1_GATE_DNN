"""数据不完善时，隐性推断和推断源定位还成不成立。

审稿人指出：实际工程中电网数据往往并不完善，真正的难点是缺失值、异常坏数据、
数据延迟、不同区域口径差异，而论文对此考虑欠缺。

这个脚本回答两问，顺序不能颠倒：

  第一问  数据降级之后，**隐性推断本身**还成不成立？
          看全量候选字段训练出来的测试 R² 掉多少。掉到没有推断能力，
          后面就不用问了。
  第二问  如果推断仍然成立，**推断源定位**还成不成立？
          看门控选出来的字段还是不是原来那些（用 Jaccard 重合度衡量），
          以及选中子集重训后的 R² 掉多少。

## 必须有基准波动这一栏

只看"缺失 30% 时重合度 0.7"是无法解读的——不知道 0.7 是降级造成的，
还是什么都不改、只换个随机种子本来就会这样。所以先用 5 个随机种子在
原始数据上各跑一遍，两两之间的重合度给出**天然波动的参照线**，
后面所有降级条件都跟这条线比。

## 降级条件怎么设计

缺失值做成**块状**而不是零星：电网数据缺失通常是表计掉线、通道中断造成的
整段缺失，不是每个点独立丢失。块长取 6 小时。补法用线性插值，
因为实际处理就是这么补的——如果直接丢弃缺失行，比较的就不是同一批样本了。

数据延迟只对一半字段施加：实际中不会所有数据一起延迟，
而是部分来源慢、部分来源快，这种参差才是真正影响分析的情形。
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
import runlock  # noqa: E402

warnings.filterwarnings("ignore")


def _load(name: str, path: Path):
    saved = sys.argv
    try:
        sys.argv = [name]
        spec = ilu.spec_from_file_location(name, path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.argv = saved


sl = _load("sl", SCRIPTS / "04_source_location.py")

OUT_VERSION = "pjm_2025_v2"
YEAR = 2025
# 三个目标覆盖不同的可推断程度：容易、中等、困难
TARGETS = ["net_actual_interchange_mw",          # 全量 R² 0.9673
           "da_as_total_mw_primary_reserve",     # 0.8588
           "congestion_price_da"]                # 0.6069
BLOCK = 6          # 缺失按 6 小时一段


def block_missing(X: np.ndarray, frac: float, rng) -> np.ndarray:
    """按段制造缺失，再用线性插值补回。

    逐字段独立地挑若干个长为 BLOCK 的时间段置为缺失——对应表计掉线、
    通道中断这类真实情形，而不是每个点独立丢失。
    补法用线性插值：实际工程就是这么处理的，而且直接丢弃缺失行会让
    各条件下比较的样本不一致。
    """
    Z = X.copy()
    n, p = Z.shape
    n_blk = max(1, int(round(frac * n / BLOCK)))
    for j in range(p):
        starts = rng.choice(max(1, n - BLOCK), size=n_blk, replace=False)
        m = np.zeros(n, bool)
        for s in starts:
            m[s:s + BLOCK] = True
        if m.all():
            continue
        idx = np.arange(n)
        Z[m, j] = np.interp(idx[m], idx[~m], Z[~m, j])
    return Z


def delay(X: np.ndarray, hours: int, rng) -> np.ndarray:
    """随机一半字段整体后移若干小时，开头用第一个有效值填。"""
    Z = X.copy()
    p = Z.shape[1]
    pick = rng.choice(p, size=max(1, p // 2), replace=False)
    for j in pick:
        Z[hours:, j] = X[:-hours, j]
        Z[:hours, j] = X[0, j]
    return Z


def run_one(X, y, cfg, seed: int):
    """训练一次门控，返回（选中下标集合，全量 R²，选中子集 R²，门控值）。"""
    c = gates.TrainConfig(**{**cfg.__dict__, "seed": seed})
    full = gates.train("DNN", X, y, c).best_test_r2
    r = gates.train("DGatingDNN", X, y, c)
    g = r.final_gates
    order = np.argsort(-g)
    act = [int(i) for i in order if g[i] >= c.dgate_threshold]
    sub = gates.retrain_subset(X, y, act, c) if act else float("nan")
    return set(act), full, sub, g


def jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=5, help="基准波动用几个种子")
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    args = ap.parse_args()

    with runlock.single_instance("robustness"):
        tm = runlock.Timer("数据不完善时的鲁棒性")
        cfg = gates.TrainConfig(epochs=args.epochs, lambda_dgate=0.005)
        df = dataio.load_clean(YEAR, "main")
        out = (dataio.OUTPUTS / OUT_VERSION / "07_robustness" /
               f"run_{datetime.now():%Y%m%d_%H%M%S}")
        out.mkdir(parents=True, exist_ok=True)
        FIG, DAT = dataio.split_dirs(out)

        rows = []
        for t in args.targets:
            pool, note = sl.build_pool(df, t, True, False, YEAR, True, OUT_VERSION)
            X0 = df[pool].to_numpy(float)
            y = df[t].to_numpy(float)
            print(f"\n{'=' * 70}\n{sl.CN[t]}　候选 {len(pool)} 个（{note}）",
                  flush=True)

            # ---- 基准：只换随机种子，数据不动 ----
            base_sets, base_full, base_sub = [], [], []
            for s in range(args.seeds):
                a, f, sb, _ = run_one(X0, y, cfg, 42 + s)
                base_sets.append(a); base_full.append(f); base_sub.append(sb)
                print(f"  基准 种子{42 + s}　全量 {f:.4f}　选中 {len(a)}　子集 {sb:.4f}",
                      flush=True)
            pairs = [jac(base_sets[i], base_sets[j])
                     for i in range(len(base_sets)) for j in range(i + 1, len(base_sets))]
            ref_jac = float(np.mean(pairs))
            ref = base_sets[0]
            print(f"  → 基准天然波动：重合度 {ref_jac:.3f}（{np.min(pairs):.2f}~{np.max(pairs):.2f}），"
                  f"全量 R² {np.mean(base_full):.4f}±{np.std(base_full):.4f}", flush=True)
            rows.append({"目标": t, "中文名": sl.CN[t], "条件": "基准（换种子）",
                         "水平": f"{args.seeds} 个种子", "全量R2": float(np.mean(base_full)),
                         "全量R2标准差": float(np.std(base_full)),
                         "选中数": float(np.mean([len(a) for a in base_sets])),
                         "子集R2": float(np.mean(base_sub)),
                         "与基准重合度": ref_jac, "是否基准": 1})
            tm.mark(f"{sl.CN[t]} 基准")

            # ---- 各降级条件 ----
            conds = ([("块状缺失", f"{int(v * 100)}%", lambda X, r, v=v: block_missing(X, v, r))
                      for v in (0.05, 0.15, 0.30)] +
                     [("整字段缺失", f"{int(v * 100)}%", None) for v in (0.10, 0.20)] +
                     [("数据延迟", f"{h} 小时", lambda X, r, h=h: delay(X, h, r))
                      for h in (1, 3)])
            for name, lvl, fn in conds:
                rng = np.random.default_rng(2025)
                if name == "整字段缺失":
                    frac = float(lvl.rstrip("%")) / 100
                    k = max(1, int(round(len(pool) * frac)))
                    drop = rng.choice(len(pool), size=k, replace=False)
                    keep = [i for i in range(len(pool)) if i not in set(drop)]
                    Xd, kept = X0[:, keep], [pool[i] for i in keep]
                    a, f, sb, _ = run_one(Xd, y, cfg, 42)
                    a = {pool.index(kept[i]) for i in a}      # 换回原池的下标
                else:
                    Xd = fn(X0, rng)
                    a, f, sb, _ = run_one(Xd, y, cfg, 42)
                j = jac(a, ref)
                rows.append({"目标": t, "中文名": sl.CN[t], "条件": name, "水平": lvl,
                             "全量R2": f, "全量R2标准差": np.nan, "选中数": float(len(a)),
                             "子集R2": sb, "与基准重合度": j, "是否基准": 0})
                flag = "" if j >= ref_jac - 0.05 else "  ← 明显低于天然波动"
                print(f"  {name}{lvl:>6s}　全量 {f:.4f}（{f - np.mean(base_full):+.4f}）"
                      f"　选中 {len(a):2d}　子集 {sb:.4f}　重合度 {j:.3f}{flag}", flush=True)
                tm.mark(f"{sl.CN[t]} {name}{lvl}")

        res = pd.DataFrame(rows)
        res.to_csv(DAT / "robustness.csv", index=False)
        (DAT / "config.json").write_text(json.dumps(
            {"targets": args.targets, "轮次": args.epochs, "种子数": args.seeds,
             "缺失块长小时": BLOCK, "候选池": "排除其余关注字段、两层剥离、不初筛"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        write_doc(out / "说明.md", res, args)
        plot(FIG, res)
        (out / "timing.md").write_text("# 耗时\n\n" + tm.report() + "\n",
                                       encoding="utf-8")
        print("\n" + tm.report())
        print(f"\n写入 {out}")


def plot(FIG, res):
    import matplotlib.pyplot as plt
    tg = res.中文名.unique()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    conds = [c for c in res.条件.unique() if c != "基准（换种子）"]
    w = 0.8 / max(1, len(tg))
    for k, t in enumerate(tg):
        d = res[res.中文名 == t]
        b = d[d.是否基准 == 1].iloc[0]
        xs, ys1, ys2 = [], [], []
        for i, c in enumerate(conds):
            for _, r in d[d.条件 == c].iterrows():
                xs.append(len(xs)); ys1.append(r.全量R2 - b.全量R2)
                ys2.append(r.与基准重合度)
        ax[0].bar(np.arange(len(ys1)) + k * w, ys1, width=w, label=t)
        ax[1].plot(np.arange(len(ys2)), ys2, "o-", label=t, ms=5)
        ax[1].axhline(b.与基准重合度, ls="--", lw=1, alpha=.5,
                      color=ax[1].lines[-1].get_color())
    lab = [f"{r.条件}\n{r.水平}" for _, r in
           res[(res.中文名 == tg[0]) & (res.是否基准 == 0)].iterrows()]
    for a in ax:
        a.set_xticks(np.arange(len(lab))); a.set_xticklabels(lab, fontsize=8)
        a.legend(fontsize=8)
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_ylabel("全量 R² 相对基准的变化")
    ax[0].set_title("第一问：数据降级后，隐性推断还成不成立")
    ax[1].set_ylabel("选中字段与基准的重合度")
    ax[1].set_title("第二问：推断源定位还成不成立\n（虚线为只换种子的天然波动）")
    fig.tight_layout()
    fig.savefig(FIG / "fig_robustness.png", dpi=150)
    plt.close(fig)


def write_doc(path, res, args):
    L = ["# 数据不完善时的鲁棒性", "",
         "审稿人指出实际工程中真正的难点是缺失值、异常坏数据、数据延迟与口径差异。"
         "这一节回答两问，顺序不能颠倒：**降级之后隐性推断本身还成不成立**；"
         "若仍成立，**推断源定位还成不成立**。", "",
         "## 一、为什么必须先测天然波动", "",
         f"只看「缺失 30% 时重合度 0.7」是没法解读的——不知道这 0.7 是降级造成的，"
         f"还是什么都不改、只换个随机种子本来就会这样。所以先用 {args.seeds} 个随机种子"
         "在原始数据上各跑一遍，两两之间的重合度作为参照线，"
         "后面所有降级条件都跟这条线比。", "",
         "## 二、降级怎么造", "",
         "| 条件 | 设置 | 为什么这么设 |", "|---|---|---|",
         f"| 块状缺失 | 5% / 15% / 30%，块长 {BLOCK} 小时，线性插值补回 | "
         "电网缺失通常是表计掉线、通道中断造成的整段缺失，不是每点独立丢失；"
         "补法用插值是因为实际就这么处理，直接丢弃缺失行会让各条件比较的样本不一致 |",
         "| 整字段缺失 | 随机去掉 10% / 20% 的候选字段 | 对应某类数据这次没有发布 |",
         "| 数据延迟 | 随机一半字段整体后移 1 / 3 小时 | 实际中不会所有数据一起延迟，"
         "部分来源快、部分慢，这种参差才真正影响分析 |", "",
         "## 三、结果", ""]
    for t in res.中文名.unique():
        d = res[res.中文名 == t]
        b = d[d.是否基准 == 1].iloc[0]
        L += [f"### {t}", "",
              f"基准：全量 R² {b.全量R2:.4f}（{args.seeds} 个种子标准差 {b.全量R2标准差:.4f}），"
              f"平均选中 {b.选中数:.1f} 个，**天然重合度 {b.与基准重合度:.3f}**", "",
              "| 条件 | 水平 | 全量 R² | 相对基准 | 选中数 | 子集 R² | 与基准重合度 |",
              "|---|---|---|---|---|---|---|"]
        for _, r in d[d.是否基准 == 0].iterrows():
            L.append(f"| {r.条件} | {r.水平} | {r.全量R2:.4f} | {r.全量R2 - b.全量R2:+.4f} | "
                     f"{int(r.选中数)} | {r.子集R2:.4f} | {r.与基准重合度:.3f} |")
        L.append("")
    L += ["## 四、图", "",
          "- `fig_robustness.png` 左：全量 R² 相对基准的变化（第一问）；"
          "右：选中字段与基准的重合度，虚线是只换种子的天然波动（第二问）", ""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
