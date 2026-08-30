"""未选中字段在时序划分下的还原精度（图3 面板数据，支持 RTS 与 PJM）。

背景：随机划分下 RTS 未选中字段 R² 平均 0.9745，与选中字段（0.9743）持平，
原因是 RTS 公开字段互为汇总口径、冗余极高。为给出更严苛口径下的对比，
本脚本补算时序划分（前 80% 时段训练、后 20% 时段测试）下三种情形：

    全量字段 / 门控选中字段 / 其余未选中字段

选中字段与划分方式沿用既有结果定义；仅划分方式不同。
结果写入 04_outputs/{数据集}/07_unselected_temporal/run_时间戳/。
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

B15 = ROOT / "03_scripts" / "15b_mitigation_timesplit.py"
_saved_argv = sys.argv
try:
    sys.argv = ["b15"]
    spec = ilu.spec_from_file_location("b15", B15)
    b15 = ilu.module_from_spec(spec)
    spec.loader.exec_module(b15)
finally:
    sys.argv = _saved_argv

_orig_split = gates.split


def temporal_split(n: int, cfg):
    k = int(n * cfg.train_ratio)
    return np.arange(k), np.arange(k, n)


def run(ds: str) -> None:
    tm = runlock.Timer(f"{ds} 未选中字段·时序划分")
    cfg = gates.TrainConfig(epochs=200, lambda_dgate=0.005)
    ver, df, summ, cn, pool_of = b15.build(ds)

    out = dataio.OUTPUTS / ver / "07_unselected_temporal" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    FIG, DAT = dataio.split_dirs(out)

    rows = []
    gates.split = temporal_split
    try:
        for _, r in summ.iterrows():
            t = r["target"]
            pool = pool_of(t)
            sel = [c for c in str(r["选中字段"]).split("|") if c]
            idx_sel = [pool.index(c) for c in sel if c in pool]
            idx_rest = [j for j in range(len(pool)) if j not in idx_sel]
            X = df[pool].to_numpy(float)
            y = df[t].to_numpy(float)
            r_full = gates.retrain_subset(X, y, list(range(len(pool))), cfg)
            r_sel = gates.retrain_subset(X, y, idx_sel, cfg)
            r_rest = gates.retrain_subset(X, y, idx_rest, cfg)
            rows.append({
                "target": t, "中文名": cn[t], "候选数": len(pool),
                "选中数": len(idx_sel), "未选中数": len(idx_rest),
                "全量R2_时序": r_full, "选中R2_时序": r_sel,
                "未选中R2_时序": r_rest,
                "选中减未选中": r_sel - r_rest,
            })
            print(f"  {cn[t]:18s} 全量 {r_full:.4f}　选中 {r_sel:.4f}"
                  f"　未选中 {r_rest:.4f}　差 {r_sel - r_rest:+.4f}", flush=True)
            tm.mark(cn[t])
    finally:
        gates.split = _orig_split

    res = pd.DataFrame(rows)
    res.to_csv(DAT / "unselected_temporal.csv", index=False)
    (DAT / "config.json").write_text(json.dumps({
        "数据集": ver, "划分": "时间顺序 8:2（前 80% 时段训练、后 20% 时段测试）",
        "说明": "全量/选中/未选中三种情形均在该划分下重新训练普通 DNN",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    L = [f"# {ver} 未选中字段·时序划分结果", "",
         "随机划分下未选中字段精度见 source_location_summary；",
         "本目录给出更严苛的时序划分口径：前 80% 时段训练、后 20% 时段测试。", "",
         "| 目标 | 候选 | 选中 | 全量 R² | 选中 R² | 未选中 R² | 选中−未选中 |",
         "|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r['中文名']} | {int(r.候选数)} | {int(r.选中数)} | "
                 f"{r.全量R2_时序:.4f} | {r.选中R2_时序:.4f} | "
                 f"{r.未选中R2_时序:.4f} | {r.选中减未选中:+.4f} |")
    L += ["", f"**平均**：全量 {res.全量R2_时序.mean():.4f}　"
          f"选中 {res.选中R2_时序.mean():.4f}　"
          f"未选中 {res.未选中R2_时序.mean():.4f}　"
          f"选中−未选中 {res.选中减未选中.mean():+.4f}", ""]
    (out / "说明.md").write_text("\n".join(L), encoding="utf-8")
    print("\n" + tm.report())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["rts", "pjm"], default="rts")
    args = ap.parse_args()
    with runlock.single_instance("unselected_temporal"):
        run(args.dataset)


if __name__ == "__main__":
    main()
