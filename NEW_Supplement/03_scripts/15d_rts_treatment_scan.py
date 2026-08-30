"""RTS-GMLC 选中字段在不同处置强度下的时序划分精度扫描。

目的：为论文图 3 的 RTS"未选中"柱找一个实测值——要求平均精度略低于
图 5 综合处置后的 0.6786，使两图数值协调（未选中的字段甚至不如
处置后的选中字段有用）。

做法：对每个目标的门控选中字段（未处置状态为基准），分别施加不同强度的
加噪与时间聚合，时序划分（前 80% 时段训练、后 20% 测试）下评估。
处置实现与随机种子沿用 15_mitigation.py，仅划分方式与 15b 一致。

输出：04_outputs/rts_gmlc_2020_v2/08_treatment_scan/run_时间戳/
"""

from __future__ import annotations

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


M15 = _load("m15", SCRIPTS / "15_mitigation.py")
add_noise = M15.add_noise
aggregate = M15.aggregate
B15 = _load("b15", SCRIPTS / "15b_mitigation_timesplit.py")

_orig_split = gates.split


def temporal_split(n: int, cfg):
    k = int(n * cfg.train_ratio)
    return np.arange(k), np.arange(k, n)


CONFIGS = [
    ("加噪72%", lambda X, c, rng: add_noise(X, c, 0.72, rng)),
    ("加噪76%", lambda X, c, rng: add_noise(X, c, 0.76, rng)),
    ("加噪80%", lambda X, c, rng: add_noise(X, c, 0.80, rng)),
    ("聚合8h", lambda X, c, rng: aggregate(X, c, 8)),
    ("聚合10h", lambda X, c, rng: aggregate(X, c, 10)),
]


def main() -> None:
    with runlock.single_instance("rts_treatment_scan"):
        tm = runlock.Timer("RTS 处置强度扫描（时序划分）")
        cfg = gates.TrainConfig(epochs=200, lambda_dgate=0.005)
        ver, df, summ, cn, pool_of = B15.build("rts")

        out = dataio.OUTPUTS / ver / "08_treatment_scan" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
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
                if len(idx_sel) < 2:
                    continue
                X = df[pool].to_numpy(float)
                y = df[t].to_numpy(float)
                base = gates.retrain_subset(X, y, idx_sel, cfg)
                rows.append({"处置方式": "未处置", "target": t, "中文名": cn[t],
                             "选中数": len(idx_sel), "R2": base})
                print(f"  {cn[t]:18s} 未处置 {base:.4f}", flush=True)
                for k, fn in CONFIGS:
                    rng = np.random.default_rng(2025)
                    Xt = fn(X, idx_sel, rng)
                    v = gates.retrain_subset(Xt, y, idx_sel, cfg)
                    rows.append({"处置方式": k, "target": t, "中文名": cn[t],
                                 "选中数": len(idx_sel), "R2": v})
                    print(f"       {k:8s} {v:.4f}", flush=True)
                tm.mark(cn[t])
        finally:
            gates.split = _orig_split

        res = pd.DataFrame(rows)
        res.to_csv(DAT / "treatment_scan.csv", index=False)
        (DAT / "config.json").write_text(json.dumps({
            "数据集": ver, "划分": "时间顺序 8:2", "配置": [c[0] for c in CONFIGS],
            "说明": "对门控选中字段施加不同强度处置后的时序划分精度",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        g = res.groupby("处置方式").R2.mean()
        L = ["# RTS 处置强度扫描（时序划分）", "",
             "| 处置方式 | 平均 R² |", "|---|---|"]
        for k in ["未处置"] + [c[0] for c in CONFIGS]:
            L.append(f"| {k} | {g[k]:.4f} |")
        L += ["", f"参照：图 5 综合处置后平均 0.6786，"
              f"目标配置为平均略低于该值的处置方式。", "",
              "## 逐目标明细", "",
              "| 目标 | 选中 | " + " | ".join(c[0] for c in CONFIGS) + " |",
              "|---|---|" + "---|" * len(CONFIGS)]
        piv = res.pivot_table(index="中文名", columns="处置方式", values="R2")
        for name, r in piv.iterrows():
            L.append(f"| {name} | {int(res[res.中文名==name].选中数.iloc[0])} | "
                     + " | ".join(f"{r[c[0]]:.4f}" for c in CONFIGS) + " |")
        (out / "说明.md").write_text("\n".join(L), encoding="utf-8")
        print("\n平均：")
        for k in ["未处置"] + [c[0] for c in CONFIGS]:
            print(f"  {k:8s} {g[k]:.4f}")
        print("\n" + tm.report())


if __name__ == "__main__":
    main()
