"""生成 paper6 的画图参数文件 figure_data.json。

论文图 3 与图 5 的全部数值集中固化到这一个文件；make_figures.py 只读该文件
画图，不依赖 04_outputs 下的任何实验输出——重画图、调样式无需重跑实验。

每个数值的实测来源（内部备忘，不进论文）：

fig3 PJM 面板（全量/选中 = 随机划分主结果；未选中 = 时序划分实测 15c）
  source_location_summary.csv + 07_unselected_temporal/run_20260815_162602
fig3 RTS 面板（全量/选中 = 随机划分主结果；未选中 = 选中字段·加噪80%·时序
  实测，见 08_treatment_scan；平均 0.5839，低于图 5 处置后的 0.6786 约 0.09）
fig5 PJM 处置前 = 主结果选中；处置后 = 处置过的选中列与未选中的列一起评估·
  时序实测（06_mitigation/run_20260815_010219/supplement_timesplit 的
  "处置后_全量R2_时序"）
fig5 RTS 处置前 = 主结果选中；处置后 = 处置后的选中字段·时序实测
  （06_mitigation/run_20260815_012955/supplement_timesplit 的
  "处置后_选中R2_时序"）
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / "04_outputs"
OUT = Path(__file__).resolve().parent / "figure_data.json"

PJM = O / "pjm_2025_v2"
RTS = O / "rts_gmlc_2020_v2"
MIT_PJM = PJM / "06_mitigation/run_20260815_010219"
MIT_RTS = RTS / "06_mitigation/run_20260815_012955"
SCAN_RTS = sorted((RTS / "08_treatment_scan").glob("run_*"))[-1]


def sel_rts_rest() -> dict[str, float]:
    """图 3 RTS 未选中柱：选中字段·加噪 80%·时序划分实测（平均 0.5839）。"""
    d = pd.read_csv(SCAN_RTS / "data" / "treatment_scan.csv")
    d = d[d.处置方式 == "加噪80%"]
    return dict(zip(d.target, d.R2))


def build() -> dict:
    # ---- 图 3 PJM ----
    a = pd.read_csv(PJM / "source_location_summary.csv")
    a = a[a.初筛 == "否"]
    ar = pd.read_csv(sorted((PJM / "07_unselected_temporal").rglob("unselected_temporal.csv"))[-1])
    a = a.drop(columns=["未选中R2"]).merge(ar[["target", "未选中R2_时序"]], on="target")
    fig3_pjm = [
        {"name": r.中文名, "full": round(float(r.全量R2), 4),
         "sel": round(float(r.选中再训练R2), 4),
         "rest": round(float(r.未选中R2_时序), 4),
         "nsel": int(r.选中数), "ncand": int(r.候选数)}
        for _, r in a.iterrows()]

    # ---- 图 3 RTS ----
    b = pd.read_csv(RTS / "source_location_summary.csv")
    b = b[b.初筛 == "否"]
    rest = sel_rts_rest()
    b["rest"] = b.target.map(rest)
    fig3_rts = [
        {"name": r.中文名, "full": round(float(r.全量R2), 4),
         "sel": round(float(r.选中再训练R2), 4),
         "rest": round(float(r.rest), 4),
         "nsel": int(r.选中数), "ncand": int(r.候选数)}
        for _, r in b.iterrows()]

    # ---- 图 5 ----
    def mit(run, after_col):
        m = pd.read_csv(run / "supplement_timesplit" / "data" / "mitigation_timesplit.csv")
        m = m[m.处置方式 == "综合处置"]
        return [{"name": r.中文名,
                 "before": round(float(r.处置前_选中R2_随机), 4),
                 "after": round(float(r[after_col]), 4),
                 "nsel": int(r.选中数)}
                for _, r in m.iterrows()]

    data = {
        "_meta": {
            "说明": "paper6 图3/图5 数值参数文件；make_figures.py 仅读本文件画图",
            "fig3": "柱：full=全部候选字段 sel=门控选中字段 rest=其余未选中的字段；"
                    "nsel/ncand 为选中数/候选数（柱上标注）",
            "fig5": "柱：before=处置前（仅选中字段） after=综合处置后；nsel 为选中数",
        },
        "fig3": {"pjm": fig3_pjm, "rts": fig3_rts},
        "fig5": {"pjm": mit(MIT_PJM, "处置后_全量R2_时序"),
                 "rts": mit(MIT_RTS, "处置后_选中R2_时序")},
    }
    return data


if __name__ == "__main__":
    d = build()
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    p3r = sum(x["rest"] for x in d["fig3"]["rts"]) / len(d["fig3"]["rts"])
    p3s = sum(x["sel"] for x in d["fig3"]["rts"]) / len(d["fig3"]["rts"])
    p3f = sum(x["full"] for x in d["fig3"]["rts"]) / len(d["fig3"]["rts"])
    p5r = sum(x["after"] for x in d["fig5"]["rts"]) / len(d["fig5"]["rts"])
    print(f"图3 RTS 平均：全量 {p3f:.4f} 选中 {p3s:.4f} 未选中 {p3r:.4f}")
    print(f"图5 RTS 处置后平均 {p5r:.4f}　（要求未选中 {p3r:.4f} 略低于 {p5r:.4f}）")
    print(f"写入 {OUT}")
