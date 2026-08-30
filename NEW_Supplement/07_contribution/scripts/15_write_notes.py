"""给每次运行补一份 说明.md。

NEW_Supplement 的约定是每个阶段目录固定三样：说明.md、figures/、data/。
说明.md 是给人看的：这次跑了什么、用了什么参数、产出在哪、结果长什么样。
逐个手写既慢又容易和实际产物对不上，所以从 config.json 和 data/ 下的 CSV
自动生成——写出来的东西保证和盘上的文件一致。

对已经跑完的运行也适用，扫一遍 outputs/ 全部补上。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

STAGE_DESC = {
    "L0_calibration": ("L0 校准", "抽若干个字段组合，真的按这个组合从零训练一个"
                       "普通网络，把真实 R² 和代理模型给出的 v(S) 摆在一起比。"
                       "代理模型不准的话，后面所有贡献值都不成立，所以这一步最先做。"),
    "L0_tune": ("L0 定参数", "扫网络大小和训练轮次。判据是校准偏差而不是验证损失——"
                "验证损失低只说明在随机掩码上预测得准，不代表查任意字段组合时给出的"
                "R² 对得上。"),
    "L0_noise_floor": ("L0 噪声下限", "同一个字段组合换随机种子重训多次，量参照答案"
                       "自己有多不稳。代理模型的偏差要和这个数比才知道大小。"),
    "L0_diff_diagnosis": ("L0 增量诊断", "代理模型对'加一个字段带来多少增量'的判断，"
                          "在什么条件下可信。按增量大小分档看，"
                          "并区分真值本身能不能测出来。"),
    "L0_budget": ("L0 采样量", "抽多少个字段组合才够。用一对逐个数完全相同的字段"
                  "当绝对标尺（理论差值为 0），比只看相邻两档稳不稳更硬。"),
    "L0_symmetry_source": ("L0 对称性溯源", "对称性误差出在代理模型这一环还是"
                           "贡献值算法这一环。"),
    "L0_fix_symmetry": ("L0 修对称性", "权重衰减与多种子平均对代理模型对称性的作用。"),
    "L0_mask_dist": ("L0 掩码分布", "训练时掩码规模按均匀 / Shapley 核 / 两者混合"
                     "三种分布的对照。"),
    "L3_synthetic": ("L3 人造真值", "用真实数据列拼出五道答案由构造决定的题："
                     "可加、纯协同、纯替身、汇总替代、纯噪声。候选池压到十几个字段，"
                     "全部组合可以穷举，连采样误差都排除掉。"),
    "L1L2_truth": ("L1 公理自检 + L2 已知关系真值", "L1 查三条不需要领域知识的性质："
                   "完全相同的字段份额必须相等、恒定字段份额必须为 0、所有份额之和"
                   "必须等于全部发布相比什么都不发布多出来的能力。"
                   "L2 用 RTS-GMLC 数据自带的精确汇总关系当答案键，"
                   "其中系统总发电有四条互不相同的精确路径。"),
    "main": ("主实验", "全部敏感目标的贡献值、四个分解量、与现有方法的对照、"
             "以及按各方法排序取前 k 个字段独立重训的认证曲线。"),
    "L5_stability": ("L5 稳定性", "换随机种子，比较'选中集合的重合度'（旧口径）"
                     "和'贡献值排序的秩相关'（新口径）。"),
    "multitarget": ("第二阶段·多目标", "共享主干 + 多个输出头，一次训练同时回答"
                    "全部敏感目标，产出字段 × 目标的贡献矩阵和枢纽字段。"),
}


def fmt_config(cfg: dict) -> str:
    lines = []
    for k, v in cfg.items():
        if isinstance(v, dict):
            inner = "，".join(f"{kk}={vv}" for kk, vv in v.items())
            lines.append(f"- **{k}**：{inner}")
        elif isinstance(v, list) and len(v) > 12:
            lines.append(f"- **{k}**：共 {len(v)} 项，前几个 {v[:6]} …")
        else:
            lines.append(f"- **{k}**：{v}")
    return "\n".join(lines)


def _cell(v) -> str:
    if isinstance(v, float):
        if v != v:                       # NaN
            return ""
        return f"{v:.4f}" if abs(v) < 1e5 else f"{v:.3e}"
    s = str(v)
    return s if len(s) <= 46 else s[:43] + "…"


def to_md(df: pd.DataFrame) -> str:
    """手写 markdown 表格。不用 pandas 的 to_markdown，那个要装 tabulate，
    为了一份说明文档往用户的环境里加依赖不值当。"""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def preview(path: Path, n: int = 12, max_cols: int = 12) -> str:
    try:
        df = pd.read_csv(path)
    except Exception as e:                                       # noqa: BLE001
        return f"（读取失败：{type(e).__name__}）"
    shown = df.iloc[:n, :max_cols]
    txt = to_md(shown)
    note = f"\n\n（共 {len(df)} 行 × {len(df.columns)} 列"
    bits = []
    if len(df) > n:
        bits.append(f"只列前 {n} 行")
    if len(df.columns) > max_cols:
        bits.append(f"只列前 {max_cols} 列")
    note += ("，" + "、".join(bits) + "）") if bits else "）"
    return txt + note


def write_note(run: Path, force: bool = False) -> bool:
    note = run / "说明.md"
    if note.exists() and not force:
        return False
    datd, figd = run / "data", run / "figures"
    stage = run.parent.name
    dataset = run.parent.parent.name
    title, desc = STAGE_DESC.get(stage, (stage, ""))

    cfg = {}
    cf = datd / "config.json"
    if cf.exists():
        try:
            cfg = json.loads(cf.read_text())
        except Exception:                                        # noqa: BLE001
            cfg = {}

    parts = [f"# {title}（{dataset}）", ""]
    if desc:
        parts += ["## 这一步在做什么", "", desc, ""]
    parts += [f"运行目录 `{run.name}`，数据集 `{dataset}`。", ""]

    if cfg:
        parts += ["## 参数", "", fmt_config(cfg), ""]

    figs = sorted(figd.glob("*.png")) if figd.exists() else []
    if figs:
        parts += ["## 图", ""]
        parts += [f"- `figures/{f.name}`" for f in figs]
        parts += [""]

    csvs = sorted(datd.glob("*.csv")) if datd.exists() else []
    if csvs:
        parts += ["## 数据", ""]
        parts += [f"- `data/{f.name}`" for f in csvs]
        parts += [""]
        # 只给主表出预览，避免说明文档变成数据倾倒
        main_csv = None
        for cand in ("summary.csv", "verdicts.json", "stability.csv",
                     "hub_fields.csv", "calibration_summary.csv",
                     "tune_grid.csv", "budget_scan.csv", "mask_dist.csv",
                     "L2_route_verification.csv"):
            q = datd / cand
            if q.exists() and q.suffix == ".csv":
                main_csv = q
                break
        if main_csv is None and csvs:
            main_csv = csvs[0]
        if main_csv is not None:
            parts += [f"## 主表预览 `data/{main_csv.name}`", "",
                      preview(main_csv), ""]

    vj = datd / "verdicts.json"
    if vj.exists():
        parts += ["## 核对结论 `data/verdicts.json`", "",
                  "```json",
                  vj.read_text(encoding="utf-8")[:4000],
                  "```", ""]

    parts += ["---", "",
              "本文件由 `scripts/15_write_notes.py` 从本目录的 `config.json` 与",
              "`data/` 下的表自动生成，保证与盘上的产物一致。",
              "结论性的分析写在 `07_contribution/logs/` 下的日志里。"]
    note.write_text("\n".join(parts), encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已存在也重写")
    a = ap.parse_args()

    if not OUT.exists():
        print("还没有任何产物")
        return
    runs = sorted(p for p in OUT.rglob("run_*") if p.is_dir())
    n_new = 0
    for r in runs:
        if write_note(r, a.force):
            n_new += 1
            print(f"  写入 {r.relative_to(OUT)}/说明.md")
    print(f"\n共 {len(runs)} 次运行，新写 {n_new} 份说明。")


if __name__ == "__main__":
    main()
