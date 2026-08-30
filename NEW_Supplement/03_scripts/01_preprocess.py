"""第 1 步：数据清洗。

只做机械处理：分离时间列、缺失严重的列整列删（main 版本）、缺失少的列插值补齐。
全零字段和重复字段留给第 2 步的第一类检测按判据处理。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_src"))
import dataio  # noqa: E402

YEARS = [2024, 2025]


def main() -> None:
    lines: list[str] = ["# 数据清洗报告", ""]
    for year in YEARS:
        rep = dataio.build(year)
        d = dataio.out_dir(year, "01_preprocess")

        print(f"\n===== pjm_{year} =====")
        print(f"  原始         {rep.raw_rows} 行 × {rep.raw_cols} 列")
        print(f"  整列删除     {rep.dropped_cols}")
        print(f"  插值补齐     {rep.interpolated}")
        print(f"  main 版本    {rep.main_shape[0]} 行 × {rep.main_shape[1]} 列")
        print(f"  allcols 版本 {rep.allcols_shape[0]} 行 × {rep.allcols_shape[1]} 列")

        lines += [
            f"## pjm_{year}",
            "",
            f"- 原始：{rep.raw_rows} 行 × {rep.raw_cols} 列",
            f"- 整列删除（缺失率 > {dataio.HIGH_MISSING:.0%}）：{rep.dropped_cols or '无'}",
            f"- 插值补齐（缺失率 ≤ {dataio.INTERP_MAX:.0%}）：共 {len(rep.interpolated)} 列",
        ]
        for c, n in sorted(rep.interpolated.items()):
            lines.append(f"    - `{c}`：补 {n} 个点")
        lines += [
            f"- **main 版本**（主实验用）：{rep.main_shape[0]} 行 × {rep.main_shape[1]} 列",
            f"- **allcols 版本**（对照用）：{rep.allcols_shape[0]} 行 × {rep.allcols_shape[1]} 列",
            f"- 样本增益：main 比 allcols 多 "
            f"{rep.main_shape[0] - rep.allcols_shape[0]} 行 "
            f"（{(rep.main_shape[0] / rep.allcols_shape[0] - 1) * 100:.1f}%）",
            "",
        ]
        pd.DataFrame([rep.to_dict()]).to_json(
            d / "clean_report.json", orient="records", force_ascii=False, indent=2
        )

    (dataio.ROOT / "04_outputs" / "preprocess_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("\n报告写入 04_outputs/preprocess_report.md")


if __name__ == "__main__":
    main()
