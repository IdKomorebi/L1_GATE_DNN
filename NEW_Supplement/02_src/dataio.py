"""数据读取与预处理。

原料放在 01_data/raw/pjm_<year>/，清洗结果写到 01_data/clean/pjm_<year>/。

清洗只做机械处理，不做任何依赖判据的删除：
  - 时间列分离，单独保存，供后续按时间切分实验使用
  - 缺失比例高的列整列删除（main 版本）或保留后按行删（allcols 版本）
  - 缺失比例低的列按时间线性插值补齐

全零字段和完全重复字段不在这里删除，它们由第一类检测给出判据后再处理。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "01_data" / "raw"
CLEAN = ROOT / "01_data" / "clean"
OUTPUTS = ROOT / "04_outputs"

TIME_COLS = ["datetime_beginning_utc", "datetime_beginning_ept"]

# 缺失率超过这个比例的列，在 main 版本里整列删除
HIGH_MISSING = 0.10
# 缺失率低于这个比例的列，直接插值补齐
INTERP_MAX = 0.01


@dataclass
class CleanReport:
    year: int
    raw_rows: int
    raw_cols: int
    dropped_cols: list[str] = field(default_factory=list)
    interpolated: dict[str, int] = field(default_factory=dict)
    main_shape: tuple[int, int] = (0, 0)
    allcols_shape: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "raw_rows": self.raw_rows,
            "raw_cols": self.raw_cols,
            "dropped_cols": self.dropped_cols,
            "interpolated": self.interpolated,
            "main_shape": list(self.main_shape),
            "allcols_shape": list(self.allcols_shape),
        }


def raw_csv(year: int) -> Path:
    return RAW / f"pjm_{year}" / f"pjm_rto_hourly_{year}_aligned_processed_one_header.csv"


def load_raw(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (时间列, 数值字段)。"""
    df = pd.read_csv(raw_csv(year))
    time = df[[c for c in TIME_COLS if c in df.columns]].copy()
    data = df.drop(columns=[c for c in TIME_COLS if c in df.columns])
    data = data.apply(pd.to_numeric, errors="coerce")
    return time, data


def clean(year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, CleanReport]:
    """返回 (时间列, main 版本, allcols 版本, 报告)。"""
    time, data = load_raw(year)
    rep = CleanReport(year=year, raw_rows=len(data), raw_cols=data.shape[1])

    miss = data.isna().mean()
    high = sorted(miss[miss > HIGH_MISSING].index)
    low = sorted(miss[(miss > 0) & (miss <= INTERP_MAX)].index)
    rep.dropped_cols = high
    rep.interpolated = {c: int(data[c].isna().sum()) for c in low}

    # 低缺失列按时间顺序线性插值；两端用最近值填充
    filled = data.copy()
    if low:
        filled[low] = filled[low].interpolate(method="linear", limit_direction="both")

    main = filled.drop(columns=high).dropna()
    allcols = filled.dropna()

    rep.main_shape = main.shape
    rep.allcols_shape = allcols.shape
    return time, main, allcols, rep


def build(year: int) -> CleanReport:
    time, main, allcols, rep = clean(year)
    out = CLEAN / f"pjm_{year}"
    out.mkdir(parents=True, exist_ok=True)

    time.loc[main.index].to_csv(out / "time_index_main.csv", index=False)
    time.loc[allcols.index].to_csv(out / "time_index_allcols.csv", index=False)
    main.to_csv(out / "main.csv", index=False)
    allcols.to_csv(out / "allcols.csv", index=False)
    (out / "clean_report.json").write_text(
        json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rep


def load_clean(year: int, version: str = "main") -> pd.DataFrame:
    return pd.read_csv(CLEAN / f"pjm_{year}" / f"{version}.csv")


def load_time(year: int, version: str = "main") -> pd.DataFrame:
    return pd.read_csv(CLEAN / f"pjm_{year}" / f"time_index_{version}.csv")


def out_dir(year: int, stage: str, version: str | None = None) -> Path:
    """阶段输出目录。version 传 "pjm_2025_v2" 之类可切到重跑版本，不覆盖原有结果。"""
    d = OUTPUTS / (version or f"pjm_{year}") / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def split_dirs(base: Path) -> tuple[Path, Path]:
    """把一个输出目录分成 figures/ 与 data/ 两个子目录。

    文档留在 base 下（`说明.md` 或 `总结.md`），图进 figures/，其余文件进 data/。
    这样只看图和文档的人不必在一堆 CSV 里翻。
    """
    fig, dat = base / "figures", base / "data"
    fig.mkdir(parents=True, exist_ok=True)
    dat.mkdir(parents=True, exist_ok=True)
    return fig, dat
