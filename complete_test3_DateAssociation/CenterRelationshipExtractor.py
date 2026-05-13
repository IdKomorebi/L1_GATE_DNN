import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["extract_center_relationships", "DEFAULT_THRESHOLDS", "DEFAULT_METRICS_ORDER"]


@dataclass(frozen=True)
class ThresholdConfig:
    """每种关联指标的阈值配置（按绝对值比较）。"""

    thresholds: Dict[str, float]

    def threshold_for(self, metric: str) -> Optional[float]:
        return self.thresholds.get(metric)


DEFAULT_METRICS_ORDER: List[str] = ["nmi", "spearman", "pearson", "kendall", "distance_corr", "hsic"]
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "nmi": 0.05,
    "spearman": 0.1,
    "pearson": 0.1,
    "kendall": 0.08,
    "distance_corr": 0.1,
    "hsic": 0.05,
}


def _normalize_name(s: str) -> str:
    return str(s).strip()


def _extract_secondary_name(full_name: str) -> str:
    """
    从完整列名中提取二级列名（"|"后面的部分）。
    例如："总发电量 | 地区电力总发电量" -> "地区电力总发电量"
    如果没有"|"，则返回原名称（去除首尾空格）。
    """
    full_name = _normalize_name(full_name)
    if "|" in full_name:
        parts = full_name.split("|", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return full_name


def _detect_center_matches(df: pd.DataFrame, center_name: str) -> pd.DataFrame:
    """
    匹配中心列名：只比较二级列名（"|"后面的部分）。
    center_name 应该是二级列名，例如"地区电力总发电量"。
    """
    center_name = _normalize_name(center_name)
    a_secondary = df["column_a"].astype(str).map(_extract_secondary_name)
    b_secondary = df["column_b"].astype(str).map(_extract_secondary_name)
    mask = (a_secondary == center_name) | (b_secondary == center_name)
    return df.loc[mask].copy()


def _other_column(row: pd.Series, center_name: str) -> str:
    """
    返回与 center_name 匹配的另一列（完整列名）。
    center_name 应该是二级列名。
    """
    center_name = _normalize_name(center_name)
    a_full = _normalize_name(row["column_a"])
    b_full = _normalize_name(row["column_b"])
    a_secondary = _extract_secondary_name(a_full)
    b_secondary = _extract_secondary_name(b_full)
    
    if a_secondary == center_name:
        return b_full
    elif b_secondary == center_name:
        return a_full
    else:
        # 理论上不应该到这里，但为了安全返回一个值
        return a_full


def _available_metrics(df: pd.DataFrame, preferred_order: Sequence[str]) -> List[str]:
    return [m for m in preferred_order if m in df.columns]


def _generate_output_path(input_csv: str, center_name: str) -> str:
    """
    根据输入文件名和中心列名自动生成输出路径。
    例如：DataAnalyze/realdata4_relationships.csv + "地区电力总发电量"
     -> CenterDataRelationships/center_realdata4_地区电力总发电量.csv
    """
    # 从输入文件名提取数据编号
    basename = os.path.basename(input_csv)
    match = re.search(r'realdata(\d+)', basename, re.IGNORECASE)
    data_num = match.group(1) if match else "unknown"
    
    # 清理中心列名（用于文件名）
    center_safe = "".join(c for c in center_name if c.isalnum() or c in (" ", "_", "-")).strip()
    center_safe = center_safe.replace(" ", "_")
    if not center_safe:
        center_safe = "center"
    
    output_dir = "CenterDataRelationships"
    output_filename = f"center_realdata{data_num}_{center_safe}.csv"
    return os.path.join(output_dir, output_filename)


def extract_center_relationships(
    input_csv: str,
    center_name: str,
    output_csv: Optional[str] = None,
    thresholds: Optional[Dict[str, float]] = None,
    metrics_order: Optional[Sequence[str]] = None,
    require_any_metric: bool = False,
) -> str:
    """
    从 realdata*_relationships.csv 中提取“以 center_name 为中心”的所有候选关系，并输出明细 CSV。

    输出列结构：
      - index
      - center
      - related
      - 对每个 metric：metric 值列 + metric_达阈值(阈值=xx) 的 0/1 列

    Args:
        input_csv: 关系文件路径（包含 column_a/column_b 以及若干指标列）
        center_name: 中心列名（二级列名，例如"地区电力总发电量"；会匹配"|"后面的部分）
        output_csv: 输出文件路径；如果为 None，则根据 input_csv 和 center_name 自动生成
        thresholds: 各指标阈值 dict；不传则用默认
        metrics_order: 指标顺序；不传则用默认
        require_any_metric: 若为 True，则只输出至少有一个指标达阈值的关系行
    """
    if output_csv is None:
        output_csv = _generate_output_path(input_csv, center_name)
    df = pd.read_csv(input_csv)
    if "column_a" not in df.columns or "column_b" not in df.columns:
        raise ValueError("输入CSV必须包含 column_a 和 column_b 列。")

    thresholds_cfg = ThresholdConfig(thresholds or DEFAULT_THRESHOLDS)
    metrics = _available_metrics(df, metrics_order or DEFAULT_METRICS_ORDER)
    if not metrics:
        raise ValueError("输入CSV中没有找到任何可用的指标列。")

    df_center = _detect_center_matches(df, center_name)
    if df_center.empty:
        raise ValueError(f"未找到中心列名 '{center_name}' 的任何关系（请检查是否与CSV一致）。")

    # 每个 related 可能出现多行（不同方向/重复），这里按“每个 related 取最大绝对值”聚合
    df_center["related"] = df_center.apply(lambda r: _other_column(r, center_name), axis=1)

    agg_map = {m: (lambda s: float(np.nanmax(np.abs(pd.to_numeric(s, errors="coerce").to_numpy())))) for m in metrics}
    grouped = df_center.groupby("related", dropna=False).agg(agg_map).reset_index()
    # 保存时使用完整的中心列名（如果找到的话），否则使用传入的二级列名
    center_full_name = None
    if not df_center.empty:
        # 尝试从匹配的行中找到完整的中心列名
        first_row = df_center.iloc[0]
        a_secondary = _extract_secondary_name(str(first_row["column_a"]))
        b_secondary = _extract_secondary_name(str(first_row["column_b"]))
        if a_secondary == center_name:
            center_full_name = _normalize_name(str(first_row["column_a"]))
        elif b_secondary == center_name:
            center_full_name = _normalize_name(str(first_row["column_b"]))
    grouped.insert(0, "center", center_full_name if center_full_name else center_name)

    # 生成达阈值标记列
    any_pass = np.zeros(len(grouped), dtype=bool)
    for m in metrics:
        th = thresholds_cfg.threshold_for(m)
        if th is None:
            # 没配置阈值就认为都不通过
            flag = np.zeros(len(grouped), dtype=int)
            grouped[f"{m}_达阈值(阈值=NA)"] = flag
            continue

        vals = pd.to_numeric(grouped[m], errors="coerce").to_numpy()
        passed = np.isfinite(vals) & (np.abs(vals) >= float(th))
        any_pass |= passed
        grouped[f"{m}_达阈值(阈值={th})"] = passed.astype(int)

    if require_any_metric:
        grouped = grouped.loc[any_pass].copy()

    # 排序：先按“通过的指标数”降序，再按 nmi（或第一个指标）降序
    flag_cols = [c for c in grouped.columns if c.endswith(")")]
    if flag_cols:
        grouped["_pass_count"] = grouped[flag_cols].sum(axis=1)
        primary_metric = "nmi" if "nmi" in metrics else metrics[0]
        grouped.sort_values(by=["_pass_count", primary_metric], ascending=[False, False], inplace=True)
        grouped.drop(columns=["_pass_count"], inplace=True)

    # 插入 index
    grouped.insert(0, "index", range(1, len(grouped) + 1))

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    grouped.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_csv


def _parse_thresholds(s: Optional[str]) -> Optional[Dict[str, float]]:
    """
    解析形如 "nmi=0.05,spearman=0.1,pearson=0.1,kendall=0.08,distance_corr=0.1,hsic=0.05"
    """
    if not s:
        return None
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"阈值参数格式错误: {part}，应为 key=value")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def main(
    input_csv: str,
    center_name: str,
    output_csv: Optional[str] = None,
    thresholds: Optional[Dict[str, float]] = None,
    require_any_metric: bool = True,
) -> str:
    """
    直接在当前文件里通过函数参数运行。

    你只需要修改文件底部的 INPUT_CSV / CENTER_NAME / THRESHOLDS，
    输出路径会根据输入文件名和中心列名自动生成。
    """
    output = extract_center_relationships(
        input_csv=input_csv,
        center_name=center_name,
        output_csv=output_csv,
        thresholds=thresholds,
        metrics_order=DEFAULT_METRICS_ORDER,
        require_any_metric=require_any_metric,
    )
    print(f"已输出: {output}")
    return output


if __name__ == "__main__":
    # =========================
    # 在这里改参数（推荐方式）
    # =========================
    INPUT_CSV = "DataAnalyze/realdata4_relationships.csv"
    CENTER_NAME = "地区节点阻塞价格"  # 只需提供二级列名（"|"后面的部分）

    # 可选：自定义六种阈值；不填则使用 DEFAULT_THRESHOLDS
    THRESHOLDS = DEFAULT_THRESHOLDS

    # 可选：True=只保留至少一个指标达阈值的关系；False=输出全部关系明细
    REQUIRE_ANY_METRIC = True

    # 输出路径会自动根据 INPUT_CSV 和 CENTER_NAME 生成
    main(
        input_csv=INPUT_CSV,
        center_name=CENTER_NAME,
        output_csv=None,  # None 表示自动生成
        thresholds=THRESHOLDS,
        require_any_metric=REQUIRE_ANY_METRIC,
    )

