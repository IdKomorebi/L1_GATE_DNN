"""数据集登记表：把两个数据集统一成同一副面孔。

本目录的所有实验都只通过这里拿数据，好处是：换数据集只改一个名字，
候选池怎么来的、哪些是敏感目标、哪些字段是恒定的或者互为重复，
全部在这一个文件里说清楚，不散落在各个脚本里。

与 03_scripts 那一套的关系：那边的候选池要先做第一类公式剥离，
这里**默认不剥离**。原因是本目录的验证正需要把已知公式关系留在池子里当答案键——
剥离之后答案就没了。需要剥离的场景通过 strip= 参数单独打开。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # NEW_Supplement/
CLEAN = ROOT / "01_data" / "clean"
OUT = ROOT / "07_contribution" / "outputs"

# 复用 02_src 里的既有实现（只读，不修改那边任何文件）
sys.path.insert(0, str(ROOT / "02_src"))
import identity as idt        # noqa: E402


# ---------------------------------------------------------------- 目标字段

# RTS-GMLC v2：公开的是系统/分区/燃料级汇总，敏感的是具名设备状态。
# 这条边界不是人为指定的密级，是电网实际发布惯例——汇总量对外发布，
# 具体到某条线路、某台机组的运行状态不发布。
RTS_TARGETS = [
    "bus_115_va_deg", "bus_215_va_deg", "bus_315_va_deg",
    "branch_ab1_loading_pct", "branch_ab2_loading_pct", "branch_ab3_loading_pct",
    "branch_ca_1_loading_pct", "branch_cb_1_loading_pct",
    "gen_121_nuclear_1_pg_mw", "gen_218_cc_1_pg_mw",
    "gen_317_wind_1_pg_mw", "gen_321_cc_1_status",
]

RTS_CN = {
    "bus_115_va_deg": "115号母线电压相角",
    "bus_215_va_deg": "215号母线电压相角",
    "bus_315_va_deg": "315号母线电压相角",
    "branch_ab1_loading_pct": "AB1联络线负载率",
    "branch_ab2_loading_pct": "AB2联络线负载率",
    "branch_ab3_loading_pct": "AB3联络线负载率",
    "branch_ca_1_loading_pct": "CA1联络线负载率",
    "branch_cb_1_loading_pct": "CB1联络线负载率",
    "gen_121_nuclear_1_pg_mw": "121号核电机组出力",
    "gen_218_cc_1_pg_mw": "218号联合循环机组出力",
    "gen_317_wind_1_pg_mw": "317号风电机组出力",
    "gen_321_cc_1_status": "321号联合循环机组启停状态",
}

# PJM：12 个目标沿用 03_scripts 那一套（五处重复定义，取值完全一致）
PJM_TARGETS = [
    "net_actual_interchange_mw", "gross_actual_interchange_mw",
    "net_sched_interchange_mw", "total_gen", "metered_load_mw", "total_losses",
    "congestion_price_da", "congestion_price_rt", "marginal_loss_price_da",
    "total_lmp_da", "da_as_total_mw_primary_reserve",
    "da_as_total_mw_thirty_minutes_reserve",
]

PJM_CN = {
    "net_actual_interchange_mw": "净实际交换功率",
    "gross_actual_interchange_mw": "总实际交换功率",
    "net_sched_interchange_mw": "净计划交换功率",
    "total_gen": "总发电量",
    "metered_load_mw": "计量负荷",
    "total_losses": "网损",
    "congestion_price_da": "日前阻塞价",
    "congestion_price_rt": "实时阻塞价",
    "marginal_loss_price_da": "日前边际损耗价",
    "total_lmp_da": "日前总电价",
    "da_as_total_mw_primary_reserve": "日前主用备用总量",
    "da_as_total_mw_thirty_minutes_reserve": "日前30分钟备用总量",
}


@dataclass
class Dataset:
    """一个数据集的全部信息。"""
    name: str
    df: pd.DataFrame                 # 全部数值列
    targets: list[str]               # 敏感目标
    cn: dict[str, str]               # 字段中文名（只有目标有，其余回落到英文名）
    constants: list[str]             # 全期恒定字段（贡献值必须为 0）
    dup_groups: list[list[str]]      # 完全重复的字段组（组内贡献值必须相等）
    note: str = ""

    def pool(self, target: str, drop_constants: bool = True,
             drop_other_targets: bool = True, strip_formula: bool = False) -> list[str]:
        """构造某个目标的候选字段池。

        drop_constants     恒定字段没有任何信息，留着只是浪费算力（但 L1 检验要留）
        drop_other_targets 其余敏感目标本身也不发布，不能当推断来源
        strip_formula      是否剥离已知公式关系。**默认关闭**——本目录的真值验证
                           正需要这些关系留在池子里当答案键
        """
        cols = [c for c in self.df.columns if c != target]
        if drop_other_targets:
            cols = [c for c in cols if c not in self.targets]
        if drop_constants:
            cols = [c for c in cols if c not in self.constants]
        if strip_formula:
            drop, _hit = idt.formula_drop(target)
            cols = [c for c in cols if c not in set(drop)]
        return cols

    def label(self, col: str) -> str:
        return self.cn.get(col, col)


def _find_constants(df: pd.DataFrame) -> list[str]:
    n = df.nunique()
    return sorted(n[n <= 1].index.tolist())


def _find_dup_groups(df: pd.DataFrame, decimals: int = 9) -> list[list[str]]:
    """找出取值完全相同的字段组。

    用四舍五入到 9 位小数后的整列做哈希键。这个精度足以区分真正不同的字段，
    又能容忍 CSV 读写引入的最后一两位差异。
    """
    seen: dict[tuple, list[str]] = {}
    for c in df.columns:
        key = tuple(np.round(df[c].to_numpy(dtype=float), decimals))
        seen.setdefault(key, []).append(c)
    return [g for g in seen.values() if len(g) > 1]


def load(name: str) -> Dataset:
    """按名字加载数据集。可用名字见 REGISTRY。"""
    if name == "rts_v2":
        df = pd.read_csv(CLEAN / "rts_2020_v2" / "base.csv")
        targets, cn = RTS_TARGETS, RTS_CN
        note = ("RTS-GMLC 2020 扩展版，由 pandapower 交流潮流求解生成。"
                "物理关系已知，是本目录唯一有确切真值的数据集。")
    elif name == "rts_v1":
        df = pd.read_csv(CLEAN / "rts_2020" / "base.csv")
        targets, cn = RTS_TARGETS, RTS_CN
        note = "RTS-GMLC 2020 原版，公开候选字段较少。"
    elif name == "pjm_2025":
        df = pd.read_csv(CLEAN / "pjm_2025" / "main.csv")
        targets, cn = PJM_TARGETS, PJM_CN
        note = "PJM 2025 真实市场数据，无真值，作主结果数据集。"
    elif name == "pjm_2024":
        df = pd.read_csv(CLEAN / "pjm_2024" / "main.csv")
        targets, cn = PJM_TARGETS, PJM_CN
        note = "PJM 2024，用于跨年复验。"
    else:
        raise ValueError(f"未知数据集：{name}；可用：{list(REGISTRY)}")

    df = df.select_dtypes(include=[np.number]).astype(float)
    targets = [t for t in targets if t in df.columns]
    return Dataset(name=name, df=df, targets=targets, cn=dict(cn),
                   constants=_find_constants(df),
                   dup_groups=_find_dup_groups(df), note=note)


REGISTRY = ["rts_v2", "rts_v1", "pjm_2025", "pjm_2024"]


# ---------------------------------------------------------------- 输出目录

def run_dir(dataset: str, stage: str, stamp: str | None = None) -> tuple[Path, Path, Path]:
    """建立 outputs/<数据集>/<阶段>/run_<时间戳>/ 并返回 (根, figures, data)。

    沿用 NEW_Supplement 的既有约定：每次运行进各自带时间戳的目录，从不覆盖。
    """
    from datetime import datetime
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUT / dataset / stage / f"run_{stamp}"
    fig, dat = base / "figures", base / "data"
    for d in (base, fig, dat):
        d.mkdir(parents=True, exist_ok=True)
    return base, fig, dat


def latest_run(dataset: str, stage: str) -> Path | None:
    """取某个阶段最近一次运行的目录，供下游脚本消费上游产物。"""
    d = OUT / dataset / stage
    if not d.exists():
        return None
    runs = sorted(p for p in d.iterdir() if p.name.startswith("run_"))
    return runs[-1] if runs else None
