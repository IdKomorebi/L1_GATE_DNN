"""真实数据里的答案键：哪些字段组合确实能精确推出哪个字段。

这些不是猜的，是用最小二乘逐条实测出来的（残差比见每条的 rr 字段，
全部在 1e-9 到 1e-10 量级，系数精确等于 1.0000）。RTS-GMLC 的数据由
交流潮流仿真生成，各级汇总量本来就是加出来的，所以这些关系是数据的定义，
不是拟合出来的巧合。

为什么这是最有说服力的检验
--------------------------
`system_generation_mw`（系统总发电）在候选池里**至少有四条互不相同的精确路径**：

  路径一  六个分燃料出力之和
  路径二  三个分区发电总量之和
  路径三  系统负荷 + 系统网损（功率平衡）
  路径四  十五个"分区×分燃料"出力之和

四条路径**各自单独就够**。正确的贡献值必须把份额分给全部四条路径上的字段，
并且把它们全部标成替身型（抽掉任何一个都不痛，因为别的路径还在）。
而稀疏门控的目标函数是"用最少的字段达到足够精度"，它必然只留下一条路径，
把另外三条路径上的字段判为零贡献——于是"把这些字段处理掉就安全了"这个结论
就是错的：另外三条路径原封不动地留在发布清单里。

这正好解释了处置实验里那个反常的结果：只处置被选中的字段、其余照常发布，
整体可推断性几乎不降（RTS 只降 0.007，PJM 甚至升了 0.012）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_src"))
import identity as idt          # noqa: E402

FUELS = ["coal", "natural_gas", "nuclear", "hydro", "wind", "solar"]


# 每个条目：伪目标字段 → 若干条各自单独就够的精确路径
RTS_ROUTES: dict[str, dict] = {
    "system_load_actual_mw": {
        "中文名": "系统负荷",
        "路径": {
            "三个分区负荷之和": [f"area_{i}_load_actual_mw" for i in (1, 2, 3)],
            "系统发电减网损": ["system_generation_mw", "system_losses_mw"],
        },
        "说明": "两条路径。第二条是功率平衡，属于电力系统常识。",
    },
    "system_generation_mw": {
        "中文名": "系统总发电",
        "路径": {
            "六个分燃料之和": [f"gen_fuel_{f}_mw" for f in FUELS],
            "三个分区发电之和": [f"area_{i}_generation_mw" for i in (1, 2, 3)],
            "负荷加网损": ["system_load_actual_mw", "system_losses_mw"],
            "分区分燃料之和": [f"area_{i}_fuel_{f}_mw"
                        for i in (1, 2, 3) for f in FUELS],
        },
        "说明": "四条互不相同的精确路径，是本目录判别力最强的一道真实题。",
    },
    "area_1_generation_mw": {
        "中文名": "1区发电总量",
        "路径": {
            "本区各燃料之和": [f"area_1_fuel_{f}_mw" for f in FUELS],
            "系统发电减其余两区": ["system_generation_mw", "area_2_generation_mw",
                          "area_3_generation_mw"],
        },
        "说明": "两条路径，一条向下拆到分燃料，一条向上借道系统汇总。",
    },
    "gen_fuel_coal_mw": {
        "中文名": "系统煤电出力",
        "路径": {
            "三个分区煤电之和": [f"area_{i}_fuel_coal_mw" for i in (1, 2, 3)],
        },
        "说明": "单条路径，用来对照——只有一条路时各方法应当都能找对。",
    },
}


def verify_routes(df: pd.DataFrame, routes: dict, tol: float = 1e-6) -> pd.DataFrame:
    """逐条核对路径确实成立，返回实测残差比。

    这一步每次跑实验都要重做一遍。写死在代码里的"已知关系"是最容易腐烂的东西：
    换个数据集版本、少一个字段，关系就不成立了，而代码不会报错，
    只会安静地给出错误的答案键。
    """
    rows = []
    for tgt, info in routes.items():
        for rname, fields in info["路径"].items():
            miss = [f for f in fields if f not in df.columns]
            if miss or tgt not in df.columns:
                rows.append(dict(伪目标=tgt, 路径=rname, 字段数=len(fields),
                                 残差比=np.nan, 成立=False, 缺失字段=";".join(miss)))
                continue
            coefs, const, rr = idt._refit(df, tgt, list(fields))
            rows.append(dict(伪目标=tgt, 路径=rname, 字段数=len(fields),
                             残差比=rr, 成立=bool(rr < tol),
                             系数最小=min(coefs.values()),
                             系数最大=max(coefs.values()), 缺失字段=""))
    return pd.DataFrame(rows)


def route_fields(info: dict) -> set[str]:
    """某个伪目标的全部路径字段并集——这些都应当拿到份额。"""
    out: set[str] = set()
    for fields in info["路径"].values():
        out |= set(fields)
    return out


def exact_twins(df: pd.DataFrame, target: str, pool: list[str],
                tol: float = 1e-9) -> list[str]:
    """池子里哪些字段和目标**逐个数完全相同**。

    这类字段是最干净的答案键：它们互为完美替身，正确的贡献值必须给它们相等的份额，
    而各自的不可替代性必须为 0。
    """
    if target not in df.columns:
        return []
    y = df[target].to_numpy(float)
    out = []
    for c in pool:
        v = df[c].to_numpy(float)
        scale = max(np.abs(y).max(), 1e-12)
        if np.max(np.abs(v - y)) / scale < tol:
            out.append(c)
    return out


def duplicate_groups_in_pool(df: pd.DataFrame, pool: list[str],
                             decimals: int = 9) -> list[list[str]]:
    """池子内部取值完全相同的字段组——对称性检验用。"""
    seen: dict[tuple, list[str]] = {}
    for c in pool:
        key = tuple(np.round(df[c].to_numpy(dtype=float), decimals))
        seen.setdefault(key, []).append(c)
    return [g for g in seen.values() if len(g) > 1]
