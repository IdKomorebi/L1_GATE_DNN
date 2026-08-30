"""第一类（公式型）关系的检测。

核心思路：字段之间若存在"若干字段加权相加正好等于零"的精确关系，等价于把字段
摆成表格后，这张表格的秩比列数少。秩少多少，就有多少条互相独立的关系。

流程：
  1. 求秩 —— 得到"总共有几条"，作为完整性标尺
  2. 门槛扫描 —— 找出"门槛怎么调结果都不变"的平台段，据此定门槛
  3. 高斯消元化简 —— 把抽象的关系变成"某字段 = 某几个字段的组合"的可读形式
  4. 按支撑集重新最小二乘拟合 —— 得到干净的系数和诚实的残差
  5. 分类 —— 恒定字段 / 完全重复字段 / 真正的公式关系
  6. 换到对数空间重跑 —— 捕获比例型和乘积型关系
  7. 跨年复验 —— 真关系换一年也成立
  8. 反复剥离 —— 删到目标字段再也无法被精确算出为止
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONST_NAME = "<常数项>"


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

def design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """字段矩阵后面接一列全 1，用来容纳公式里的常数项。"""
    X = np.column_stack([df.to_numpy(float), np.ones(len(df))])
    return X, list(df.columns) + [CONST_NAME]


def singular_ratios(df: pd.DataFrame) -> np.ndarray:
    """列归一化后的奇异值谱（除以最大值）。列归一化是必要的，否则量级小的列
    会因为数值小而假装成'接近零方向'。"""
    X, _ = design_matrix(df)
    nr = np.linalg.norm(X, axis=0)
    nr[nr == 0] = 1.0
    s = np.linalg.svd(X / nr, compute_uv=False)
    return s / s[0]


def threshold_scan(df: pd.DataFrame, tols: list[float]) -> pd.DataFrame:
    """门槛扫描：每个门槛下有多少条关系。平台段说明门槛不敏感。"""
    r = singular_ratios(df)
    rows = [{"tol": t, "n_relations": int((r < t).sum())} for t in tols]
    out = pd.DataFrame(rows)
    out["plateau"] = out["n_relations"].eq(out["n_relations"].shift()) | out[
        "n_relations"
    ].eq(out["n_relations"].shift(-1))
    return out


# --------------------------------------------------------------------------
# 关系对象
# --------------------------------------------------------------------------

@dataclass
class Identity:
    lead: str
    coefs: dict[str, float] = field(default_factory=dict)
    const: float = 0.0
    residual_ratio: float = np.nan
    kind: str = ""
    space: str = "linear"
    cross_year: float = np.nan

    @property
    def support(self) -> list[str]:
        return sorted(self.coefs)

    @property
    def clean_coefs(self) -> bool:
        """系数是否接近整数。账目关系的系数天然是 ±1、±2 这种干净的数，
        统计上凑出来的关系系数往往是难看的小数。"""
        if not self.coefs:
            return True
        v = np.array(list(self.coefs.values()))
        return bool(np.all(np.abs(v - np.round(v)) < 1e-6))

    def text(self) -> str:
        if not self.coefs:
            body = f"{self.const:.6g}"
        else:
            body = " ".join(f"{c:+.6g}*{n}" for n, c in sorted(self.coefs.items()))
            if abs(self.const) > 1e-9:
                body += f" {self.const:+.6g}"
        op = "=" if self.space == "linear" else "= exp"
        return f"{self.lead} {op} {body}"


# --------------------------------------------------------------------------
# 提取
# --------------------------------------------------------------------------

def _rref(A: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, list[int]]:
    """高斯消元化成行最简形，主元按绝对值最大挑选。"""
    A = A.copy()
    row, piv = 0, []
    for col in range(A.shape[1]):
        if row >= A.shape[0]:
            break
        j = row + int(np.argmax(np.abs(A[row:, col])))
        if abs(A[j, col]) < tol:
            continue
        A[[row, j]] = A[[j, row]]
        A[row] = A[row] / A[row, col]
        for i in range(A.shape[0]):
            if i != row and abs(A[i, col]) > 1e-12:
                A[i] = A[i] - A[i, col] * A[row]
        piv.append(col)
        row += 1
    return A[:row], piv


def _refit(df: pd.DataFrame, lead: str, support: list[str]) -> tuple[dict, float, float]:
    """在给定支撑集上重新最小二乘拟合，返回干净的系数、常数项和残差比。"""
    y = df[lead].to_numpy(float)
    sigma = y.std()
    if not support:
        return {}, float(y.mean()), 0.0 if sigma == 0 else float(np.nan)
    A = np.column_stack([df[s].to_numpy(float) for s in support] + [np.ones(len(df))])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ c
    rr = float(np.sqrt(np.mean(res**2)) / sigma) if sigma > 0 else float("nan")
    return {s: float(c[k]) for k, s in enumerate(support)}, float(c[-1]), rr


def _prune(df: pd.DataFrame, lead: str, support: list[str], tol: float) -> list[str]:
    """精简支撑集：按"贡献量 = |系数| × 该字段标准差"从小到大尝试删除，
    只要删掉之后残差仍在门槛内就真的删掉。

    化简出来的关系常带一些量级极小的伪系数（例如 1e-7），它们对关系没有实质
    贡献，却会让支撑集看起来很大、并在后续剥离时选错字段。
    """
    support = list(support)
    while len(support) > 0:
        coefs, _, rr = _refit(df, lead, support)
        if not np.isfinite(rr) or rr >= tol:
            return support
        contrib = {s: abs(coefs[s]) * df[s].std() for s in support}
        weakest = min(contrib, key=contrib.get)
        trial = [s for s in support if s != weakest]
        _, _, rr2 = _refit(df, lead, trial)
        if np.isfinite(rr2) and rr2 < tol:
            support = trial
        else:
            return support
    return support


def _classify(df: pd.DataFrame, ident: Identity) -> str:
    if df[ident.lead].nunique() <= 1:
        return "constant"
    if len(ident.coefs) == 1:
        (other, c), = ident.coefs.items()
        if abs(c - 1.0) < 1e-9 and abs(ident.const) < 1e-9:
            return "duplicate"
    return "formula"


def extract(
    df: pd.DataFrame,
    tol_rank: float = 1e-8,
    tol_resid: float = 1e-4,
    coef_eps: float = 1e-6,
) -> tuple[list[Identity], dict]:
    """求秩 + 化简，把全部精确关系提取成可读形式。

    两个门槛量纲不同，必须分开设：
      tol_rank  比较的是归一化后的奇异值比，用来数"总共有几条"
      tol_resid 比较的是某个字段自身的相对残差，用来判断一条关系写出来是否成立

    化简用的高斯消元按列顺序挑主导字段，可能把某条关系分配给不合适的字段，
    导致写出来不成立。所以化简之后再补一轮逐字段扫描，把漏掉的形式找回来。

    返回 (关系列表, 统计信息)。n_directions 是求秩给出的应有条数，n_valid 是
    最终写出成立的条数，两者的差额如实记在 stats 里。
    """
    X, names = design_matrix(df)
    nr = np.linalg.norm(X, axis=0)
    nr[nr == 0] = 1.0
    _, s, Vt = np.linalg.svd(X / nr, full_matrices=False)
    r = s / s[0]
    k = int((r < tol_rank).sum())
    stats = {"n_directions": k, "n_valid": 0, "n_from_rref": 0,
             "n_recovered": 0, "unresolved": []}
    if k == 0:
        return [], stats

    out: list[Identity] = []
    seen: set[frozenset] = set()

    def add(lead: str, support: list[str]) -> bool:
        key = frozenset([lead] + list(support))
        if key in seen:
            return False
        support = _prune(df, lead, support, tol_resid)
        coefs, const, rr = _refit(df, lead, support)
        if df[lead].nunique() > 1 and not (np.isfinite(rr) and rr < tol_resid):
            return False
        key = frozenset([lead] + list(support))
        if key in seen:
            return False
        seen.add(key)
        it = Identity(lead=lead, coefs=coefs, const=const, residual_ratio=rr)
        it.kind = _classify(df, it)
        out.append(it)
        return True

    # 第一轮：高斯消元化简
    basis = Vt[-k:] / nr
    A, piv = _rref(basis)
    for i in range(A.shape[0]):
        lead = names[piv[i]]
        if lead == CONST_NAME:
            continue
        v = A[i].copy()
        v[np.abs(v) < coef_eps * np.max(np.abs(v))] = 0.0
        support = [names[j] for j in np.nonzero(v)[0]
                   if j != piv[i] and names[j] != CONST_NAME]
        if add(lead, support):
            stats["n_from_rref"] += 1
    # 第二轮：逐字段扫描，补足化简没能写出来的关系
    if len(out) < k:
        covered = {c for it in out for c in [it.lead] + it.support}
        for lead in df.columns:
            if len(out) >= k:
                break
            if lead in covered:
                continue
            pool = [c for c in df.columns if c != lead]
            support = _greedy_support(df, lead, pool, tol_resid)
            if support and add(lead, support):
                stats["n_recovered"] += 1
                covered |= {lead, *support}

    stats["n_valid"] = len(out)
    if len(out) < k:
        stats["unresolved"] = [k - len(out)]
    return out, stats


# --------------------------------------------------------------------------
# 对数空间：比例型与乘积型
# --------------------------------------------------------------------------

def log_frame(df: pd.DataFrame, min_positive_ratio: float = 1.0) -> pd.DataFrame:
    """取全部取值恒为正的字段，取对数。乘除关系在取对数之后变成加减关系。"""
    ok = [c for c in df.columns if (df[c] > 0).mean() >= min_positive_ratio]
    return np.log(df[ok])


def extract_log(
    df: pd.DataFrame, tol_rank: float = 1e-8, tol_resid: float = 1e-4
) -> tuple[list[Identity], dict]:
    lg = log_frame(df)
    if lg.shape[1] < 3:
        return [], {"n_directions": 0, "n_valid": 0}
    out, stats = extract(lg, tol_rank=tol_rank, tol_resid=tol_resid)
    for it in out:
        it.space = "log"
    return out, stats


# --------------------------------------------------------------------------
# 跨年复验
# --------------------------------------------------------------------------

def verify(ident: Identity, other: pd.DataFrame) -> float:
    """把关系原样搬到另一年的数据上重算残差比。真关系不会变差几个数量级。"""
    need = [ident.lead] + ident.support
    if any(c not in other.columns for c in need):
        return float("nan")
    d = other[need].dropna()
    if ident.space == "log":
        if (d <= 0).any().any():
            return float("nan")
        d = np.log(d)
    y = d[ident.lead].to_numpy(float)
    sigma = y.std()
    if sigma == 0:
        return 0.0
    pred = np.full(len(d), ident.const)
    for n, c in ident.coefs.items():
        pred = pred + c * d[n].to_numpy(float)
    return float(np.sqrt(np.mean((y - pred) ** 2)) / sigma)


# --------------------------------------------------------------------------
# 已有公式验算
# --------------------------------------------------------------------------

def check_formula(df: pd.DataFrame, target: str, expr, calibrate: bool = False) -> dict:
    """把一条已有的公式代入数据算误差。

    calibrate=False 用公式原样（零自由参数），calibrate=True 允许最小二乘标定系数。
    两者之差量化了'口径差异的代价'。
    """
    y = df[target].to_numpy(float)
    sigma = y.std()
    parts = expr(df)
    if calibrate:
        A = np.column_stack([np.asarray(p, float) for p in parts] + [np.ones(len(df))])
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ c
        coef = [float(x) for x in c]
    else:
        pred = np.sum([np.asarray(p, float) for p in parts], axis=0)
        coef = None
    rr = float(np.sqrt(np.mean((y - pred) ** 2)) / sigma)
    return {"target": target, "calibrated": calibrate, "residual_ratio": rr, "coef": coef}


# --------------------------------------------------------------------------
# 反复剥离直到切断
# --------------------------------------------------------------------------

def exact_fit_ratio(df: pd.DataFrame, target: str, pool: list[str]) -> float:
    """用 pool 里的字段整体拟合 target，返回残差比。"""
    y = df[target].to_numpy(float)
    sigma = y.std()
    if sigma == 0 or not pool:
        return float("nan")
    A = np.column_stack([df[p].to_numpy(float) for p in pool] + [np.ones(len(df))])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.sqrt(np.mean((y - A @ c) ** 2)) / sigma)


def strip_closure(
    df: pd.DataFrame, target: str, tol: float = 1e-6, max_iter: int = 20
) -> tuple[list[str], list[dict]]:
    """反复剥离：删掉参与公式关系的字段后重新检测，直到目标字段再也无法被
    剩下的字段精确算出为止。返回 (被删字段, 每一轮的记录)。

    只删一个直接分量是不够的：若 Y = A - B 而 A = C + D，删掉 A 之后
    Y = C + D - B 仍然精确成立，关系只是换了个形式。
    """
    pool = [c for c in df.columns if c != target]
    removed: list[str] = []
    trace: list[dict] = []
    for step in range(max_iter):
        rr = exact_fit_ratio(df, target, pool)
        trace.append({"step": step, "n_pool": len(pool), "residual_ratio": rr})
        if not np.isfinite(rr) or rr >= tol:
            break
        # 找出这一轮里让目标可被精确算出的最小支撑集
        sub = df[[target] + pool]
        found, _ = extract(sub, tol_rank=1e-8, tol_resid=tol)
        idents = [i for i in found if i.lead == target]
        if idents:
            support = idents[0].support
        else:
            # 秩层面看不到（属于含舍入的关系），退回用逐步选择找支撑
            support = _greedy_support(df, target, pool, tol)
            support = _prune(df, target, support, tol)
        if not support:
            break
        # 删掉支撑集里贡献最大的那个：贡献 = |系数| × 该字段标准差
        coefs, _, _ = _refit(df, target, support)
        victim = max(support, key=lambda c: abs(coefs.get(c, 0.0)) * df[c].std())
        pool.remove(victim)
        removed.append(victim)
        trace[-1]["removed"] = victim
        trace[-1]["support"] = support
    return removed, trace


def _greedy_support(
    df: pd.DataFrame, target: str, pool: list[str], tol: float, max_k: int = 8
) -> list[str]:
    y = df[target].to_numpy(float)
    sigma = y.std()
    resid = y - y.mean()
    chosen: list[str] = []
    for _ in range(max_k):
        best, score = None, -1.0
        for c in pool:
            if c in chosen:
                continue
            x = df[c].to_numpy(float)
            if x.std() == 0:
                continue
            v = abs(np.corrcoef(x, resid)[0, 1])
            if np.isfinite(v) and v > score:
                best, score = c, v
        if best is None:
            break
        chosen.append(best)
        A = np.column_stack([df[c].to_numpy(float) for c in chosen] + [np.ones(len(df))])
        cc, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ cc
        if np.sqrt(np.mean(resid**2)) / sigma < tol:
            return chosen
    return chosen


# --------------------------------------------------------------------------
# 稳定性判据：一条关系换一批数据还成不成立
# --------------------------------------------------------------------------

def stability(
    df: pd.DataFrame, lead: str, support: list[str], n_blocks: int = 5
) -> dict:
    """把数据按时间切成若干块，轮流留一块出来验算，看误差被放大多少倍。

    真公式关系在哪块数据上都成立，放大倍数接近 1；
    只在某段数据上凑出来的伪关系，换一块就失效，放大倍数是几百倍甚至更高。

    按时间切块而不是随机抽样，是因为电力数据前后小时高度相似，
    随机抽样会把相邻小时分到训练和验算两边，等于泄题。
    """
    if not support:
        return {"fit": float("nan"), "check": float("nan"), "amplify": float("inf")}
    y = df[lead].to_numpy(float)
    A = np.column_stack([df[s].to_numpy(float) for s in support] + [np.ones(len(df))])
    edges = np.linspace(0, len(df), n_blocks + 1).astype(int)
    fits, checks = [], []
    for b in range(n_blocks):
        te = np.zeros(len(df), bool)
        te[edges[b]:edges[b + 1]] = True
        tr = ~te
        if df[lead][tr].std() == 0 or df[lead][te].std() == 0:
            continue
        c, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        fits.append(np.sqrt(np.mean((y[tr] - A[tr] @ c) ** 2)) / y[tr].std())
        checks.append(np.sqrt(np.mean((y[te] - A[te] @ c) ** 2)) / y[te].std())
    if not fits:
        return {"fit": float("nan"), "check": float("nan"), "amplify": float("inf")}
    f, k = float(np.median(fits)), float(np.median(checks))
    return {"fit": f, "check": k,
            "amplify": float(k / f) if f > 0 else float("inf"),
            "worst_amplify": float(max(ck / fi for fi, ck in zip(fits, checks) if fi > 0))}


def background_level(
    df: pd.DataFrame, lead: str, pool: list[str], k: int,
    n_draws: int = 30, seed: int = 0
) -> float:
    """随机挑 k 个字段拟合目标能达到的残差水平（中位数）。

    用来判断"找到的这条关系到底特不特别"：如果它的残差和随便挑 k 个字段差不多，
    那它就不是一条关系，只是多元回归本来就有的拟合能力。
    """
    if k <= 0 or k >= len(pool):
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draws):
        sub = list(rng.choice(pool, k, replace=False))
        r = exact_fit_ratio(df, lead, sub)
        if np.isfinite(r):
            vals.append(r)
    return float(np.median(vals)) if vals else float("nan")


def participates_in_relation(
    df: pd.DataFrame, target: str, tol: float
) -> int:
    """这个字段参不参与关系。只能给出是非答案，**不能用来计数**。

    做法是比较去掉该字段前后的"接近零方向"数量。注意这个差值在数学上必然只能是
    0 或 1：任意两条涉及该字段的关系，缩放后相减就能把它消掉，变成一条不涉及它的
    关系，所以"涉及该字段"这件事在维数上最多贡献 1。

    松门槛下偶尔会算出 2，那是删掉一列后整条奇异值序列平移造成的数值假象，
    同样不能当计数用。"某字段有几条推断路径"是组合枚举问题，求秩给不出答案。
    """
    if target not in df.columns:
        return 0
    a = int((singular_ratios(df) < tol).sum())
    b = int((singular_ratios(df.drop(columns=[target])) < tol).sum())
    return a - b


# --------------------------------------------------------------------------
# 第一层剥离：按已知公式，不看残差
# --------------------------------------------------------------------------

FUELS = ["coal", "gas", "hydro", "multiple_fuels", "nuclear", "oil",
         "other_renewables", "solar", "storage", "wind"]

KNOWN_FORMULAS = [
    {"name": "日前电价三分量分解", "source": "PJM Data Miner 官方定义",
     "fields": ["total_lmp_da", "system_energy_price_da",
                "congestion_price_da", "marginal_loss_price_da"]},
    {"name": "实时电价三分量分解", "source": "PJM Data Miner 官方定义",
     "fields": ["total_lmp_rt", "system_energy_price_rt",
                "congestion_price_rt", "marginal_loss_price_rt"]},
    {"name": "净交换功率账目关系", "source": "零空间检测：残差 2.0e-15，系数 ±1",
     "fields": ["net_actual_interchange_mw", "net_sched_interchange_mw",
                "net_inadv_interchange_mw"]},
    {"name": "总交换功率同式", "source": "与净交换同构",
     "fields": ["gross_actual_interchange_mw", "gross_sched_interchange_mw",
                "gross_inadv_interchange_mw"]},
    {"name": "分燃料出力求和", "source": "PJM Generation by Fuel Type 定义",
     "fields": ["total_gen"] + [f"gen_fuel_{f}_mw" for f in FUELS]},
    {"name": "备用容量构成", "source": "零空间检测：残差 2.4e-05，系数 ±1",
     "fields": ["da_as_total_mw_primary_reserve",
                "da_as_total_mw_synchronized_reserve",
                "da_as_nsr_mw_primary_reserve"]},
    {"name": "功率平衡", "source": "电力系统常识",
     "fields": ["total_gen", "metered_load_mw", "total_losses",
                "net_actual_interchange_mw"]},
]


def formula_drop(target: str) -> tuple[list[str], list[str]]:
    """目标出现在哪条已知公式里，就把该公式涉及的其余字段全部剔除。

    **这一步不看残差。** 依据是官方文档和电力常识，不是数据——
    所以不需要任何阈值，也就不存在"门槛定多少"的主观性。

    这样做是为了堵住一个漏洞：日前总电价的 `总电价 ≈ 能量价格` 残差是 0.116，
    刚好高于按残差剥离的门槛 0.10 而漏了过去，导致"剥离之后"仍有一个字段
    就能把目标推到 R² = 0.9865。它本来就是官方公式的分量，
    应该在这一层被剔除，轮不到按残差去判。

    返回 (要剔除的字段, 命中了哪几条公式)。
    """
    drop, hit = set(), []
    for f in KNOWN_FORMULAS:
        if target in f["fields"]:
            drop |= set(f["fields"]) - {target}
            hit.append(f["name"])
    return sorted(drop), hit
