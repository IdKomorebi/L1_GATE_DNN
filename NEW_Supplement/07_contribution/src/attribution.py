"""从 v(S) 读出每个字段的贡献值，并把这个值拆开看。

为什么用 Shapley 值
------------------
手头有了函数 v(S)——任意字段组合的还原能力。现在要把 v(全部) 这块总能力
分摊到每个字段头上。"分摊"听起来随意，但其实只要提四条要求，分法就唯一确定了：

  加和性   所有字段的份额加起来，正好等于全部发布相比什么都不发布多出来的能力
  对称性   两个字段如果在任何组合里都可以互换而不改变结果，份额必须相等
  零贡献   一个字段如果加进任何组合都不带来任何提升，份额为 0
  可加性   两个目标合起来看时，每个字段的份额等于分别看时的份额之和

满足这四条的分法**有且只有一种**，就是 Shapley 值（Shapley 1953）。
它的算式是：把字段 j 加进各种规模的组合里，看每次带来多少增量，再按规模加权平均。

这正是这个项目需要的性质。关键在"按各种规模加权平均"这一句：
- 组合规模很小时，看到的是这个字段**自己**能推出多少；
- 组合规模很大时，看到的是**抽掉它**会损失多少；
- 中间规模看到的是它和别人**配合**才兑现的那部分。
三者被同一个数吸收，不需要人为决定谁占多大比重——权重由那四条要求推出来。

怎么算
------
字段有 46 个，穷举 2^46 个组合不可能。两条路：
- 字段少（≤ 18 个）时**直接穷举**，得到精确值，用来检验采样法准不准；
- 字段多时用 KernelSHAP：把 Shapley 值改写成一个带约束的加权最小二乘问题，
  抽若干个组合去解它。抽样按 Shapley 核的规模分布做（见 surrogate.shapley_kernel_sizes），
  这样加权就退化成不加权，数值更稳。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from . import surrogate as sg


# ================================================================= 精确值

def exact_shapley(v_of_sets, p: int, k_out: int = 1) -> np.ndarray:
    """穷举所有 2^p 个组合，算精确 Shapley 值。只在 p 较小时可用。

    v_of_sets: 接受 (n, p) 掩码数组、返回 (n,) 或 (n, k) 的可调用对象。
    返回 (p,) 或 (p, k)。
    """
    if p > 20:
        raise ValueError(f"p={p} 太大，穷举不可行（2^{p} 个组合）")
    n_sub = 1 << p
    masks = np.zeros((n_sub, p), dtype=np.float32)
    for s in range(n_sub):
        for j in range(p):
            if s >> j & 1:
                masks[s, j] = 1.0
    v = np.asarray(v_of_sets(masks), dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]

    # 规模权重：|S|!(p-|S|-1)!/p!
    fact = [math.factorial(i) for i in range(p + 1)]
    w = np.array([fact[s] * fact[p - s - 1] / fact[p] for s in range(p)])

    sizes = masks.sum(1).astype(int)
    phi = np.zeros((p, v.shape[1]))
    for j in range(p):
        has = masks[:, j] > 0.5
        idx_wo = np.where(~has)[0]
        idx_w = idx_wo + (1 << j)               # 加上 j 之后的组合下标
        gain = v[idx_w] - v[idx_wo]             # (m, k)
        phi[j] = (w[sizes[idx_wo]][:, None] * gain).sum(0)
    return phi[:, 0] if phi.shape[1] == 1 else phi


# ================================================================= KernelSHAP

@dataclass
class ShapResult:
    phi: np.ndarray                 # (p,) 或 (p, k) 贡献值
    v_full: np.ndarray              # 全部发布时的还原能力
    v_empty: np.ndarray             # 什么都不发布时（应当接近 0）
    n_coalitions: int
    efficiency_gap: np.ndarray      # Σφ − (v_full − v_empty)，越接近 0 越好
    solo: np.ndarray | None = None      # v({j})           独立能力
    marginal: np.ndarray | None = None  # v(F) − v(F∖{j})  不可替代性


def _enumerate_layer(p: int, k: int) -> np.ndarray:
    """列出全部恰好含 k 个字段的组合。只在数量不大时调用。"""
    Z = np.zeros((math.comb(p, k), p), dtype=np.float32)
    for i, c in enumerate(itertools.combinations(range(p), k)):
        Z[i, list(c)] = 1.0
    return Z


def build_coalitions(p: int, budget: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """凑出用于回归的字段组合和它们的权重。

    Shapley 核给规模为 k 的组合的权重正比于 (p-1)/(k(p-k))：
    **两头最重、中间最轻**。只含一两个字段的组合，看的是这些字段自己的能力；
    只差一两个字段的组合，看的是抽掉它们的损失。这两头信息量最大，
    偏偏数量又最少（p=46 时规模 1 和 45 各只有 46 个）。

    所以先从两头往里，把能穷举得起的整层**全部列出来**并给它们精确的权重，
    剩下的预算再到中间层去抽样。这样最重要的那部分完全没有抽样误差，
    同样的预算下方差小很多。纯抽样是把预算平摊到各层，最重的那几层
    反而只抽到寥寥几个，这正是对称性误差的主要来源。
    """
    ks = np.arange(1, p)
    w_k = (p - 1) / (ks * (p - ks))
    w_k = w_k / w_k.sum()

    Zs, Ws = [], []
    left, remaining = budget, set(range(1, p))
    for k in range(1, (p + 1) // 2 + 1):
        pair = {k, p - k} & remaining
        if not pair:
            continue
        cnt = sum(math.comb(p, j) for j in pair)
        if cnt > left * 0.6 or cnt > 200000:
            break                              # 这一层太大，留给抽样
        for j in sorted(pair):
            Z = _enumerate_layer(p, j)
            Zs.append(Z)
            # 整层被完全列出，每个组合分到该层总权重的均等一份
            Ws.append(np.full(len(Z), w_k[j - 1] / len(Z)))
        left -= cnt
        remaining -= pair

    if left > 0 and remaining:
        rem = np.array(sorted(remaining))
        pr = w_k[rem - 1]
        w_rem_total = float(pr.sum())
        pr = pr / pr.sum()
        sizes = rng.choice(rem, size=left, p=pr)
        Z = sg.masks_from_sizes(sizes, p, rng)
        Zs.append(Z)
        # 按核分布抽出来的样本，彼此等权；整体占剩余层的总权重
        Ws.append(np.full(len(Z), w_rem_total / len(Z)))

    return np.concatenate(Zs, axis=0), np.concatenate(Ws)


def kernel_shap(vf, p: int, n_coalitions: int = 8192, seed: int = 0,
                ridge: float = 1e-10, paired: bool = True) -> ShapResult:
    """用加权最小二乘（KernelSHAP）估计 Shapley 值。

    做法：取若干个字段组合，把 v(S) 对"哪些字段在组合里"做一次加权线性回归，
    回归系数就是 Shapley 值。约束条件是所有系数之和必须等于 v(全部)−v(空)，
    这一条让加和性严格成立，不靠采样碰运气。

    组合怎么取见 build_coalitions：权重最高的那几层直接穷举，其余抽样。
    """
    rng = np.random.default_rng(seed)
    v_full = np.atleast_1d(np.asarray(vf(np.ones((1, p), dtype=np.float32))[0], dtype=float))
    v_empty = np.atleast_1d(np.asarray(vf(np.zeros((1, p), dtype=np.float32))[0], dtype=float))

    Z, w = build_coalitions(p, n_coalitions, rng)
    vz = np.asarray(vf(Z), dtype=np.float64)
    if vz.ndim == 1:
        vz = vz[:, None]
    k = vz.shape[1]

    # 目标量：相对于"什么都不发布"的增量
    ytab = vz - v_empty[None, :]
    tot = (v_full - v_empty)                                # (k,)

    # 带等式约束 1ᵀφ = tot 的加权最小二乘，闭式解（SHAP 原文附录）
    wn = w / w.sum()
    Zw = Z * wn[:, None]
    A = Z.T @ Zw + ridge * np.eye(p)
    b = Zw.T @ ytab                                          # (p, k)
    Ainv = np.linalg.inv(A)
    ones = np.ones(p)
    Ai1 = Ainv @ ones
    denom = float(ones @ Ai1)
    phi = np.zeros((p, k))
    for c in range(k):
        u = Ainv @ b[:, c]
        phi[:, c] = u - Ai1 * ((ones @ u) - tot[c]) / denom

    gap = phi.sum(0) - tot
    n_coalitions = len(Z)

    # 两个分解量，各只要 p+1 次查询，很便宜
    eye = np.eye(p, dtype=np.float32)
    solo = np.asarray(vf(eye), dtype=np.float64)
    if solo.ndim == 1:
        solo = solo[:, None]
    solo = solo - v_empty[None, :]
    marg = v_full[None, :] - np.asarray(vf(1.0 - eye), dtype=np.float64).reshape(p, k)

    sq = lambda a: a[:, 0] if a.shape[1] == 1 else a        # noqa: E731
    return ShapResult(phi=sq(phi),
                      v_full=v_full[0] if k == 1 else v_full,
                      v_empty=v_empty[0] if k == 1 else v_empty,
                      n_coalitions=len(Z),
                      efficiency_gap=gap[0] if k == 1 else gap,
                      solo=sq(solo), marginal=sq(marg))


# ================================================================= 分解

def decompose(res: ShapResult) -> dict[str, np.ndarray]:
    """把贡献值拆开，得到"这个字段值多少"和"它是哪一类"两个维度。

    直接查出来的两个量（各只要 p 次查询，很便宜）：

      独立能力   solo_j     = v({j}) − v(∅)        只发布它一个，能推出多少
      不可替代性 marg_j     = v(F) − v(F∖{j})      全部发布、单独抽掉它，损失多少

    份额维度：

      贡献值     φ_j                                它该分到多少份额
      归一化份额 φ_j / (v(F)−v(∅))                  占总还原能力的百分之多少

    性质维度——**协同–冗余指数** = marg_j − solo_j：

      指数 < 0  这个字段在别人在场时贬值 ⇒ 有替身顶着，删了它别人补上
      指数 ≈ 0  它的贡献和别人无关，可加
      指数 > 0  它只有在别人在场时才值钱 ⇒ 协同型

    三种极端情形能把这个指数的刻度定住（已在 00_selftest 里用解析解验证）：
      完美替身   solo=1, marg=0  → 指数 = −1
      纯协同     solo=0, marg=1  → 指数 = +1
      独立可加   solo=marg=w     → 指数 = 0

    这两个维度合起来正是要的那张坐标图：横轴看这个字段值多少，
    纵轴看它属于哪一类。它直接回答了两个此前答不上来的问题——
    为什么只处置被选中的字段、整体可推断性几乎不降（因为它们指数为负，有替身）；
    为什么按两两相关性排序会漏掉最关键的字段（因为那些字段指数为正，
    单独看什么都看不出来）。

    另外保留两个便于对照的差值：
      synergy          = φ − solo   （与指数同号，但带上了份额的尺度）
      substitutability = φ − marg
    """
    phi, solo, marg = res.phi, res.solo, res.marginal
    total = res.v_full - res.v_empty
    denom = np.where(np.abs(total) < 1e-12, 1.0, total)
    return {
        "phi": phi,
        "share": phi / denom,
        "solo": solo,
        "marginal": marg,
        "interaction": marg - solo,
        "synergy": phi - solo,
        "substitutability": phi - marg,
    }


def classify(dec: dict[str, np.ndarray], zero_tol: float,
             inter_tol: float = 0.05) -> np.ndarray:
    """按贡献值和协同–冗余指数给每个字段贴一个标签。

    zero_tol   判零门槛，由注入噪声字段的**贡献值**分布量出来（见 zero_band）
    inter_tol  指数落在 ±inter_tol 内算"独立可加"。同样应当由噪声字段的
               **指数**分布量出来（见 interaction_band）——噪声字段的真实指数
               就是 0，它们观测到的散布宽度就是"多接近 0 算 0"的客观标尺。
               这里保留一个默认值只是为了让不注入噪声字段时函数仍可调用，
               正式结果一律传实测值，不用这个默认数。
    """
    phi = np.abs(np.asarray(dec["phi"]).ravel())
    inter = np.asarray(dec["interaction"]).ravel()
    out = np.empty(len(phi), dtype=object)
    for i in range(len(phi)):
        if phi[i] < zero_tol:
            out[i] = "无贡献"
        elif inter[i] < -inter_tol:
            out[i] = "替身型"
        elif inter[i] > inter_tol:
            out[i] = "协同型"
        else:
            out[i] = "独立可加"
    return out


# ================================================================= 收敛诊断

def convergence_curve(vf, p: int, budgets: list[int], seed: int = 0) -> dict:
    """在若干个采样量下各算一次，看贡献值稳定到什么程度。

    用来给"抽多少个组合够用"找一个有依据的数字，而不是拍脑袋定。
    报告两个量：相邻采样量之间贡献值向量的秩相关，以及最大绝对差。
    """
    out = {"budget": [], "phi": [], "efficiency_gap": []}
    for nb in budgets:
        r = kernel_shap(vf, p, n_coalitions=nb, seed=seed)
        out["budget"].append(nb)
        out["phi"].append(np.asarray(r.phi))
        out["efficiency_gap"].append(float(np.max(np.abs(r.efficiency_gap))))
    from scipy.stats import spearmanr
    rho, dmax = [np.nan], [np.nan]
    for i in range(1, len(budgets)):
        a, b = out["phi"][i - 1].ravel(), out["phi"][i].ravel()
        rho.append(float(spearmanr(a, b).statistic))
        dmax.append(float(np.max(np.abs(a - b))))
    out["spearman_vs_prev"] = rho
    out["maxdiff_vs_prev"] = dmax
    return out


# ================================================================= 判零标尺

def interaction_band(inter_noise: np.ndarray, q: float = 0.95) -> float:
    """用注入的噪声字段的协同–冗余指数给出"多接近 0 算 0"的客观标尺。

    噪声字段和目标独立，它的独立能力和不可替代性都应当是 0，所以指数的真值也是 0。
    实际观测到的散布完全来自估计误差，取分布的 q 分位数当门槛，
    超过它才算真的偏向协同或替身。和判零门槛是同一个思路——
    门槛从数据里量出来，不是人为定的。
    """
    return float(np.quantile(np.abs(np.asarray(inter_noise).ravel()), q))


def zero_band(phi_noise: np.ndarray, q: float = 0.95) -> float:
    """用注入的噪声字段的贡献值给出"多小算 0"的客观标尺。

    噪声字段和目标独立，真实贡献是 0，它们的贡献值分布宽度就完全来自
    估计误差。取这个分布的 q 分位数作为判零门槛——超过它才算真有贡献。
    这样门槛是从数据里量出来的，不是人为定的。
    """
    return float(np.quantile(np.abs(np.asarray(phi_noise).ravel()), q))
