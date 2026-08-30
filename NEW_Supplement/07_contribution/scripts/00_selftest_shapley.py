"""自检：用有解析答案的玩具例子验证 Shapley 实现本身没写错。

这一步必须先做。后面所有结论都建立在"贡献值算得对"之上，
如果算法本身有 bug，在真实数据上是看不出来的——没有答案可以对照。

四个例子，答案都能用笔算出来：
  1. 线性可加      v(S) = Σ_{j∈S} w_j          → φ_j = w_j
  2. 纯协同（异或） v(S) = 1 当且仅当 {0,1} ⊆ S  → φ_0 = φ_1 = 1/2，而各自单独为 0
  3. 纯冗余（替身） v(S) = 1 当 0∈S 或 1∈S       → φ_0 = φ_1 = 1/2，而各自抽掉损失为 0
  4. 虚拟字段      加进去永远不改变 v            → φ = 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import attribution as att      # noqa: E402
from src import surrogate as sg         # noqa: E402


def check(name: str, got, want, tol: float) -> bool:
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    err = float(np.max(np.abs(got - want)))
    ok = err <= tol
    print(f"  {'✓' if ok else '✗'} {name:<28} 最大偏差 {err:.2e}  (容差 {tol:.0e})")
    if not ok:
        print(f"      得到 {np.round(got, 4)}")
        print(f"      应为 {np.round(want, 4)}")
    return ok


def main() -> int:
    print("=" * 66)
    print("Shapley 实现自检")
    print("=" * 66)
    allok = True

    # ---------------------------------------------------------- 1 线性可加
    p = 8
    w = np.array([0.30, 0.20, 0.15, 0.12, 0.10, 0.07, 0.04, 0.02])

    def v_lin(masks):
        return np.asarray(masks, dtype=float) @ w

    print("\n[1] 线性可加：v(S) = Σ w_j，理论上 φ_j = w_j")
    allok &= check("穷举精确解", att.exact_shapley(v_lin, p), w, 1e-12)
    r = att.kernel_shap(v_lin, p, n_coalitions=4096, seed=0)
    allok &= check("KernelSHAP 估计", r.phi, w, 1e-6)
    allok &= check("加和性偏差", [r.efficiency_gap], [0.0], 1e-9)
    allok &= check("独立能力 v({j})", r.solo, w, 1e-9)
    allok &= check("不可替代性", r.marginal, w, 1e-9)

    # ---------------------------------------------------------- 2 纯协同
    p = 6

    def v_syn(masks):
        m = np.asarray(masks, dtype=float)
        return (m[:, 0] * m[:, 1])          # 只有两个都在才有值

    print("\n[2] 纯协同：只有 0 和 1 同时发布才能推出目标")
    ex = att.exact_shapley(v_syn, p)
    allok &= check("穷举：两者各半", ex[:2], [0.5, 0.5], 1e-12)
    allok &= check("穷举：其余为 0", ex[2:], np.zeros(p - 2), 1e-12)
    r = att.kernel_shap(v_syn, p, n_coalitions=4096, seed=0)
    allok &= check("KernelSHAP", r.phi[:2], [0.5, 0.5], 1e-6)
    d = att.decompose(r)
    allok &= check("独立能力应为 0", d["solo"][:2], [0.0, 0.0], 1e-9)
    allok &= check("协同度应为 0.5", d["synergy"][:2], [0.5, 0.5], 1e-6)
    print("      → 协同型字段的特征：贡献值高，但单独看能力为 0。"
          "任何基于两两相关性的方法在这里必然漏掉它们。")

    # ---------------------------------------------------------- 3 纯冗余
    p = 6

    def v_dup(masks):
        m = np.asarray(masks, dtype=float)
        return np.maximum(m[:, 0], m[:, 1])     # 有任意一个就够

    print("\n[3] 纯冗余：0 和 1 互为完美替身，有任意一个就够")
    ex = att.exact_shapley(v_dup, p)
    allok &= check("穷举：两者各半", ex[:2], [0.5, 0.5], 1e-12)
    r = att.kernel_shap(v_dup, p, n_coalitions=4096, seed=0)
    allok &= check("KernelSHAP", r.phi[:2], [0.5, 0.5], 1e-6)
    d = att.decompose(r)
    allok &= check("不可替代性应为 0", d["marginal"][:2], [0.0, 0.0], 1e-9)
    allok &= check("可替代度应为 0.5", d["substitutability"][:2], [0.5, 0.5], 1e-6)
    print("      → 替身字段的特征：贡献值高，但单独抽掉毫无损失。"
          "稀疏门控在这里只会留一个、把另一个判为零——这正是现有方法的缺陷。")

    # ---------------------------------------------------------- 4 虚拟字段
    print("\n[4] 虚拟字段：加进任何组合都不改变结果")
    p = 7

    def v_dummy(masks):
        m = np.asarray(masks, dtype=float)
        return m[:, 0] * 0.5 + m[:, 1] * 0.3        # 字段 2..6 完全无关

    ex = att.exact_shapley(v_dummy, p)
    allok &= check("穷举：无关字段为 0", ex[2:], np.zeros(p - 2), 1e-12)
    r = att.kernel_shap(v_dummy, p, n_coalitions=4096, seed=0)
    allok &= check("KernelSHAP：无关字段为 0", r.phi[2:], np.zeros(p - 2), 1e-6)

    # ---------------------------------------------------------- 5 掩码采样
    print("\n[5] 掩码采样的规模分布")
    P = 20
    ref = {}
    for dist in ("uniform", "kernel"):
        rng = np.random.default_rng(0)
        m = sg.sample_masks(20000, P, rng, p_full=0.10, p_empty=0.02, dist=dist)
        ks = m.sum(1)
        frac_full = float((ks == P).mean())
        frac_empty = float((ks == 0).mean())
        mid = ks[(ks > 0) & (ks < P)]
        cnt = np.bincount(mid.astype(int), minlength=P)[1:P]
        ratio = float(cnt.max() / cnt.min())          # 两头档 / 中间档
        ref[dist] = ks

        # 两个比例旋钮在任何分布下都必须名副其实
        ok = (abs(frac_full - 0.10) < 0.005 and abs(frac_empty - 0.02) < 0.005)
        # 形状则各测各的：均匀应当平，核分布应当两头翘
        theo = sg.size_probs(P, dist)
        want_ratio = float(theo.max() / theo.min())
        ok &= abs(ratio - want_ratio) < 0.25 * want_ratio
        allok &= ok
        print(f"  {dist:<8} 全1占比 {frac_full:.4f}  全0占比 {frac_empty:.4f}  "
              f"两头/中间计数比 {ratio:.2f}（理论 {want_ratio:.2f}）  "
              f"{'✓' if ok else '✗'}")

    print("      → 默认用 kernel：贡献值全部由 Shapley 核加权算出，"
          "而这个核极度偏向两头。")
    print("        均匀采样下'只发布某一个指定字段'在训练样本里占比约 "
          f"{1/(P-1)/P:.4%}，模型最没训到的地方恰恰是估计量最依赖的地方。")

    # 每行开的字段数必须精确等于抽到的规模（向量化实现容易在这里出错）
    rng = np.random.default_rng(1)
    ks2 = np.array([0, 1, 5, 19, 20, 20, 3])
    m2 = sg.masks_from_sizes(ks2, P, rng)
    allok &= check("每行开启数=指定规模", m2.sum(1), ks2, 0)

    # 对照：逐字段抛硬币会把规模压在中间，两端采不到
    coin = (rng.random((20000, P)) < 0.5).sum(1)
    print(f"  对照·逐字段抛硬币：规模范围 {coin.min()}~{coin.max()}，"
          f"规模≤2 的占比 {float((coin<=2).mean()):.5f}"
          f"（均匀 {float((ref['uniform']<=2).mean()):.5f}，"
          f"核分布 {float((ref['kernel']<=2).mean()):.5f}）")
    print("      → 这就是为什么不能逐字段抛硬币：小组合几乎采不到，"
          "而小组合恰恰是衡量单字段独立能力的地方。")

    print("\n" + "=" * 66)
    print("全部通过 ✓" if allok else "有检查未通过 ✗")
    print("=" * 66)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
