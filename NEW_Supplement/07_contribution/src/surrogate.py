"""随机屏蔽代理模型：把"某组字段能推出目标多少"变成一个可查的函数。

要解决的问题
------------
现有做法训练一个门控网络，得到**一组**够用的字段，然后说"这些就是推断源"。
但它只回答"有没有一组够用"，不回答"每个字段各占多少份"。后果在数据上已经看到了：
没被选中的字段照样能把目标推出来，只处置选中字段等于没处置，换个随机种子选中集合就变。

根子在于：要衡量单个字段的份额，得知道**任意字段组合**的推断能力，
也就是要知道函数 v(S) = 只发布 S 里的字段时目标能被还原到什么程度。
字段有 46 个，组合有 2^46 种，一个一个训练是不可能的。

做法
----
训练一个模型，它每一步看到的输入都被随机屏蔽掉一部分字段，
于是它被迫学会"在任意字段组合下都尽力还原目标"。训练完之后，
想查任何一个字段组合的能力，只要把对应的掩码喂进去做一次前向就行。
一次训练，任意子集可查。

三个关键实现点（做错任何一个结论都不成立）
------------------------------------------
1. **掩码本身必须作为额外输入喂进去**。数据是标准化过的，被屏蔽的位置填 0，
   而 0 恰好是该字段的均值。不把掩码告诉模型，它就分不清
   "这个字段的值正好等于均值"和"这个字段根本没发布"。输入维度因此是 2p。

2. **屏蔽用"标准化后填 0"，等价于用该字段的长期平均值填补**。这是在定义
   "移除一个字段"是什么意思——外部拿不到这个字段时，他能做的最好猜测就是它的平均水平。
   不同的移除定义会给出不同的贡献值，所以这个口径必须写在结果说明里。

3. **掩码规模要均匀采样**：先均匀抽子集大小 k，再在该大小上均匀抽子集。
   不能对每个字段独立抛硬币——那样 k 会集中在 p/2 附近，
   只有一两个字段的小组合和几乎全发布的大组合都采不到，
   而这两端恰恰是衡量单字段贡献时权重最高的地方。
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_src"))


# ================================================================= 配置

@dataclass
class SurrogateConfig:
    # 网络。比主流程的 (64,32,16) 宽一些——屏蔽把任务变难了，
    # 模型要同时拟合所有字段组合下的映射，容量不够会在小组合上失准。
    hidden: tuple = (384, 256, 192, 128)
    dropout: float = 0.0

    # 训练。这几个数字都是扫出来的，依据记在 logs/20260815_02。
    epochs: int = 1000
    batch_size: int = 256
    lr: float = 1e-3
    # 权重衰减不是随手填的 1e-5。实测 3e-3 把"两个完全相同的字段被给出不同
    # 还原能力"这个毛病从 0.071 压到 0.011，同时校准偏差也降了（0.0387→0.0349）。
    # 机制上说得通：两个相同输入 w₁x+w₂x=(w₁+w₂)x，总和固定时让 w₁²+w₂² 最小的解
    # 必然是 w₁=w₂，衰减本身就在把网络往对称解上推。注意这条**不是单调的**，
    # 1e-4 和 3e-4 反而比 1e-5 更差，所以只能扫不能猜。
    weight_decay: float = 3e-3
    patience: int = 150          # 早停：验证损失连续多少轮不降就停
    min_epochs: int = 150
    seed: int = 42

    # 划分。主口径用时序划分（对齐 08_rts_core_pipeline 的严格版）：
    # 用历史训练、推断新数据，对应"外部只有历史公开数据"的攻击者假设。
    # 随机划分对应"外部已掌握同批次的部分配对样本"，作对照口径单独跑。
    split: str = "temporal"      # temporal | random
    train_frac: float = 0.70
    val_frac: float = 0.10

    # 掩码采样
    masks_per_sample: int = 1    # 每个样本每轮抽几个掩码
    p_full: float = 0.10         # 强制全 1 掩码的比例（保证"全部发布"这一点准）
    p_empty: float = 0.02        # 强制全 0 掩码的比例（保证基准点准）
    mask_dist: str = "kernel"    # 掩码规模分布：uniform | kernel | mixture
    kernel_w: float = 0.5        # mixture 里 Shapley 核那一半的占比

    # 门控（通道 A）
    use_gate: bool = False
    lambda_dgate: float = 0.005
    dgate_depth: int = 4

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


# ================================================================= 掩码采样

def masks_from_sizes(ks: np.ndarray, p: int, rng: np.random.Generator) -> np.ndarray:
    """给定每行要开几个字段，生成 (n, p) 的 0/1 掩码。

    实现上用"给每列抽一个随机数、按名次取前 k 名"来做，
    等价于每行独立抽一个不重复的 k 元子集，但整批一次算完，没有逐行循环。
    训练时每个批次都要抽掩码，这里是热路径，值得向量化。
    """
    n = len(ks)
    rank = rng.random((n, p)).argsort(axis=1).argsort(axis=1)   # 每行是 0..p-1 的随机排名
    return (rank < np.asarray(ks)[:, None]).astype(np.float32)


def size_probs(p: int, dist: str = "mixture", kernel_w: float = 0.5) -> np.ndarray:
    """规模 1..p-1 的抽取概率。

    dist 取值：
      uniform   各规模等概率。直觉上"公平"，但和后面用它的方式不匹配（见下）
      kernel    正比于 Shapley 核权重 (p-1)/(k(p-k))，两头重中间轻
      mixture   两者各一半，默认

    为什么不能只用 uniform
    ----------------------
    贡献值全部由 Shapley 核加权算出来，而这个核**极度偏向两头**：
    只含一两个字段的组合看的是字段自己的能力，只差一两个字段的组合看的是
    抽掉它的损失，这两头权重最高。可是均匀采样下，"只发布某一个指定字段"
    这种情形在训练样本里的占比约为 1/(p-1)/p——p=46 时是 0.05%，
    每轮六千个样本里只摊到两三个。**模型最没训到的地方，恰恰是估计量最依赖的地方。**

    实测后果：两列**逐个数完全相同**的字段，模型给出的"只发布它一个"的还原能力
    差了 0.071，而"全部发布再抽掉它"只差 0.0038——不对称完全集中在小组合上。

    为什么也不只用 kernel
    ---------------------
    纯核分布把几乎所有力气压在两头，中间规模训不好，而 v(S) 在中间规模上
    仍然要被查（风险坐标图、重训认证都要用）。各占一半是个折中。
    """
    ks = np.arange(1, p)
    uni = np.ones(p - 1) / (p - 1)
    ker = (p - 1) / (ks * (p - ks))
    ker = ker / ker.sum()
    if dist == "uniform":
        return uni
    if dist == "kernel":
        return ker
    if dist == "mixture":
        return (1 - kernel_w) * uni + kernel_w * ker
    raise ValueError(f"未知的掩码规模分布：{dist}")


def sample_masks(n: int, p: int, rng: np.random.Generator,
                 p_full: float = 0.10, p_empty: float = 0.02,
                 dist: str = "mixture", kernel_w: float = 0.5) -> np.ndarray:
    """抽 n 个长度为 p 的 0/1 掩码。1 = 该字段发布，0 = 未发布。

    p_full 的比例是"全部发布"，p_empty 的比例是"什么都不发布"，
    剩下的按 size_probs 给出的分布在 1..p-1 之间取。

    注意剩下那部分要从 **1..p-1** 抽而不是 0..p——否则它自己还会再抽出一些
    全 1 和全 0，两头的实际比例就会超过设定值（p=20 时全 1 会从 10% 涨到 14%），
    这两个旋钮就名不副实了。
    """
    n_full = int(round(n * p_full))
    n_empty = int(round(n * p_empty))
    n_mid = max(n - n_full - n_empty, 0)
    if p > 1 and n_mid:
        mid = rng.choice(np.arange(1, p), size=n_mid,
                         p=size_probs(p, dist, kernel_w))
    else:
        mid = np.zeros(n_mid, dtype=np.int64)
    ks = np.concatenate([
        np.full(n_full, p, dtype=np.int64),
        np.zeros(n_empty, dtype=np.int64),
        mid.astype(np.int64),
    ])
    rng.shuffle(ks)                      # 打散，避免批次内前若干行总是全 1
    return masks_from_sizes(ks, p, rng)


def shapley_kernel_sizes(p: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """按 Shapley 核权重的规模分布抽 n 个子集规模，取值范围 1..p-1。

    Shapley 值给不同规模的组合不同权重，权重 ∝ (p-1) / (k(p-k))，
    也就是**极小和极大的组合最重要**：只有一两个字段时看得出这个字段自己的能力，
    几乎全发布时看得出抽掉它的损失。中间规模的组合信息量反而低。
    按这个分布采样，后面的加权最小二乘就退化成普通最小二乘，数值上更稳。
    """
    ks = np.arange(1, p)
    w = (p - 1) / (ks * (p - ks))
    w = w / w.sum()
    return rng.choice(ks, size=n, p=w)


def sample_shapley_masks(p: int, n: int, rng: np.random.Generator,
                         paired: bool = True) -> np.ndarray:
    """为 KernelSHAP 抽子集：规模按 Shapley 核分布，规模内均匀。

    paired=True 时成对采样——抽到 S 就同时把补集 F\\S 也放进来。
    这一招能显著降低方差，因为 Shapley 核对 S 和它的补集给的权重相同，
    成对之后估计量的偏差项互相抵消。
    """
    m_each = n // 2 if paired else n
    ks = shapley_kernel_sizes(p, m_each, rng)
    base = masks_from_sizes(ks, p, rng)
    return np.concatenate([base, 1.0 - base], axis=0) if paired else base


# ================================================================= 模型

def _mlp(in_dim: int, hidden, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MaskedSurrogate(nn.Module):
    """普通版：把被屏蔽的字段值和掩码拼在一起送进多层网络。"""

    def __init__(self, p: int, hidden, n_out: int = 1, dropout: float = 0.0):
        super().__init__()
        self.p = p
        self.n_out = n_out
        self.net = _mlp(2 * p, hidden, n_out, dropout)

    def forward(self, x, m):
        return self.net(torch.cat([x * m, m], dim=1))

    def gates(self):
        return None

    def penalty(self, cfg):
        return torch.zeros((), device=next(self.parameters()).device)


class MaskedDGatingSurrogate(nn.Module):
    """带分解式门控的版本（通道 A）。

    门控参数化照搬 02_src/gates.py:118 的 DGating：门控值是 D-1 个因子的乘积，
    对因子加 L2 惩罚。等价于对该字段的整组权重施加一个比 L1 更强的稀疏惩罚，
    收敛后大部分门控值精确为 0，形成断崖。

    和原版的关键区别在训练方式上，不在参数化上：
    原版看到的永远是完整输入，所以门控一旦找到一条够用的路径就可以把其余全压到 0；
    这里每一批的输入都被随机屏蔽，那条路径的字段被屏蔽掉的批次里，
    替代字段必须顶上来承担损失，否则损失降不下去。
    门控因此没法塌缩到单条路径，它的值变成"跨所有字段组合平均的贡献强度"。

    掩码那一路不加门控也不受惩罚——掩码本身不携带 x 的信息，
    它只是告诉模型"这一位是缺失还是恰好等于均值"。
    """

    def __init__(self, p: int, hidden, n_out: int = 1, depth: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.p = p
        self.n_out = n_out
        self.depth = depth
        h0 = hidden[0]
        self.omega = nn.Parameter(torch.empty(p, h0))
        nn.init.kaiming_normal_(self.omega, nonlinearity="relu")
        self.gamma = nn.Parameter(torch.ones(depth - 1, p))   # 初值全 1 ⇒ 门控从 1 出发
        self.omega_m = nn.Parameter(torch.zeros(p, h0))       # 掩码通路，不受惩罚
        nn.init.normal_(self.omega_m, std=0.01)
        self.bias = nn.Parameter(torch.zeros(h0))
        self.rest = _mlp(h0, hidden[1:], n_out, dropout) if len(hidden) > 1 \
            else nn.Linear(h0, n_out)

    def _g(self):
        return self.gamma.prod(dim=0)

    def forward(self, x, m):
        eff = self.omega * self._g().unsqueeze(1)
        h = torch.relu((x * m) @ eff + m @ self.omega_m + self.bias)
        return self.rest(h)

    def gates(self) -> np.ndarray:
        """报告因子乘积的绝对值。初值恰好是 1，不需要任何归一化。"""
        return self.gamma.detach().prod(dim=0).abs().cpu().numpy()

    def penalty(self, cfg):
        return cfg.lambda_dgate * (self.omega.pow(2).sum() + self.gamma.pow(2).sum())


# ================================================================= 训练

@dataclass
class FitResult:
    model: nn.Module
    cfg: SurrogateConfig
    fields: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    history: dict = field(default_factory=dict)
    best_epoch: int = -1
    gates: np.ndarray | None = None


def _split_idx(n: int, cfg: SurrogateConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_tr = int(n * cfg.train_frac)
    n_va = int(n * cfg.val_frac)
    if cfg.split == "temporal":
        order = np.arange(n)
    elif cfg.split == "random":
        order = np.random.default_rng(cfg.seed).permutation(n)
    else:
        raise ValueError(f"未知划分方式：{cfg.split}")
    return order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit(X: np.ndarray, Y: np.ndarray, fields: list[str],
        cfg: SurrogateConfig, verbose: bool = False) -> FitResult:
    """训练一个随机屏蔽代理模型。

    X (n, p) 候选字段，Y (n,) 或 (n, k) 目标。返回可用于查询 v(S) 的结果对象。
    """
    set_seed(cfg.seed)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, p = X.shape
    k_out = Y.shape[1]

    i_tr, i_va, i_te = _split_idx(n, cfg)

    # 标准化只用训练段的统计量，避免把验证/测试段的信息漏进来
    xm, xs = X[i_tr].mean(0), X[i_tr].std(0)
    xs[xs < 1e-12] = 1.0
    ym, ys = Y[i_tr].mean(0), Y[i_tr].std(0)
    ys[ys < 1e-12] = 1.0
    Xs = ((X - xm) / xs).astype(np.float32)
    Ys = ((Y - ym) / ys).astype(np.float32)

    dev = torch.device("cpu")
    xt = torch.tensor(Xs, device=dev)
    yt = torch.tensor(Ys, device=dev)

    if cfg.use_gate:
        model = MaskedDGatingSurrogate(p, cfg.hidden, k_out, cfg.dgate_depth, cfg.dropout)
    else:
        model = MaskedSurrogate(p, cfg.hidden, k_out, cfg.dropout)
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = nn.MSELoss()
    rng = np.random.default_rng(cfg.seed)

    # 验证用的掩码固定下来，否则验证损失自己在抖，早停判据就不可靠了
    val_masks = torch.tensor(
        sample_masks(len(i_va), p, np.random.default_rng(cfg.seed + 999),
                     cfg.p_full, cfg.p_empty, cfg.mask_dist, cfg.kernel_w),
        device=dev)

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "n_active": []}
    best = (math.inf, -1, None)

    tr_idx = torch.tensor(i_tr, device=dev)
    va_idx = torch.tensor(i_va, device=dev)

    for ep in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(i_tr))
        tot, nb = 0.0, 0
        for s in range(0, len(i_tr), cfg.batch_size):
            b = tr_idx[perm[s:s + cfg.batch_size]]
            xb, yb = xt[b], yt[b]
            mb = torch.tensor(
                sample_masks(len(b), p, rng, cfg.p_full, cfg.p_empty,
                             cfg.mask_dist, cfg.kernel_w), device=dev)
            pred = model(xb, mb)
            loss = lossf(pred, yb) + model.penalty(cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1

        model.eval()
        with torch.no_grad():
            vp = model(xt[va_idx], val_masks)
            # 早停判据用**和训练目标一致**的量（含惩罚项），
            # 这样"什么时候停"这件事完全不碰测试段
            vl = float(lossf(vp, yt[va_idx]) + model.penalty(cfg))
        g = model.gates()
        na = int((g > 0.01).sum()) if g is not None else p

        hist["epoch"].append(ep)
        hist["train_loss"].append(tot / max(nb, 1))
        hist["val_loss"].append(vl)
        hist["n_active"].append(na)

        if vl < best[0] - 1e-7:
            best = (vl, ep, copy.deepcopy(model.state_dict()))
        if verbose and ep % 25 == 0:
            print(f"  ep {ep:4d}  train {tot/max(nb,1):.5f}  val {vl:.5f}  active {na}")
        if ep >= cfg.min_epochs and ep - best[1] >= cfg.patience:
            break

    if best[2] is not None:
        model.load_state_dict(best[2])
    model.eval()

    return FitResult(model=model, cfg=cfg, fields=list(fields),
                     x_mean=xm, x_std=xs, y_mean=ym, y_std=ys,
                     idx_train=i_tr, idx_val=i_va, idx_test=i_te,
                     history=hist, best_epoch=best[1],
                     gates=model.gates())


# ================================================================= v(S) 查询

def retrain_reference(X: np.ndarray, y: np.ndarray, cols: list[int],
                      cfg: SurrogateConfig, eval_idx: np.ndarray,
                      hidden=(64, 32, 16), epochs: int = 200,
                      seed: int | None = None) -> float:
    """只用 cols 这几个字段，从零训练一个普通网络，返回测试段 R²。

    这是代理模型的**参照答案**：代理模型说某个字段组合能推到 v(S)，
    到底准不准，只能靠真的按这个组合训练一个模型来核对。

    三个口径必须和 ValueFunction 完全一致，否则比出来的差异是口径差异不是模型误差：
      - 一样的时序划分
      - 一样的标准化（只用训练段统计量）
      - **一样的评估行**（eval_idx），R² 的分母也用同一批行算

    网络结构故意用主流程那套 (64,32,16)——参照答案应当代表
    "按论文既有做法训练出来能达到什么水平"，而不是代理模型自己的结构。
    """
    seed = cfg.seed if seed is None else seed
    set_seed(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(X)
    i_tr, i_va, _ = _split_idx(n, cfg)

    if len(cols) == 0:
        # 空组合：只能猜训练段均值
        pred = np.full(len(eval_idx), y[i_tr].mean())
        yy = y[eval_idx]
        sst = ((yy - yy.mean()) ** 2).sum()
        return float(1.0 - ((pred - yy) ** 2).sum() / max(sst, 1e-12))

    Xc = X[:, list(cols)]
    xm, xs = Xc[i_tr].mean(0), Xc[i_tr].std(0)
    xs[xs < 1e-12] = 1.0
    ym, ys = y[i_tr].mean(), y[i_tr].std()
    ys = ys if ys > 1e-12 else 1.0
    Xs = ((Xc - xm) / xs).astype(np.float32)
    Ys = ((y - ym) / ys).astype(np.float32)

    xt = torch.tensor(Xs)
    yt = torch.tensor(Ys).view(-1, 1)
    model = _mlp(len(cols), hidden, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    lossf = nn.MSELoss()

    tr = torch.tensor(i_tr)
    va = torch.tensor(i_va)
    best = (math.inf, None)
    bad = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(i_tr))
        for s in range(0, len(i_tr), cfg.batch_size):
            b = tr[perm[s:s + cfg.batch_size]]
            loss = lossf(model(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(xt[va]), yt[va]))
        if vl < best[0] - 1e-7:
            best, bad = (vl, copy.deepcopy(model.state_dict())), 0
        else:
            bad += 1
            if ep >= 40 and bad >= 25:
                break
    if best[1] is not None:
        model.load_state_dict(best[1])

    model.eval()
    with torch.no_grad():
        pr = model(xt[torch.tensor(eval_idx)]).numpy().ravel() * ys + ym
    yy = y[eval_idx]
    sst = ((yy - yy.mean()) ** 2).sum()
    return float(1.0 - ((pr - yy) ** 2).sum() / max(sst, 1e-12))


class ValueFunction:
    """把训练好的代理模型包装成"给一个字段组合，返回它的还原能力"的函数。

    还原能力用测试段 R² 衡量：R² = 1 − 残差平方和/总平方和。
    R² = 1 表示完全还原，R² = 0 表示只相当于猜平均值，可能为负。

    评估用的行**固定下来**：所有字段组合共用同一批测试样本。
    不固定的话，两个组合之间的差异里会混进抽样噪声，
    而贡献值恰恰是靠组合之间的差异算出来的。
    """

    def __init__(self, res: FitResult, X: np.ndarray, Y: np.ndarray,
                 n_eval: int = 2000, seed: int = 0):
        self.res = res
        self.p = X.shape[1]
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        idx = res.idx_test
        if n_eval and len(idx) > n_eval:
            idx = np.random.default_rng(seed).choice(idx, size=n_eval, replace=False)
            idx = np.sort(idx)
        self.idx = idx
        Xs = ((np.asarray(X, dtype=np.float64) - res.x_mean) / res.x_std)
        self.xt = torch.tensor(Xs[idx].astype(np.float32))
        self.y = Y[idx]
        # 总平方和以**测试段自身均值**为基准
        self.sst = ((self.y - self.y.mean(0)) ** 2).sum(0)
        self.sst[self.sst < 1e-12] = 1.0
        self.n_calls = 0

    @torch.no_grad()
    def _predict(self, masks: np.ndarray) -> np.ndarray:
        """一批掩码一次算完，返回 (n_mask, n_eval, k) 的原尺度预测。

        把 B 个掩码和 n 行样本铺成 B×n 行一次前向，而不是循环 B 次。
        全量实验一个目标要查上万个字段组合，这里是最热的一段。
        """
        B, n_row = len(masks), len(self.idx)
        xrep = self.xt.expand(B, n_row, self.p).reshape(B * n_row, self.p)
        mrep = torch.tensor(masks).unsqueeze(1).expand(B, n_row, self.p) \
                    .reshape(B * n_row, self.p)
        pr = self.res.model(xrep, mrep).numpy().astype(np.float64)
        pr = pr.reshape(B, n_row, -1)
        return pr * self.res.y_std + self.res.y_mean

    def __call__(self, masks: np.ndarray, batch: int = 64) -> np.ndarray:
        """masks (n, p) → v 值。单目标返回 (n,)，多目标返回 (n, k)。"""
        masks = np.atleast_2d(np.asarray(masks, dtype=np.float32))
        # 每批的行数控制在 ~1.5e6 以内，避免一次铺开吃掉太多内存
        batch = max(1, min(batch, int(1.5e6 // max(len(self.idx), 1))))
        vals = []
        for s in range(0, len(masks), batch):
            pr = self._predict(masks[s:s + batch])
            sse = ((pr - self.y[None]) ** 2).sum(axis=1)      # (b, k)
            vals.append(1.0 - sse / self.sst[None])
        self.n_calls += len(masks)
        v = np.concatenate(vals, axis=0)
        return v[:, 0] if v.shape[1] == 1 else v

    def of_sets(self, sets: list[list[int]]) -> np.ndarray:
        """按字段下标列表查询，等价于 __call__ 但入参更直观。"""
        m = np.zeros((len(sets), self.p), dtype=np.float32)
        for i, s in enumerate(sets):
            if len(s):
                m[i, list(s)] = 1.0
        return self(m)

    def full(self) -> np.ndarray:
        return self(np.ones((1, self.p), dtype=np.float32))[0] if self.res.model.n_out == 1 \
            else self(np.ones((1, self.p), dtype=np.float32))[0]

    def empty(self) -> np.ndarray:
        return self(np.zeros((1, self.p), dtype=np.float32))[0] if self.res.model.n_out == 1 \
            else self(np.zeros((1, self.p), dtype=np.float32))[0]
