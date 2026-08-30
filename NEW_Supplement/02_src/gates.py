"""门控推断模型：在网络输入端给每个字段配一个开关，训练中自动把没用的字段关掉。

三种模型：

  DNN         普通全连接网络，不做任何字段选择，作为可推断性的上限参照
  L1Gate      每个字段配一个门控系数，对门控系数加 L1 惩罚推动它们趋零
  DGating     把每个门控系数拆成 D 个因子相乘，改对因子加 L2 惩罚。
              效果等价于对乘积施加比 L1 更强的稀疏惩罚，好处是门控值分布更贴近
              "要么接近零、要么明显非零"的两极，中间地带很少，
              于是"门控值多大算活跃"这个阈值落在空档里，取多少都不影响结果。
              （Differentiable Sparsity via D-Gating, NeurIPS 2025）

选出字段之后，还要用普通 DNN 只拿这些字段重新训练一遍。这一步是为了排除
"高精度是门控层自身训练带来的"这种可能——只有重训后仍然准，才说明这组字段
本身确实携带了推断目标所需的信息。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from torch import nn


@dataclass
class TrainConfig:
    hidden: tuple = (64, 32, 16)
    epochs: int = 200
    batch_size: int = 50
    lr: float = 1e-3
    train_ratio: float = 0.8
    seed: int = 42
    # L1Gate
    lambda_l1: float = 0.01
    active_threshold: float = 0.05
    # DGating
    lambda_dgate: float = 0.005
    dgate_depth: int = 4
    dgate_threshold: float = 0.01
    # STG（随机门控）
    lambda_stg: float = 0.05
    stg_sigma: float = 0.5
    stg_threshold: float = 0.0   # 原始 μ 尺度上，正负之间是一片空档

    def to_dict(self) -> dict:
        return asdict(self)


def _mlp(in_dim: int, hidden) -> nn.Sequential:
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class DNN(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = _mlp(in_dim, hidden)

    def forward(self, x):
        return self.net(x)

    def gates(self):
        return None


class L1Gate(nn.Module):
    """每个字段一个门控系数，逐元素乘在输入上。"""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(in_dim))
        self.net = _mlp(in_dim, hidden)
        self.rescale()                       # 先做一次尺度补偿，确定初始基准
        self._gate0 = self.gate.detach().abs().cpu().numpy().copy()

    def forward(self, x):
        return self.net(x * self.gate)

    def gates(self):
        """门控值以"训练开始时的水平"为 1。

        不能除以当前最大值来归一化——最大值本身在训练中会换字段，
        分母一变，所有曲线都会跟着出现假的涨落（实测在最大值易主的那一轮，
        全部字段的门控值同时出现一个驼峰，看上去像是突然涨了）。
        改成除以初始值：每个字段都从 1 出发，之后的升降都是它自己的真实变化。
        """
        v = self.gate.detach().abs().cpu().numpy()
        return v / self._gate0

    def penalty(self, cfg):
        return cfg.lambda_l1 * self.gate.abs().sum()

    @torch.no_grad()
    def rescale(self):
        """尺度补偿：把第一层权重按列归一化，把范数吸收进门控系数。

        不做这一步，门控值的绝对大小是没有意义的——网络可以把某个字段的门控
        压到 0.01，同时把第一层对应的权重放大 100 倍，输出一模一样。
        训练轮次一多，所有门控值都会被压到阈值以下，"门控值超过多少算活跃"
        这个判据就失效了（实测 200 轮后活跃字段数变成 0，而模型 R² 依然很高）。

        归一化之后，字段的重要性只能体现在门控值上，L1 惩罚才真正作用在
        字段重要性上，活跃阈值也才有意义。这一步不改变网络输出。
        """
        W = self.net[0].weight            # (第一隐层维度, 字段数)
        nrm = W.norm(dim=0).clamp(min=1e-8)
        W.div_(nrm)
        self.gate.mul_(nrm)


class DGating(nn.Module):
    """门控系数拆成 D 个因子相乘，对因子加 L2 惩罚。

    第一层的等效权重是 omega[i,j] × (gamma[0,i] × gamma[1,i] × …)，
    某个字段的门控强度就是它那一行等效权重的范数。
    """

    def __init__(self, in_dim, hidden, depth=4):
        super().__init__()
        h0 = hidden[0]
        self.depth = depth
        self.omega = nn.Parameter(torch.empty(in_dim, h0))
        nn.init.kaiming_normal_(self.omega, nonlinearity="relu")
        self.gamma = nn.Parameter(torch.ones(depth - 1, in_dim))
        self.bias = nn.Parameter(torch.zeros(h0))
        self.rest = _mlp(h0, hidden[1:]) if len(hidden) > 1 else nn.Linear(h0, 1)

    def _eff(self):
        g = self.gamma.prod(dim=0)          # 每个字段的因子乘积
        return self.omega * g.unsqueeze(1)

    def forward(self, x):
        h = torch.relu(x @ self._eff() + self.bias)
        return self.rest(h)

    def gates(self):
        """报告的是各因子的乘积 ∏γ，初始值恰好是 1（γ 全部初始化为 1）。

        这样每个字段都从 1 出发，能直接看出它是被压到零还是被保留下来，
        不需要再做任何归一化。被淘汰的字段这个乘积会精确收敛到 0。
        """
        return self.gamma.detach().prod(dim=0).abs().cpu().numpy()

    def penalty(self, cfg):
        return cfg.lambda_dgate * (self.omega.pow(2).sum() + self.gamma.pow(2).sum())


class STG(nn.Module):
    """随机门控（Stochastic Gates, ICML 2020）。

    给每个字段一个可学的中心值 μ，训练时在它上面加高斯噪声再截断到 [0,1] 当作门。
    这样做的用意是：真正想优化的是"用了几个字段"（也就是 L0），但 L0 不可导，
    于是用"门被打开的概率"来代替——某个字段的门大于零的概率是可以写出解析式的，
    对 μ 可导，就能跟着梯度一起训练。

    与本文方法的区别：STG 的门控值分布在 0 到 1 之间连续变化，
    所以"门控值多大算选中"仍然要人为定一个阈值（通常取 0.5）。
    """

    def __init__(self, in_dim, hidden, sigma: float = 0.5):
        super().__init__()
        self.mu = nn.Parameter(torch.full((in_dim,), 0.5))
        self.sigma = sigma
        self.net = _mlp(in_dim, hidden)

    def forward(self, x):
        z = self.mu
        if self.training:
            z = z + self.sigma * torch.randn_like(z)
        return self.net(x * torch.clamp(z, 0.0, 1.0))

    def gates(self):
        """返回未经裁剪的 μ。

        前向传播里门是 clamp(μ+σε, 0, 1)，但**报告重要性时不能用裁剪后的值**：
        所有被选中的字段裁剪后都恰好等于 1，彼此并列，排序完全失效
        （实测按裁剪值取前 5 个，R² 只有 0.5436；按原始 μ 取前 5 个是 0.9252）。
        原始 μ 保留了字段之间的强弱差别，而且被选中与未被选中之间有很宽的间隔
        （实测选中的都在 +2.4 以上、未选中的都在 −2.2 以下），
        所以判定阈值放在这个间隔里任何位置都不影响选中结果。
        """
        return self.mu.detach().cpu().numpy()

    def penalty(self, cfg):
        # 门被打开的概率之和，是"预计用到几个字段"的可导替代
        p = 0.5 * (1.0 + torch.erf(self.mu / (self.sigma * math.sqrt(2.0))))
        return cfg.lambda_stg * p.sum()



class ImprovedDGating(nn.Module):
    """门控值由六类相关系数生成的 D-Gating。

    标准 D-Gating 给每个字段一组自由的门控因子，因子个数随字段数增长；
    这里改成让因子由字段的相关系数向量算出来：

        γ[d, j] = c[j] · W[d] + b[d]          d = 1 … depth-1
        门控值 = |∏_d γ[d, j]|

    c[j] 是第 j 个字段的六个相关系数（先按列标准化）。可学参数只有
    (depth-1) × (6+1) 个，**与字段数无关**——这正是它能推广到新字段的原因：
    来了一个没参与训练的字段，只要算出它的六个相关系数，代入同一组 W、b
    就能直接得到门控值，不需要重新训练。

    W 初始化为 0、b 初始化为 1，于是训练开始时每个字段的 γ 都是 1、
    门控值都是 1，与标准 D-Gating 的起点完全一致，两者才好比较。

    代价是表达能力受限：标准 D-Gating 能把任意单个字段单独压到 0，
    这里的门控值完全由相关系数决定，相关系数相近的字段只能得到相近的门控值。
    能不能压到 0，取决于该字段在六维相关系数空间里的位置。
    """

    def __init__(self, in_dim, hidden, depth=4, corr=None):
        super().__init__()
        if corr is None:
            raise ValueError("ImprovedDGating 需要相关系数矩阵 corr")
        c = torch.as_tensor(np.asarray(corr, dtype=np.float32))
        c = (c - c.mean(0, keepdim=True)) / (c.std(0, keepdim=True) + 1e-8)
        self.register_buffer("corr", c)
        h0 = hidden[0]
        self.depth = depth
        self.omega = nn.Parameter(torch.empty(in_dim, h0))
        nn.init.kaiming_normal_(self.omega, nonlinearity="relu")
        self.W = nn.Parameter(torch.zeros(depth - 1, c.shape[1]))
        self.b = nn.Parameter(torch.ones(depth - 1))
        self.bias = nn.Parameter(torch.zeros(h0))
        self.rest = _mlp(h0, hidden[1:]) if len(hidden) > 1 else nn.Linear(h0, 1)

    def _factors(self):
        """(depth-1, n_fields)：每个字段在每一层的门控因子。"""
        return self.corr @ self.W.t() + self.b            # (n_fields, depth-1)

    def _eff(self):
        g = self._factors().prod(dim=1)                   # (n_fields,)
        return self.omega * g.unsqueeze(1)

    def forward(self, x):
        h = torch.relu(x @ self._eff() + self.bias)
        return self.rest(h)

    def gates(self):
        return self._factors().prod(dim=1).detach().abs().cpu().numpy()

    def penalty(self, cfg):
        f = self._factors()
        return cfg.lambda_dgate * (self.omega.pow(2).sum() + f.pow(2).sum())

    def gate_from_corr(self, corr_new: np.ndarray, corr_ref: np.ndarray) -> np.ndarray:
        """给没参与训练的字段算门控值。

        corr_new 是新字段的原始相关系数，corr_ref 是训练时那批字段的原始
        相关系数——标准化必须沿用训练时的均值和标准差，否则两边不在同一把尺子上。
        """
        ref = np.asarray(corr_ref, dtype=np.float32)
        mu, sd = ref.mean(0), ref.std(0) + 1e-8
        c = torch.as_tensor((np.asarray(corr_new, dtype=np.float32) - mu) / sd)
        with torch.no_grad():
            f = c @ self.W.t() + self.b
            return f.prod(dim=1).abs().cpu().numpy()


MODELS = {"DNN": DNN, "L1GateDNN": L1Gate, "DGatingDNN": DGating, "STG": STG,
          "ImprovedDGating": ImprovedDGating}


def split(n: int, cfg: TrainConfig):
    k = int(n * cfg.train_ratio)
    perm = np.random.default_rng(cfg.seed).permutation(n)
    return perm[:k], perm[k:]


def r2(y, p):
    ss = float(((y - p) ** 2).sum())
    tt = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return 1.0 - ss / tt


@dataclass
class TrainResult:
    model_name: str
    history: dict = field(default_factory=dict)
    gate_history: np.ndarray | None = None
    final_gates: np.ndarray | None = None
    best_test_r2: float = float("nan")
    best_epoch: int = -1
    final_train_r2: float = float("nan")
    final_test_r2: float = float("nan")


def train(model_name: str, X: np.ndarray, y: np.ndarray, cfg: TrainConfig,
          corr: np.ndarray | None = None) -> TrainResult:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    tr, te = split(len(X), cfg)

    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    ym, ys = y[tr].mean(), y[tr].std()
    ys = ys if ys > 0 else 1.0
    yn = (y - ym) / ys

    Xtr = torch.tensor(Xs[tr], dtype=torch.float32)
    ytr = torch.tensor(yn[tr], dtype=torch.float32).view(-1, 1)
    Xte = torch.tensor(Xs[te], dtype=torch.float32)
    yte_raw = y[te]

    kw = ({"depth": cfg.dgate_depth} if model_name == "DGatingDNN"
          else {"depth": cfg.dgate_depth, "corr": corr} if model_name == "ImprovedDGating"
          else {"sigma": cfg.stg_sigma} if model_name == "STG" else {})
    model = MODELS[model_name](X.shape[1], cfg.hidden, **kw)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.MSELoss()

    hist = {"epoch": [], "train_r2": [], "test_r2": [], "n_active": []}
    gate_hist = []
    best, best_ep = -np.inf, -1
    n = len(Xtr)

    # 记录训练开始前的门控值，这样演化图的第一个点是真正的起点（全部为 1）
    g0 = model.gates()
    if g0 is not None:
        gate_hist.append(g0.copy())

    for ep in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            if hasattr(model, "penalty"):
                loss = loss + model.penalty(cfg)
            loss.backward()
            opt.step()
            if hasattr(model, "rescale"):
                model.rescale()

        model.eval()
        with torch.no_grad():
            ptr = model(Xtr).squeeze(-1).numpy() * ys + ym
            pte = model(Xte).squeeze(-1).numpy() * ys + ym
        rtr, rte = r2(y[tr], ptr), r2(yte_raw, pte)
        g = model.gates()
        thr = {"L1GateDNN": cfg.active_threshold, "STG": cfg.stg_threshold}.get(
            model_name, cfg.dgate_threshold)
        na = int((g >= thr).sum()) if g is not None else X.shape[1]
        hist["epoch"].append(ep); hist["train_r2"].append(rtr)
        hist["test_r2"].append(rte); hist["n_active"].append(na)
        if g is not None:
            gate_hist.append(g.copy())
        if rte > best:
            best, best_ep = rte, ep

    return TrainResult(
        model_name=model_name, history=hist,
        gate_history=np.array(gate_hist) if gate_hist else None,
        final_gates=model.gates(), best_test_r2=best, best_epoch=best_ep,
        final_train_r2=hist["train_r2"][-1], final_test_r2=hist["test_r2"][-1],
    )


def retrain_subset(X: np.ndarray, y: np.ndarray, idx: list[int],
                   cfg: TrainConfig) -> float:
    """只用选中的字段重新训练一个普通 DNN，返回最优测试 R²。"""
    if not idx:
        return float("nan")
    return train("DNN", X[:, idx], y, cfg).best_test_r2


# --------------------------------------------------------------------------
# 增量：发布口径里新增一个字段时，不必从头重训
# --------------------------------------------------------------------------

def train_incremental(X: np.ndarray, y: np.ndarray, cfg: TrainConfig,
                      new_pos: int, warm: "DGating | None" = None,
                      epochs: int | None = None,
                      init_gate: float | None = None,
                      reset_gates: bool = False) -> TrainResult:
    """在已训练好的门控模型上插入一个新字段，只训练很少的轮次。

    X 是**加进新字段之后**的完整矩阵，new_pos 是新字段在其中的列号。
    warm 是在去掉该字段的池子上训练好的模型；传 None 就是随机初始化，
    用来做对照——分清收益到底来自"沿用旧解"还是仅仅来自"少训练几轮"。

    新字段的门控因子初值设为 1，与全新训练时每个字段的起点完全一致，
    不给它任何优待。已有字段沿用各自训练出来的值，其中没用的那些早就
    被压到接近 0 了，所以新字段实际上是从最有利的位置出发去竞争——
    它如果没用，惩罚项会很快把它压下去；压不下去才说明它真有用。

    第一层权重按列标准化，各列的均值方差互不影响，所以加一列不会让
    已有列的权重失效，旧解可以直接沿用。
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    tr, te = split(len(X), cfg)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    ym, ys = y[tr].mean(), y[tr].std()
    ys = ys if ys > 0 else 1.0
    yn = (y - ym) / ys

    Xtr = torch.tensor(Xs[tr], dtype=torch.float32)
    ytr = torch.tensor(yn[tr], dtype=torch.float32).view(-1, 1)
    Xte = torch.tensor(Xs[te], dtype=torch.float32)

    model = DGating(X.shape[1], cfg.hidden, depth=cfg.dgate_depth)
    if warm is not None:
        with torch.no_grad():
            keep = [i for i in range(X.shape[1]) if i != new_pos]
            model.omega[keep] = warm.omega.detach().clone()
            model.gamma[:, keep] = warm.gamma.detach().clone()
            # 新字段的门控初值。设成 1.0 是错的：全新训练时所有字段都从 1 出发，
            # 那是公平的；热启动时其余字段已经被 200 轮压到接近 0，只有新字段
            # 从 1 出发，等于给它天大的便宜——实测它 30 轮后仍在 0.88，
            # 按衰减速度外推还要几千轮才跌破阈值，而且排序都是乱的。
            # 改成从判定阈值出发：升上去还是掉下来是个对称的问题，
            # 字段必须靠自己挣到位置。∏γ = init_gate，所以每个因子开 depth-1 次方根。
            if reset_gates:
                # 第三种做法：只沿用学到的网络（第一层权重与后续各层），
                # 门控全部重置为 1，让新旧字段从同一起跑线重新竞争。
                # 省下的是"学函数"那部分工夫，门控的取舍重新来过。
                model.gamma[:] = 1.0
            else:
                g0 = cfg.dgate_threshold if init_gate is None else init_gate
                model.gamma[:, new_pos] = float(g0) ** (1.0 / model.gamma.shape[0])
            model.bias.copy_(warm.bias.detach())
            model.rest.load_state_dict(warm.rest.state_dict())

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.MSELoss()
    ne = epochs if epochs is not None else cfg.epochs
    hist = {"epoch": [], "train_r2": [], "test_r2": [], "n_active": []}
    gate_hist = [model.gates().copy()]
    best, best_ep = -np.inf, -1
    n = len(Xtr)

    for ep in range(1, ne + 1):
        model.train()
        order = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx]) + model.penalty(cfg)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ptr = model(Xtr).squeeze(-1).numpy() * ys + ym
            pte = model(Xte).squeeze(-1).numpy() * ys + ym
        rtr, rte = r2(y[tr], ptr), r2(y[te], pte)
        g = model.gates()
        hist["epoch"].append(ep); hist["train_r2"].append(rtr)
        hist["test_r2"].append(rte)
        hist["n_active"].append(int((g >= cfg.dgate_threshold).sum()))
        gate_hist.append(g.copy())
        if rte > best:
            best, best_ep = rte, ep

    res = TrainResult(
        model_name="DGatingDNN(增量)", history=hist,
        gate_history=np.array(gate_hist), final_gates=model.gates(),
        best_test_r2=best, best_epoch=best_ep,
        final_train_r2=hist["train_r2"][-1], final_test_r2=hist["test_r2"][-1])
    res.model = model
    return res


def train_keep_model(model_name: str, X, y, cfg: TrainConfig,
                     corr: np.ndarray | None = None) -> TrainResult:
    """跟 train() 一样，但把训练好的模型对象也带回来，供增量热启动使用。"""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    tr, te = split(len(X), cfg)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    ym, ys = y[tr].mean(), y[tr].std()
    ys = ys if ys > 0 else 1.0
    yn = (y - ym) / ys
    Xtr = torch.tensor(Xs[tr], dtype=torch.float32)
    ytr = torch.tensor(yn[tr], dtype=torch.float32).view(-1, 1)
    Xte = torch.tensor(Xs[te], dtype=torch.float32)
    model = (ImprovedDGating(X.shape[1], cfg.hidden, depth=cfg.dgate_depth, corr=corr)
             if model_name == "ImprovedDGating"
             else DGating(X.shape[1], cfg.hidden, depth=cfg.dgate_depth))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.MSELoss()
    hist = {"epoch": [], "train_r2": [], "test_r2": [], "n_active": []}
    gate_hist = [model.gates().copy()]
    best, best_ep = -np.inf, -1
    n = len(Xtr)
    for ep in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(n)
        for i in range(0, n, cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx]) + model.penalty(cfg)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ptr = model(Xtr).squeeze(-1).numpy() * ys + ym
            pte = model(Xte).squeeze(-1).numpy() * ys + ym
        rtr, rte = r2(y[tr], ptr), r2(y[te], pte)
        g = model.gates()
        hist["epoch"].append(ep); hist["train_r2"].append(rtr)
        hist["test_r2"].append(rte)
        hist["n_active"].append(int((g >= cfg.dgate_threshold).sum()))
        gate_hist.append(g.copy())
        if rte > best:
            best, best_ep = rte, ep
    res = TrainResult(
        model_name=model_name, history=hist, gate_history=np.array(gate_hist),
        final_gates=model.gates(), best_test_r2=best, best_epoch=best_ep,
        final_train_r2=hist["train_r2"][-1], final_test_r2=hist["test_r2"][-1])
    res.model = model
    return res
