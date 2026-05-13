"""
基于统计先验知识驱动的自适应门控机制的中心变量预测网络。

使用方式（直接在本文件底部改参数后运行）：
    python MetaGating_CenterDataPredictor.py
"""

import os
import re
import json
# 兼容部分环境中 torch._dynamo / torch.onnx 的导入兼容性问题：
# 禁用 Dynamo/compile，避免在构建 optimizer 时触发 torch._dynamo -> torch.onnx 的导入链。
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleAdam:
    """
    一个最小可用的 Adam 实现，用于绕开某些环境里 torch.optim 触发 torch._compile/torch._dynamo 导入链导致的异常。
    支持参数组（类似 torch.optim.Adam 的参数组）。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        # 支持参数组：如果传入的是字典列表，则处理参数组
        if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
            # 参数组格式：[{'params': ..., 'lr': ...}, ...]
            self.param_groups = []
            for group in params:
                group_params = [p for p in group['params'] if p.requires_grad]
                self.param_groups.append({
                    'params': group_params,
                    'lr': group.get('lr', lr),
                    'betas': group.get('betas', betas),
                    'eps': group.get('eps', eps),
                    'weight_decay': group.get('weight_decay', weight_decay),
                })
        else:
            # 单个参数组格式
            group_params = [p for p in params if p.requires_grad]
            self.param_groups = [{
                'params': group_params,
                'lr': lr,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay,
            }]
        
        self.t = 0
        self.m = {}
        self.v = {}

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1

        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                pid = id(p)
                if pid not in self.m:
                    self.m[pid] = torch.zeros_like(p)
                    self.v[pid] = torch.zeros_like(p)

                m = self.m[pid]
                v = self.v[pid]

                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                # bias correction
                m_hat = m / (1 - b1 ** self.t)
                v_hat = v / (1 - b2 ** self.t)

                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)


class MetaGatedRegressor(nn.Module):
    """
    基于统计先验知识驱动的自适应门控回归网络。
    
    对每个特征 i，使用其6维相关系数向量 Ri 生成门控 gi：
    gi = Sigmoid(W_meta^T * Ri + b_meta)
    
    然后对特征进行加权：X_i^new = X_i * gi
    """

    def __init__(self, in_dim: int, correlation_vectors: np.ndarray):
        """
        Args:
            in_dim: 输入特征维度
            correlation_vectors: 形状为 (in_dim, 6) 的数组，每行是一个特征的6维相关系数向量
                               顺序为: [nmi, spearman, pearson, kendall, distance_corr, hsic]
        """
        super().__init__()
        
        # === 核心修改1：强制 Z-Score 标准化 ===
        # 即使原始数据是 0-1，也要变成 均值0 方差1
        # 这样差异会被放大，且有正有负，利于 W 区分
        cor_tensor = torch.from_numpy(correlation_vectors).float()
        mean = cor_tensor.mean(dim=0, keepdim=True)
        std = cor_tensor.std(dim=0, keepdim=True) + 1e-8
        norm_cor_vectors = (cor_tensor - mean) / std
        
        self.register_buffer('correlation_vectors', norm_cor_vectors)
        
        # === 核心修改2：增大 W 初始化范围 ===
        # 使用均匀分布初始化，范围稍微大一点，让 W 一开始就有话语权
        self.W_meta = nn.Parameter(torch.rand(6, 1) - 0.5)  # [-0.5, 0.5]
        
        # b 初始化为 0，配合 Warm-up 策略
        self.b_meta = nn.Parameter(torch.tensor([0.0]))
        
        # 后续 MLP
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入特征，形状为 (batch_size, in_dim)
        
        Returns:
            预测值，形状为 (batch_size, 1)
        """
        # 计算门控：gi = Sigmoid(W_meta^T * Ri + b_meta)
        # correlation_vectors: (in_dim, 6)
        # W_meta: (6, 1)
        # correlation_vectors @ W_meta: (in_dim, 1) -> squeeze -> (in_dim,)
        gate_logits = torch.matmul(self.correlation_vectors, self.W_meta).squeeze(-1) + self.b_meta
        gates = torch.sigmoid(gate_logits)  # (in_dim,)
        
        # 特征加权：X_i^new = X_i * gi
        gated_x = x * gates.unsqueeze(0)  # (batch_size, in_dim)
        
        return self.net(gated_x)
    
    def get_gates(self) -> torch.Tensor:
        """获取当前的门控值 gi"""
        gate_logits = torch.matmul(self.correlation_vectors, self.W_meta).squeeze(-1) + self.b_meta
        return torch.sigmoid(gate_logits)


def gated_loss(pred: torch.Tensor, target: torch.Tensor, gates: torch.Tensor, lambda_l1: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    带 L1 正则化的损失函数。
    
    Args:
        pred: 预测值
        target: 真实值
        gates: 门控值 gi
        lambda_l1: L1 正则化系数
    
    Returns:
        total_loss, mse_loss, l1_loss
    """
    mse = nn.functional.mse_loss(pred, target)
    # 使用 Mean 避免梯度过大
    l1 = torch.mean(torch.abs(gates))
    return mse + lambda_l1 * l1, mse, l1


@dataclass
class TrainConfig:
    center_rel_path: str
    data_path: str
    correlation_path: str  # 相关系数文件路径
    batch_size: int = 50
    epochs: int = 200
    lr: float = 1e-3
    lambda_l1: float = 0.05  # L1 正则化系数（因为用了 Mean，可以稍微大一点）
    warmup_epochs: int = 50  # === 核心修改3：预热轮数 ===
    active_threshold: float = 0.06  # 活跃特征阈值（门控参数绝对值大于此值视为活跃）
    train_ratio: float = 0.8  # 4:1 训练:测试
    random_state: int = 42


def _select_related_columns(center_rel_path: str) -> Tuple[str, List[str]]:
    """
    从 center 关系明细文件中选出中心列名和所有"达阈值"的 related 列名。
    若文件中存在多个指标达阈值列，则只要任意一个指标标记为 1 即被选中。
    """
    df = pd.read_csv(center_rel_path)
    if "center" not in df.columns or "related" not in df.columns:
        raise ValueError("中心关系文件缺少 center 或 related 列")

    # 找出标记列：列名中包含 '达阈值(' 的列
    flag_cols = [c for c in df.columns if "达阈值(" in c]
    if not flag_cols:
        raise ValueError("中心关系文件中没有达阈值标记列，请先使用 CenterRelationshipExtractor 生成")

    center_name = str(df["center"].iloc[0]).strip()
    # 任意标记列为1即视为达阈值
    mask = df[flag_cols].fillna(0).astype(int).sum(axis=1) > 0
    related_cols = df.loc[mask, "related"].astype(str).str.strip().tolist()

    if not related_cols:
        raise ValueError("没有任何 related 列满足阈值，请检查阈值或中心名称")

    return center_name, related_cols


def _load_correlation_vectors(correlation_path: str, center: str, related: List[str]) -> np.ndarray:
    """
    从相关系数文件中加载每个特征的6维相关系数向量。
    
    Args:
        correlation_path: 相关系数文件路径
        center: 中心变量名
        related: 相关变量名列表
    
    Returns:
        形状为 (len(related), 6) 的数组，每行是一个特征的6维相关系数向量
        顺序为: [nmi, spearman, pearson, kendall, distance_corr, hsic]
    """
    df = pd.read_csv(correlation_path)
    
    # 检查必需的列
    required_cols = ['nmi', 'spearman', 'pearson', 'kendall', 'distance_corr', 'hsic']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"相关系数文件缺少必需的列: {missing_cols}")
    
    # 筛选出中心变量对应的行
    if 'center' not in df.columns:
        raise ValueError("相关系数文件缺少 center 列")
    
    center_df = df[df['center'] == center].copy()
    if center_df.empty:
        raise ValueError(f"相关系数文件中没有找到中心变量: {center}")
    
    # 为每个 related 变量提取相关系数向量
    correlation_vectors = []
    for rel_col in related:
        rel_row = center_df[center_df['related'] == rel_col]
        if rel_row.empty:
            # 如果找不到，使用NaN填充
            print(f"警告: 在相关系数文件中未找到 {rel_col}，使用NaN填充")
            correlation_vectors.append([np.nan] * 6)
        else:
            row = rel_row.iloc[0]
            vec = [
                float(row['nmi']),
                float(row['spearman']),
                float(row['pearson']),
                float(row['kendall']),
                float(row['distance_corr']),
                float(row['hsic'])
            ]
            correlation_vectors.append(vec)
    
    correlation_array = np.array(correlation_vectors, dtype=np.float32)
    
    # 处理NaN值：用0填充（或者可以用均值填充）
    if np.isnan(correlation_array).any():
        print("警告: 发现NaN值，使用0填充")
        correlation_array = np.nan_to_num(correlation_array, nan=0.0)
    
    return correlation_array


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    if ss_tot.item() == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def _is_numeric(value: str) -> bool:
    """判断字符串是否可转换为浮点数。"""
    try:
        float(str(value).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def _detect_data_start(raw_df: pd.DataFrame) -> int:
    """检测从哪一行开始出现数值数据。"""
    for idx in range(len(raw_df)):
        row = raw_df.iloc[idx]
        numeric_count = sum(_is_numeric(value) for value in row)
        if numeric_count >= max(2, len(row) * 0.5):
            return idx
    return min(2, len(raw_df))


def _compose_column_names(raw_df: pd.DataFrame, data_start: int) -> List[str]:
    """根据数据起始行之前的内容构造列名（支持前两行都是列名的情况）。"""
    header_rows = raw_df.iloc[:data_start]
    columns: List[str] = []

    for col_idx in header_rows.columns:
        values = [
            str(value).strip()
            for value in header_rows[col_idx]
            if str(value).strip() not in ("", "nan", "NaN")
        ]
        if values:
            # 用 " | " 连接多行列名（例如：一级列名 | 二级列名）
            name = " | ".join(dict.fromkeys(values))
        else:
            name = f"column_{col_idx + 1}"

        # 处理重名列
        original = name
        suffix = 1
        while name in columns:
            suffix += 1
            name = f"{original}_{suffix}"
        columns.append(name)

    return columns


def _prepare_dataset(
    data_path: str,
    center: str,
    related: List[str],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取数据文件，支持前两行都是列名的结构（与 analyzer.py 的处理方式一致）。
    """
    # 兼容中文数据文件编码：优先 utf-8，失败则回退 gb18030
    try:
        raw_df = pd.read_csv(data_path, header=None, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        raw_df = pd.read_csv(data_path, header=None, dtype=str, keep_default_na=False, encoding="gb18030")
    
    if raw_df.empty:
        raise ValueError("数据文件为空")

    # 检测数据起始行并构造列名
    data_start = _detect_data_start(raw_df)
    column_names = _compose_column_names(raw_df, data_start)

    # 提取数值数据并设置列名
    numeric_df = raw_df.iloc[data_start:].replace("", np.nan)
    numeric_df.columns = column_names
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
    numeric_df.dropna(axis=0, how="all", inplace=True)
    numeric_df.dropna(axis=1, how="all", inplace=True)

    if numeric_df.empty:
        raise ValueError("数据经过清洗后为空，请检查数据质量")

    # 检查所需列是否存在
    cols_needed = [center] + related
    missing = [c for c in cols_needed if c not in numeric_df.columns]
    if missing:
        raise ValueError(f"数据文件缺少列: {missing}")

    # 只保留需要的列并丢弃含缺失值的行
    sub = numeric_df[cols_needed].dropna()
    if sub.empty:
        raise ValueError("数据经过清洗后为空，请检查数据质量")

    X_np = sub[related].to_numpy(dtype=np.float32)
    y_np = sub[center].to_numpy(dtype=np.float32).reshape(-1, 1)
    return X_np, y_np


def _extract_data_number(data_path: str) -> str:
    """从数据路径提取编号，例如 'RealData/data4.csv' -> '4'"""
    basename = os.path.basename(data_path)
    match = re.search(r'data(\d+)', basename, re.IGNORECASE)
    if match:
        return match.group(1)
    return "unknown"


def _create_name_mapping(names: List[str]) -> Dict[str, int]:
    """创建名称到数字序号的映射表。"""
    return {name: idx + 1 for idx, name in enumerate(names)}


def train_model(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.random_state)
    center, related = _select_related_columns(cfg.center_rel_path)
    
    # 加载相关系数向量
    correlation_vectors = _load_correlation_vectors(cfg.correlation_path, center, related)
    print(f"已加载 {len(related)} 个特征的相关系数向量，形状: {correlation_vectors.shape}")
    
    # 自动构建输出目录结构：MetaGatingModelOutput/BasedOnData{i}/CenteredOn{中心变量}(epochs={},lambda={})/
    data_num = _extract_data_number(cfg.data_path)
    # 清理中心变量名（去掉特殊字符，用于文件夹名）
    center_safe = "".join(c for c in center if c.isalnum() or c in (" ", "_", "-")).strip()
    center_safe = center_safe.replace(" ", "_")
    if not center_safe:
        center_safe = "center"
    
    # 格式化 lambda 值（避免科学计数法，保留适当精度）
    lambda_str = f"{cfg.lambda_l1:.0e}" if cfg.lambda_l1 < 0.01 else f"{cfg.lambda_l1:.4f}".rstrip('0').rstrip('.')
    # 在目录名中包含 active_threshold 参数
    center_dir_name = f"CenteredOn{center_safe}(epochs={cfg.epochs},lambda={lambda_str},active={cfg.active_threshold})"
    base_dir = os.path.join("MetaGatingModelOutput", f"BasedOnData{data_num}", center_dir_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # 简化文件名
    output_model_path = os.path.join(base_dir, "model.pth")
    log_path = os.path.join(base_dir, "log.csv")
    loss_fig_path = os.path.join(base_dir, "loss.png")
    r2_fig_path = os.path.join(base_dir, "r2.png")
    gate_fig_path = os.path.join(base_dir, "gate_params.png")
    active_features_fig_path = os.path.join(base_dir, "active_features.png")
    w_meta_fig_path = os.path.join(base_dir, "W_meta_evolution.png")
    b_meta_fig_path = os.path.join(base_dir, "b_meta_evolution.png")
    wb_snapshot_path = os.path.join(base_dir, "W_b_snapshots.json")
    name_mapping_path = os.path.join(base_dir, "name_mapping.csv")
    keydata_path = os.path.join(base_dir, "keydata_for_pointdata.json")

    # 创建名称映射表（用于可视化时避免中文）
    all_names = [center] + related
    name_mapping = _create_name_mapping(all_names)
    mapping_df = pd.DataFrame([
        {"index": idx, "name": name}
        for name, idx in name_mapping.items()
    ])
    mapping_df.to_csv(name_mapping_path, index=False, encoding="utf-8-sig")

    X_np, y_np = _prepare_dataset(cfg.data_path, center, related, cfg.random_state)

    # 先划分训练/测试，再只用训练集统计量进行标准化（避免数据泄漏）
    n = X_np.shape[0]
    rng = np.random.default_rng(cfg.random_state)
    perm = rng.permutation(n)
    train_size = int(n * cfg.train_ratio)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    X_train = X_np[train_idx]
    y_train = y_np[train_idx]
    X_test = X_np[test_idx]
    y_test = y_np[test_idx]

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True) + 1e-8
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-8)

    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MetaGatedRegressor(in_dim=len(related), correlation_vectors=correlation_vectors).to(DEVICE)
    # === 核心修改4：给 W_meta 设置大一点的学习率，让它学得快一点 ===
    # 注意：必须将单个参数用列表包裹，否则会被迭代成切片（非叶子节点）
    optimizer = SimpleAdam([
        {'params': [model.W_meta], 'lr': cfg.lr * 5},  # W 学快点
        {'params': [model.b_meta], 'lr': cfg.lr},
        {'params': model.net.parameters(), 'lr': cfg.lr}
    ])

    log_rows = []
    gate_history = []  # 记录每个 epoch 的门控参数
    active_features_history = []  # 记录每个 epoch 的活跃特征数量
    w_meta_history = []  # 记录每个 epoch 的 W_meta 参数 (6个值)
    b_meta_history = []  # 记录每个 epoch 的 b_meta 参数
    # 记录特定 epoch 的活跃特征（用于 keydata 分析）
    key_epochs = [10, 20, 50, 100, 200]
    keydata_records = {}  # {epoch: {"active_indices": [...], "active_names": [...], "count": int}}
    wb_snapshots = {}  # {epoch: {"W_meta": [...], "b_meta": float}}
    best_state = None
    best_r2 = -1e9

    print(f"开始训练... 前 {cfg.warmup_epochs} 轮为 Warm-up (无 L1 惩罚)")
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_mse_loss = 0.0
        train_l1_loss = 0.0
        
        # === 核心修改5：动态 Lambda ===
        # 在 Warm-up 期间，lambda = 0，只优化 MSE
        # 这样模型会先学会"哪些特征对预测有用"，从而调整 W
        current_lambda = 0.0 if epoch <= cfg.warmup_epochs else cfg.lambda_l1
        
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            gates = model.get_gates()
            total_loss, mse_loss, l1_loss = gated_loss(pred, yb, gates, current_lambda)
            total_loss.backward()
            optimizer.step()
            train_total_loss += total_loss.item() * xb.size(0)
            train_mse_loss += mse_loss.item() * xb.size(0)
            train_l1_loss += l1_loss.item() * xb.size(0)
        
        train_total_loss /= len(train_ds)
        train_mse_loss /= len(train_ds)
        train_l1_loss /= len(train_ds)

        # 记录门控参数
        gate_values = model.get_gates().detach().cpu().numpy()
        gate_history.append(gate_values.copy())
        
        # 记录 W_meta 和 b_meta
        w_meta_values = model.W_meta.detach().cpu().numpy().flatten()  # (6,)
        b_meta_value = model.b_meta.detach().cpu().item()
        w_meta_history.append(w_meta_values.copy())
        b_meta_history.append(b_meta_value)
        
        # 计算活跃特征数量（绝对值大于阈值）
        active_mask = np.abs(gate_values) > cfg.active_threshold
        active_count = np.sum(active_mask)
        active_features_history.append(active_count)
        
        # 记录特定 epoch 的活跃特征和 W/b 快照
        if epoch in key_epochs:
            active_indices = np.where(active_mask)[0].tolist()
            active_names = [related[i] for i in active_indices]
            
            # 计算推断风险：W_meta^T * Ri + b_meta > 0 说明存在推断风险
            gate_logits = np.matmul(correlation_vectors, w_meta_values.reshape(-1, 1)).flatten() + b_meta_value
            inference_risk_mask = gate_logits > 0
            inference_risk_indices = np.where(inference_risk_mask)[0].tolist()
            inference_risk_names = [related[i] for i in inference_risk_indices]
            
            keydata_records[epoch] = {
                "active_indices": active_indices,
                "active_names": active_names,
                "count": int(active_count),
                "inference_risk_indices": inference_risk_indices,
                "inference_risk_names": inference_risk_names,
                "inference_risk_count": int(np.sum(inference_risk_mask))
            }
            wb_snapshots[epoch] = {
                "W_meta": w_meta_values.tolist(),
                "b_meta": float(b_meta_value)
            }

        # 评估（只用 MSE，不加 L1 惩罚）
        def _eval(loader):
            model.eval()
            total_loss = 0.0
            ys = []
            ps = []
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    pred = model(xb)
                    loss = nn.functional.mse_loss(pred, yb)  # 只用 MSE
                    total_loss += loss.item() * xb.size(0)
                    ys.append(yb.cpu())
                    ps.append(pred.cpu())
            total_loss /= len(loader.dataset)
            y_cat = torch.cat(ys, dim=0)
            p_cat = torch.cat(ps, dim=0)
            r2 = _r2_score(y_cat, p_cat)
            return total_loss, r2

        train_eval_loss, train_r2 = _eval(train_loader)
        test_loss, test_r2 = _eval(test_loader)

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_eval_loss,
                "train_r2": train_r2,
                "test_loss": test_loss,
                "test_r2": test_r2,
                "train_l1_loss": train_l1_loss,
                "active_features": active_count,
            }
        )

        # 保存最好模型（按 test_r2）
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_state = {
                "model_state": model.state_dict(),
                "center": center,
                "related": related,
                "correlation_vectors": correlation_vectors,
                # 保存训练集标准化参数，便于推理时做同样的变换 / 反变换
                "x_mean": x_mean.squeeze(),
                "x_std": x_std.squeeze(),
                "y_mean": float(y_mean),
                "y_std": float(y_std),
            }

        print(
            f"Epoch {epoch:02d} | "
            f"Lambda: {current_lambda:.4f} | "
            f"train_loss={train_eval_loss:.4f} train_r2={train_r2:.4f} | "
            f"test_loss={test_loss:.4f} test_r2={test_r2:.4f} | "
            f"active_features={active_count}/{len(related)} | "
            f"Mean Gate: {gate_values.mean():.3f}"
        )

    # 写日志
    pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    # 绘图（loss & R2），使用数字序号避免中文
    log_df = pd.DataFrame(log_rows)

    plt.figure(figsize=(7, 4))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    plt.plot(log_df["epoch"], log_df["test_loss"], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train/Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig_path, dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(log_df["epoch"], log_df["train_r2"], label="train_r2")
    plt.plot(log_df["epoch"], log_df["test_r2"], label="test_r2")
    plt.xlabel("Epoch")
    plt.ylabel("R^2")
    plt.title("Train/Test R^2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(r2_fig_path, dpi=150)
    plt.close()

    # 门控参数随 epoch 变化
    gate_array = np.array(gate_history)  # shape: (epochs, n_features)
    plt.figure(figsize=(10, 6))
    for feat_idx in range(gate_array.shape[1]):
        feat_num = name_mapping.get(related[feat_idx], feat_idx + 1)
        plt.plot(log_df["epoch"], gate_array[:, feat_idx], label=f"Feature_{feat_num}", alpha=0.7)
    # 画一条竖线标记 Warm-up 结束
    plt.axvline(x=cfg.warmup_epochs, color='r', linestyle='--', label='Warm-up End', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Gate Parameter Value (gi)")
    plt.title("Gate Parameters Over Epochs (with Warm-up)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(gate_fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 活跃特征数量随 epoch 变化
    plt.figure(figsize=(7, 4))
    plt.plot(log_df["epoch"], log_df["active_features"], marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Number of Active Features")
    plt.title("Number of Active Features Over Epochs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(active_features_fig_path, dpi=150)
    plt.close()

    # W_meta 的6个值随 epoch 变化
    w_meta_array = np.array(w_meta_history)  # shape: (epochs, 6)
    plt.figure(figsize=(10, 6))
    w_labels = ['w1 (nmi)', 'w2 (spearman)', 'w3 (pearson)', 'w4 (kendall)', 'w5 (distance_corr)', 'w6 (hsic)']
    for i in range(6):
        plt.plot(log_df["epoch"], w_meta_array[:, i], label=w_labels[i], alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("W_meta Value")
    plt.title("W_meta Parameters Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(w_meta_fig_path, dpi=150)
    plt.close()

    # b_meta 随 epoch 变化
    b_meta_array = np.array(b_meta_history)  # shape: (epochs,)
    plt.figure(figsize=(7, 4))
    plt.plot(log_df["epoch"], b_meta_array, marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("b_meta Value")
    plt.title("b_meta Parameter Over Epochs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(b_meta_fig_path, dpi=150)
    plt.close()

    # 保存 keydata 记录（JSON 格式）
    if keydata_records:
        with open(keydata_path, 'w', encoding='utf-8') as f:
            json.dump(keydata_records, f, ensure_ascii=False, indent=2)
        print(f"Keydata 记录已保存: {keydata_path}")

    # 保存 W 和 b 的快照（JSON 格式）
    if wb_snapshots:
        with open(wb_snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(wb_snapshots, f, ensure_ascii=False, indent=2)
        print(f"W 和 b 快照已保存: {wb_snapshot_path}")

    # 保存最佳模型
    if best_state is not None:
        torch.save(best_state, output_model_path)
        print(f"最佳模型已保存: {output_model_path} (test_r2={best_r2:.4f})")
        print(f"日志已保存: {log_path}")
        print(f"可视化图表已保存: {loss_fig_path}, {r2_fig_path}, {gate_fig_path}, {active_features_fig_path}")
        print(f"W_meta 和 b_meta 演化图已保存: {w_meta_fig_path}, {b_meta_fig_path}")
        print(f"名称映射表已保存: {name_mapping_path}")


if __name__ == "__main__":
    # 在这里改参数（输出路径会自动生成：MetaGatingModelOutput/BasedOnData{i}/CenteredOn{中心变量}(epochs={},lambda={})/）
    CONFIG = TrainConfig(
        center_rel_path="CenterDataRelationships/center_realdata4_地区节点阻塞价格.csv",
        data_path="RealData/data4.csv",
        correlation_path="CenterDataRelationships/center_realdata4_地区节点阻塞价格.csv",  # 相关系数文件路径
        batch_size=50,
        epochs=200,
        lr=1e-3,
        lambda_l1=0.03,  # L1 正则化系数（因为用了 Mean，可以稍微大一点，试 0.02 - 0.1）
        warmup_epochs=50,  # 预热 50 轮
        active_threshold=0.06,  # 活跃特征阈值
        train_ratio=0.8,
        random_state=42,
    )

    train_model(CONFIG)
