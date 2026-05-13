"""
基于 L1 门控的特征筛选回归网络，用于预测中心变量。

使用方式（直接在本文件底部改参数后运行）：
    python L1Gate_CenterDataPredictor.py
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

# 活跃特征阈值（门控参数绝对值大于此值视为活跃）
ACTIVE_FEATURE_THRESHOLD = 0.06


class SimpleAdam:
    """
    一个最小可用的 Adam 实现，用于绕开某些环境里 torch.optim 触发 torch._compile/torch._dynamo 导入链导致的异常。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        lr = self.lr

        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad

            if self.weight_decay != 0.0:
                g = g.add(p, alpha=self.weight_decay)

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

            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-lr)


class GatedRegressor(nn.Module):
    """带 L1 门控的回归网络，用于特征筛选。"""

    def __init__(self, in_dim: int):
        super().__init__()
        # 输入门控参数（一维、可解释）
        self.gate = nn.Parameter(torch.ones(in_dim))

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
        # 门控（逐特征缩放）
        gated_x = x * self.gate
        return self.net(gated_x)


def gated_loss(pred: torch.Tensor, target: torch.Tensor, gate: torch.Tensor, lambda_l1: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    带 L1 正则化的损失函数。
    
    Returns:
        total_loss, mse_loss, l1_loss
    """
    mse = nn.functional.mse_loss(pred, target)
    l1 = torch.norm(gate, p=1)
    return mse + lambda_l1 * l1, mse, l1


@dataclass
class TrainConfig:
    center_rel_path: str
    data_path: str
    batch_size: int = 50
    epochs: int = 20
    lr: float = 1e-3
    lambda_l1: float = 1e-3  # L1 正则化系数
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
    
    # 自动构建输出目录结构：L1GateModelOutput/BasedOnData{i}/CenteredOn{中心变量}(epochs={},lambda={})/
    data_num = _extract_data_number(cfg.data_path)
    # 清理中心变量名（去掉特殊字符，用于文件夹名）
    center_safe = "".join(c for c in center if c.isalnum() or c in (" ", "_", "-")).strip()
    center_safe = center_safe.replace(" ", "_")
    if not center_safe:
        center_safe = "center"
    
    # 格式化 lambda 值（避免科学计数法，保留适当精度）
    lambda_str = f"{cfg.lambda_l1:.0e}" if cfg.lambda_l1 < 0.01 else f"{cfg.lambda_l1:.4f}".rstrip('0').rstrip('.')
    center_dir_name = f"CenteredOn{center_safe}VCASE(epochs={cfg.epochs},lambda={lambda_str})"
    base_dir = os.path.join("L1GateModelOutput", f"BasedOnData{data_num}", center_dir_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # 简化文件名
    output_model_path = os.path.join(base_dir, "model.pth")
    log_path = os.path.join(base_dir, "log.csv")
    loss_fig_path = os.path.join(base_dir, "loss.png")
    r2_fig_path = os.path.join(base_dir, "r2.png")
    gate_fig_path = os.path.join(base_dir, "gate_params.png")
    active_features_fig_path = os.path.join(base_dir, "active_features.png")
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

    model = GatedRegressor(in_dim=len(related)).to(DEVICE)
    optimizer = SimpleAdam(model.parameters(), lr=cfg.lr)

    log_rows = []
    gate_history = []  # 记录每个 epoch 的门控参数
    active_features_history = []  # 记录每个 epoch 的活跃特征数量
    # 记录特定 epoch 的活跃特征（用于 keydata 分析）
    key_epochs = [10, 20, 50, 100, 200]
    keydata_records = {}  # {epoch: {"active_indices": [...], "active_names": [...], "count": int}}
    best_state = None
    best_r2 = -1e9

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_total_loss = 0.0
        train_mse_loss = 0.0
        train_l1_loss = 0.0
        
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            total_loss, mse_loss, l1_loss = gated_loss(pred, yb, model.gate, cfg.lambda_l1)
            total_loss.backward()
            optimizer.step()
            train_total_loss += total_loss.item() * xb.size(0)
            train_mse_loss += mse_loss.item() * xb.size(0)
            train_l1_loss += l1_loss.item() * xb.size(0)
        
        train_total_loss /= len(train_ds)
        train_mse_loss /= len(train_ds)
        train_l1_loss /= len(train_ds)

        # 记录门控参数
        gate_values = model.gate.detach().cpu().numpy()
        gate_history.append(gate_values.copy())
        
        # 计算活跃特征数量（绝对值大于阈值）
        active_mask = np.abs(gate_values) > ACTIVE_FEATURE_THRESHOLD
        active_count = np.sum(active_mask)
        active_features_history.append(active_count)
        
        # 记录特定 epoch 的活跃特征
        if epoch in key_epochs:
            active_indices = np.where(active_mask)[0].tolist()
            active_names = [related[i] for i in active_indices]
            keydata_records[epoch] = {
                "active_indices": active_indices,
                "active_names": active_names,
                "count": int(active_count)
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
                # 保存训练集标准化参数，便于推理时做同样的变换 / 反变换
                "x_mean": x_mean.squeeze(),
                "x_std": x_std.squeeze(),
                "y_mean": float(y_mean),
                "y_std": float(y_std),
            }

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_eval_loss:.4f} train_r2={train_r2:.4f} | "
            f"test_loss={test_loss:.4f} test_r2={test_r2:.4f} | "
            f"active_features={active_count}/{len(related)}"
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
    plt.plot(log_df["epoch"], log_df["train_r2"], label=r"Train $R^2$")
    plt.plot(log_df["epoch"], log_df["test_r2"], label=r"Test $R^2$")
    plt.xlabel("Epoch")
    plt.ylabel(r"$R^2$")
    plt.title(r"Train/Test $R^2$")
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
    plt.xlabel("Epoch")
    plt.ylabel("Gate Parameter Value")
    plt.title("Gate Parameters Over Epochs")
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

    # 保存 keydata 记录（JSON 格式）
    if keydata_records:
        with open(keydata_path, 'w', encoding='utf-8') as f:
            json.dump(keydata_records, f, ensure_ascii=False, indent=2)
        print(f"Keydata 记录已保存: {keydata_path}")

    # 保存最佳模型
    if best_state is not None:
        torch.save(best_state, output_model_path)
        print(f"最佳模型已保存: {output_model_path} (test_r2={best_r2:.4f})")
        print(f"日志已保存: {log_path}")
        print(f"可视化图表已保存: {loss_fig_path}, {r2_fig_path}, {gate_fig_path}, {active_features_fig_path}")
        print(f"名称映射表已保存: {name_mapping_path}")


if __name__ == "__main__":
    # 在这里改参数（输出路径会自动生成：L1GateModelOutput/BasedOnData{i}/CenteredOn{中心变量}(epochs={},lambda={})/）
    CONFIG = TrainConfig(
        center_rel_path="CenterDataRelationships/center_realdata4_地区节点阻塞价格.csv",
        data_path="RealData/data4.csv",
        batch_size=50,
        epochs=200,
        lr=1e-3,
        lambda_l1=2e-3,  # L1 正则化系数 total_loss = mse + lambda_l1 * torch.sum(|g|)
        train_ratio=0.8,
        random_state=42,
    )

    train_model(CONFIG)
