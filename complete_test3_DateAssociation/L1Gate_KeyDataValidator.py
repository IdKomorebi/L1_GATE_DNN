"""
L1 门控模型的关键特征验证工具。

使用方式（直接在本文件底部改参数后运行）：
    python L1Gate_KeyDataValidator.py
"""

import os
import json
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Regressor(nn.Module):
    """全连接回归网络（用于全量模型）。"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedRegressor(nn.Module):
    """带 L1 门控的回归网络（用于 L1 模型）。"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(in_dim))
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
        gated_x = x * self.gate
        return self.net(gated_x)


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
            name = " | ".join(dict.fromkeys(values))
        else:
            name = f"column_{col_idx + 1}"

        original = name
        suffix = 1
        while name in columns:
            suffix += 1
            name = f"{original}_{suffix}"
        columns.append(name)

    return columns


def _prepare_test_data(
    data_path: str,
    center: str,
    related: List[str],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: float,
    y_std: float,
    random_state: int,
    train_ratio: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    读取测试数据并标准化。
    """
    # 兼容中文数据文件编码
    try:
        raw_df = pd.read_csv(data_path, header=None, dtype=str, keep_default_na=False, encoding="utf-8")
    except UnicodeDecodeError:
        raw_df = pd.read_csv(data_path, header=None, dtype=str, keep_default_na=False, encoding="gb18030")
    
    if raw_df.empty:
        raise ValueError("数据文件为空")

    data_start = _detect_data_start(raw_df)
    column_names = _compose_column_names(raw_df, data_start)

    numeric_df = raw_df.iloc[data_start:].replace("", np.nan)
    numeric_df.columns = column_names
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
    numeric_df.dropna(axis=0, how="all", inplace=True)
    numeric_df.dropna(axis=1, how="all", inplace=True)

    if numeric_df.empty:
        raise ValueError("数据经过清洗后为空")

    cols_needed = [center] + related
    missing = [c for c in cols_needed if c not in numeric_df.columns]
    if missing:
        raise ValueError(f"数据文件缺少列: {missing}")

    sub = numeric_df[cols_needed].dropna()
    if sub.empty:
        raise ValueError("数据经过清洗后为空")

    X_np = sub[related].to_numpy(dtype=np.float32)
    y_np = sub[center].to_numpy(dtype=np.float32).reshape(-1, 1)

    # 划分测试集（使用相同的 random_state 和 train_ratio）
    n = X_np.shape[0]
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    train_size = int(n * train_ratio)
    test_idx = perm[train_size:]

    X_test = X_np[test_idx]
    y_test = y_np[test_idx]

    # 标准化（使用训练集的统计量）
    X_test = (X_test - x_mean) / x_std
    y_test = (y_test - y_mean) / y_std

    return torch.from_numpy(X_test), torch.from_numpy(y_test)


def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    """评估模型，返回 R² 分数。"""
    model.eval()
    with torch.no_grad():
        pred = model(X_test.to(DEVICE))
        r2 = _r2_score(y_test.to(DEVICE), pred)
    return r2


def validate_keydata(
    full_model_path: str,
    l1_model_path: str,
    keydata_path: str,
    data_path: str,
    random_state: int = 42,
    train_ratio: float = 0.8,
) -> None:
    """
    验证关键特征的有效性。
    
    Args:
        full_model_path: 全量模型路径（ModelOutput下的）
        l1_model_path: L1 门控模型路径（L1GateModelOutput下的）
        keydata_path: keydata JSON 文件路径
        data_path: 数据文件路径
        random_state: 随机种子
        train_ratio: 训练集比例（用于划分测试集）
    """
    # 加载全量模型
    print(f"加载全量模型: {full_model_path}")
    full_state = torch.load(full_model_path, map_location=DEVICE)
    center = full_state["center"]
    related = full_state["related"]
    
    full_model = Regressor(in_dim=len(related)).to(DEVICE)
    full_model.load_state_dict(full_state["model_state"])
    
    # 加载 L1 模型（使用 L1 模型的标准化参数，因为这是最新的训练）
    print(f"加载 L1 模型: {l1_model_path}")
    l1_state = torch.load(l1_model_path, map_location=DEVICE)
    l1_model = GatedRegressor(in_dim=len(related)).to(DEVICE)
    l1_model.load_state_dict(l1_state["model_state"])
    
    # 使用 L1 模型的标准化参数（因为测试集应该用训练时的参数）
    x_mean = l1_state["x_mean"]
    x_std = l1_state["x_std"]
    y_mean = l1_state["y_mean"]
    y_std = l1_state["y_std"]
    
    # 转换为 numpy（如果需要）
    if isinstance(x_mean, torch.Tensor):
        x_mean = x_mean.numpy()
    if isinstance(x_std, torch.Tensor):
        x_std = x_std.numpy()
    
    # 加载 keydata
    print(f"加载 keydata: {keydata_path}")
    with open(keydata_path, 'r', encoding='utf-8') as f:
        keydata_records = json.load(f)
    
    # 准备测试数据
    print("准备测试数据...")
    X_test, y_test = _prepare_test_data(
        data_path, center, related, x_mean, x_std, y_mean, y_std, random_state, train_ratio
    )
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=100, shuffle=False)
    
    # 评估结果
    results = []
    
    # 1. 全量模型评估
    print("评估 1/6: 全量模型（全特征）...")
    r2_full = evaluate_model(full_model, X_test, y_test)
    results.append({"name": "Full Model", "r2": r2_full, "epoch": None})
    print(f"  R² = {r2_full:.4f}")
    
    # 2-6. 使用 keydata 评估（其他特征设为0）
    key_epochs = sorted([int(k) for k in keydata_records.keys()])
    for epoch in key_epochs:
        keydata = keydata_records[str(epoch)]
        active_indices = keydata["active_indices"]
        active_count = keydata["count"]
        
        print(f"评估 {len(results)+1}/6: Keydata from epoch {epoch} ({active_count} features)...")
        
        # 创建掩码：只保留活跃特征
        mask = torch.zeros(len(related), dtype=torch.float32)
        mask[active_indices] = 1.0
        mask = mask.to(DEVICE)
        
        # 评估 L1 模型（应用掩码）
        l1_model.eval()
        with torch.no_grad():
            X_masked = X_test.to(DEVICE) * mask.unsqueeze(0)
            pred = l1_model(X_masked)
            r2 = _r2_score(y_test.to(DEVICE), pred)
        
        results.append({
            "name": f"Epoch {epoch} ({active_count} features)",
            "r2": r2,
            "epoch": epoch
        })
        print(f"  R² = {r2:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    output_dir = os.path.dirname(l1_model_path)
    results_path = os.path.join(output_dir, "keydata_validation_results.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {results_path}")
    
    # 绘制 R² 直方图
    fig_path = os.path.join(output_dir, "keydata_validation_r2.png")
    plt.figure(figsize=(10, 6))
    names = [r["name"] for r in results]
    r2_values = [r["r2"] for r in results]
    
    bars = plt.bar(range(len(results)), r2_values, alpha=0.7)
    plt.xlabel("Model Configuration")
    plt.ylabel("R^2 Score")
    plt.title("R^2 Comparison: Full Model vs Keydata Models")
    plt.xticks(range(len(results)), names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数值
    for i, (bar, r2) in enumerate(zip(bars, r2_values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化图表已保存: {fig_path}")


if __name__ == "__main__":
    # =========================
    # 在这里改参数
    # =========================
    FULL_MODEL_PATH = "ModelOutput/BasedOnData4/CenteredOn电压节点消耗功率(epochs=200)/model.pth"
    L1_MODEL_PATH = "L1GateModelOutput/BasedOnData4/CenteredOn电压节点消耗功率(epochs=200,lambda=3e-03)/model.pth"
    KEYDATA_PATH = "L1GateModelOutput/BasedOnData4/CenteredOn电压节点消耗功率(epochs=200,lambda=3e-03)/keydata_for_pointdata.json"
    DATA_PATH = "RealData/data4.csv"
    RANDOM_STATE = 42
    TRAIN_RATIO = 0.8
    
    validate_keydata(
        full_model_path=FULL_MODEL_PATH,
        l1_model_path=L1_MODEL_PATH,
        keydata_path=KEYDATA_PATH,
        data_path=DATA_PATH,
        random_state=RANDOM_STATE,
        train_ratio=TRAIN_RATIO,
    )
