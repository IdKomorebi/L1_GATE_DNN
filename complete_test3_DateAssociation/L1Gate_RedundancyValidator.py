"""
L1 门控模型的“去冗余”验证工具。

功能概述：
- 从 keydata_for_pointdata.json 中读取不同 epoch 的关键特征集合；
- 使用原始数据 DATA_PATH，针对每一组 keydata 重新训练一个普通全连接回归模型；
- 对最后一个 epoch 的 keydata，再额外训练 N 个“去掉其中一个特征”的模型，用于检查冗余；
- 将所有模型的 R² 结果汇总到同一目录下，并绘制对比柱状图。

使用方式（直接在本文件底部改参数后运行）：
    python L1Gate_RedundancyValidator.py
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


class Regressor(nn.Module):
    """全连接回归网络（与 CenterDataPredictor 中一致）。"""

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


def _prepare_dataset(
    data_path: str,
    center: str,
    features: List[str],
    random_state: int,
    train_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """为给定特征集合准备训练/测试集（含标准化）。"""
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

    cols_needed = [center] + features
    missing = [c for c in cols_needed if c not in numeric_df.columns]
    if missing:
        raise ValueError(f"数据文件缺少列: {missing}")

    sub = numeric_df[cols_needed].dropna()
    if sub.empty:
        raise ValueError("数据经过清洗后为空")

    X_np = sub[features].to_numpy(dtype=np.float32)
    y_np = sub[center].to_numpy(dtype=np.float32).reshape(-1, 1)

    # 划分训练/测试
    n = X_np.shape[0]
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    train_size = int(n * train_ratio)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    X_train = X_np[train_idx]
    y_train = y_np[train_idx]
    X_test = X_np[test_idx]
    y_test = y_np[test_idx]

    # 标准化（基于训练集）
    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True) + 1e-8
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-8)

    X_train = (X_train - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    return (
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )


def _extract_data_number(data_path: str) -> str:
    """从数据路径提取编号，例如 'RealData/data4.csv' -> '4'"""
    basename = os.path.basename(data_path)
    match = re.search(r"data(\d+)", basename, re.IGNORECASE)
    if match:
        return match.group(1)
    return "unknown"


def _build_validation_root(keydata_path: str) -> str:
    """
    根据 keydata 路径构造验证输出根目录：
    L1GateRedundancyOutput/BasedOnData{i}/CenteredOn...
    """
    base_dir = os.path.dirname(os.path.normpath(keydata_path))
    parts = base_dir.split(os.sep)

    based_on = None
    centered_on = None

    for p in parts:
        if re.match(r"BasedOnData\d+", p):
            based_on = p
        elif p.startswith("CenteredOn"):
            centered_on = p

    if based_on is None or centered_on is None:
        raise ValueError(
            f"无法从路径中解析 BasedOnData / CenteredOn 信息: {keydata_path}\n"
            f"解析得到 based_on={based_on}, centered_on={centered_on}"
        )

    root = os.path.join("L1GateRedundancyOutput", based_on, centered_on)
    os.makedirs(root, exist_ok=True)
    return root

def _train_and_evaluate(
    center: str,
    features: List[str],
    data_path: str,
    random_state: int,
    train_ratio: float,
    epochs: int,
    lr: float,
    batch_size: int,
    output_dir: str,
) -> float:
    """针对给定特征集合训练一个模型，并返回测试集 R²。"""
    X_train, y_train, X_test, y_test = _prepare_dataset(
        data_path, center, features, random_state, train_ratio
    )

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = Regressor(in_dim=len(features)).to(DEVICE)
    optimizer = SimpleAdam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    log_rows = []
    best_r2 = -1e9

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # 评估
        model.eval()
        with torch.no_grad():
            ys = []
            ps = []
            total_loss = 0.0
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                total_loss += loss.item() * xb.size(0)
                ys.append(yb.cpu())
                ps.append(pred.cpu())
            total_loss /= len(test_ds)
            y_cat = torch.cat(ys, dim=0)
            p_cat = torch.cat(ps, dim=0)
            r2 = _r2_score(y_cat, p_cat)

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "test_loss": total_loss, "test_r2": r2})
        best_r2 = max(best_r2, r2)

    # 保存日志与模型
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.csv")
    pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    # 绘制训练过程（Loss & R²），仅使用英文标签，避免中文字体问题
    log_df = pd.DataFrame(log_rows)

    # Loss 曲线
    loss_fig_path = os.path.join(output_dir, "loss.png")
    plt.figure(figsize= (7, 4))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    plt.plot(log_df["epoch"], log_df["test_loss"], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train/Test Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_fig_path, dpi=150)
    plt.close()

    # R² 曲线
    r2_fig_path = os.path.join(output_dir, "r2.png")
    plt.figure(figsize=(7, 4))
    plt.plot(log_df["epoch"], log_df["test_r2"], label="test_R2")
    plt.xlabel("Epoch")
    plt.ylabel("R^2")
    plt.title("Test R^2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(r2_fig_path, dpi=150)
    plt.close()

    # 保存模型
    torch.save(
        {
            "model_state": model.state_dict(),
            "center": center,
            "features": features,
        },
        os.path.join(output_dir, "model.pth"),
    )
    # 记录使用的特征
    with open(os.path.join(output_dir, "used_features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)

    return best_r2


@dataclass
class RedundancyConfig:
    keydata_path: str
    data_path: str
    random_state: int = 42
    train_ratio: float = 0.8
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 50


def run_redundancy_validation(cfg: RedundancyConfig) -> None:
    # 加载 keydata
    with open(cfg.keydata_path, "r", encoding="utf-8") as f:
        keydata_records: Dict[str, Dict] = json.load(f)

    # 解析所有 epoch，最多取 5 个（通常是 10,20,50,100,200）
    all_epochs = sorted(int(e) for e in keydata_records.keys())
    if len(all_epochs) > 5:
        epochs_to_use = all_epochs[:5]
    else:
        epochs_to_use = all_epochs

    last_epoch = max(epochs_to_use)

    # 从同目录下的 name_mapping.csv 取中心变量名（index=1）
    mapping_path = os.path.join(os.path.dirname(cfg.keydata_path), "name_mapping.csv")
    mapping_df = pd.read_csv(mapping_path)
    center_row = mapping_df.loc[mapping_df["index"] == 1]
    if center_row.empty:
        raise ValueError("name_mapping.csv 中没有 index=1 的中心变量记录")
    center_name = str(center_row["name"].iloc[0])

    # 构建验证输出根目录
    root_dir = _build_validation_root(cfg.keydata_path)

    results = []
    exp_id = 0

    # 1. 针对每个 epoch 的 keydata 训练一个“全量 keydata”模型
    for epoch in epochs_to_use:
        record = keydata_records[str(epoch)]
        active_names = record.get("active_names", [])
        if not active_names:
            continue
        exp_id += 1
        exp_name = f"E{epoch}_full"
        exp_dir = os.path.join(root_dir, exp_name)
        n_feats = len(active_names)
        print(f"[{exp_name}] 训练使用 {n_feats} 个特征...")
        r2 = _train_and_evaluate(
            center=center_name,
            features=active_names,
            data_path=cfg.data_path,
            random_state=cfg.random_state,
            train_ratio=cfg.train_ratio,
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            output_dir=exp_dir,
        )
        results.append(
            {
                "exp_name": exp_name,
                "epoch_source": epoch,
                "type": "full",
                "dropped": "",
                "n_features": n_feats,
                "r2": r2,
            }
        )

    # 2. 对最后一个 epoch 的 keydata 做“去冗余”实验：全量 + 逐个去掉一个特征
    last_record = keydata_records[str(last_epoch)]
    last_active = last_record.get("active_names", [])
    if last_active:
        # 先确保已经有 last_epoch 的 full 结果（若前面被跳过，则现在补一个）
        if not any(r["exp_name"] == f"E{last_epoch}_full" for r in results):
            exp_id += 1
            exp_name = f"E{last_epoch}_full"
            exp_dir = os.path.join(root_dir, exp_name)
            n_feats_last = len(last_active)
            print(f"[{exp_name}] 训练使用 {n_feats_last} 个特征...")
            r2 = _train_and_evaluate(
                center=center_name,
                features=last_active,
                data_path=cfg.data_path,
                random_state=cfg.random_state,
                train_ratio=cfg.train_ratio,
                epochs=cfg.epochs,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                output_dir=exp_dir,
            )
            results.append(
                {
                    "exp_name": exp_name,
                    "epoch_source": last_epoch,
                    "type": "full",
                    "dropped": "",
                    "n_features": n_feats_last,
                    "r2": r2,
                }
            )

        # 逐个去掉一个特征
        n_last = len(last_active)
        for i in range(n_last):
            dropped = last_active[i]
            kept = [f for j, f in enumerate(last_active) if j != i]
            n_kept = len(kept)
            exp_id += 1
            exp_name = f"E{last_epoch}_drop{i+1}"
            exp_dir = os.path.join(root_dir, exp_name)
            print(f"[{exp_name}] 训练使用 {n_kept} 个特征（去掉: {dropped}）...")
            r2 = _train_and_evaluate(
                center=center_name,
                features=kept,
                data_path=cfg.data_path,
                random_state=cfg.random_state,
                train_ratio=cfg.train_ratio,
                epochs=cfg.epochs,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                output_dir=exp_dir,
            )
            results.append(
                {
                    "exp_name": exp_name,
                    "epoch_source": last_epoch,
                    "type": "drop_one",
                    "dropped": dropped,
                    "n_features": n_kept,
                    "r2": r2,
                }
            )

    # 汇总结果
    results_df = pd.DataFrame(results)
    summary_path = os.path.join(root_dir, "redundancy_validation_results.csv")
    results_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n冗余验证结果已保存: {summary_path}")

    # 绘制 R² 对比柱状图（仅使用英文/数字标签）
    fig_path = os.path.join(root_dir, "redundancy_validation_r2.png")
    plt.figure(figsize=(10, 6))
    # 在标签中加入特征数量信息
    names = [f"{r['exp_name']} (n={r.get('n_features', '')})" for r in results]
    r2_values = [r["r2"] for r in results]
    bars = plt.bar(range(len(results)), r2_values, alpha=0.7)
    plt.xlabel("Experiment")
    plt.ylabel("R^2 Score")
    plt.title("R^2 Comparison Across Experiments")
    plt.xticks(range(len(results)), names, rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")

    for bar, r2 in zip(bars, r2_values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{r2:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"R² 对比图已保存: {fig_path}")


if __name__ == "__main__":
    # 在这里改参数
    CONFIG = RedundancyConfig(
        keydata_path="L1GateModelOutput/BasedOnData4/CenteredOn电压节点消耗功率(epochs=200,lambda=3e-03)/keydata_for_pointdata.json",
        data_path="RealData/data4.csv",
        random_state=42,
        train_ratio=0.8,
        epochs=100,  # 每个验证模型的训练轮数
        lr=1e-3,
        batch_size=50,
    )

    run_redundancy_validation(CONFIG)

