import importlib
import itertools
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr, entropy
from sklearn.metrics.pairwise import rbf_kernel

DCOR_AVAILABLE = importlib.util.find_spec("dcor") is not None
dcor = importlib.import_module("dcor") if DCOR_AVAILABLE else None


class RealDataCorrelationAnalyzer:
    """面向真实CSV数据的通用关联分析器。"""

    def __init__(self, realdata_dir: str = "realdata") -> None:
        self.realdata_dir = "RealData\data4.csv"
        self.relationships: List[Dict[str, float]] = []

    def analyze_pair(self, series_a: pd.Series, series_b: pd.Series, col_a: str, col_b: str) -> Optional[Dict[str, float]]:
        """计算两个字段之间的相关系数（NMI_min、Spearman、Pearson、Kendall、dCor、HSIC）。"""
        df_pair = pd.DataFrame({col_a: series_a, col_b: series_b}).dropna()
        if len(df_pair) < 2:
            return None

        x = df_pair[col_a].astype(float).to_numpy()
        y = df_pair[col_b].astype(float).to_numpy()

        try:
            nmi = self._normalized_mutual_info(x, y)
        except Exception:
            nmi = np.nan

        try:
            spearman_corr = spearmanr(x, y).statistic
        except Exception:
            spearman_corr = np.nan

        try:
            pearson_corr = pearsonr(x, y).statistic
        except Exception:
            pearson_corr = np.nan

        try:
            kendall_corr = kendalltau(x, y).statistic
        except Exception:
            kendall_corr = np.nan

        try:
            if DCOR_AVAILABLE and dcor is not None:
                dcor_value = dcor.distance_correlation(x, y)
            else:
                dcor_value = self._fallback_dcor(x, y)
        except Exception:
            dcor_value = np.nan

        try:
            hsic_value = self._hsic_normalized(x, y)
        except Exception:
            hsic_value = np.nan

        return {
            "column_a": col_a,
            "column_b": col_b,
            "nmi": nmi,
            "spearman": spearman_corr,
            "pearson": pearson_corr,
            "kendall": kendall_corr,
            "distance_corr": dcor_value,
            "hsic": hsic_value,
        }

    def analyze_csv(self, csv_path: str) -> None:
        """遍历单个CSV文件中的所有列组合，累积关联结果。"""
        numeric_df = self._prepare_numeric_dataframe(csv_path)
        if numeric_df is None or numeric_df.shape[1] < 2:
            return

        for col_a, col_b in itertools.combinations(numeric_df.columns, 2):
            result = self.analyze_pair(numeric_df[col_a], numeric_df[col_b], col_a, col_b)
            if result:
                self.relationships.append(result)

    def analyze_path(self, path: Optional[str] = None) -> None:
        """分析单个文件或整个目录。"""
        target = path or self.realdata_dir
        if not target:
            raise ValueError("必须提供CSV文件路径或realdata目录。")

        if os.path.isdir(target):
            for name in sorted(os.listdir(target)):
                if name.lower().endswith(".csv"):
                    self.analyze_csv(os.path.join(target, name))
        elif os.path.isfile(target) and target.lower().endswith(".csv"):
            self.analyze_csv(target)
        else:
            raise FileNotFoundError(f"无法找到有效的CSV路径: {target}")

    def export_relationships(self, output_csv: str) -> str:
        """将累积的关系写入CSV，包含索引列。"""
        if not self.relationships:
            raise ValueError("没有可导出的关联数据，请先运行分析。")

        df = pd.DataFrame(self.relationships)
        df.insert(0, "index", range(1, len(df) + 1))
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return output_csv

    def _prepare_numeric_dataframe(self, csv_path: str) -> Optional[pd.DataFrame]:
        """加载CSV，检测数据起始行，并返回纯数值DataFrame。

        优先尝试按 UTF-8 读取，如果失败则回退到常见的中文编码（gb18030）。
        """
        try:
            raw_df = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False, encoding="utf-8")
        except UnicodeDecodeError:
            raw_df = pd.read_csv(csv_path, header=None, dtype=str, keep_default_na=False, encoding="gb18030")
        if raw_df.empty:
            return None

        data_start = self._detect_data_start(raw_df)
        column_names = self._compose_column_names(raw_df, data_start)

        numeric_df = raw_df.iloc[data_start:].replace("", np.nan)
        numeric_df.columns = column_names
        numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
        numeric_df.dropna(axis=0, how="all", inplace=True)
        numeric_df.dropna(axis=1, how="all", inplace=True)
        return numeric_df if not numeric_df.empty else None

    def _detect_data_start(self, raw_df: pd.DataFrame) -> int:
        """检测从哪一行开始出现数值数据。"""
        for idx in range(len(raw_df)):
            row = raw_df.iloc[idx]
            numeric_count = sum(self._is_numeric(value) for value in row)
            if numeric_count >= max(2, len(row) * 0.5):
                return idx
        return min(2, len(raw_df))

    def _compose_column_names(self, raw_df: pd.DataFrame, data_start: int) -> List[str]:
        """根据数据起始行之前的内容构造列名。"""
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

    @staticmethod
    def _is_numeric(value: str) -> bool:
        """判断字符串是否可转换为浮点数。"""
        try:
            float(str(value).replace(",", ""))
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _normalized_mutual_info(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
        """
        基于同一离散空间的归一化互信息：
            NMI_min(X, Y) = I(X; Y) / min(H(X), H(Y))
        - X、Y 先做等频离散化（quantile binning）
        - I、H 均使用离散香农熵（nats）
        - 结果严格在 [0, 1] 内（数值误差除外）
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        if x.shape[0] != y.shape[0] or x.shape[0] < 2:
            return np.nan

        def _discretize_quantile(arr: np.ndarray, bins: int) -> np.ndarray:
            """等频离散化，将连续变量映射为 {0, 1, ..., k-1}。"""
            arr = np.asarray(arr).ravel()
            finite_mask = np.isfinite(arr)
            arr_valid = arr[finite_mask]

            if arr_valid.size < 2 or np.all(arr_valid == arr_valid[0]):
                return np.zeros_like(arr, dtype=int)

            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = np.quantile(arr_valid, quantiles)
            bin_edges = np.unique(bin_edges)  # 去重避免空箱
            if bin_edges.size <= 2:
                return np.zeros_like(arr, dtype=int)

            discretized = np.zeros_like(arr, dtype=int)
            discretized[finite_mask] = np.digitize(arr_valid, bin_edges[1:-1], right=False)
            return discretized

        # 离散化
        xd = _discretize_quantile(x, n_bins)
        yd = _discretize_quantile(y, n_bins)

        kx = int(np.max(xd)) + 1
        ky = int(np.max(yd)) + 1
        if kx <= 1 or ky <= 1:
            return 0.0

        # 联合分布
        joint_xy, _, _ = np.histogram2d(xd, yd, bins=(kx, ky), range=[[0, kx], [0, ky]])
        total = joint_xy.sum()
        if total <= 0:
            return 0.0
        joint_xy = joint_xy / total

        px = joint_xy.sum(axis=1)
        py = joint_xy.sum(axis=0)

        px_nz = px[px > 0]
        py_nz = py[py > 0]
        joint_nz = joint_xy[joint_xy > 0]

        hx = entropy(px_nz) if px_nz.size else 0.0
        hy = entropy(py_nz) if py_nz.size else 0.0
        hxy = entropy(joint_nz) if joint_nz.size else 0.0

        mi = hx + hy - hxy
        h_min = min(hx, hy)

        if h_min <= 0:
            return 0.0

        nmi = mi / h_min
        # 轻微的数值漂移压回区间
        if nmi < 0:
            nmi = 0.0
        if nmi > 1 and nmi < 1 + 1e-6:
            nmi = 1.0

        return float(nmi)

    @staticmethod
    def _hsic_normalized(
        x: np.ndarray,
        y: np.ndarray,
        max_samples: int = 800,
        sigma: Optional[float] = None,
        random_state: int = 42,
    ) -> float:
        """
        计算归一化 HSIC（采用 RBF 核）。

        为避免核矩阵在大样本下造成 O(n^3) 的计算开销，这里采用子采样：
        当 n > max_samples 时随机抽取 max_samples 个样本后再计算。
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        n = x.shape[0]
        if n < 5 or y.shape[0] != n:
            return np.nan

        # 子采样
        if n > max_samples:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, size=max_samples, replace=False)
            x = x[idx]
            y = y[idx]
            n = max_samples

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        if sigma is None:
            sigma_x = np.median(np.abs(x - np.median(x))) + 1e-8
            sigma_y = np.median(np.abs(y - np.median(y))) + 1e-8
        else:
            sigma_x = sigma_y = sigma

        if sigma_x <= 0 or sigma_y <= 0:
            return np.nan

        Kx = rbf_kernel(x, gamma=1.0 / (2.0 * sigma_x**2))
        Ky = rbf_kernel(y, gamma=1.0 / (2.0 * sigma_y**2))

        H = np.eye(n) - np.ones((n, n)) / n
        Kc = H @ Kx @ H
        Lc = H @ Ky @ H

        hsic = np.trace(Kc @ Lc)
        hsic_xx = np.trace(Kc @ Kc)
        hsic_yy = np.trace(Lc @ Lc)

        # 加入稳定项避免除零
        return float(hsic / np.sqrt(hsic_xx * hsic_yy + 1e-12))

    @staticmethod
    def _fallback_dcor(x: np.ndarray, y: np.ndarray) -> float:
        """dcor库缺失时的距离相关替代实现。"""
        from scipy.spatial.distance import pdist, squareform

        x_dist = squareform(pdist(x.reshape(-1, 1)))
        y_dist = squareform(pdist(y.reshape(-1, 1)))

        def _center(dist_matrix: np.ndarray) -> np.ndarray:
            mean_row = dist_matrix.mean(axis=1, keepdims=True)
            mean_col = dist_matrix.mean(axis=0, keepdims=True)
            mean_total = dist_matrix.mean()
            return dist_matrix - mean_row - mean_col + mean_total

        ax = _center(x_dist)
        ay = _center(y_dist)

        dcov_xy = np.sqrt(np.mean(ax * ay))
        dcov_xx = np.sqrt(np.mean(ax * ax))
        dcov_yy = np.sqrt(np.mean(ay * ay))

        if dcov_xx == 0 or dcov_yy == 0:
            return 0.0
        return dcov_xy / np.sqrt(dcov_xx * dcov_yy)


if __name__ == "__main__":
    analyzer = RealDataCorrelationAnalyzer()
    analyzer.analyze_path()
    output = analyzer.export_relationships("DataAnalyze/realdata4_relationships.csv")
    print(f"关联结果已导出: {output}")
