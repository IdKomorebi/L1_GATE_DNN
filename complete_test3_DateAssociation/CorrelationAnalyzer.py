import pandas as pd
import numpy as np
import os
import re
from scipy.stats import spearmanr, pearsonr, kendalltau, entropy
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 尝试导入dcor库，如果没有则使用备用实现
try:
    import dcor
    DCOR_AVAILABLE = True
except ImportError:
    DCOR_AVAILABLE = False
    print("警告: dcor库未安装，Distance Correlation将使用备用实现")

class CorrelationAnalyzer:
    """
    关联度分析器类
    
    功能：
    - 计算MI、Spearman、Pearson、Kendall、dCor五种相关系数
    - 分析y{i}与所有x{j}的关联度
    - 保存结果为CSV文件
    """
    
    def __init__(self):
        """
        初始化分析器
        """
        self.results = {}
        
    def calculate_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算归一化互信息 NMI_min
        
        使用公式：
            NMI_min(X, Y) = I(X; Y) / min(H(X), H(Y))
        其中 I(X; Y) 使用 mutual_info_regression 估计，
        H(X)、H(Y) 通过直方图估计熵。
        
        Args:
            x: 自变量数组
            y: 因变量数组
            
        Returns:
            float: 归一化互信息 NMI_min，范围约为 [0, 1]
        """
        try:
            # 确保是一维向量
            x = np.asarray(x).ravel()
            y = np.asarray(y).ravel()

            # 如果长度不一致或样本过少，直接返回 NaN
            if x.shape[0] != y.shape[0] or x.shape[0] < 2:
                return np.nan

            # 使用 sklearn 的 mutual_info_regression 计算互信息 I(X; Y)
            # 需要将 x 重新整形为 sklearn 期望的格式 (n_samples, n_features)
            x_reshaped = x.reshape(-1, 1)
            mi_score = mutual_info_regression(x_reshaped, y, random_state=42)[0]

            # 使用直方图估计 H(X) 和 H(Y)
            # 不设置 base，默认使用自然对数，与 mutual_info_regression 的单位一致（nats）
            def _estimate_entropy(arr: np.ndarray, n_bins: int = 20) -> float:
                # 去除 NaN
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return 0.0

                # 如果所有值相同，熵为 0
                if np.all(arr == arr[0]):
                    return 0.0

                hist, _ = np.histogram(arr, bins=n_bins, density=True)
                # 去掉为 0 的概率，以避免 log(0)
                hist = hist[hist > 0]
                if hist.size == 0:
                    return 0.0
                return entropy(hist)

            hx = _estimate_entropy(x)
            hy = _estimate_entropy(y)

            h_min = min(hx, hy)

            # 如果最小熵为 0，则无法进行归一化，此时互信息应为 0
            if h_min <= 0 or not np.isfinite(mi_score):
                return 0.0

            nmi_min = mi_score / h_min

            # 数值上可能略超出 [0,1]，这里做一下截断
            nmi_min = max(0.0, min(1.0, float(nmi_min)))

            return nmi_min
        except Exception as e:
            print(f"MI计算错误: {e}")
            return np.nan
    
    def calculate_spearman(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        计算Spearman相关系数
        
        Args:
            x: 自变量数组
            y: 因变量数组
            
        Returns:
            Tuple[float, float]: (相关系数, p值)
        """
        try:
            correlation, p_value = spearmanr(x, y)
            return correlation, p_value
        except Exception as e:
            print(f"Spearman计算错误: {e}")
            return np.nan, np.nan
    
    def calculate_pearson(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        计算Pearson相关系数
        
        Args:
            x: 自变量数组
            y: 因变量数组
            
        Returns:
            Tuple[float, float]: (相关系数, p值)
        """
        try:
            correlation, p_value = pearsonr(x, y)
            return correlation, p_value
        except Exception as e:
            print(f"Pearson计算错误: {e}")
            return np.nan, np.nan
    
    def calculate_kendall(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        计算Kendall τ相关系数
        
        Args:
            x: 自变量数组
            y: 因变量数组
            
        Returns:
            Tuple[float, float]: (相关系数, p值)
        """
        try:
            correlation, p_value = kendalltau(x, y)
            return correlation, p_value
        except Exception as e:
            print(f"Kendall计算错误: {e}")
            return np.nan, np.nan
    
    def calculate_dcor(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算Distance Correlation (dCor)
        
        Args:
            x: 自变量数组
            y: 因变量数组
            
        Returns:
            float: dCor值
        """
        try:
            if DCOR_AVAILABLE:
                # 使用dcor库
                dcor_value = dcor.distance_correlation(x, y)
                return dcor_value
            else:
                # 备用实现：使用简化的距离相关性
                from scipy.spatial.distance import pdist, squareform
                # 计算距离矩阵
                x_matrix = x.reshape(-1, 1)
                y_matrix = y.reshape(-1, 1)
                
                # 计算欧氏距离
                x_dist = squareform(pdist(x_matrix))
                y_dist = squareform(pdist(y_matrix))
                
                # 中心化距离矩阵
                n = len(x)
                x_centered = x_dist - np.mean(x_dist, axis=1, keepdims=True) - np.mean(x_dist, axis=0, keepdims=True) + np.mean(x_dist)
                y_centered = y_dist - np.mean(y_dist, axis=1, keepdims=True) - np.mean(y_dist, axis=0, keepdims=True) + np.mean(y_dist)
                
                # 计算dCor
                dcov_xy = np.sqrt(np.mean(x_centered * y_centered))
                dcov_xx = np.sqrt(np.mean(x_centered ** 2))
                dcov_yy = np.sqrt(np.mean(y_centered ** 2))
                
                if dcov_xx * dcov_yy > 0:
                    dcor_value = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
                else:
                    dcor_value = 0.0
                    
                return dcor_value
        except Exception as e:
            print(f"dCor计算错误: {e}")
            return np.nan
    
    def analyze_correlations(self, csv_file_path: str) -> Dict[str, pd.DataFrame]:
        """
        分析CSV文件中的关联度
        
        Args:
            csv_file_path: CSV文件路径
            
        Returns:
            Dict[str, pd.DataFrame]: 包含三种相关系数结果的字典
        """
        # 读取CSV文件
        try:
            df = pd.read_csv(csv_file_path)
            print(f"成功读取文件: {csv_file_path}")
            print(f"数据形状: {df.shape}")
        except Exception as e:
            print(f"读取文件错误: {e}")
            return {}
        
        # 分离特征和目标
        feature_cols = [col for col in df.columns if col.startswith('x')]
        target_cols = [col for col in df.columns if col.startswith('y')]
        
        print(f"特征列: {feature_cols}")
        print(f"目标列: {target_cols}")
        
        # 初始化结果矩阵
        n_targets = len(target_cols)
        n_features = len(feature_cols)
        
        # 创建结果DataFrame
        mi_results = pd.DataFrame(index=target_cols, columns=feature_cols)
        spearman_results = pd.DataFrame(index=target_cols, columns=feature_cols)
        pearson_results = pd.DataFrame(index=target_cols, columns=feature_cols)
        kendall_results = pd.DataFrame(index=target_cols, columns=feature_cols)
        dcor_results = pd.DataFrame(index=target_cols, columns=feature_cols)
        
        # 计算所有y与所有x的关联度
        print("开始计算关联度...")
        for i, target_col in enumerate(target_cols):
            print(f"处理目标变量: {target_col} ({i+1}/{n_targets})")
            
            for j, feature_col in enumerate(feature_cols):
                # 获取数据
                x_data = df[feature_col].values
                y_data = df[target_col].values
                
                # 计算MI (互信息)
                mi_value = self.calculate_mi(x_data, y_data)
                mi_results.loc[target_col, feature_col] = mi_value
                
                # 计算Spearman
                spearman_corr, spearman_p = self.calculate_spearman(x_data, y_data)
                spearman_results.loc[target_col, feature_col] = spearman_corr
                
                # 计算Pearson
                pearson_corr, pearson_p = self.calculate_pearson(x_data, y_data)
                pearson_results.loc[target_col, feature_col] = pearson_corr
                
                # 计算Kendall
                kendall_corr, kendall_p = self.calculate_kendall(x_data, y_data)
                kendall_results.loc[target_col, feature_col] = kendall_corr
                
                # 计算dCor
                dcor_value = self.calculate_dcor(x_data, y_data)
                dcor_results.loc[target_col, feature_col] = dcor_value
        
        # 转换数据类型
        mi_results = mi_results.astype(float)
        spearman_results = spearman_results.astype(float)
        pearson_results = pearson_results.astype(float)
        kendall_results = kendall_results.astype(float)
        dcor_results = dcor_results.astype(float)
        
        # 添加yi列
        mi_results.insert(0, 'yi', target_cols)
        spearman_results.insert(0, 'yi', target_cols)
        pearson_results.insert(0, 'yi', target_cols)
        kendall_results.insert(0, 'yi', target_cols)
        dcor_results.insert(0, 'yi', target_cols)
        
        return {
            'MI': mi_results,
            'Spearman': spearman_results,
            'Pearson': pearson_results,
            'Kendall': kendall_results,
            'dCor': dcor_results
        }
    
    def save_results(self, results: Dict[str, pd.DataFrame], source_filename: str, 
                     output_dir: str = "DataAnalyze") -> str:
        """
        保存分析结果到CSV文件
        
        Args:
            results: 分析结果字典
            source_filename: 源文件名（不含扩展名）
            output_dir: 输出目录
            
        Returns:
            str: 结果文件夹路径
        """
        # 创建结果文件夹
        result_folder = f"AnalyzeResult_{source_filename}"
        result_path = os.path.join(output_dir, result_folder)
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            print(f"创建结果文件夹: {result_path}")
        
        # 保存三种相关系数结果
        for corr_type, df in results.items():
            filename = f"{corr_type.lower()}_correlation.csv"
            filepath = os.path.join(result_path, filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"保存{corr_type}结果到: {filepath}")
        
        return result_path
    
    def analyze_folder(self, folder_path: str, output_dir: str = "DataAnalyze") -> str:
        """
        分析整个文件夹中的CSV文件并生成汇总结果
        
        Args:
            folder_path: 包含CSV文件的文件夹路径
            output_dir: 输出目录
            
        Returns:
            str: 结果文件路径
        """
        print(f"开始分析文件夹: {folder_path}")
        
        # 获取文件夹中的所有CSV文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        # 自定义排序：MaxCorrelationXY1, XY2, ..., XY10
        def sort_key(filename):
            # 提取MaxCorrelationXY后面的数字
            match = re.search(r'MaxCorrelationXY(\d+)', filename)
            if match:
                return int(match.group(1))
            return 999
        csv_files.sort(key=sort_key)  # 按数字排序
        
        if not csv_files:
            print("文件夹中没有找到CSV文件")
            return None
        
        print(f"找到{len(csv_files)}个CSV文件: {csv_files}")
        
        # 存储所有分析结果
        all_results = []
        
        # 函数名称映射
        func_names = {
            1: "Linear", 2: "Sine", 3: "Quadratic", 4: "Exponential", 5: "Logarithmic",
            6: "Trigonometric", 7: "Polynomial", 8: "Piecewise", 9: "Composite", 10: "Complex_Nonlinear"
        }
        
        for i, csv_file in enumerate(csv_files):
            csv_path = os.path.join(folder_path, csv_file)
            print(f"分析文件 {i+1}/{len(csv_files)}: {csv_file}")
            
            # 从CSV第一行读取weight列
            weight_str = self._read_weight_from_csv(csv_path)
            
            # 分析单个文件
            results = self.analyze_correlations(csv_path)
            
            if results:
                # 为每个结果添加文件标识
                for corr_type, df in results.items():
                    # 移除yi列，因为我们要重新组织数据
                    df_without_yi = df.drop('yi', axis=1)
                    
                    for j, (_, row) in enumerate(df_without_yi.iterrows()):
                        # 创建汇总行
                        summary_row = {
                            'index': f'csv{i+1}',
                            'method': corr_type.ljust(8),  # 确保method列对齐
                            'weight': weight_str,  # 从CSV读取的权重
                            'y': f'y{j+1}',
                            'func(默认)': func_names[j+1].ljust(18)  # 确保func列对齐
                        }
                        
                        # 添加x1到x10的相关系数
                        for k in range(10):
                            summary_row[f'x{k+1}'] = f"{row.iloc[k]:.8f}"
                        
                        all_results.append(summary_row)
        
        # 创建汇总DataFrame
        if all_results:
            summary_df = pd.DataFrame(all_results)
            
            # 保存汇总结果
            output_filename = f"correlation_summary_{os.path.basename(folder_path)}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"汇总结果已保存到: {output_path}")
            
            return output_path
        else:
            print("没有生成任何分析结果")
            return None
    
    def _read_weight_from_csv(self, csv_file_path: str) -> str:
        """
        从CSV文件的weight列读取权重向量
        
        Args:
            csv_file_path: CSV文件路径
            
        Returns:
            str: 权重向量的字符串表示
        """
        try:
            # 只读取第一行以获取weight列的值
            df_first_row = pd.read_csv(csv_file_path, nrows=1)
            
            if 'weight' in df_first_row.columns:
                weight_str = df_first_row['weight'].iloc[0]
                return str(weight_str)
            else:
                print(f"警告: CSV文件 {csv_file_path} 中没有weight列，使用默认权重")
                return "[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]"
        except Exception as e:
            print(f"读取weight列失败: {e}，使用默认权重")
            return "[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]"
    
    def analyze_file(self, csv_file_path: str, output_dir: str = "DataAnalyze") -> str:
        """
        分析单个CSV文件并保存结果
        
        Args:
            csv_file_path: CSV文件路径
            output_dir: 输出目录
            
        Returns:
            str: 结果文件夹路径
        """
        # 获取源文件名（不含扩展名）
        source_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        print(f"开始分析文件: {csv_file_path}")
        print(f"源文件名: {source_filename}")
        
        # 分析关联度
        results = self.analyze_correlations(csv_file_path)
        
        if not results:
            print("分析失败，没有生成结果")
            return None
        
        # 保存结果
        result_path = self.save_results(results, source_filename, output_dir)
        
        # 显示结果摘要
        self._print_summary(results, source_filename)
        
        return result_path
    
    def _print_summary(self, results: Dict[str, pd.DataFrame], source_filename: str):
        """
        打印结果摘要
        
        Args:
            results: 分析结果
            source_filename: 源文件名
        """
        print(f"\n{'='*60}")
        print(f"分析结果摘要 - {source_filename}")
        print(f"{'='*60}")
        
        for corr_type, df in results.items():
            print(f"\n{corr_type} 相关系数:")
            print(f"数据形状: {df.shape}")
            
            # 显示前几行
            print("前5行数据:")
            print(df.head())
            
            # 显示统计信息
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                print(f"\n{corr_type} 统计信息:")
                print(f"平均值: {numeric_df.mean().mean():.4f}")
                print(f"最大值: {numeric_df.max().max():.4f}")
                print(f"最小值: {numeric_df.min().min():.4f}")
        
        print(f"\n{'='*60}")


# 使用示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = CorrelationAnalyzer()
    
    # 分析RawSample1文件夹中的所有CSV文件
    raw_sample_folder = "DataGenerate/RawSample1"
    if os.path.exists(raw_sample_folder):
        print("分析RawSample1文件夹...")
        result_path = analyzer.analyze_folder(raw_sample_folder)
        if result_path:
            print(f"汇总分析结果已保存到: {result_path}")
    else:
        print(f"文件夹不存在: {raw_sample_folder}")
    
    print("\n分析完成！")

