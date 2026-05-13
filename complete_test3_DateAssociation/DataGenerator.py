import numpy as np
import pandas as pd
import os
from typing import Tuple, List

class SyntheticDataGenerator:
    """
    自动生成关联数据的生成器类
    
    功能：
    - 生成x1~x10特征数据（确保特征间有差别）
    - 生成y1~y5线性关系目标值（使用make_regression）
    - 生成y6~y10非线性关系目标值（使用friedman方法）
    - 保存CSV和规则文件
    """
    
    def __init__(self, n_samples: int = 1000, noise: float = 0.1, random_state: int = 42, 
                 weights: np.ndarray = None, flag: bool = True):
        """
        初始化生成器
        
        Args:
            n_samples: 生成样本数量
            noise: 噪声水平
            random_state: 随机种子
            weights: 权重向量，形状为(10,)，每个y与所有x都有关系
            flag: 是否生成10个CSV文件（循环右移权重）
        """
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.flag = flag
        
        # 设置默认权重向量
        if weights is None:
            # 默认权重向量：10个元素，每个y与所有x都有关系
            self.weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        else:
            self.weights = weights
        
    def generate_features(self) -> np.ndarray:
        """
        生成x1~x10特征数据，确保特征间有差别
        
        Returns:
            np.ndarray: 形状为(n_samples, 10)的特征矩阵
        """
        np.random.seed(self.random_state)
        
        # 生成10个不同的特征，每个特征有不同的分布和范围
        features = np.zeros((self.n_samples, 10))
        
        # x1: 正态分布
        features[:, 0] = np.random.normal(0, 1, self.n_samples)
        
        # x2: 均匀分布
        features[:, 1] = np.random.uniform(-2, 2, self.n_samples)
        
        # x3: 指数分布
        features[:, 2] = np.random.exponential(1, self.n_samples)
        
        # x4: 对数正态分布
        features[:, 3] = np.random.lognormal(0, 1, self.n_samples)
        
        # x5: 伽马分布
        features[:, 4] = np.random.gamma(2, 1, self.n_samples)
        
        # x6: 贝塔分布
        features[:, 5] = np.random.beta(2, 5, self.n_samples)
        
        # x7: 卡方分布
        features[:, 6] = np.random.chisquare(3, self.n_samples)
        
        # x8: 学生t分布
        features[:, 7] = np.random.standard_t(3, self.n_samples)
        
        # x9: 威布尔分布
        features[:, 8] = np.random.weibull(2, self.n_samples)
        
        # x10: 拉普拉斯分布
        features[:, 9] = np.random.laplace(0, 1, self.n_samples)
        
        return features
    
    def generate_target_function(self, X: np.ndarray, func_type: int, weights: np.ndarray) -> np.ndarray:
        """
        根据函数类型生成目标值
        
        Args:
            X: 特征矩阵
            func_type: 函数类型 (1-10)
            weights: 权重向量
            
        Returns:
            np.ndarray: 目标值
        """
        np.random.seed(self.random_state)
        
        if func_type == 1:
            # 线性函数
            return np.dot(X, weights) + np.random.normal(0, self.noise, self.n_samples)
        
        elif func_type == 2:
            # 正弦函数
            return (np.sin(np.pi * np.dot(X, weights)) + 
                    np.random.normal(0, self.noise, self.n_samples))
        
        elif func_type == 3:
            # 二次函数
            return (np.dot(X, weights)**2 + 
                    np.random.normal(0, self.noise, self.n_samples))
        
        elif func_type == 4:
            # 指数函数
            return (np.exp(np.dot(X, weights)) + 
                    np.random.normal(0, self.noise, self.n_samples))
        
        elif func_type == 5:
            # 对数函数
            return (np.log(np.abs(np.dot(X, weights)) + 1) + 
                    np.random.normal(0, self.noise, self.n_samples))
        
        elif func_type == 6:
            # 三角函数组合
            return (np.sin(np.dot(X, weights)) * np.cos(np.dot(X, weights)) + 
                    np.random.normal(0, self.noise, self.n_samples))
        
        elif func_type == 7:
            # 多项式函数
            linear_part = np.dot(X, weights)
            return (linear_part + 0.5 * linear_part**2 + 0.1 * linear_part**3 + 
                    np.random.normal(0, self.noise, self.n_samples))
        
        elif func_type == 8:
            # 分段函数
            linear_part = np.dot(X, weights)
            return (np.where(linear_part > 0, linear_part**2, -linear_part**2) + 
                    np.random.normal(0, self.noise, self.n_samples))
        
        elif func_type == 9:
            # 复合函数
            return (np.sin(np.dot(X, weights)) * np.exp(-np.dot(X, weights)**2) + 
                    np.random.normal(0, self.noise, self.n_samples))
        
        elif func_type == 10:
            # 复杂非线性函数
            linear_part = np.dot(X, weights)
            return (np.tanh(linear_part) + 0.5 * np.sin(2 * np.pi * linear_part) + 
                    np.random.normal(0, self.noise, self.n_samples))
        
        else:
            raise ValueError(f"不支持的函数类型: {func_type}")
    
    def generate_all_targets(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        生成所有目标值（y1~y10），每个y使用不同的函数类型
        
        Args:
            X: 特征矩阵
            weights: 权重向量
            
        Returns:
            np.ndarray: 形状为(n_samples, 10)的目标矩阵
        """
        targets = np.zeros((self.n_samples, 10))
        
        for i in range(10):
            # y1使用函数1（线性），y2使用函数2，...，y10使用函数10
            func_type = i + 1
            targets[:, i] = self.generate_target_function(X, func_type, weights)
        
        return targets
    
   
    
    def get_next_sample_id(self) -> int:
        """
        自动获取下一个可用的样本ID
        
        Returns:
            int: 下一个可用的样本ID
        """
        output_dir = "DataGenerate"
        if not os.path.exists(output_dir):
            return 1
        
        sample_id = 1
        while True:
            # 检查RawSample文件夹是否存在
            raw_sample_dir = os.path.join(output_dir, f"RawSample{sample_id}")
            if not os.path.exists(raw_sample_dir):
                return sample_id
            sample_id += 1
    
    def generate_data(self, sample_id: int = None) -> Tuple[List[pd.DataFrame], str]:
        """
        生成完整的关联数据集
        
        Args:
            sample_id: 样本ID，用于文件命名，如果为None则自动获取
            
        Returns:
            Tuple[List[pd.DataFrame], str]: 数据框列表和规则描述
        """
        if sample_id is None:
            sample_id = self.get_next_sample_id()
        
        # 设置当前样本ID
        self.current_sample_id = sample_id
        
        # 生成特征
        X = self.generate_features()
        
        # 创建列名
        feature_names = [f'x{i+1}' for i in range(10)]
        target_names = [f'y{i+1}' for i in range(10)]
        column_names = feature_names + target_names
        
        dataframes = []
        
        if self.flag:
            # 生成10个CSV文件，每次循环右移权重
            for j in range(10):
                # 循环右移权重
                shifted_weights = np.roll(self.weights, j)
                
                # 生成目标值
                y = self.generate_all_targets(X, shifted_weights)
                
                # 创建数据框
                data = np.hstack([X, y])
                df = pd.DataFrame(data, columns=column_names)
                
                # 保留8位小数
                df = df.round(8)
                
                # 在最后一列之后添加weight列（字符串格式）
                weight_str = str(shifted_weights.tolist())
                df.insert(len(df.columns), 'weight', weight_str)
                
                dataframes.append(df)
        else:
            # 只生成一个CSV文件
            y = self.generate_all_targets(X, self.weights)
            
            # 创建数据框
            data = np.hstack([X, y])
            df = pd.DataFrame(data, columns=column_names)
            df = df.round(8)
            
            # 在最后一列之后添加weight列（字符串格式）
            weight_str = str(self.weights.tolist())
            df.insert(len(df.columns), 'weight', weight_str)
            
            dataframes.append(df)
        
        # 生成规则描述
        rules = self._generate_rules()
        
        return dataframes, rules
    
    def _generate_rules(self) -> str:
        """
        生成数据生成规则描述
        
        Returns:
            str: 规则描述文本
        """
        # 函数名称映射
        func_names = {
            1: "Linear",
            2: "Sine", 
            3: "Quadratic",
            4: "Exponential",
            5: "Logarithmic",
            6: "Trigonometric",
            7: "Polynomial",
            8: "Piecewise",
            9: "Composite",
            10: "Complex_Nonlinear"
        }
        
        rules = f"""数据生成规则说明 (样本ID: {getattr(self, 'current_sample_id', 1)})

=== 固定配置 ===
特征生成函数: x1~x10 (10种不同分布)
目标生成函数: y1~y10 (10种不同函数类型)
- y1: {func_names[1]} (线性函数)
- y2: {func_names[2]} (正弦函数)  
- y3: {func_names[3]} (二次函数)
- y4: {func_names[4]} (指数函数)
- y5: {func_names[5]} (对数函数)
- y6: {func_names[6]} (三角函数组合)
- y7: {func_names[7]} (多项式函数)
- y8: {func_names[8]} (分段函数)
- y9: {func_names[9]} (复合函数)
- y10: {func_names[10]} (复杂非线性函数)

=== 可选参数 ===
权重向量: {self.weights}
样本量: {self.n_samples}
随机种子: {self.random_state}
噪声水平: {self.noise}
数据精度: 8位小数

=== 权重循环说明 ===
当flag=True时，生成10个文件，每次循环右移权重向量：
- 文件1中y1=f1([{self.weights[0]:.1f}, {self.weights[1]:.1f}, {self.weights[2]:.1f}, {self.weights[3]:.1f}, {self.weights[4]:.1f}, {self.weights[5]:.1f}, {self.weights[6]:.1f}, {self.weights[7]:.1f}, {self.weights[8]:.1f}, {self.weights[9]:.1f}], x1-x10)
- 文件2中y1=f1([{np.roll(self.weights, 1)[0]:.1f}, {np.roll(self.weights, 1)[1]:.1f}, {np.roll(self.weights, 1)[2]:.1f}, {np.roll(self.weights, 1)[3]:.1f}, {np.roll(self.weights, 1)[4]:.1f}, {np.roll(self.weights, 1)[5]:.1f}, {np.roll(self.weights, 1)[6]:.1f}, {np.roll(self.weights, 1)[7]:.1f}, {np.roll(self.weights, 1)[8]:.1f}, {np.roll(self.weights, 1)[9]:.1f}], x1-x10)
- 文件3中y1=f1([{np.roll(self.weights, 2)[0]:.1f}, {np.roll(self.weights, 2)[1]:.1f}, {np.roll(self.weights, 2)[2]:.1f}, {np.roll(self.weights, 2)[3]:.1f}, {np.roll(self.weights, 2)[4]:.1f}, {np.roll(self.weights, 2)[5]:.1f}, {np.roll(self.weights, 2)[6]:.1f}, {np.roll(self.weights, 2)[7]:.1f}, {np.roll(self.weights, 2)[8]:.1f}, {np.roll(self.weights, 2)[9]:.1f}], x1-x10)
- ...
- 文件10中y1=f1([{np.roll(self.weights, 9)[0]:.1f}, {np.roll(self.weights, 9)[1]:.1f}, {np.roll(self.weights, 9)[2]:.1f}, {np.roll(self.weights, 9)[3]:.1f}, {np.roll(self.weights, 9)[4]:.1f}, {np.roll(self.weights, 9)[5]:.1f}, {np.roll(self.weights, 9)[6]:.1f}, {np.roll(self.weights, 9)[7]:.1f}, {np.roll(self.weights, 9)[8]:.1f}, {np.roll(self.weights, 9)[9]:.1f}], x1-x10)

每个y变量都使用相同的权重循环模式，但使用不同的函数类型。
        """
        return rules.strip()
    
    def save_data(self, sample_id: int = None):
        """
        生成数据并保存到文件
        
        Args:
            sample_id: 样本ID，如果为None则自动获取下一个可用ID
        """
        # 生成数据
        dataframes, rules = self.generate_data(sample_id)
        
        # 获取实际使用的sample_id
        actual_sample_id = self.current_sample_id
        
        # 创建DataGenerate文件夹
        output_dir = "DataGenerate"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建RawSample{i}文件夹
        raw_sample_dir = os.path.join(output_dir, f"RawSample{actual_sample_id}")
        if not os.path.exists(raw_sample_dir):
            os.makedirs(raw_sample_dir)
        
        # 保存CSV文件
        saved_files = []
        for j, df in enumerate(dataframes):
            if self.flag:
                # 保存为MaxCorrelationXY{j}.csv
                csv_filename = os.path.join(raw_sample_dir, f"MaxCorrelationXY{j+1}.csv")
            else:
                # 只保存一个文件
                csv_filename = os.path.join(raw_sample_dir, f"MaxCorrelationXY1.csv")
            
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            saved_files.append(csv_filename)
            print(f"数据已保存到: {csv_filename}")
        
        # 保存规则文件到RawSample文件夹中
        txt_filename = os.path.join(raw_sample_dir, f"rule{actual_sample_id}.txt")
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(rules)
        
        print(f"规则已保存到: {txt_filename}")
        print(f"生成了{len(dataframes)}个CSV文件")
        
        return dataframes, rules


