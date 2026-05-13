# 测试文件说明

本目录包含多个测试文件，用于测试系统的各个模块。

## 测试文件列表

### 1. test_data_generation.py
**功能**: 测试数据生成模块
- **输入**: 无（使用内置参数）
- **输出**: 在 `DataGenerate/RawSample{N}/` 目录下生成10个CSV文件
- **用法**: 
  ```bash
  python test_data_generation.py
  ```

### 2. test_correlation_analysis.py
**功能**: 测试关联度分析模块（5种相关系数）
- **输入**: 包含10个CSV文件的文件夹路径（可选参数）
- **输出**: 在 `DataAnalyze/` 目录下生成分析结果CSV文件
- **用法**: 
  ```bash
  # 使用默认路径
  python test_correlation_analysis.py
  
  # 指定文件夹路径
  python test_correlation_analysis.py DataGenerate/RawSample8
  ```

### 3. test_knowledge_graph.py
**功能**: 测试知识图谱生成模块
- **输入**: 关联度分析结果CSV文件路径（可选参数）
- **输出**: 在 `KnowledgeGraph/` 目录下生成HTML可视化文件
- **用法**: 
  ```bash
  # 使用默认路径
  python test_knowledge_graph.py
  
  # 指定分析文件路径
  python test_knowledge_graph.py DataAnalyze/correlation_summary_RawSample8.csv
  ```

### 4. test_complete_system.py
**功能**: 完整系统测试（数据生成 -> 关联分析 -> 知识图谱生成）
- **输入**: 无（自动生成新的样本ID）
- **输出**: 
  - 数据生成目录: `DataGenerate/RawSample{N}/`
  - 分析结果: `DataAnalyze/correlation_summary_RawSample{N}.csv`
  - 知识图谱: `KnowledgeGraph/KG_Correlation{N}.html`
- **用法**: 
  ```bash
  python test_complete_system.py
  ```

### 5. test_correlation_speed.py
**功能**: 测试五种相关系数的计算速度
- **输入**: MaxCorrelationXY文件路径（必需），迭代次数（可选）
- **输出**: 控制台输出各方法的计算速度对比
- **用法**: 
  ```bash
  # 使用默认路径，迭代1次
  python test_correlation_speed.py DataGenerate/RawSample8/MaxCorrelationXY1.csv
  
  # 指定迭代次数（用于更准确的计时）
  python test_correlation_speed.py DataGenerate/RawSample8/MaxCorrelationXY1.csv 3
  ```

## 速度测试结果示例

根据测试结果，五种相关系数的计算速度排序（从快到慢）：
1. **Kendall** - 最快（~0.41毫秒/次）
2. **Pearson** - 很快（~0.61毫秒/次）
3. **Spearman** - 较快（~1.36毫秒/次）
4. **MI** - 较慢（~7.42毫秒/次）
5. **dCor** - 最慢（~134.50毫秒/次，比Kendall慢约330倍）

**注意**: dCor计算最慢，建议在数据量较大时考虑优化或使用并行计算。

## 支持的相关系数

系统支持以下5种相关系数：
- **MI** (Mutual Information) - 互信息
- **Spearman** - Spearman等级相关系数
- **Pearson** - Pearson线性相关系数
- **Kendall** - Kendall τ相关系数
- **dCor** - Distance Correlation（距离相关系数）

## 系统特性

- ✅ 自动识别数据块（csv1-csv10）
- ✅ 自动识别可用的相关系数方法
- ✅ 交互式HTML可视化（支持切换数据块和方法）
- ✅ 每种方法的阈值独立配置
- ✅ 完整的测试覆盖

