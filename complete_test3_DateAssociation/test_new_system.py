from DataGenerator import SyntheticDataGenerator
from CorrelationAnalyzer import CorrelationAnalyzer
import numpy as np

if __name__ == "__main__":
    print("=== 测试新的数据生成和分析系统 ===")
    
    # 1. 创建数据生成器（使用默认权重和flag=True）
    print("\n1. 创建数据生成器...")
    generator = SyntheticDataGenerator(
        n_samples=1000, 
        noise=0.1, 
        random_state=43,
        weights=np.array([1.0, 0.1, 0.08, 0.07, 0.06, 0.05, 0.01, 0.01, 0.005, 0.001]),
        flag=True
    )
    
    # 2. 生成数据（生成新的样本ID）
    print("\n2. 生成数据...")
    dataframes, rules = generator.save_data()  # 自动获取下一个可用ID
    
    print(f"生成了{len(dataframes)}个数据框")
    print(f"第一个数据框形状: {dataframes[0].shape}")
    print(f"第一个数据框列名: {dataframes[0].columns.tolist()}")
    
    # 获取实际使用的sample_id
    actual_sample_id = generator.current_sample_id
    print(f"使用的样本ID: {actual_sample_id}")
    
    # 3. 创建分析器
    print("\n3. 创建分析器...")
    analyzer = CorrelationAnalyzer()
    
    # 4. 分析新生成的文件夹
    print(f"\n4. 分析RawSample{actual_sample_id}文件夹...")
    raw_sample_folder = f"DataGenerate/RawSample{actual_sample_id}"
    result_path = analyzer.analyze_folder(raw_sample_folder)
    
    if result_path:
        print(f"汇总分析结果已保存到: {result_path}")
        
        # 读取并显示结果摘要
        import pandas as pd
        summary_df = pd.read_csv(result_path)
        print(f"\n汇总结果形状: {summary_df.shape}")
        print("汇总结果列名:", summary_df.columns.tolist())
        print("\n前5行数据:")
        print(summary_df.head())
    else:
        print("分析失败")
    
    print("\n=== 测试完成 ===")
