#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据生成测试
测试 SyntheticDataGenerator 的数据生成功能
"""

from DataGenerator import SyntheticDataGenerator
import numpy as np

if __name__ == "__main__":
    print("="*70)
    print("数据生成测试")
    print("="*70)
    
    # 创建数据生成器
    print("\n创建数据生成器...")
    generator = SyntheticDataGenerator(
        n_samples=1000, 
        noise=0.01, 
        random_state=45,
        weights=np.array([1.0, 0.1, 0.08, 0.07, 0.06, 0.05, 0.01, 0.01, 0.005, 0.001]),
        flag=True  # 生成10个CSV文件
    )
    
    # 生成数据
    print("\n开始生成数据...")
    dataframes, rules = generator.save_data()
    actual_sample_id = generator.current_sample_id
    
    print(f"\n[OK] 数据生成完成！")
    print(f"  - 使用的样本ID: {actual_sample_id}")
    print(f"  - 生成的文件数量: {len(dataframes)}")
    print(f"  - 每个文件的形状: {dataframes[0].shape if dataframes else 'N/A'}")
    print(f"  - 输出目录: DataGenerate/RawSample{actual_sample_id}/")
    
    # 显示第一个数据框的列名
    if dataframes:
        print(f"  - 列名: {list(dataframes[0].columns)}")
        print(f"  - 第一行weight列值: {dataframes[0]['weight'].iloc[0]}")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)

