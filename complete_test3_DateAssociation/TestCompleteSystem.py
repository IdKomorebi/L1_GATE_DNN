#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
完整系统测试
自动化流程: 数据生成 -> 关联分析 -> 知识图谱生成
根据已存在的数据序号自动生成新的数据分析以及知识图谱
"""

from DataGenerator import SyntheticDataGenerator
from CorrelationAnalyzer import CorrelationAnalyzer
from ImprovedKnowledgeGraphSystem import ImprovedKnowledgeGraphSystem
import numpy as np
import os

def find_latest_sample_id():
    """查找最新的样本ID"""
    base_dir = "DataGenerate"
    if not os.path.exists(base_dir):
        return 0
    
    max_id = 0
    for folder in os.listdir(base_dir):
        if folder.startswith("RawSample"):
            try:
                folder_id = int(folder.replace("RawSample", ""))
                max_id = max(max_id, folder_id)
            except:
                pass
    return max_id

if __name__ == "__main__":
    print("="*70)
    print("完整系统测试：数据生成 -> 关联分析 -> 知识图谱生成")
    print("="*70)
    
    # 1. 创建数据生成器
    print("\n[步骤1/6] 创建数据生成器...")
    generator = SyntheticDataGenerator(
        n_samples=1000, 
        noise=0.1, 
        random_state=42,
        weights=np.array([1.0, 0.1, 0.08, 0.07, 0.06, 0.05, 0.01, 0.01, 0.005, 0.001]),
        flag=True
    )
    
    # 2. 生成数据
    print("\n[步骤2/6] 生成数据...")
    dataframes, rules = generator.save_data()
    actual_sample_id = generator.current_sample_id
    print(f"[OK] 使用的样本ID: {actual_sample_id}")
    print(f"[OK] 生成了{len(dataframes)}个CSV文件")
    
    # 3. 创建分析器
    print("\n[步骤3/6] 创建关联度分析器...")
    analyzer = CorrelationAnalyzer()
    
    # 4. 分析新生成的文件夹
    print(f"\n[步骤4/6] 分析RawSample{actual_sample_id}文件夹...")
    raw_sample_folder = f"DataGenerate/RawSample{actual_sample_id}"
    result_path = analyzer.analyze_folder(raw_sample_folder)
    
    if result_path:
        print(f"[OK] 汇总分析结果已保存到: {result_path}")
        
        # 5. 创建知识图谱系统
        print(f"\n[步骤5/6] 创建知识图谱系统...")
        kg_system = ImprovedKnowledgeGraphSystem(data_file=result_path)
        
        # 6. 生成可视化
        print(f"\n[步骤6/6] 生成知识图谱可视化...")
        kg_output_path = kg_system.generate_visualization()
        
        if kg_output_path:
            print("\n" + "="*70)
            print("[OK] 测试完成！")
            print("="*70)
            print(f"\n输出文件:")
            print(f"  - 数据生成目录: DataGenerate/RawSample{actual_sample_id}/")
            print(f"  - 分析结果: {result_path}")
            print(f"  - 知识图谱: {kg_output_path}")
            print(f"\n系统特性:")
            print(f"  - 支持5种相关系数: MI, Spearman, Pearson, Kendall, dCor")
            print(f"  - 自动识别数据块: {len(kg_system.available_blocks)}个")
            print(f"  - 自动识别方法: {len(kg_system.available_methods)}种")
            print(f"  - 交互式可视化: 支持切换数据块和方法")
            print(f"\n请打开 {kg_output_path} 查看知识图谱！")
        else:
            print("[ERROR] 知识图谱生成失败")
    else:
        print("[ERROR] 关联分析失败")
    
    print()

