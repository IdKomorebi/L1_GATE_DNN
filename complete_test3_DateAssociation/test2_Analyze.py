#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
关联度分析测试
测试 CorrelationAnalyzer 的分析功能
输入: 包含10个CSV文件的文件夹路径
"""

from CorrelationAnalyzer import CorrelationAnalyzer
import sys
import os

if __name__ == "__main__":
    print("="*70)
    print("关联度分析测试")
    print("="*70)
    
    # 获取输入文件夹路径
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        # 默认使用最新的RawSample文件夹
        folder_path = "DataGenerate/RawSample12"
        print(f"\n未指定文件夹路径，使用默认路径: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"[ERROR] 文件夹不存在: {folder_path}")
        sys.exit(1)
    
    print(f"\n分析文件夹: {folder_path}")
    
    # 创建分析器
    print("\n创建关联度分析器...")
    analyzer = CorrelationAnalyzer()
    
    # 执行分析
    print("\n开始分析...")
    result_path = analyzer.analyze_folder(folder_path)
    
    if result_path:
        print(f"\n[OK] 分析完成！")
        print(f"  - 分析结果已保存到: {result_path}")
        
        # 读取并显示摘要信息
        import pandas as pd
        summary_df = pd.read_csv(result_path)
        print(f"  - 结果数据形状: {summary_df.shape}")
        print(f"  - 包含的方法: {sorted(summary_df['method'].str.strip().unique().tolist())}")
        print(f"  - 数据块数量: {len(summary_df['index'].unique())}")
        
        # 显示前5行
        print(f"\n前5行数据预览:")
        print(summary_df.head().to_string())
    else:
        print("[ERROR] 分析失败")
    
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)

