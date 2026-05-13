#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识图谱生成测试
测试 ImprovedKnowledgeGraphSystem 的可视化功能
输入: 关联度分析结果CSV文件路径
"""

from ImprovedKnowledgeGraphSystem import ImprovedKnowledgeGraphSystem
import sys
import os

if __name__ == "__main__":
    print("="*70)
    print("知识图谱生成测试")
    print("="*70)
    
    # 获取输入文件路径
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # 默认使用最新的分析结果文件
        data_file = "DataAnalyze/correlation_summary_RawSample12.csv"
        print(f"\n未指定分析文件路径，使用默认路径: {data_file}")
    
    if not os.path.exists(data_file):
        print(f"[ERROR] 文件不存在: {data_file}")
        sys.exit(1)
    
    print(f"\n读取分析文件: {data_file}")
    
    # 创建知识图谱系统
    print("\n创建知识图谱系统...")
    kg_system = ImprovedKnowledgeGraphSystem(data_file=data_file)
    
    # 生成可视化
    print("\n开始生成知识图谱...")
    output_path = kg_system.generate_visualization()
    
    if output_path:
        print(f"\n[OK] 知识图谱生成完成！")
        print(f"  - 输出文件: {output_path}")
        print(f"  - 样本ID: {kg_system.sample_id}")
        print(f"  - 可用数据块: {len(kg_system.available_blocks)}个")
        print(f"  - 可用方法: {len(kg_system.available_methods)}种")
        print(f"    {', '.join(kg_system.available_methods)}")
        print(f"  - 阈值设置: {kg_system.thresholds}")
        print(f"\n请打开 {output_path} 查看知识图谱！")
    else:
        print("[ERROR] 知识图谱生成失败")
     
    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)

