#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
相关系数计算速度测试
测试五种相关系数（MI, Spearman, Pearson, Kendall, dCor）的计算速度
输入: 某个MaxCorrelationXY文件路径
"""

from CorrelationAnalyzer import CorrelationAnalyzer
import pandas as pd
import numpy as np
import time
import sys
import os

def test_correlation_speed(csv_file_path, num_iterations=1):
    """
    测试单个相关系数的计算速度
    
    Args:
        csv_file_path: CSV文件路径
        num_iterations: 迭代次数（用于更准确的计时）
    
    Returns:
        dict: 各种方法的计算时间（秒）
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 分离特征和目标
    feature_cols = [col for col in df.columns if col.startswith('x')]
    target_cols = [col for col in df.columns if col.startswith('y')]
    
    print(f"数据文件: {csv_file_path}")
    print(f"数据形状: {df.shape}")
    print(f"特征列数: {len(feature_cols)}")
    print(f"目标列数: {len(target_cols)}")
    print(f"迭代次数: {num_iterations}")
    print("\n开始测试各相关系数的计算速度...\n")
    
    # 创建分析器
    analyzer = CorrelationAnalyzer()
    
    # 测试各种方法
    methods = ['MI', 'Spearman', 'Pearson', 'Kendall', 'dCor']
    results = {}
    
    for method in methods:
        print(f"测试 {method}...", end=" ", flush=True)
        start_time = time.time()
        
        for iteration in range(num_iterations):
            # 计算所有y与所有x的关联度
            for target_col in target_cols:
                for feature_col in feature_cols:
                    x_data = df[feature_col].values
                    y_data = df[target_col].values
                    
                    if method == 'MI':
                        analyzer.calculate_mi(x_data, y_data)
                    elif method == 'Spearman':
                        analyzer.calculate_spearman(x_data, y_data)
                    elif method == 'Pearson':
                        analyzer.calculate_pearson(x_data, y_data)
                    elif method == 'Kendall':
                        analyzer.calculate_kendall(x_data, y_data)
                    elif method == 'dCor':
                        analyzer.calculate_dcor(x_data, y_data)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        avg_time = elapsed_time / num_iterations
        
        results[method] = {
            'total_time': elapsed_time,
            'avg_time': avg_time,
            'per_calculation': avg_time / (len(target_cols) * len(feature_cols))
        }
        
        print(f"完成 - 总时间: {elapsed_time:.2f}秒, 平均: {avg_time:.2f}秒, "
              f"单次计算: {results[method]['per_calculation']*1000:.2f}毫秒")
    
    return results

def print_speed_summary(results):
    """打印速度对比摘要"""
    print("\n" + "="*70)
    print("速度测试结果摘要")
    print("="*70)
    
    # 按平均时间排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_time'])
    
    print(f"\n{'方法':<15} {'总时间(秒)':<15} {'平均时间(秒)':<15} {'单次计算(毫秒)':<15}")
    print("-" * 70)
    
    for method, stats in sorted_results:
        print(f"{method:<15} {stats['total_time']:<15.2f} "
              f"{stats['avg_time']:<15.2f} {stats['per_calculation']*1000:<15.2f}")
              
    
    # 找出最快和最慢的方法
    fastest = sorted_results[0]
    slowest = sorted_results[-1]
    
    print("\n" + "="*70)
    print(f"最快方法: {fastest[0]} (平均: {fastest[1]['avg_time']:.2f}秒)")
    print(f"最慢方法: {slowest[0]} (平均: {slowest[1]['avg_time']:.2f}秒)")
    print(f"速度比: {slowest[1]['avg_time']/fastest[1]['avg_time']:.2f}x")
    print("="*70)

if __name__ == "__main__":
    # 获取输入文件路径
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # 默认使用示例文件
        csv_file = "DataGenerate/RawSample8/MaxCorrelationXY1.csv"
        print(f"未指定文件路径，使用默认路径: {csv_file}\n")
    
    if not os.path.exists(csv_file):
        print(f"[ERROR] 文件不存在: {csv_file}")
        print("\n使用方法:")
        print("  python test_correlation_speed.py <CSV文件路径> [迭代次数]")
        print("\n示例:")
        print("  python test_correlation_speed.py DataGenerate/RawSample8/MaxCorrelationXY1.csv")
        sys.exit(1)
    
    # 获取迭代次数（可选）
    num_iterations = 1
    if len(sys.argv) > 2:
        try:
            num_iterations = int(sys.argv[2])
        except:
            print(f"警告: 无效的迭代次数参数，使用默认值1")
    
    # 执行速度测试
    results = test_correlation_speed(csv_file, num_iterations)
    
    # 打印摘要
    print_speed_summary(results)

