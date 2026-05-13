#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import numpy as np
import re

class ImprovedKnowledgeGraphSystem:
    """
    改进的知识图谱可视化系统
    
    改进点：
    1. y节点在中间，x节点分在左右两侧
    2. 不同方法使用不同阈值
    3. 合并为一个HTML文件，支持JS切换
    4. 根据文件名自动命名
    """
    
    def __init__(self, data_file):
        """
        初始化系统
        
        Args:
            data_file: 关联度数据文件路径
        """
        self.data_file = data_file
        # 读取完整文件，不限制行数
        self.df = pd.read_csv(data_file)
        
        # 不同方法的阈值（默认值，可根据数据调整）
        self.thresholds = {
            "MI": 0.06,
            "Spearman": 0.1,
            "Pearson": 0.16,
            "Kendall": 0.1,
            "dCor": 0.1
        }
        
        # 从文件名提取样本ID
        self.sample_id = self.extract_sample_id(data_file)
        
        # 自动获取所有可用的数据块（基于index列）
        # 自定义排序：先按数字部分排序（csv1, csv2, ..., csv10）
        blocks = self.df['index'].unique().tolist()
        def sort_key(block):
            # 提取数字部分进行排序
            num = block.replace('csv', '')
            try:
                return int(num)
            except:
                return 999
        self.available_blocks = sorted(blocks, key=sort_key)
        
        # 自动获取所有可用的方法（基于method列）
        self.available_methods = sorted(self.df['method'].str.strip().unique().tolist())
        
        print(f"知识图谱系统初始化完成")
        print(f"数据文件: {data_file}")
        print(f"数据形状: {self.df.shape}")
        print(f"样本ID: {self.sample_id}")
        print(f"可用数据块数量: {len(self.available_blocks)}")
        print(f"数据块列表: {self.available_blocks}")
        print(f"可用方法数量: {len(self.available_methods)}")
        print(f"方法列表: {self.available_methods}")
        print(f"阈值设置: {self.thresholds}")
    
    def extract_sample_id(self, filename):
        """从文件名提取样本ID"""
        # 从correlation_summary_RawSample{i}.csv中提取i
        match = re.search(r'correlation_summary_RawSample(\d+)', filename)
        if match:
            return match.group(1)
        # 备用匹配模式
        match2 = re.search(r'RawSample(\d+)', filename)
        if match2:
            return match2.group(1)
        return "1"
    
    def create_network_graph(self, method, block_index=None):
        """
        创建NetworkX网络图
        
        Args:
            method: 关联度方法
            block_index: 数据块索引（例如 'csv1', 'csv2'），如果为None则使用全部数据
            
        Returns:
            nx.Graph: NetworkX图对象
        """
        # 过滤数据
        method_data = self.df[self.df['method'].str.strip() == method].copy()
        
        # 如果指定了数据块，进一步过滤
        if block_index is not None:
            method_data = method_data[method_data['index'] == block_index].copy()
        
        if method_data.empty:
            print(f"没有找到方法 {method} 在块 {block_index} 的数据")
            return nx.Graph()
        
        # 获取阈值
        threshold = self.thresholds.get(method, 0.1)
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点和边
        for _, row in method_data.iterrows():
            y_node = row['y']
            weight_vector = eval(row['weight'])
            
            # 添加y节点（目标变量）
            G.add_node(y_node, 
                      node_type='target',
                      func=row['func(默认)'].strip(),
                      size=30,
                      color='#ff6b6b')
            
            # 添加x节点和边（特征变量）
            for i, x_name in enumerate([f'x{j+1}' for j in range(10)]):
                if x_name in row and not pd.isna(row[x_name]):
                    correlation_value = float(row[x_name])
                    
                    # 使用对应方法的阈值
                    if abs(correlation_value) >= threshold:
                        # 添加x节点（如果不存在）
                        if not G.has_node(x_name):
                            G.add_node(x_name, 
                                      node_type='feature',
                                      size=20,
                                      color='#4ecdc4')
                        
                        # 添加边
                        edge_weight = abs(correlation_value)
                        G.add_edge(y_node, x_name, 
                                  weight=edge_weight,
                                  correlation=correlation_value,
                                  original_weight=weight_vector[i])
        
        return G
    
    def create_improved_layout(self, G):
        """
        创建改进的布局：y在中间一列，x分居两列（左侧x1-x5，右侧x6-x10）
        
        Args:
            G: NetworkX图对象
            
        Returns:
            dict: 节点位置字典
        """
        if len(G.nodes()) == 0:
            return {}
        
        # 分离y节点和x节点
        y_nodes = [n for n in G.nodes() if n.startswith('y')]
        x_nodes = [n for n in G.nodes() if n.startswith('x')]
        
        pos = {}
        
        # y节点排列在中间一列
        if len(y_nodes) > 0:
            center_x = 0  # 中间列
            for i, y_node in enumerate(y_nodes):
                x = center_x
                y = -3 + i * 0.6  # 垂直分布，间距0.6
                pos[y_node] = (x, y)
        
        # x节点分在左右两列
        if len(x_nodes) > 0:
            # 按x节点编号排序
            x_nodes_sorted = sorted(x_nodes, key=lambda x: int(x[1:]))
            
            # 左侧x节点 (x1-x5)
            left_x_nodes = [x for x in x_nodes_sorted if int(x[1:]) <= 5]
            # 右侧x节点 (x6-x10)
            right_x_nodes = [x for x in x_nodes_sorted if int(x[1:]) > 5]
            
            # 左侧列布局
            for i, x_node in enumerate(left_x_nodes):
                x = -3  # 左侧列x坐标
                y = -3 + i * 0.6  # 垂直分布，间距0.6
                pos[x_node] = (x, y)
            
            # 右侧列布局
            for i, x_node in enumerate(right_x_nodes):
                x = 3   # 右侧列x坐标
                y = -3 + i * 0.6  # 垂直分布，间距0.6
                pos[x_node] = (x, y)
        
        return pos
    
    def create_combined_visualization(self, output_path):
        """
        创建合并的可视化HTML文件，支持JS切换
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            bool: 是否成功
        """
        # 为每个数据块和每种方法创建数据
        # 数据结构: all_data[block_index][method] = {nodes, edges, threshold}
        all_data = {}
        
        for block_index in self.available_blocks:
            block_data = {}
            block_num = block_index.replace('csv', '')  # 提取数字部分
            
            for method in self.available_methods:
                G = self.create_network_graph(method, block_index)
                if len(G.nodes()) == 0:
                    print(f"块 {block_index} 方法 {method} 没有节点")
                    continue
                
                print(f"块 {block_index} {method} 图谱: {len(G.nodes())} 个节点, {len(G.edges())} 条边")
                
                # 创建布局
                pos = self.create_improved_layout(G)
                
                # 准备节点和边数据
                nodes_data = []
                edges_data = []
                
                # 节点数据
                for node in G.nodes():
                    node_data = G.nodes[node]
                    x, y = pos.get(node, (0, 0))
                    
                    if node_data.get('node_type') == 'target':
                        nodes_data.append({
                            'id': node,
                            'label': node,
                            'size': 30,
                            'color': '#ff6b6b',
                            'title': f"目标变量: {node}<br>函数类型: {node_data.get('func', 'Unknown')}",
                            'x': x,
                            'y': y
                        })
                    else:
                        nodes_data.append({
                            'id': node,
                            'label': node,
                            'size': 20,
                            'color': '#4ecdc4',
                            'title': f"特征变量: {node}",
                            'x': x,
                            'y': y
                        })
                
                # 边数据
                for edge in G.edges(data=True):
                    source, target, data = edge
                    correlation = data.get('correlation', 0)
                    weight = data.get('weight', 0)
                    
                    # 根据关联强度设置边的颜色和粗细
                    if abs(correlation) >= 0.7:
                        color = "#ff4444"
                        width = 8
                    elif abs(correlation) >= 0.5:
                        color = "#ff8844"
                        width = 6
                    elif abs(correlation) >= 0.3:
                        color = "#ffaa44"
                        width = 4
                    else:
                        color = "#ffcc88"
                        width = 2
                    
                    edges_data.append({
                        'from': source,
                        'to': target,
                        'width': width,
                        'color': color,
                        'title': f"关联度 ({method}): {correlation:.4f}<br>"
                                 f"权重: {weight:.4f}<br>"
                                 f"原始权重: {data.get('original_weight', 0):.2f}"
                    })
                
                block_data[method] = {
                    'nodes': nodes_data,
                    'edges': edges_data,
                    'threshold': self.thresholds[method]
                }
            
            all_data[block_index] = block_data
        
        # 生成HTML内容
        html_content = self.generate_combined_html(all_data)
        
        # 保存文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"[OK] 合并知识图谱已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] 保存失败: {e}")
            return False
    
    def generate_combined_html(self, all_data):
        """生成合并的HTML内容
        
        Args:
            all_data: 数据结构为 all_data[block_index][method] = {nodes, edges, threshold}
        """
        import json
        
        # 将数据转换为JSON字符串，保持嵌套结构
        # 使用json.dumps确保正确的JSON格式
        all_data_json = json.dumps(all_data, ensure_ascii=False)
        
        # 生成X{i}YMAX选项的HTML
        block_options_html = ""
        for block_index in self.available_blocks:
            block_num = block_index.replace('csv', '')
            block_options_html += f'<option value="{block_index}">X{block_num}YMAX</option>\n'
        
        # 生成方法选项的HTML
        method_options_html = ""
        method_names = {
            "MI": "互信息 (MI)",
            "Spearman": "Spearman相关系数",
            "Pearson": "Pearson相关系数",
            "Kendall": "Kendall τ相关系数",
            "dCor": "Distance Correlation (dCor)"
        }
        for method in self.available_methods:
            method_name = method_names.get(method, method)
            threshold = self.thresholds.get(method, 0.1)
            method_options_html += f'<option value="{method}">{method_name} - 阈值: {threshold}</option>\n'
        
        # 生成阈值说明HTML
        threshold_info_html = ""
        for method in self.available_methods:
            method_name = method_names.get(method, method)
            threshold = self.thresholds.get(method, 0.1)
            threshold_info_html += f'<li><strong>{method}:</strong> 绝对值小于 {threshold} 的边不显示</li>\n'
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>知识图谱可视化 - RawSample{self.sample_id}</title>
    <meta charset="utf-8">
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
            text-align: center;
        }}
        .controls {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .control-group {{
            margin-bottom: 15px;
        }}
        label {{
            display: inline-block;
            width: 120px;
            font-weight: bold;
        }}
        select {{
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-right: 10px;
            font-size: 16px;
        }}
        button {{
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }}
        button:hover {{
            background-color: #2980b9;
        }}
        .graph-container {{
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        #network {{
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }}
        .stats {{
            background-color: #e8f5e8;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>知识图谱可视化</h1>
        <p>RawSample{self.sample_id}</p>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label for="blockSelect">数据块选择:</label>
            <select id="blockSelect" onchange="switchBlock()">
                {block_options_html}
            </select>
        </div>
        
        <div class="control-group">
            <label for="method">关联度方法:</label>
            <select id="method" onchange="switchMethod()">
                {method_options_html}
            </select>
        </div>
        
        <div class="control-group">
            <button onclick="resetView()">重置视图</button>
            <button onclick="fitView()">适应窗口</button>
        </div>
        
        <div class="stats" id="stats">
            请选择数据块和关联度方法查看图谱
        </div>
    </div>
    
    <div class="graph-container">
        <div id="network"></div>
    </div>
    
    <div class="info">
         <h3>使用说明:</h3>
         <ul>
             <li><strong>红色节点:</strong> 目标变量 (y1-y10) - 排列在中间一列</li>
             <li><strong>青色节点:</strong> 特征变量 (x1-x10) - 左侧列x1-x5，右侧列x6-x10</li>
             <li><strong>边的颜色:</strong> 表示关联强度 (红色=强，橙色=中，黄色=弱)</li>
             <li><strong>边的粗细:</strong> 表示关联程度</li>
             <li><strong>点击边:</strong> 查看详细的关联度信息</li>
             <li><strong>拖拽节点:</strong> 调整布局</li>
             <li><strong>滚轮:</strong> 缩放图谱</li>
         </ul>
         
        <h3>阈值设置:</h3>
        <ul>
            {threshold_info_html}
        </ul>
    </div>

    <script>
        // 数据 - 结构为 allData[block_index][method] = {{nodes, edges, threshold}}
        const allData = {all_data_json};
        
        // 网络对象
        let network = null;
        let currentBlock = '{self.available_blocks[0]}';
        let currentMethod = '{self.available_methods[0]}';
        
        // 获取当前数据
        function getCurrentData() {{
            if (allData[currentBlock] && allData[currentBlock][currentMethod]) {{
                return allData[currentBlock][currentMethod];
            }}
            return null;
        }}
        
        // 初始化
        function init() {{
            const container = document.getElementById('network');
            const data = getCurrentData();
            
            if (!data) {{
                document.getElementById('network').innerHTML = 
                    '<div style="text-align: center; padding: 50px;">没有找到 块' + currentBlock + ' 方法 ' + currentMethod + ' 的数据</div>';
                updateStats();
                return;
            }}
            
            const nodes = new vis.DataSet(data.nodes);
            const edges = new vis.DataSet(data.edges);
            
            const networkData = {{
                nodes: nodes,
                edges: edges
            }};
            
            const options = {{
                physics: {{
                    enabled: true,
                    stabilization: {{ iterations: 100 }},
                    barnesHut: {{
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 200,
                        springConstant: 0.04,
                        damping: 0.09
                    }}
                }},
                interaction: {{
                    hover: true,
                    hoverConnectedEdges: true
                }},
                nodes: {{
                    font: {{
                        size: 16,
                        color: '#000000'
                    }}
                }},
                edges: {{
                    font: {{
                        size: 12,
                        color: '#000000'
                    }}
                }}
            }};
            
            network = new vis.Network(container, networkData, options);
            
            // 更新统计信息
            updateStats();
        }}
        
        // 切换数据块
        function switchBlock() {{
            currentBlock = document.getElementById('blockSelect').value;
            init();
        }}
        
        // 切换方法
        function switchMethod() {{
            currentMethod = document.getElementById('method').value;
            init();
        }}
        
        // 重置视图
        function resetView() {{
            if (network) {{
                network.setOptions({{ physics: {{ enabled: true }} }});
                setTimeout(() => {{
                    network.setOptions({{ physics: {{ enabled: false }} }});
                }}, 1000);
            }}
        }}
        
        // 适应窗口
        function fitView() {{
            if (network) {{
                network.fit();
            }}
        }}
        
        // 更新统计信息
        function updateStats() {{
            const data = getCurrentData();
            if (data) {{
                const nodeCount = data.nodes.length;
                const edgeCount = data.edges.length;
                const threshold = data.threshold;
                const blockNum = currentBlock.replace('csv', '');
                
                document.getElementById('stats').innerHTML = 
                    `<strong>X${{blockNum}}YMAX</strong> | <strong>${{currentMethod}}</strong> | 节点: ${{nodeCount}} | 边: ${{edgeCount}} | 阈值: ${{threshold}}`;
            }} else {{
                const blockNum = currentBlock.replace('csv', '');
                document.getElementById('stats').innerHTML = 
                    `<strong>X${{blockNum}}YMAX</strong> | <strong>${{currentMethod}}</strong> | 没有数据`;
            }}
        }}
        
        // 页面加载完成后初始化
        window.onload = function() {{
            init();
        }};
    </script>
</body>
</html>"""
        
        return html_content
    
    def generate_visualization(self, output_dir="KnowledgeGraph"):
        """
        生成可视化文件
        
        Args:
            output_dir: 输出目录
            
        Returns:
            str: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        output_filename = f"KG_Correlation{self.sample_id}.html"
        output_path = os.path.join(output_dir, output_filename)
        
        # 创建合并的可视化
        success = self.create_combined_visualization(output_path)
        
        if success:
            print(f"\n=== 知识图谱生成完成 ===")
            print(f"输出文件: {output_path}")
            print(f"样本ID: {self.sample_id}")
            print(f"阈值设置: {self.thresholds}")
            return output_path
        else:
            print("生成失败")
            return None

# 主程序
if __name__ == "__main__":
    print("=== 改进的知识图谱可视化系统 ===")
    
    # 创建系统
    kg_system = ImprovedKnowledgeGraphSystem(
        data_file="DataAnalyze/correlation_summary_RawSample7.csv"
    )
    
    # 生成可视化
    output_path = kg_system.generate_visualization()
    
    if output_path:
        print(f"\n请打开 {output_path} 查看知识图谱！")
