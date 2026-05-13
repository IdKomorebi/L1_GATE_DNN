#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List

import numpy as np
import pandas as pd


class RealDataKnowledgeGraphSystem:
    """
    针对真实数据关联结果(realdata*_relationships.csv)的知识图谱可视化系统。

    - 节点：表示一列数据，节点标签使用“|”后面的短名称，完整名称放在悬浮提示里。
    - 边：表示两列之间的关联关系，粗细和颜色表示关联强度。
    - 支持切换视角：
        * 整体图谱：展示所有列和全部边。
        * 以某一列为视角：只展示该列与其它列的边。
    - 支持切换不同的关联度方法：NMI_min / Spearman / Pearson / Kendall / Distance Corr。
    """

    def __init__(self, data_file: str):
        """
        Args:
            data_file: analyzer 生成的 realdata*_relationships.csv 路径
        """
        self.data_file = data_file
        self.df = pd.read_csv(data_file)

        # 关联度方法及阈值（可按需要调整）
        # 支持两种格式：
        #   - 新格式：使用列名 nmi（归一化互信息 NMI_min）
        #   - 旧格式：仍然是 mi（原始互信息）
        # 会根据 CSV 实际包含的列自动选择。
        base_metric_order = ["nmi", "mi", "spearman", "pearson", "kendall", "distance_corr", "hsic"]
        self.metrics = [m for m in base_metric_order if m in self.df.columns]

        self.metric_display_names: Dict[str, str] = {
            "nmi": "归一化互信息 (NMI_min)",
            "mi": "互信息 (MI)",
            "spearman": "Spearman 相关系数",
            "pearson": "Pearson 相关系数",
            "kendall": "Kendall τ 相关系数",
            "distance_corr": "Distance Correlation (dCor)",
            "hsic": "归一化 HSIC",
        }
        self.thresholds: Dict[str, float] = {
            "nmi": 0.05,
            "mi": 0.05,
            "spearman": 0.1,
            "pearson": 0.1,
            "kendall": 0.08,
            "distance_corr": 0.1,
            "hsic": 0.05,
        }

        # 列名称处理：从 column_a / column_b 中解析出短名称
        self.columns_info = self._extract_columns_info()
        self.column_keys: List[str] = sorted(self.columns_info.keys())

        print("实数知识图谱系统初始化完成")
        print(f"数据文件: {data_file}")
        print(f"数据形状: {self.df.shape}")
        print(f"列数量: {len(self.column_keys)}")

    def _extract_columns_info(self) -> Dict[str, Dict[str, str]]:
        """
        从 column_a / column_b 提取列的完整名称与短名称。

        Returns:
            dict: {full_name: {"full": full_name, "short": short_name}}
        """
        columns_info: Dict[str, Dict[str, str]] = {}

        for col in pd.concat([self.df["column_a"], self.df["column_b"]]).unique():
            full_name = str(col)
            if "|" in full_name:
                # 使用 | 后面的名称作为短名称
                short_name = full_name.split("|", 1)[1].strip()
            else:
                short_name = full_name.strip()

            # 节点标签过长时做一个截断，完整名称放在 title 里
            max_len = 14
            if len(short_name) > max_len:
                display_name = short_name[: max_len - 1] + "…"
            else:
                display_name = short_name

            columns_info[full_name] = {
                "full": full_name,
                "short": display_name,
                "desc": short_name,
            }

        return columns_info

    def _build_all_data(self) -> Dict[str, Dict]:
        """
        构建前端 JS 使用的数据结构。

        Returns:
            all_data[metric] = {
                "global": {"nodes": [...], "edges": [...], "threshold": float},
                "focus": {column_key: {"nodes": [...], "edges": [...]}},
            }
        """
        all_data: Dict[str, Dict] = {}

        # 先准备节点（对所有 metric 共用）
        base_nodes = []
        for key in self.column_keys:
            info = self.columns_info[key]
            node = {
                "id": key,
                "label": info["short"],
                "title": f"<b>{info['desc']}</b><br>{info['full']}",
                "size": 20,
                "color": "#4ecdc4",
            }
            base_nodes.append(node)

        # 每种 metric 分别构建边
        for metric in self.metrics:
            if metric not in self.df.columns:
                continue

            threshold = self.thresholds.get(metric, 0.1)
            metric_edges_global = []

            # focus 视角：为每列准备一个子图
            focus_edges: Dict[str, List[Dict]] = {k: [] for k in self.column_keys}

            for _, row in self.df.iterrows():
                a = str(row["column_a"])
                b = str(row["column_b"])
                if a not in self.columns_info or b not in self.columns_info:
                    continue

                value = row[metric]
                if pd.isna(value):
                    continue

                try:
                    val = float(value)
                except Exception:
                    continue

                if abs(val) < threshold:
                    continue

                color, width = self._edge_style(abs(val))
                edge = {
                    "from": a,
                    "to": b,
                    "width": width,
                    "color": color,
                    "title": f"{self.metric_display_names.get(metric, metric)}: {val:.4f}",
                }

                metric_edges_global.append(edge)

                # 加入 focus 结构：a/b 视角都要包含这条边
                focus_edges[a].append(edge)
                focus_edges[b].append(edge)

            # 为每个 focus 列构建 nodes 子集
            focus_subgraphs: Dict[str, Dict] = {}
            for col_key, edges in focus_edges.items():
                if not edges:
                    focus_subgraphs[col_key] = {"nodes": [], "edges": []}
                    continue

                # 节点：包含自身以及所有边的另一端
                neighbor_ids = {col_key}
                for e in edges:
                    neighbor_ids.add(e["from"])
                    neighbor_ids.add(e["to"])

                nodes = [
                    n for n in base_nodes
                    if n["id"] in neighbor_ids
                ]

                focus_subgraphs[col_key] = {"nodes": nodes, "edges": edges}

            all_data[metric] = {
                "global": {
                    "nodes": base_nodes,
                    "edges": metric_edges_global,
                    "threshold": threshold,
                },
                "focus": focus_subgraphs,
            }

        return all_data

    @staticmethod
    def _edge_style(abs_value: float):
        """根据关联强度返回颜色和粗细。"""
        if abs_value >= 0.7:
            return "#ff4444", 8
        if abs_value >= 0.5:
            return "#ff8844", 6
        if abs_value >= 0.3:
            return "#ffaa44", 4
        return "#ffcc88", 2

    def generate_html(self) -> str:
        """生成 HTML 字符串。"""
        import json

        all_data = self._build_all_data()
        all_data_json = json.dumps(all_data, ensure_ascii=False)

        # 列选择下拉框 HTML
        column_options_html = '<option value="__all__">整体图谱（全部列）</option>\n'
        for key in self.column_keys:
            info = self.columns_info[key]
            column_options_html += f'<option value="{key}">{info["desc"]}</option>\n'

        # 方法选择下拉框 HTML
        metric_options_html = ""
        for m in self.metrics:
            if m not in all_data:
                continue
            name = self.metric_display_names.get(m, m)
            th = self.thresholds.get(m, 0.1)
            metric_options_html += f'<option value="{m}">{name} - 阈值: {th}</option>\n'

        # 阈值说明
        threshold_info_html = ""
        for m in self.metrics:
            if m not in all_data:
                continue
            name = self.metric_display_names.get(m, m)
            th = self.thresholds.get(m, 0.1)
            threshold_info_html += f'<li><strong>{m}:</strong> 绝对值小于 {th} 的边不显示</li>\n'

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>真实数据关联知识图谱</title>
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
            height: 650px;
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
        <h1>真实数据关联知识图谱</h1>
        <p>{os.path.basename(self.data_file)}</p>
    </div>

    <div class="controls">
        <div class="control-group">
            <label for="metricSelect">关联度方法:</label>
            <select id="metricSelect" onchange="switchMetric()">
                {metric_options_html}
            </select>
        </div>

        <div class="control-group">
            <label for="columnSelect">视角列:</label>
            <select id="columnSelect" onchange="switchColumn()">
                {column_options_html}
            </select>
        </div>

        <div class="control-group">
            <button onclick="resetView()">重置视图</button>
            <button onclick="fitView()">适应窗口</button>
        </div>

        <div class="stats" id="stats">
            请选择关联度方法与视角列。
        </div>
    </div>

    <div class="graph-container">
        <div id="network"></div>
    </div>

    <div class="info">
        <h3>使用说明:</h3>
        <ul>
            <li><strong>节点:</strong> 每个节点表示一列数据，节点标签为简短名称，悬浮提示显示完整名称。</li>
            <li><strong>整体图谱:</strong> 选择“整体图谱（全部列）”可查看所有列之间的关系。</li>
            <li><strong>视角模式:</strong> 选择某一列作为视角时，只显示该列与其它列的关联边。</li>
            <li><strong>边的颜色/粗细:</strong> 颜色越红、边越粗，表示关联越强。</li>
            <li><strong>操作:</strong> 拖拽节点移动布局，滚轮缩放，点击边可查看关联度数值。</li>
        </ul>

        <h3>阈值说明:</h3>
        <ul>
            {threshold_info_html}
        </ul>
    </div>

    <script>
        const allData = {all_data_json};
        let network = null;
        let currentMetric = Object.keys(allData)[0];
        let currentColumn = "__all__";

        function getCurrentData() {{
            const metricData = allData[currentMetric];
            if (!metricData) return null;

            if (currentColumn === "__all__") {{
                return metricData["global"];
            }} else {{
                return metricData["focus"][currentColumn] || {{"nodes": [], "edges": []}};
            }}
        }}

        function initNetwork() {{
            const container = document.getElementById('network');
            const data = getCurrentData();

            if (!data || data.nodes.length === 0) {{
                container.innerHTML = '<div style="text-align:center;padding:50px;">当前选择没有可显示的边或节点（可能所有边都低于阈值）。</div>';
                document.getElementById('stats').innerHTML =
                    `<strong>${{currentMetric}}</strong> | 视角: ${{currentColumn === "__all__" ? "整体图谱" : currentColumn}} | 没有数据`;
                return;
            }}

            const nodes = new vis.DataSet(data.nodes);
            const edges = new vis.DataSet(data.edges);
            const networkData = {{nodes: nodes, edges: edges}};

            const options = {{
                physics: {{
                    enabled: true,
                    stabilization: {{ iterations: 200 }},
                    barnesHut: {{
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 220,
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
                    }},
                    smooth: {{
                        type: 'continuous'
                    }}
                }}
            }};

            container.innerHTML = "";
            network = new vis.Network(container, networkData, options);
            updateStats();
        }}

        function switchMetric() {{
            currentMetric = document.getElementById('metricSelect').value;
            initNetwork();
        }}

        function switchColumn() {{
            currentColumn = document.getElementById('columnSelect').value;
            initNetwork();
        }}

        function resetView() {{
            if (network) {{
                network.setOptions({{ physics: {{ enabled: true }} }});
                setTimeout(() => {{
                    network.setOptions({{ physics: {{ enabled: false }} }});
                }}, 800);
            }}
        }}

        function fitView() {{
            if (network) {{
                network.fit();
            }}
        }}

        function updateStats() {{
            const data = getCurrentData();
            if (!data) {{
                document.getElementById('stats').innerHTML = '没有数据';
                return;
            }}

            const nodeCount = data.nodes.length;
            const edgeCount = data.edges.length;
            const th = (allData[currentMetric] && allData[currentMetric]["global"])
                ? allData[currentMetric]["global"]["threshold"] : 0;

            const colLabel = currentColumn === "__all__" ? "整体图谱" : currentColumn;
            document.getElementById('stats').innerHTML =
                `<strong>${{currentMetric}}</strong> | 视角: ${{colLabel}} | 节点: ${{nodeCount}} | 边: ${{edgeCount}} | 阈值: ${{th}}`;
        }}

        window.onload = function() {{
            initNetwork();
        }};
    </script>
</body>
</html>
"""

        return html

    def generate_visualization(self, output_dir: str = "KnowledgeGraph") -> str:
        """
        生成 HTML 文件。

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        # 例如 data_file = "DataAnalyze/realdata2_relationships.csv"
        # basename = "realdata2_relationships.csv" -> stem = "realdata2_relationships"
        # 去掉尾部的 "_relationships" 让文件名更简洁：KG_RealData_realdata2.html
        stem = os.path.splitext(os.path.basename(self.data_file))[0]
        suffix = "_relationships"
        base = stem[:-len(suffix)] if stem.endswith(suffix) else stem
        output_path = os.path.join(output_dir, f"KG_RealData_{base}.html")

        html = self.generate_html()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"真实数据知识图谱已生成: {output_path}")
        return output_path


if __name__ == "__main__":
    kg = RealDataKnowledgeGraphSystem("DataAnalyze/realdata4_relationships.csv")
    kg.generate_visualization()


