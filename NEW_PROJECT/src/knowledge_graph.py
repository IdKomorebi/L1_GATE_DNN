from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .data_utils import ensure_dir


def _edge_columns(df: pd.DataFrame) -> tuple[str, str]:
    if {"column_a", "column_b"}.issubset(df.columns):
        return "column_a", "column_b"
    if {"center", "related"}.issubset(df.columns):
        return "center", "related"
    raise ValueError("Relationship table must contain column_a/column_b or center/related columns.")


def _best_metric(row: pd.Series, metrics: Sequence[str]) -> tuple[str, float]:
    best_name = ""
    best_value = 0.0
    for metric in metrics:
        if metric not in row:
            continue
        value = row.get(metric)
        if pd.isna(value):
            continue
        value = float(value)
        if abs(value) > abs(best_value):
            best_name = metric
            best_value = value
    return best_name, best_value


def _node_levels(edges: list[dict], center: str | None) -> Dict[str, int]:
    if center:
        return {center: 0, **{edge["target"]: 1 for edge in edges}}
    levels: Dict[str, int] = {}
    for edge in edges:
        levels.setdefault(edge["source"], 0)
        levels.setdefault(edge["target"], 0)
    return levels


def generate_knowledge_graph(
    relationships: pd.DataFrame,
    output_html: str | Path,
    metrics: Sequence[str],
    thresholds: Dict[str, float],
    title: str,
    center: str | None = None,
    min_pass_count: int = 1,
) -> Path:
    source_col, target_col = _edge_columns(relationships)
    edge_rows = relationships.copy()
    if "pass_count" in edge_rows.columns:
        edge_rows = edge_rows[edge_rows["pass_count"] >= min_pass_count]

    edges = []
    nodes = set()
    for _, row in edge_rows.iterrows():
        source = str(row[source_col])
        target = str(row[target_col])
        best_name, best_value = _best_metric(row, metrics)
        metric_values = {
            metric: None if metric not in row or pd.isna(row[metric]) else float(row[metric])
            for metric in metrics
        }
        pass_count = int(row.get("pass_count", 0)) if not pd.isna(row.get("pass_count", 0)) else 0
        max_abs = float(row.get("max_abs_score", np.nan)) if not pd.isna(row.get("max_abs_score", np.nan)) else 0.0
        edges.append(
            {
                "source": source,
                "target": target,
                "pass_count": pass_count,
                "max_abs_score": max_abs,
                "best_metric": best_name,
                "best_value": best_value,
                "metrics": metric_values,
            }
        )
        nodes.add(source)
        nodes.add(target)

    node_levels = _node_levels(edges, center)
    nodes_data = [
        {
            "id": node,
            "label": node,
            "level": node_levels.get(node, 0),
            "is_center": bool(center and node == center),
        }
        for node in sorted(nodes)
    ]

    payload = {
        "title": title,
        "nodes": nodes_data,
        "edges": edges,
        "metrics": list(metrics),
        "thresholds": thresholds,
        "center": center,
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #17202a; background: #f7f8fa; }}
    header {{ padding: 14px 18px; background: #ffffff; border-bottom: 1px solid #d9dee7; }}
    h1 {{ margin: 0 0 8px; font-size: 18px; font-weight: 700; }}
    .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; font-size: 13px; }}
    select, input {{ height: 28px; border: 1px solid #c8d0dc; border-radius: 4px; padding: 0 8px; background: white; }}
    #stats {{ color: #5b6573; }}
    #wrap {{ display: grid; grid-template-columns: minmax(680px, 1fr) 360px; height: calc(100vh - 78px); }}
    #graph {{ width: 100%; height: 100%; background: #ffffff; }}
    #side {{ border-left: 1px solid #d9dee7; background: #ffffff; overflow: auto; padding: 12px; }}
    .edge {{ stroke: #7f8fa6; stroke-opacity: 0.36; }}
    .edge.strong {{ stroke-opacity: 0.74; }}
    .node {{ stroke: #ffffff; stroke-width: 1.5px; cursor: pointer; }}
    .label {{ font-size: 10px; fill: #263241; pointer-events: none; }}
    .center {{ fill: #d64545; }}
    .normal {{ fill: #278f83; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid #edf0f5; padding: 6px 4px; text-align: left; }}
    th {{ position: sticky; top: 0; background: #ffffff; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="controls">
      <label>Metric <select id="metric"></select></label>
      <label>Min abs score <input id="minScore" type="number" value="0" min="0" max="1" step="0.01"></label>
      <label>Search node <input id="search" type="text" placeholder="column name"></label>
      <span id="stats"></span>
    </div>
  </header>
  <div id="wrap">
    <svg id="graph"></svg>
    <aside id="side"><table id="edgeTable"></table></aside>
  </div>
  <script>
    const payload = {json.dumps(payload, ensure_ascii=False)};
    const svg = document.getElementById("graph");
    const metricSel = document.getElementById("metric");
    const minScore = document.getElementById("minScore");
    const search = document.getElementById("search");
    const stats = document.getElementById("stats");
    const edgeTable = document.getElementById("edgeTable");

    metricSel.innerHTML = '<option value="any">any passed metric</option>' +
      payload.metrics.map(m => `<option value="${{m}}">${{m}}</option>`).join("");

    function filteredEdges() {{
      const metric = metricSel.value;
      const min = Number(minScore.value || 0);
      const q = search.value.trim().toLowerCase();
      return payload.edges.filter(e => {{
        const metricOk = metric === "any"
          ? e.pass_count > 0
          : e.metrics[metric] !== null && Math.abs(e.metrics[metric]) >= (payload.thresholds[metric] || 0);
        const scoreOk = Math.abs(metric === "any" ? e.max_abs_score : (e.metrics[metric] || 0)) >= min;
        const searchOk = !q || e.source.toLowerCase().includes(q) || e.target.toLowerCase().includes(q);
        return metricOk && scoreOk && searchOk;
      }}).sort((a, b) => b.max_abs_score - a.max_abs_score);
    }}

    function layout(nodes, width, height) {{
      const pos = {{}};
      if (payload.center) {{
        pos[payload.center] = {{x: width / 2, y: height / 2}};
        const others = nodes.filter(n => n.id !== payload.center);
        const r = Math.min(width, height) * 0.38;
        others.forEach((n, i) => {{
          const angle = 2 * Math.PI * i / Math.max(others.length, 1) - Math.PI / 2;
          pos[n.id] = {{x: width / 2 + r * Math.cos(angle), y: height / 2 + r * Math.sin(angle)}};
        }});
      }} else {{
        const r = Math.min(width, height) * 0.42;
        nodes.forEach((n, i) => {{
          const angle = 2 * Math.PI * i / Math.max(nodes.length, 1) - Math.PI / 2;
          pos[n.id] = {{x: width / 2 + r * Math.cos(angle), y: height / 2 + r * Math.sin(angle)}};
        }});
      }}
      return pos;
    }}

    function render() {{
      const width = svg.clientWidth || 900;
      const height = svg.clientHeight || 650;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = "";

      const edges = filteredEdges();
      const used = new Set();
      edges.forEach(e => {{ used.add(e.source); used.add(e.target); }});
      const nodes = payload.nodes.filter(n => used.has(n.id));
      const pos = layout(nodes, width, height);

      edges.forEach(e => {{
        if (!pos[e.source] || !pos[e.target]) return;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", pos[e.source].x);
        line.setAttribute("y1", pos[e.source].y);
        line.setAttribute("x2", pos[e.target].x);
        line.setAttribute("y2", pos[e.target].y);
        line.setAttribute("class", "edge" + (e.pass_count >= 3 ? " strong" : ""));
        line.setAttribute("stroke-width", String(0.7 + Math.min(5, e.max_abs_score * 5)));
        line.innerHTML = `<title>${{e.source}} -> ${{e.target}}\\n${{e.best_metric}}=${{e.best_value.toFixed(4)}}\\npass_count=${{e.pass_count}}</title>`;
        svg.appendChild(line);
      }});

      nodes.forEach(n => {{
        const p = pos[n.id];
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", p.x);
        circle.setAttribute("cy", p.y);
        circle.setAttribute("r", n.is_center ? 13 : 8);
        circle.setAttribute("class", "node " + (n.is_center ? "center" : "normal"));
        circle.innerHTML = `<title>${{n.id}}</title>`;
        svg.appendChild(circle);

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.setAttribute("x", p.x + 10);
        label.setAttribute("y", p.y + 4);
        label.setAttribute("class", "label");
        label.textContent = n.label.length > 30 ? n.label.slice(0, 30) + "..." : n.label;
        svg.appendChild(label);
      }});

      stats.textContent = `${{nodes.length}} nodes, ${{edges.length}} edges`;
      edgeTable.innerHTML = "<tr><th>source</th><th>target</th><th>best</th><th>score</th></tr>" +
        edges.slice(0, 300).map(e =>
          `<tr><td>${{e.source}}</td><td>${{e.target}}</td><td>${{e.best_metric}}</td><td>${{e.best_value.toFixed(4)}}</td></tr>`
        ).join("");
    }}

    metricSel.addEventListener("change", render);
    minScore.addEventListener("input", render);
    search.addEventListener("input", render);
    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>
"""

    output_html = Path(output_html)
    ensure_dir(output_html.parent)
    output_html.write_text(html, encoding="utf-8")
    return output_html
