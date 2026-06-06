from __future__ import annotations

import re
from pathlib import Path


TEX = Path(__file__).resolve().parent / "main.tex"


def find_env(text: str, label: str) -> tuple[int, int]:
    pos = text.find(r"\label{" + label + "}")
    if pos < 0:
        raise ValueError(f"label not found: {label}")
    candidates: list[tuple[int, int]] = []
    for env in ("figure", "table"):
        begin_token = rf"\begin{{{env}}}"
        end_token = rf"\end{{{env}}}"
        begin = text.rfind(begin_token, 0, pos)
        end_pos = text.find(end_token, pos)
        if begin >= 0 and end_pos >= 0:
            candidates.append((begin, end_pos + len(end_token)))
    if not candidates:
        raise ValueError(f"environment not found for label: {label}")
    return max(candidates, key=lambda item: item[0])


def expand_multicols_gap(text: str, start: int, end: int) -> tuple[int, int]:
    end_tag = r"\end{multicols}"
    pre = text[:start]
    pre_pos = pre.rfind(end_tag)
    if pre_pos >= 0 and pre[pre_pos + len(end_tag) : start].strip() == "":
        start = pre_pos

    post = text[end:]
    match = re.match(r"\s*\\begin\{multicols\}\{2\}", post)
    if match:
        end += match.end()
    return start, end


def replace_label_group(text: str, labels: list[str], replacement: str, expand: bool = True) -> str:
    ranges = [find_env(text, label) for label in labels]
    start = min(item[0] for item in ranges)
    end = max(item[1] for item in ranges)
    if expand:
        start, end = expand_multicols_gap(text, start, end)
    return text[:start] + replacement.strip() + "\n\n" + text[end:].lstrip()


def replace_single_env(text: str, label: str, replacement: str) -> str:
    start, end = find_env(text, label)
    return text[:start] + replacement.strip() + "\n\n" + text[end:].lstrip()


TABLE1 = r"""
\begin{table}[H]
\centering
\small
\tabbicaption{PJM 2025 实验数据概况}{Overview of the PJM 2025 dataset}
\label{tab:data_overview}
\begin{tabularx}{\columnwidth}{lX}
\toprule
项目 & 数值或说明 \\
\midrule
数据文件 & \code{pjm\_rto\_hourly\_2025\_aligned\_processed\_one\_header.csv} \\
样本粒度 & 小时级，2025 年 8760 条记录 \\
原始字段 & 72 列，其中前 2 列为 UTC/EPT 时间字段 \\
建模字段 & 去除前两列时间字段后保留 70 个数值字段 \\
字段类型 & 发电、负荷、价格、辅助服务、交换功率等 \\
训练/测试划分 & 80\% / 20\% \\
\bottomrule
\end{tabularx}
\end{table}
"""

TABLE2 = r"""
\begin{table}[H]
\centering
\scriptsize
\setlength{\tabcolsep}{2pt}
\tabbicaption{净实际交换功率目标的多相关性初筛结果节选}{Partial screening results for net actual interchange}
\label{tab:screening_top}
\begin{tabularx}{\columnwidth}{cXrrrrc}
\toprule
排名 & 字段 & NMI & Spear. & Pear. & HSIC & 通过数 \\
\midrule
1 & 30 min 备用实际量 & 0.076 & -0.430 & -0.394 & 0.856 & 6 \\
2 & 30 min 备用总量 & 0.067 & -0.428 & -0.397 & 0.845 & 6 \\
3 & 总计划交换功率 & 0.133 & -0.465 & -0.483 & 0.813 & 6 \\
4 & 核电出力 & 0.116 & -0.354 & -0.424 & 0.795 & 6 \\
5 & 燃气出力 & 0.027 & -0.329 & -0.327 & 0.906 & 5 \\
6 & 30 min 备用需求 & 0.180 & -0.324 & -0.406 & 0.036 & 5 \\
7 & 主备用需求 & 0.190 & -0.327 & -0.402 & 0.067 & 5 \\
8 & 同步备用需求 & 0.177 & -0.324 & -0.405 & 0.065 & 5 \\
\bottomrule
\end{tabularx}
\end{table}
"""

FIG_TRAIN = r"""
\begin{figure}[H]
\centering
\begin{minipage}{0.49\columnwidth}
\centering
\includegraphics[width=\linewidth]{net_dnn_r2_zh.png}\\[-0.4em]
{\scriptsize (a) 全量 DNN}
\end{minipage}
\begin{minipage}{0.49\columnwidth}
\centering
\includegraphics[width=\linewidth]{net_l1_r2_zh.png}\\[-0.4em]
{\scriptsize (b) L1GateDNN}
\end{minipage}
\figbicaption{净实际交换功率目标上的 $R^2$ 训练过程对比}{Training $R^2$ processes for net actual interchange}
\label{fig:train_pair}
\end{figure}
"""

FIG_L1 = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{net_l1_gate_params_zh.png}
\figbicaption{净实际交换功率目标上门控参数 gate 的演化}{Gate values over epochs for net actual interchange}
\label{fig:l1_gate}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{net_l1_active_features_zh.png}
\figbicaption{净实际交换功率目标上活跃字段数量的演化}{Number of active features over epochs}
\label{fig:l1_active}
\end{figure}
"""

FIG_TOP_SERIES = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{net_top_series_r2_zh.png}
\figbicaption{净实际交换功率目标上的 Top-n 推断源验证}{Top-n inference-source validation}
\label{fig:top_series}
\end{figure}
"""

FIG_IMPROVED = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{improved_l1_gate_params_zh.png}
\figbicaption{改进 L1 门控的 gate 值演化}{Gate evolution of the improved L1 gate}
\label{fig:improved_gate}
\end{figure}

\begin{figure}[H]
\centering
\begin{minipage}{\columnwidth}
\centering
\includegraphics[width=0.92\linewidth]{improved_l1_w_meta_zh.png}\\[-0.4em]
{\scriptsize (a) 相关性指标权重 $W$}
\end{minipage}\\[-0.2em]
\begin{minipage}{\columnwidth}
\centering
\includegraphics[width=0.86\linewidth]{improved_l1_b_meta_zh.png}\\[-0.4em]
{\scriptsize (b) 偏置 $b$}
\end{minipage}
\figbicaption{改进门控中的指标权重与偏置演化}{Evolution of metric weights and bias}
\label{fig:improved_b}
\end{figure}
"""

FIG_DNN_L1_AND_TABLE = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{dnn_vs_l1_method_compare_zh.png}
\figbicaption{12 个敏感目标上的全量 DNN 与 L1DNN 对比}{Comparison between all-feature DNN and L1DNN}
\label{fig:dnn_l1_compare}
\end{figure}

\begin{table}[H]
\centering
\scriptsize
\setlength{\tabcolsep}{2pt}
\tabbicaption{全量 DNN 与 L1DNN 选中特征对比}{All-feature DNN and L1DNN-selected feature comparison}
\label{tab:dnn_l1_compare}
\begin{tabularx}{\columnwidth}{Xrrrr}
\toprule
中心目标 & 全量字段 & L1 字段 & 全量 DNN & L1DNN \\
\midrule
拥塞价 DA & 69 & 2 & 0.9408 & 0.9954 \\
主备用容量 & 67 & 14 & 0.8495 & 0.8826 \\
总实际交换 & 69 & 16 & 0.8499 & 0.8677 \\
30 min 备用容量 & 68 & 22 & 0.8967 & 0.9086 \\
边际损耗价 DA & 69 & 18 & 0.8936 & 0.8995 \\
净实际交换 & 67 & 18 & 0.9680 & 0.9728 \\
总发电量 & 42 & 16 & 0.9253 & 0.9271 \\
净计划交换 & 67 & 16 & 0.9691 & 0.9694 \\
总损耗 & 69 & 18 & 0.9413 & 0.9409 \\
计量负荷 & 42 & 15 & 0.9157 & 0.9139 \\
日前 LMP & 66 & 13 & 0.9622 & 0.9510 \\
拥塞价 RT & 69 & 18 & 0.7255 & 0.7040 \\
\midrule
平均 & -- & 15.5 & 0.9031 & 0.9111 \\
\bottomrule
\end{tabularx}
\end{table}
"""

BUDGET_FIG_TABLE = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{budget95_baseline_r2_zh.png}
\figbicaption{95\% 统一预算下不同方法的平均测试 $R^2$}{Average test $R^2$ under the 95\% feature budget}
\label{fig:budget95}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{budget97_baseline_r2_zh.png}
\figbicaption{97\% 统一预算下不同方法的平均测试 $R^2$}{Average test $R^2$ under the 97\% feature budget}
\label{fig:budget97}
\end{figure}

\begin{table}[H]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\tabbicaption{同预算 baseline 平均结果}{Average results of same-budget baselines}
\label{tab:budget_avg}
\textbf{(a) 95\% 统一预算}\\[-0.2em]
\begin{tabularx}{\columnwidth}{Xrrr}
\toprule
方法 & 字段数 & 利用率 & 平均 $R^2$ \\
\midrule
全量 DNN & 69.00 & 100.00\% & 0.9031 \\
L1GateDNN & 8.92 & 12.92\% & 0.8875 \\
NMI & 8.92 & 12.92\% & 0.6842 \\
Pearson & 8.92 & 12.92\% & 0.7419 \\
Spearman & 8.92 & 12.92\% & 0.7586 \\
Lasso & 8.92 & 12.92\% & 0.7967 \\
ElasticNet & 8.92 & 12.92\% & 0.8115 \\
RandomForest & 8.92 & 12.92\% & 0.8562 \\
XGBoost & 8.92 & 12.92\% & 0.7798 \\
\bottomrule
\end{tabularx}

\vspace{0.4em}
\textbf{(b) 97\% 统一预算}\\[-0.2em]
\begin{tabularx}{\columnwidth}{Xrrr}
\toprule
方法 & 字段数 & 利用率 & 平均 $R^2$ \\
\midrule
全量 DNN & 69.00 & 100.00\% & 0.9031 \\
L1GateDNN & 9.75 & 14.13\% & 0.8943 \\
NMI & 9.75 & 14.13\% & 0.7227 \\
Pearson & 9.75 & 14.13\% & 0.7454 \\
Spearman & 9.75 & 14.13\% & 0.7628 \\
Lasso & 9.75 & 14.13\% & 0.8071 \\
ElasticNet & 9.75 & 14.13\% & 0.8205 \\
RandomForest & 9.75 & 14.13\% & 0.8644 \\
XGBoost & 9.75 & 14.13\% & 0.7878 \\
\bottomrule
\end{tabularx}
\end{table}
"""

NOISE_FIG_TABLE = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{redundant_training_r2_curves_zh.png}
\figbicaption{全量字段与 L1 选中特征的训练 $R^2$ 曲线对比}{Training curves of all features and L1-selected features}
\label{fig:redundant_curve}
\end{figure}

\begin{table}[H]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\tabbicaption{冗余噪声解释实验的训练曲线统计}{Training-curve statistics for redundant-noise explanation}
\label{tab:noise_curve}
\textbf{(a) 拥塞价 DA}\\[-0.2em]
\begin{tabularx}{\columnwidth}{Xrrrr}
\toprule
方法 & 字段数 & Best $R^2$ & epoch & 泛化差距 \\
\midrule
全量 DNN & 69 & 0.9408 & 141 & 0.0621 \\
L1GateDNN & 69 & 1.0000 & 194 & -0.0001 \\
L1 选中特征 DNN & 2 & 0.9954 & 139 & 0.0025 \\
\bottomrule
\end{tabularx}

\vspace{0.4em}
\textbf{(b) 主备用容量}\\[-0.2em]
\begin{tabularx}{\columnwidth}{Xrrrr}
\toprule
方法 & 字段数 & Best $R^2$ & epoch & 泛化差距 \\
\midrule
全量 DNN & 67 & 0.8495 & 74 & 0.1439 \\
L1GateDNN & 67 & 0.8639 & 87 & 0.1170 \\
L1 选中特征 DNN & 35 & 0.8580 & 112 & 0.1247 \\
\bottomrule
\end{tabularx}
\end{table}
"""

TOPN_NOISE_FIG = r"""
\begin{figure}[H]
\centering
\includegraphics[width=\columnwidth]{redundant_topn_incremental_r2_zh.png}
\figbicaption{按 L1 gate 排名逐步增加字段数的测试 $R^2$}{Test $R^2$ with incrementally added Top-n features}
\label{fig:topn_noise}
\end{figure}
"""


def main() -> None:
    text = TEX.read_text(encoding="utf-8")

    start, end = find_env(text, "tab:data_overview")
    next_section = text.find(r"\subsection{参数设置}", end)
    if next_section < 0:
        raise ValueError("next section after dataset table not found")
    text = text[:start] + TABLE1.strip() + "\n\n" + text[next_section:]

    text = replace_single_env(text, "tab:screening_top", TABLE2)
    text = replace_single_env(text, "fig:train_pair", FIG_TRAIN)
    text = replace_label_group(text, ["fig:l1_gate", "fig:l1_active"], FIG_L1)
    text = replace_label_group(text, ["fig:top_series", "tab:top_series"], FIG_TOP_SERIES)
    text = replace_label_group(text, ["fig:improved_gate", "fig:improved_b"], FIG_IMPROVED)
    text = replace_label_group(text, ["fig:dnn_l1_compare", "tab:dnn_l1_compare"], FIG_DNN_L1_AND_TABLE)
    text = replace_label_group(text, ["fig:budget95", "fig:budget97", "tab:budget_avg"], BUDGET_FIG_TABLE)
    text = replace_label_group(text, ["fig:redundant_curve", "tab:noise_curve"], NOISE_FIG_TABLE)
    text = replace_label_group(text, ["fig:topn_noise", "tab:topn_noise"], TOPN_NOISE_FIG)

    text = text.replace(
        r"图~\ref{fig:topn_noise} 和表~\ref{tab:topn_noise} 显示",
        r"图~\ref{fig:topn_noise} 显示",
    )
    text = text.replace(
        r"图~\ref{fig:topn_noise} 和表~\ref{tab:topn_noise} 显示",
        r"图~\ref{fig:topn_noise} 显示",
    )
    text = text.replace("风险容忍偏置 $b$", "偏置 $b$")
    text = text.replace("风险偏置演化", "偏置演化")
    text = text.replace("risk bias", "bias")
    text = text.replace("softmax", "原始")

    TEX.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
