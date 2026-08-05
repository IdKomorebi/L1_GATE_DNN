# Agent 6 报告：电力隐私推断应用与顶刊定位（已完成）

## 最关键可用结论

1. **电力领域没有近邻，"哪些字段能一起发"是真空**：电力侧只有"能不能推断"（NILM画像、LMP→拓扑/成本）和"加噪怎么发"（Fioretto/Van Hentenryck/Dvorkin 的 DP-OPF 学派）两端。制度侧（FERC CEII、电力交易数据分类分级团标、CEEDS）只给定性清单无量化工具 → Introduction 动机来源。
2. **真正的近邻在四支文献**（Related Work 必须主动对标+差异表）：
   - D2 Pappachan et al. VLDB 2022 "Preventing Inferences through Data Dependencies"（full deniability，最小额外抑制）——方法论最像；差异：他们逻辑/精确FD+单元格级，我们统计oracle+字段级+可替代性分级+不相交路径
   - C3 Huang/Marques-Silva "Feature Necessity & Relevancy"——necessary/relevant/irrelevant ↔ 不可替代/可替代/无关一一对应，**最大新颖性风险源**；差异：他们实例级AXp+形式化模型族，我们数据集级+黑盒回归+发布策略
   - C5 TIE* + C6 SES/MXM——多极小子集枚举同骨架；差异：他们无对偶阻断、无不相交性判定、无可替代性图
   - C1 Datta et al. CCS 2017 "Use Privacy"（proxy use 检测）——差异：白盒程序分析、无极小性无全枚举
3. **新颖性三条线**：①问题形式化新组合（统计oracle+全枚举+对偶阻断+分级+不相交路径存在性，没有工作同时具备）；②"不相交推断路径数 ν(H)"（leakage redundancy / inference multiplicity）是文献中无对应物的招牌指标；③电力域首次落地（定性清单→可执行判定）。
4. **投稿**：🥇IEEE TSG（故事最顺）；🥈TKDE（须补完备性证明+复杂度+对偶定理）；🥉Applied Energy（政策落点+下游危害链条）；❌TIFS 首投（无可证明保证会被拒，需max-over-model-zoo+DP关系讨论）。
5. **致命审稿风险 R1**："你发现的路径就是功率平衡恒等式，是常识"→ 必须分层报告：(i)物理恒等类（作方法正确性验证）(ii)统计非平凡类（卖点），量化(ii)占比。不处理 TSG 大概率一审拒。
6. **P0 必做实验**：①imputation/边缘分布基线（Jayaraman & Evans CCS'22：ΔR²=R²(子集)−R²(仅边缘)）；②小规模(≤12-15字段)全穷举验证枚举 recall/precision；③多模型族 oracle（Ridge/GBDT/RF/DNN 取 max）；④R² bootstrap CI+置换检验+τ敏感性曲线。
7. **P1**：基线对照（SES/TIE*/Boruta/knockoff/SHAP-topk/SIS）；阻断集对比（贪心/ILP/删SHAP-topk/DP加噪）画隐私-效用Pareto；外部辅助知识鲁棒性（气象/日历/价格旁道）；时间切分验证（防temporal leakage指控）。
8. **可搬指标**：RAPID 相对误差容差命中率 P(|ŷ−y|/|y|≤δ)（比裸R²可辩护）；Anonymeter 攻击优势 (main−control)/(1−control)；knockoff FDR；稳定性指数。
9. **自定义新指标**：ν(H) 不相交路径数；Minimum Blocking Cost（加权最小击中集）；Irreplaceability Index；Substitution Group Coherence；Utility Retention @ Blocked。
10. **术语对齐**：推断路径→inference channel (Farkas & Jajodia 2002)；不可替代→necessary feature；替代组→statistically equivalent signature；屏蔽组合→blocking set/minimal transversal/full deniability。
11. **战术优势**：我们的 y 是系统机密量而非个人属性，绕开 B1 "只是imputation"批评射程。
12. **中文空位**：国内电力数据安全全在同态/MPC/联邦/DP，无字段组合级判定 → 可并行发中文版（电力系统自动化/中国电机工程学报），映射到分类分级标准动作。

（完整报告见 task ae3b73031c6f58e6a）
