# Agent 2 报告：数据库函数依赖/UCC/拟标识符枚举（已完成）

## 最关键可用结论（供综合方案引用）

1. **问题同构**：极小达标集枚举 ≡ 近似FD左部/近似UCC枚举。F单调 ↔ FD单调性。正边界(极小达标族M) ↔ 负边界(极大非达标族N)，两者互为极小碰集对偶（Mannila-Toivonen borders; Fredman-Khachiyan）。
2. **成本锚定边界族而非格大小**：现代算法(DFD/DUCC/Pyro/MARCO)成本 ∝ |M|+|N| × 单次验证成本，不是 2^p。
3. **推荐合成骨架**：MARCO 管完备性/终止/anytime；种子从"小端"ascend（Pyro 式，规避 MARCO 大种子 shrink 浪费）；代理模型排序+单调界传播过滤候选；序贯多 seed 统计检验只花在阈值带；输出 (M,N) 双边界。
4. **双 oracle 架构**（Pyro 核心可迁移）：便宜估计器(代理模型/子采样少epoch) + 昂贵验证器(完整DNN)；只有估计为"落在极小边界上"的候选才训完整模型。
5. **shrink 优化**：贪心 O(p) 次；QuickXplain/二分删除 → O(|S*|·log p)；删除顺序用代理边际贡献排序。
6. **单调性区间传播**：S⊆S' ⟹ R²(S)≤R²(S')，用 trie 维护已测界，免训判定。
7. **剪枝置信化**：只有高置信达标才剪超集、高置信非达标才剪子集，阈值带挂起（TANE 近似版削弱剪枝的教训）。
8. **TIE\*（JMLR 2013）**：ML 侧唯一系统研究枚举所有极小最优特征集的工作；"移除-重学+统计等价检验"，现有主路径方法可最小改动升级为 TIE* 实例。
9. **风险清单**：①近似阈值下解族爆炸(Pyro: 35列10.8万条)→需预算+排序输出+负边界证书；②噪声打破单调性→只能"高概率完备"；③真F可能不单调→用单调闭包定义 F(S)=1 iff ∃S'⊆S 达标；④τ=0.95·R̂²_full 自身有方差→多seed钉死+敏感性分析；⑤近似语义剪枝弱；⑥无廉价负证书（HyFD 的元组对证据无对应物）。

## 关键文献
- TANE: Huhtala et al., Computer Journal 1999（逐层+rhs候选集剪枝+近似g3）
- FastFDs: Wyss et al. DaWaK 2001（差集→极小碰集归约）
- DFD: Abedjan et al. CIKM 2014（随机游走+正负边界+碰集补种子）
- HyFD: Papenbrock & Naumann SIGMOD 2016（便宜负例采样+昂贵验证混合切换）
- Pyro: Kruse & Naumann PVLDB 2018（近似阈值+抽样估计+ascend/trickle-down+碰集三用）★最同构
- DUCC: Heise et al. PVLDB 2013; HyUCC: BTW 2017; GORDIAN: VLDB 2006（先枚举极大非键再对偶化）
- Motwani & Xu VLDB 2007（最小拟标识符 NP-hard+贪心近似+抽样保证）; Hildebrant et al. PODS 2023（Θ(m/√ε)样本）
- MARCO: Liffiton et al. Constraints 2016（map-solver+grow/shrink，anytime+完备）
- Fredman-Khachiyan 1996（单调对偶化拟多项式）; Bioch & Ibaraki 1995（membership query 学单调布尔函数）
- Kivinen & Mannila 1995（g1/g2/g3误差度量）; Caruccio et al. TKDE 2016（宽松FD综述）
- 开源：Metanome（Java，HPI）、Desbordante（C++/Python，快）、openclean-metanome（Python封装）、CPMpy marco 工具

（完整报告见对话记录 task ae25e6805a8d27dcf）
