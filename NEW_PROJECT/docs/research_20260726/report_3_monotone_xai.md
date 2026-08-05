# Agent 3 报告：单调布尔枚举/超图对偶/MUS/形式化XAI（已完成）

## 最关键可用结论

1. **精确映射表**：推断路径 = minimal true point = MUS = AXp/PI-explanation/sufficient reason；极大不达标集 = maximal false point = MSS；极小阻断集 = MCS = CXp。
2. **对偶定理**（Ignatiev et al. AIxIA 2020，源头 Reiter 1987）：AXp 族与 CXp 族互为极小命中集（MHS）。→ 治理问题"屏蔽哪些字段阻断所有推断"的答案就是 CXp 枚举，**不必先枚举所有路径再解命中集**。
3. **ICML 2021 "Explanations for Monotonic Classifiers"（Marques-Silva et al., PMLR 139:7469）几乎就是我们问题的原型**：
   - findAXp/findCXp：单个极小集只需 2N 次 oracle 调用，无需模型内部信息；
   - Algorithm 3 枚举：SAT 调用总数 = |AXp|+|CXp|+1（单 SAT 调用版 MARCO）；sound & complete（Thm 4）；
   - 官方代码 XMono (https://git.io/JZZBX)，把 κ 换成"训DNN测R²"即可。
   - Thm 3：判定是否存在第 ⌊N/2⌋+1 个 AXp 是 NP-complete。
4. **查询复杂度下界**（Mannila-Toivonen/PODS'97/Bioch-Ibaraki）：任何 membership-query 算法必须 ≥ |路径族P| + |极大不达标族B| 次查询；预算讨论必须围绕 P+B。
5. **implicit hitting set 循环求最优屏蔽方案**（Reiter 1987/MaxHS 范式）：解当前路径族最小命中集 H（ILP秒级）→ 测 F(U∖H)：=0 则 H 最优阻断证毕；=1 则 shrink 出新路径加入，循环。每轮 1+≤50 次 oracle，无需完整枚举。
6. **QuickXplain 分治 shrink**（Junker AAAI 2004）：单路径 oracle 数从 ≤50 降到 O(k log(50/k))≈10-25 次。
7. **Dualize & Advance**（Gunopulos et al. PODS'97）：维护极大不达标族 C → 对偶化得极小候选 T → 逐个测 F，反例扩张进 C；终止时同时得两族。对偶化用 shd（Uno）纯计算毫秒级。
8. **工程建议**：memoization+单调闭包推理省30-70%调用；两级oracle（LightGBM代理+边界带才训DNN）；k折CV+置信区间，CI跨τ加倍重复（best-arm identification式自适应）；灰区[τ-ε,τ+ε]单独报告不参与对偶推理；按尺寸从小到大枚举CXp（OptUx思路，小屏蔽方案先出，anytime）。
9. **坑**：①经验F不严格单调→显式单调修复（闭包投票）+记录违例率；②τ敏感→2-3个τ各跑或复用λ路径扫描分级；③组合爆炸无理论保底→部分枚举+覆盖率交付；④组合阻断≠统计阻断（表述写"压至R²<τ"而非"不可推断"）；⑤翻转噪声MQ无实用正面理论，重复查询+置信带是工程惯例需标注；⑥FK/shd只省计算不省查询。

## 关键文献
- Marques-Silva et al. ICML 2021 (monotonic classifiers explanations) ★核心
- Ignatiev et al. AIxIA 2020 (AXp/CXp MHS 对偶); Shih/Choi/Darwiche IJCAI 2018 (PI-explanation); Ignatiev AAAI 2019 (abduction-based)
- Liffiton et al. Constraints 2016 (MARCO); Previti & Marques-Silva AAAI 2013 (eMUS)
- Reiter AIJ 1987 (HS-tree); Ignatiev et al. ECAI 2016 (implicit hitting sets)
- Fredman-Khachiyan JAlg 1996; Eiter-Gottlob SICOMP 1995; Eiter-Makino-Gottlob DAM 2008 (综述); Murakami-Uno ALENEX 2013 (MMCS/RS→shd)
- Gunopulos et al. PODS 1997 (Dualize & Advance + 查询下界); Angluin ML 1988; Angluin-Slonim ML 1994 (incomplete MQ); Bshouty-Eiron JMLR 2002
- Gainer-Dewar & Vera-Licona SIDMA 2017 (MHS综述)
- 开源：MARCO(github.com/liffiton/MARCO)、PySAT(musx/mcsls/optux/Hitman)、shd(Uno)、XMono(ICML21官方)、PyXAI

（完整报告见 task aed1a0d2b0c214812）
