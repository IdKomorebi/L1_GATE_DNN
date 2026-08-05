# Agent 5 报告：PID/高阶交互/协同效应（已完成）

## 最关键可用结论

1. **"替代组不可组合"的信息论根源**：C 补 A 的能力若主要落在 Syn(C,B;y) 协同原子上（I(C;y)≈0 但 I(C;y|B)≫0），则撤走 B 该原子蒸发。XOR 是极端例子。这不是 bug 而是数据的固有结构。
2. **组合缺陷量 D_ij**（可作为论文新概念）：
   D_ij = [v(P∖i∪Q_i)−v(P)] + [v(P∖j∪Q_j)−v(P)] − [v(P∖{i,j}∪Q_i∪Q_j)−v(P)]
   替代不可组合 ⟺ D_ij 显著为正。两个来源：(a) 协同型失效（Q_i 中的 c 与 x_j 是超模对 Δ_{c,x_j}v>0）；(b) 冗余型失效（Red(Q_i,Q_j;y) 大，两组从同一外部信息源取水）。
3. **可组合充分条件**：跨对无超模交互 + 组间无冗余重叠 ≈ v 局部子模且替代组信息近正交。理论保证（submodularity ratio, Das & Kempe ICML 2011; Elenberg AoS 2018）恰在协同/压制变量场景下失效（γ→0）→ 无免费定理，只能检测。
4. **"compositionality of feature substitutes" 是文献空白** → 替代依赖图 + 组合缺陷量是我们的新颖性主张。
5. **替代依赖图 H**：节点=字段，边 Q_i→B 表示 Q_i 的替代有效性依赖上下文字段 B（判定量 Δ_i(B) = v(T∪Q_i)−v((T∖B)∪Q_i) 对比基线）。多缺失集 M 可行性 → 图上可达性/冲突检查：(i) dep(Q_i)∩M=∅；(ii) M 内每对组间冗余核查。把 2^|M| 排列组合坍缩为图判定。
6. **三层检测漏斗**（预算从 C(m,2)×3 压到 O(m) 次重训练）：
   - 零成本代理层：高斯 copula O-information Ω(c,B,y) 及梯度 ∂Ω（HOI包，全部19600个三元组分钟级）、NID（ICLR 2018，从已训DNN首层权重解析交互）、EBM/FAST 成对交互、MMI-PID Syn 原子（Barrett 2015 闭式）→ 50×50 协同热力图 + 每字段"协同枢纽度"
   - 代理模型层：LightGBM 作 v̂(S)（秒级），30-50 个随机子集拟合校准曲线
   - 精测层：只对 τ±margin 不确定带的对做 DNN 重训练（3 seeds+CV CI）
7. **协同枢纽度高的字段 = 治理上的"关键在场字段"**，单点管控收益最大。
8. **掩码博弈 ≠ 重训练博弈**：NID/shapiq/IH 解释"当前模型"，只能作优先级先验；达标性结论必须落在重训练 R²。我们的 v(S)=重训练 R² 对应"数据里有多少信息"——分布类量（PID/O-info/CMI）天然对应此语义。
9. **坑**：多源 PID 无共识（只在二源粒度用+两种定义交叉验证）；Ω 是净量会抵消（≈0≠无结构）；高斯 copula 抓不住非单调协同；确定性台账关系使 MI 饱和且正是多 Markov 边界成因；阈值化丢信息（中间计算保留连续 R²，只在最终裁决用 τ+margin）。

## 关键文献
- Williams & Beer 2010 (PID奠基); Bertschinger et al. Entropy 2014 (BROJA); Barrett PRE 2015 (Gaussian MMI-PID闭式); Pakman et al. NeurIPS 2021 (连续神经UI估计)
- Rosas et al. PRE 2019 (O-information); Scagliarini et al. PRR 2023 (O-info梯度→协同定位)
- Grabisch & Roubens 1999 (Shapley interaction); Sundararajan ICML 2020 (Shapley-Taylor); Tsai et al. JMLR 2023 (Faith-Shap)
- Tsang ICLR 2018 (NID); Tsang NeurIPS 2020 (Archipelago); Lou et al. KDD 2013 (GA2M/FAST); shapiq NeurIPS 2024
- Das & Kempe ICML 2011 (submodularity ratio); Elenberg et al. AoS 2018 (RSC⟹弱子模); Kelso-Crawford 1982 & Lehmann² & Nisan (gross substitutes 层级)
- CMICOT NeurIPS 2016; RelaxMRMR PR 2016
- 开源：HOI(JAX)、dit、BROJA_2PID、shapiq、NID/Archipelago、interpret(EBM)、CMICOT、IDTxl

（完整报告见 task a6fe829c540cf8c8e）
