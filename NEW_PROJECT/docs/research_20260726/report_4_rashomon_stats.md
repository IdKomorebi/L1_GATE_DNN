# Agent 4 报告：Rashomon/统计误差控制（已完成）★含对我方案的关键纠错

## ★最关键：我原方案 τ=LCB(R²_full)−ε 是错的

**问题**：R̂²(S) 与 LCB(R̂²_full) 在同一份数据上估计、高度正相关，分开处理丢掉了相关性，检验水平既非名义 α 也无方向保证。

**正确做法——直接对差值做推断**：
- ψ(S) := R²_full − R²(S)，**F(S)=1 ⟺ ψ(S) ≤ ε**
- 这与 Williamson et al. (JASA 2023) 的 oracle predictiveness 参数一比一对应（他们的 f₀,ₛ 就是"重训后的 oracle"，正好是我们的重训练语义，不是 mask）
- 差值的影响函数吸收了两项相关性

**假设方向必须翻转（当前 0.95 规则的本质错误）**：
- H₀: ψ(S) ≥ ε（性能损失不可接受）；H₁: ψ(S) < ε（非劣）
- **拒绝 H₀ 才判 F(S)=1**。证据不足默认判 0；被 α 控制的 Type I error 恰是"错误宣布坏子集合格"——正是要防的错。
- 当前做法是把非劣放在 H₀、不拒绝就接受 = 典型的"用不显著证明等价"。

## 可实现的七步检验（Williamson 路线）
1. K 折交叉拟合（K=5/10）；**时序数据必须 blocked/rolling-origin**；每折评估集切 A、B 两半且训练不共用（sample splitting 对抗 ψ=0 退化）
2. A 的训练部分训 f̂_full，B 的训练部分训 f̂_S
3. 评估：**分母固定用全体 Var(Y)**（不可每子集用各自方差，否则 R² 不可比）
4. ψ̂(S) = v̂_full − v̂_S
5. 方差：φ̂_full(zᵢ) = (1/σ̂²)·[−(yᵢ−f̂_full(xᵢ))² + (1−v̂_full)·(yᵢ−ȳ)²]，η̂²=mean(φ̂²)；ω̂(S) = η̂²_full/n_A + η̂²_S/n_B
6. T(S) = (ε − ψ̂(S))/sqrt(ω̂(S))；T > z_{1−α} 判 F(S)=1。等价：UCB_{1−α}(ψ) < ε。p 值 = 1−Φ(T)
7. ψ 远离 0 时可用不分裂联合 IF（方差更小），但退化方向偏"更易判合格"=危险方向，默认走分裂版

## ε 的定标（三选一，按推荐度）
1. **应用效用定标（首选）**：由下游决策可接受损失决定（电力：映射到调度/备用容量成本）。临床非劣性 margin 标准做法——margin 必须领域知识定，不能数据定
2. **噪声地板定标**：ε = c·ŝe(R̂²_full)，c∈{1,2}；随 n 自动收缩；须在独立定标子集算 ŝe 后冻结
3. **兼容旧口径**：ε = 0.05·LCB_α(R²_full)，LCB 独立数据上算好并冻结
辅判据：R²(S) 的 LCB > 0（防 R²_full 本身小时"人人非劣"退化）

## 多重比较（成百上千子集）——强烈推荐第 1 条
1. **搜索/确认分离（最干净）**：搜索集跑全部枚举剪枝、完全不做检验不报 p 值；只在独立确认集上对最终入围的 ≤10 个候选做检验 + Bonferroni/Holm。把多重比较从上千降到个位数，规避选择后推断
2. BH 控制"错误非劣发现率"；嵌套子集通常满足 PRDS，BH 可用，不确定用 BY
3. **利用单调性做 closed testing**：S⊂S′ ⟹ ψ(S)≥ψ(S′)，沿单调链传播判定，不额外付 α 控 FWER。搜索中反复判定改用 e-value/anytime-valid 或 LORD 在线 FDR

## masked surrogate 配方（若 V3 要用）
- **必须把 mask m 作显式输入** g_θ(x⊙m, m)，否则分不清"值为0"和"缺失"
- **必须对标签 y 训练，不能蒸馏 f_full**！蒸馏得 E[f_full(X)|X_S]，天花板被 f_full 偏差锁死；对 y 训得 E[Y|X_S] 才是重训 oracle
- mask 分布：先抽基数 k~Uniform{0..d} 再均匀抽 S；**不要 Bernoulli(0.5)**（基数集中 d/2，小子集没样本）
- 校准不可省：独立校准集上 30-100 个子集真重训，拟合 v_retrain ≈ a+b·v_surrogate，σ²_cal 加进 ω̂(S)
- 失效模式：amortization gap、mask 分布不匹配（静默失效）、off-manifold、数据泄漏、蒸馏混淆、R²分母口径、隐式正则化差异

## 其他要点
- **MCR (Fisher-Rudin-Dominici JMLR 2019)**：ε-Rashomon 集 R(ε)={f: EL(f)≤EL(f_ref)+ε}（相对 f_ref 的绝对损失增量）；MCR±=MR 在集合上的极值；**论文不给 ε 选取规则**（最大空白）；界含覆盖数 N(F,r) 很松→定位为敏感性分析工具而非判定工具
- **Stability Selection (M&B 2010)**：E(V) ≤ q_Λ²/((2π_thr−1)p)，需可交换性；电力强相关特征几乎必然不成立→用 Shah-Samworth CPSS（stabs 包）
- **HRT (Tansey)**：训一次+holdout 重采样，只前向不重训；但**是逐特征条件独立检验，不能直接判 F(S)**（子集级性能非劣性），属流水线不同环节
- **CV 检验**：朴素配对 CV t 高估显著性（折间不独立，Type I 膨胀到 10-30%）；Nadeau-Bengio 校正 = 1/K → 1/K + n₂/n₁；Alpaydin 5×2cv F ~ F(10,5)；**Bengio & Grandvalet 2004 证 K 折 CV 方差无无偏估计**→主线走 Williamson 影响函数（有渐近理论保证），5×2cv 只当旁证
- **R² omnibus 非劣性**：Campbell arXiv:2002.08476 + Campbell & Lakens 2020，反演 scaled central F 近似的单侧 CI；线性+正态假设，方向也相反；技术路线可搬，DNN 走 Williamson

## 开源
vimp (R, CRAN, bdwilliamson) `vim()/cv_vim()/sp_vim()` type="r_squared" sample_splitting=TRUE，有 delta 参数（优效方向，非劣需自己翻转）；vimpy (Python)；stabs (R, CPSS)；knockpy；tansey/hrt；dCRT (比HRT更快更有力)；aaronjfisher/mcr；TreeFARMS/TimberTrek；TOSTER；mlxtend.evaluate (paired_ttest_5x2cv, combined_ftest_5x2cv)；fastshap/sage/removal-explanations

## 头号坑
时序泄漏（电力场景）——所有 CV/子采样/mask训练/holdout 必须 blocked/rolling-origin/purged-embargo，否则所有 R²/CI/p 值全失效且方向全面乐观。

（完整报告见 task ac2c4d80f4cb9d822）
