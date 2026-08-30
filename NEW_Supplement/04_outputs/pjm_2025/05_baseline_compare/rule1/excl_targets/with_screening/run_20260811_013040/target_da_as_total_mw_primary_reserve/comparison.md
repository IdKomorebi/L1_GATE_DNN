# 各方法对比：日前主用备用总量

## 一、这次比的是什么

- 目标字段：`da_as_total_mw_primary_reserve`（日前主用备用总量）
- 候选字段：31 个
- **统一预算 n = 16**，由本文方法的门控断崖自然切出来
- 全部 31 个字段的普通 DNN：R² = 0.8632（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| Pearson ⭐ | 相关系数排序（只看两两关系） | **0.8822** |
| XGBoost | 树模型特征重要性（2016） | **0.8782** |
| Lasso | 线性稀疏（1996） | **0.8743** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.8690** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.8678** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.8605** |

> 同预算下最好的是 **Pearson**（R² = 0.8822）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=8 | k=16 | k=21 | k=31 |
|---|---|---|---|---|---|---|
| Pearson | 0.4669 | 0.4643 | 0.8731 | 0.8822 | 0.8640 | 0.8579 |
| XGBoost | 0.4669 | 0.5569 | 0.8667 | 0.8782 | 0.8695 | 0.8589 |
| Lasso | 0.4669 | 0.5073 | 0.8565 | 0.8743 | 0.8578 | 0.8573 |
| STG | 0.3396 | 0.8081 | 0.8697 | 0.8690 | 0.8691 | 0.8593 |
| LassoNet | 0.4669 | 0.6164 | 0.8393 | 0.8678 | 0.8739 | 0.8639 |
| DGatingDNN | 0.5798 | 0.8154 | 0.8744 | 0.8605 | 0.8657 | 0.8659 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.8855
- 活跃阈值 0.01，最终活跃字段 16 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.8790
- 活跃阈值 0.5，最终活跃字段 11 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 425 个点，字段数从 0–31 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 11 / 16 | 69% |
| LassoNet | 9 / 16 | 56% |
| XGBoost | 9 / 16 | 56% |
| Lasso | 10 / 16 | 62% |
| Pearson | 9 / 16 | 56% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 |
| 2 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 3 | `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 |
| 4 | `gen_fuel_nuclear_mw` | 核电出力 |
| 5 | `gen_fuel_hydro_pct` | 水电占比 |
| 6 | `total_pjm_rt_load_mwh` | 实时负荷电量 |
| 7 | `gen_fuel_coal_pct` | 燃煤占比 |
| 8 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 9 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 |
| 10 | `gen_fuel_gas_pct` | 燃气占比 |
| 11 | `gen_fuel_other_renewables_pct` | 其他可再生占比 |
| 12 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 13 | `gen_fuel_wind_pct` | 风电占比 |
| 14 | `total_pjm_self_sched_reg` | 调频自调度量 |
| 15 | `total_pjm_reg_purchases` | 调频购买量 |
| 16 | `gen_fuel_oil_mw` | 燃油出力 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
