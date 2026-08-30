# 各方法对比：总实际交换功率

## 一、这次比的是什么

- 目标字段：`gross_actual_interchange_mw`（总实际交换功率）
- 候选字段：38 个
- **统一预算 n = 16**，由本文方法的门控断崖自然切出来
- 全部 38 个字段的普通 DNN：R² = 0.8666（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| XGBoost ⭐ | 树模型特征重要性（2016） | **0.8780** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.8768** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.8730** |
| Lasso | 线性稀疏（1996） | **0.8685** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.8582** |
| Pearson | 相关系数排序（只看两两关系） | **0.8416** |

> 同预算下最好的是 **XGBoost**（R² = 0.8780）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=8 | k=16 | k=21 | k=32 | k=38 |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.6200 | 0.7087 | 0.8168 | 0.8780 | 0.8785 | 0.8789 | 0.8695 |
| DGatingDNN | 0.6200 | 0.7298 | 0.8463 | 0.8768 | 0.8627 | 0.8650 | 0.8610 |
| LassoNet | 0.6200 | 0.7101 | 0.8270 | 0.8730 | 0.8756 | 0.8723 | 0.8607 |
| Lasso | 0.1495 | 0.6756 | 0.7521 | 0.8685 | 0.8831 | 0.8732 | 0.8609 |
| STG | 0.0836 | 0.2054 | 0.8029 | 0.8582 | 0.8694 | 0.8718 | 0.8626 |
| Pearson | 0.6200 | 0.7024 | 0.8117 | 0.8416 | 0.8523 | 0.8781 | 0.8665 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.8782
- 活跃阈值 0.01，最终活跃字段 16 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.8564
- 活跃阈值 0.5，最终活跃字段 11 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 434 个点，字段数从 0–38 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 11 / 16 | 69% |
| LassoNet | 11 / 16 | 69% |
| XGBoost | 8 / 16 | 50% |
| Lasso | 7 / 16 | 44% |
| Pearson | 9 / 16 | 56% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 2 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 3 | `gen_fuel_nuclear_mw` | 核电出力 |
| 4 | `gross_sched_interchange_mw` | 总计划交换 |
| 5 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 6 | `gen_fuel_gas_pct` | 燃气占比 |
| 7 | `gen_fuel_coal_pct` | 燃煤占比 |
| 8 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 |
| 9 | `gen_fuel_nuclear_pct` | 核电占比 |
| 10 | `solar_generation_mw` | 光伏出力 |
| 11 | `system_energy_price_da` | 日前能量价格 |
| 12 | `gen_fuel_hydro_mw` | 水电出力 |
| 13 | `gen_fuel_oil_mw` | 燃油出力 |
| 14 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 |
| 15 | `total_pjm_assigned_reg` | 调频分配量 |
| 16 | `total_pjm_reg_purchases` | 调频购买量 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
