# 各方法对比：总网损

## 一、这次比的是什么

- 目标字段：`total_losses`（总网损）
- 候选字段：58 个
- **统一预算 n = 22**，由本文方法的门控断崖自然切出来
- 全部 58 个字段的普通 DNN：R² = 0.9445（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| DGatingDNN ⭐ | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.9499** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.9482** |
| XGBoost | 树模型特征重要性（2016） | **0.9475** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.9471** |
| Lasso | 线性稀疏（1996） | **0.9434** |
| Pearson | 相关系数排序（只看两两关系） | **0.9146** |

> 同预算下最好的是 **DGatingDNN**（R² = 0.9499）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=11 | k=22 | k=27 | k=44 | k=58 |
|---|---|---|---|---|---|---|---|
| DGatingDNN | 0.3356 | 0.7822 | 0.9373 | 0.9499 | 0.9509 | 0.9507 | 0.9489 |
| STG | 0.4203 | 0.7052 | 0.9149 | 0.9482 | 0.9497 | 0.9462 | 0.9479 |
| XGBoost | 0.5375 | 0.6155 | 0.9162 | 0.9475 | 0.9501 | 0.9561 | 0.9467 |
| LassoNet | 0.5375 | 0.7659 | 0.9158 | 0.9471 | 0.9556 | 0.9508 | 0.9434 |
| Lasso | 0.4760 | 0.7147 | 0.9210 | 0.9434 | 0.9400 | 0.9483 | 0.9457 |
| Pearson | 0.5375 | 0.6170 | 0.9314 | 0.9146 | 0.9385 | 0.9443 | 0.9467 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.9481
- 活跃阈值 0.01，最终活跃字段 22 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.9471
- 活跃阈值 0.5，最终活跃字段 15 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 438 个点，字段数从 0–58 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 11 / 22 | 50% |
| LassoNet | 11 / 22 | 50% |
| XGBoost | 11 / 22 | 50% |
| Lasso | 10 / 22 | 45% |
| Pearson | 9 / 22 | 41% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `marginal_loss_price_rt` | 实时边际损耗价 |
| 2 | `system_energy_price_rt` | 实时能量价格 |
| 3 | `gen_fuel_nuclear_mw` | 核电出力 |
| 4 | `gen_fuel_coal_mw` | 燃煤出力 |
| 5 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 6 | `net_actual_interchange_mw` | 净实际交换 |
| 7 | `gen_fuel_nuclear_pct` | 核电占比 |
| 8 | `gen_fuel_wind_mw` | 风电出力 |
| 9 | `marginal_loss_price_da` | 日前边际损耗价 |
| 10 | `gross_sched_interchange_mw` | 总计划交换 |
| 11 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 |
| 12 | `gen_fuel_multiple_fuels_mw` | 多燃料出力 |
| 13 | `gross_actual_interchange_mw` | 总实际交换 |
| 14 | `gen_fuel_coal_pct` | 燃煤占比 |
| 15 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 16 | `congestion_price_da` | 日前阻塞价 |
| 17 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 |
| 18 | `gen_fuel_solar_mw` | 光伏出力 |
| 19 | `gen_fuel_hydro_pct` | 水电占比 |
| 20 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 21 | `da_as_nsr_mw_primary_reserve` | 非同步备用 |
| 22 | `total_pjm_reg_purchases` | 调频购买量 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
