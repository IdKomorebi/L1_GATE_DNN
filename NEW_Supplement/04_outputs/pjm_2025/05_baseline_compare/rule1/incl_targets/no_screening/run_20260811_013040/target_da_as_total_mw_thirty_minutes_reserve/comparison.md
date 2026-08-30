# 各方法对比：日前30分钟备用总量

## 一、这次比的是什么

- 目标字段：`da_as_total_mw_thirty_minutes_reserve`（日前30分钟备用总量）
- 候选字段：58 个
- **统一预算 n = 23**，由本文方法的门控断崖自然切出来
- 全部 58 个字段的普通 DNN：R² = 0.9081（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| DGatingDNN ⭐ | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.9166** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.9035** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.8991** |
| Lasso | 线性稀疏（1996） | **0.8979** |
| XGBoost | 树模型特征重要性（2016） | **0.8877** |
| Pearson | 相关系数排序（只看两两关系） | **0.8616** |

> 同预算下最好的是 **DGatingDNN**（R² = 0.9166）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=11 | k=23 | k=28 | k=46 | k=58 |
|---|---|---|---|---|---|---|---|
| DGatingDNN | 0.3412 | 0.7143 | 0.8874 | 0.9166 | 0.9079 | 0.8983 | 0.9058 |
| LassoNet | 0.3412 | 0.7610 | 0.8431 | 0.9035 | 0.9123 | 0.9086 | 0.9110 |
| STG | 0.1122 | 0.4462 | 0.8511 | 0.8991 | 0.8997 | 0.9080 | 0.9105 |
| Lasso | 0.1023 | 0.6781 | 0.8517 | 0.8979 | 0.8911 | 0.9101 | 0.9027 |
| XGBoost | 0.3412 | 0.6399 | 0.8490 | 0.8877 | 0.8945 | 0.9151 | 0.9089 |
| Pearson | 0.3412 | 0.6919 | 0.8238 | 0.8616 | 0.8756 | 0.8824 | 0.9017 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.9170
- 活跃阈值 0.01，最终活跃字段 23 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.9023
- 活跃阈值 0.5，最终活跃字段 14 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 416 个点，字段数从 0–58 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 14 / 23 | 61% |
| LassoNet | 13 / 23 | 57% |
| XGBoost | 8 / 23 | 35% |
| Lasso | 8 / 23 | 35% |
| Pearson | 10 / 23 | 43% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `gen_fuel_nuclear_mw` | 核电出力 |
| 2 | `total_pjm_rt_load_mwh` | 实时负荷电量 |
| 3 | `total_lmp_da` | 日前总电价 |
| 4 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 5 | `gen_fuel_coal_pct` | 燃煤占比 |
| 6 | `net_sched_interchange_mw` | 净计划交换 |
| 7 | `gen_fuel_gas_pct` | 燃气占比 |
| 8 | `total_losses` | 总网损 |
| 9 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 |
| 10 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 11 | `marginal_loss_price_da` | 日前边际损耗价 |
| 12 | `gross_actual_interchange_mw` | 总实际交换 |
| 13 | `gen_fuel_oil_mw` | 燃油出力 |
| 14 | `gen_fuel_wind_mw` | 风电出力 |
| 15 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 |
| 16 | `gen_fuel_hydro_pct` | 水电占比 |
| 17 | `gross_sched_interchange_mw` | 总计划交换 |
| 18 | `da_as_as_mw_primary_reserve` | 日前主用备用实际量 |
| 19 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 |
| 20 | `total_pjm_reg_purchases` | 调频购买量 |
| 21 | `congestion_price_da` | 日前阻塞价 |
| 22 | `total_pjm_rmpcp_cr` | 调频性能信用额 |
| 23 | `congestion_price_rt` | 实时阻塞价 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
