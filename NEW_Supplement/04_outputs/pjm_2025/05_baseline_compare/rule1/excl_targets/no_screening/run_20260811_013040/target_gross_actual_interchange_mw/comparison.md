# 各方法对比：总实际交换功率

## 一、这次比的是什么

- 目标字段：`gross_actual_interchange_mw`（总实际交换功率）
- 候选字段：49 个
- **统一预算 n = 20**，由本文方法的门控断崖自然切出来
- 全部 49 个字段的普通 DNN：R² = 0.8660（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| XGBoost ⭐ | 树模型特征重要性（2016） | **0.8824** |
| Lasso | 线性稀疏（1996） | **0.8775** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.8620** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.8613** |
| Pearson | 相关系数排序（只看两两关系） | **0.8516** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.8210** |

> 同预算下最好的是 **XGBoost**（R² = 0.8824）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=10 | k=20 | k=25 | k=40 | k=49 |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.6200 | 0.7087 | 0.8224 | 0.8824 | 0.8783 | 0.8724 | 0.8601 |
| Lasso | 0.1495 | 0.6756 | 0.8409 | 0.8775 | 0.8714 | 0.8597 | 0.8645 |
| DGatingDNN | 0.6200 | 0.7175 | 0.8745 | 0.8620 | 0.8566 | 0.8574 | 0.8583 |
| LassoNet | 0.6200 | 0.7101 | 0.7914 | 0.8613 | 0.8574 | 0.8580 | 0.8577 |
| Pearson | 0.6200 | 0.7024 | 0.8288 | 0.8516 | 0.8689 | 0.8717 | 0.8524 |
| STG | 0.6200 | 0.7028 | 0.8235 | 0.8210 | 0.8198 | 0.8554 | 0.8606 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.8713
- 活跃阈值 0.01，最终活跃字段 20 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.8344
- 活跃阈值 0.5，最终活跃字段 10 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 437 个点，字段数从 0–49 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 10 / 20 | 50% |
| LassoNet | 11 / 20 | 55% |
| XGBoost | 11 / 20 | 55% |
| Lasso | 11 / 20 | 55% |
| Pearson | 10 / 20 | 50% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 2 | `gen_fuel_gas_mw` | 燃气出力 |
| 3 | `gen_fuel_nuclear_mw` | 核电出力 |
| 4 | `gen_fuel_coal_pct` | 燃煤占比 |
| 5 | `gross_sched_interchange_mw` | 总计划交换 |
| 6 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 7 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 8 | `gen_fuel_wind_mw` | 风电出力 |
| 9 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 10 | `gen_fuel_hydro_pct` | 水电占比 |
| 11 | `solar_generation_mw` | 光伏出力 |
| 12 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 |
| 13 | `total_pjm_reg_purchases` | 调频购买量 |
| 14 | `marginal_loss_price_rt` | 实时边际损耗价 |
| 15 | `gen_fuel_oil_mw` | 燃油出力 |
| 16 | `total_pjm_rmpcp_cr` | 调频性能信用额 |
| 17 | `total_pjm_rt_load_mwh` | 实时负荷电量 |
| 18 | `total_pjm_self_sched_reg` | 调频自调度量 |
| 19 | `da_as_as_mw_primary_reserve` | 日前主用备用实际量 |
| 20 | `gen_fuel_storage_mw` | 储能出力 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
