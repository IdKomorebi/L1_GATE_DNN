# 各方法对比：实时阻塞价

## 一、这次比的是什么

- 目标字段：`congestion_price_rt`（实时阻塞价）
- 候选字段：36 个
- **统一预算 n = 22**，由本文方法的门控断崖自然切出来
- 全部 36 个字段的普通 DNN：R² = 0.6514（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| XGBoost ⭐ | 树模型特征重要性（2016） | **0.6817** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.6461** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.6314** |
| Pearson | 相关系数排序（只看两两关系） | **0.6276** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.6212** |
| Lasso | 线性稀疏（1996） | **0.6045** |

> 同预算下最好的是 **XGBoost**（R² = 0.6817）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=11 | k=22 | k=27 | k=36 |
|---|---|---|---|---|---|---|
| XGBoost | 0.4603 | 0.5180 | 0.5993 | 0.6817 | 0.6519 | 0.6499 |
| DGatingDNN | 0.0894 | 0.4638 | 0.6217 | 0.6461 | 0.6457 | 0.6440 |
| STG | 0.0152 | 0.4525 | 0.6012 | 0.6314 | 0.6122 | 0.6248 |
| Pearson | 0.4603 | 0.4932 | 0.5505 | 0.6276 | 0.6291 | 0.6331 |
| LassoNet | 0.4603 | 0.4929 | 0.6045 | 0.6212 | 0.6295 | 0.6598 |
| Lasso | 0.4603 | 0.4484 | 0.5593 | 0.6045 | 0.6245 | 0.6375 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.6531
- 活跃阈值 0.01，最终活跃字段 22 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.6657
- 活跃阈值 0.5，最终活跃字段 14 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 394 个点，字段数从 0–36 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 14 / 22 | 64% |
| LassoNet | 17 / 22 | 77% |
| XGBoost | 12 / 22 | 55% |
| Lasso | 11 / 22 | 50% |
| Pearson | 15 / 22 | 68% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 2 | `marginal_loss_price_rt` | 实时边际损耗价 |
| 3 | `gen_fuel_coal_mw` | 燃煤出力 |
| 4 | `system_energy_price_rt` | 实时能量价格 |
| 5 | `gen_fuel_gas_pct` | 燃气占比 |
| 6 | `rmccp` | 调频容量出清价 |
| 7 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 8 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 |
| 9 | `gen_fuel_oil_pct` | 燃油占比 |
| 10 | `solar_generation_mw` | 光伏出力 |
| 11 | `gen_fuel_storage_mw` | 储能出力 |
| 12 | `gen_fuel_nuclear_mw` | 核电出力 |
| 13 | `gen_fuel_nuclear_pct` | 核电占比 |
| 14 | `gen_fuel_hydro_pct` | 水电占比 |
| 15 | `da_as_mcp_primary_reserve` | 日前主用备用出清价 |
| 16 | `total_pjm_loc_credit` | 机会成本信用 |
| 17 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 |
| 18 | `total_pjm_assigned_reg` | 调频分配量 |
| 19 | `system_energy_price_da` | 日前能量价格 |
| 20 | `gen_fuel_other_renewables_pct` | 其他可再生占比 |
| 21 | `gen_fuel_oil_mw` | 燃油出力 |
| 22 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
