# 各方法对比：实时阻塞价

## 一、这次比的是什么

- 目标字段：`congestion_price_rt`（实时阻塞价）
- 候选字段：45 个
- **统一预算 n = 27**，由本文方法的门控断崖自然切出来
- 全部 45 个字段的普通 DNN：R² = 0.6743（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| XGBoost ⭐ | 树模型特征重要性（2016） | **0.6923** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.6823** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.6650** |
| Lasso | 线性稀疏（1996） | **0.6632** |
| Pearson | 相关系数排序（只看两两关系） | **0.6378** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.6373** |

> 同预算下最好的是 **XGBoost**（R² = 0.6923）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=13 | k=27 | k=32 | k=45 |
|---|---|---|---|---|---|---|
| XGBoost | 0.4603 | 0.5180 | 0.6528 | 0.6923 | 0.6782 | 0.6642 |
| DGatingDNN | 0.4058 | 0.4928 | 0.6654 | 0.6823 | 0.6454 | 0.6721 |
| STG | 0.1866 | 0.2336 | 0.6605 | 0.6650 | 0.6814 | 0.6798 |
| Lasso | 0.4603 | 0.4484 | 0.5694 | 0.6632 | 0.6749 | 0.6783 |
| Pearson | 0.4603 | 0.4932 | 0.6072 | 0.6378 | 0.6362 | 0.6964 |
| LassoNet | 0.4603 | 0.4919 | 0.6226 | 0.6373 | 0.6819 | 0.7016 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.6867
- 活跃阈值 0.01，最终活跃字段 27 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.6782
- 活跃阈值 0.5，最终活跃字段 14 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 398 个点，字段数从 0–45 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 20 / 27 | 74% |
| LassoNet | 20 / 27 | 74% |
| XGBoost | 20 / 27 | 74% |
| Lasso | 16 / 27 | 59% |
| Pearson | 16 / 27 | 59% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `marginal_loss_price_rt` | 实时边际损耗价 |
| 2 | `system_energy_price_rt` | 实时能量价格 |
| 3 | `total_losses` | 总网损 |
| 4 | `gen_fuel_oil_pct` | 燃油占比 |
| 5 | `gen_fuel_gas_pct` | 燃气占比 |
| 6 | `gen_fuel_solar_pct` | 光伏占比 |
| 7 | `gen_fuel_coal_pct` | 燃煤占比 |
| 8 | `net_sched_interchange_mw` | 净计划交换 |
| 9 | `congestion_price_da` | 日前阻塞价 |
| 10 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 11 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 12 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 13 | `rmccp` | 调频容量出清价 |
| 14 | `gen_fuel_nuclear_mw` | 核电出力 |
| 15 | `marginal_loss_price_da` | 日前边际损耗价 |
| 16 | `gross_actual_interchange_mw` | 总实际交换 |
| 17 | `gen_fuel_hydro_mw` | 水电出力 |
| 18 | `system_energy_price_da` | 日前能量价格 |
| 19 | `total_pjm_loc_credit` | 机会成本信用 |
| 20 | `gen_fuel_storage_mw` | 储能出力 |
| 21 | `gen_fuel_other_renewables_pct` | 其他可再生占比 |
| 22 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 23 | `total_pjm_assigned_reg` | 调频分配量 |
| 24 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 |
| 25 | `da_as_nsr_mw_primary_reserve` | 非同步备用 |
| 26 | `da_as_mcp_primary_reserve` | 日前主用备用出清价 |
| 27 | `metered_load_mw` | 计量负荷 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
