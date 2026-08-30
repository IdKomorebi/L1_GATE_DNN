# 各方法对比：日前阻塞价

## 一、这次比的是什么

- 目标字段：`congestion_price_da`（日前阻塞价）
- 候选字段：49 个
- **统一预算 n = 26**，由本文方法的门控断崖自然切出来
- 全部 49 个字段的普通 DNN：R² = 0.6929（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| XGBoost ⭐ | 树模型特征重要性（2016） | **0.7471** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.7115** |
| Pearson | 相关系数排序（只看两两关系） | **0.7052** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.6933** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.6518** |
| Lasso | 线性稀疏（1996） | **0.5904** |

> 同预算下最好的是 **XGBoost**（R² = 0.7471）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=13 | k=26 | k=31 | k=49 |
|---|---|---|---|---|---|---|
| XGBoost | 0.1555 | 0.2490 | 0.5348 | 0.7471 | 0.7404 | 0.6826 |
| STG | 0.0048 | 0.2053 | 0.6505 | 0.7115 | 0.7255 | 0.6843 |
| Pearson | 0.1555 | 0.3404 | 0.5970 | 0.7052 | 0.6739 | 0.7033 |
| LassoNet | 0.1555 | 0.3240 | 0.6224 | 0.6933 | 0.7248 | 0.7118 |
| DGatingDNN | 0.0683 | 0.1638 | 0.7460 | 0.6518 | 0.6558 | 0.7063 |
| Lasso | 0.1555 | 0.2751 | 0.6219 | 0.5904 | 0.6471 | 0.6887 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.7451
- 活跃阈值 0.01，最终活跃字段 26 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.7302
- 活跃阈值 0.5，最终活跃字段 21 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 373 个点，字段数从 0–49 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 14 / 26 | 54% |
| LassoNet | 17 / 26 | 65% |
| XGBoost | 14 / 26 | 54% |
| Lasso | 18 / 26 | 69% |
| Pearson | 15 / 26 | 58% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 |
| 2 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 3 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 4 | `system_energy_price_da` | 日前能量价格 |
| 5 | `gen_fuel_solar_pct` | 光伏占比 |
| 6 | `gen_fuel_coal_pct` | 燃煤占比 |
| 7 | `gen_fuel_nuclear_mw` | 核电出力 |
| 8 | `gen_fuel_wind_mw` | 风电出力 |
| 9 | `gross_sched_interchange_mw` | 总计划交换 |
| 10 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 11 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 |
| 12 | `gen_fuel_gas_mw` | 燃气出力 |
| 13 | `gen_fuel_hydro_mw` | 水电出力 |
| 14 | `total_pjm_rt_load_mwh` | 实时负荷电量 |
| 15 | `total_pjm_loc_credit` | 机会成本信用 |
| 16 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 17 | `gen_fuel_oil_pct` | 燃油占比 |
| 18 | `total_pjm_self_sched_reg` | 调频自调度量 |
| 19 | `total_pjm_assigned_reg` | 调频分配量 |
| 20 | `marginal_loss_price_rt` | 实时边际损耗价 |
| 21 | `total_pjm_rmpcp_cr` | 调频性能信用额 |
| 22 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 |
| 23 | `gen_fuel_oil_mw` | 燃油出力 |
| 24 | `system_energy_price_rt` | 实时能量价格 |
| 25 | `da_as_nsr_mw_primary_reserve` | 非同步备用 |
| 26 | `net_inadv_interchange_mw` | 净非计划交换 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
