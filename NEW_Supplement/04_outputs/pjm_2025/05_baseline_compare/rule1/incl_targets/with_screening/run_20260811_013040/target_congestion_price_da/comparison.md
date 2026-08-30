# 各方法对比：日前阻塞价

## 一、这次比的是什么

- 目标字段：`congestion_price_da`（日前阻塞价）
- 候选字段：46 个
- **统一预算 n = 26**，由本文方法的门控断崖自然切出来
- 全部 46 个字段的普通 DNN：R² = 0.7576（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| XGBoost ⭐ | 树模型特征重要性（2016） | **0.7608** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.7391** |
| Pearson | 相关系数排序（只看两两关系） | **0.7155** |
| Lasso | 线性稀疏（1996） | **0.7142** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.7110** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.7062** |

> 同预算下最好的是 **XGBoost**（R² = 0.7608）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=13 | k=26 | k=31 | k=46 |
|---|---|---|---|---|---|---|
| XGBoost | 0.0683 | 0.3017 | 0.6650 | 0.7608 | 0.7669 | 0.7284 |
| STG | 0.0660 | 0.2088 | 0.6740 | 0.7391 | 0.7185 | 0.7126 |
| Pearson | 0.1227 | 0.2617 | 0.5574 | 0.7155 | 0.6999 | 0.7520 |
| Lasso | 0.1227 | 0.2749 | 0.5994 | 0.7142 | 0.7184 | 0.7251 |
| LassoNet | 0.1227 | 0.2769 | 0.6350 | 0.7110 | 0.7287 | 0.7345 |
| DGatingDNN | 0.1215 | 0.3154 | 0.7470 | 0.7062 | 0.7122 | 0.7535 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.7259
- 活跃阈值 0.01，最终活跃字段 26 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.7229
- 活跃阈值 0.5，最终活跃字段 21 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 381 个点，字段数从 0–46 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 19 / 26 | 73% |
| LassoNet | 18 / 26 | 69% |
| XGBoost | 18 / 26 | 69% |
| Lasso | 15 / 26 | 58% |
| Pearson | 13 / 26 | 50% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `system_energy_price_da` | 日前能量价格 |
| 2 | `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 |
| 3 | `total_losses` | 总网损 |
| 4 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 5 | `gen_fuel_gas_pct` | 燃气占比 |
| 6 | `gen_fuel_solar_mw` | 光伏出力 |
| 7 | `marginal_loss_price_da` | 日前边际损耗价 |
| 8 | `gross_sched_interchange_mw` | 总计划交换 |
| 9 | `gen_fuel_nuclear_mw` | 核电出力 |
| 10 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 11 | `gen_fuel_coal_pct` | 燃煤占比 |
| 12 | `da_as_mcp_primary_reserve` | 日前主用备用出清价 |
| 13 | `gen_fuel_oil_pct` | 燃油占比 |
| 14 | `forecast_load_mw_latest_available` | 最新负荷预测 |
| 15 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 |
| 16 | `congestion_price_rt` | 实时阻塞价 |
| 17 | `total_pjm_reg_purchases` | 调频购买量 |
| 18 | `total_pjm_loc_credit` | 机会成本信用 |
| 19 | `gen_fuel_hydro_pct` | 水电占比 |
| 20 | `gross_actual_interchange_mw` | 总实际交换 |
| 21 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 22 | `total_pjm_rmpcp_cr` | 调频性能信用额 |
| 23 | `gen_fuel_other_renewables_pct` | 其他可再生占比 |
| 24 | `total_pjm_rt_load_mwh` | 实时负荷电量 |
| 25 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 |
| 26 | `total_pjm_rmccp_cr` | 调频容量信用额 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
