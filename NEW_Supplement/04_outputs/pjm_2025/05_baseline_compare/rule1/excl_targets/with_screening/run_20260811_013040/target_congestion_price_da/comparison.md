# 各方法对比：日前阻塞价

## 一、这次比的是什么

- 目标字段：`congestion_price_da`（日前阻塞价）
- 候选字段：40 个
- **统一预算 n = 21**，由本文方法的门控断崖自然切出来
- 全部 40 个字段的普通 DNN：R² = 0.7050（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| XGBoost ⭐ | 树模型特征重要性（2016） | **0.7209** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.6867** |
| Lasso | 线性稀疏（1996） | **0.6808** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.6701** |
| Pearson | 相关系数排序（只看两两关系） | **0.6384** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.6354** |

> 同预算下最好的是 **XGBoost**（R² = 0.7209）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=10 | k=21 | k=26 | k=40 |
|---|---|---|---|---|---|---|
| XGBoost | 0.1555 | 0.2490 | 0.4910 | 0.7209 | 0.7422 | 0.7174 |
| STG | 0.0660 | 0.0960 | 0.4255 | 0.6867 | 0.7262 | 0.6956 |
| Lasso | 0.1555 | 0.2751 | 0.6693 | 0.6808 | 0.6462 | 0.6840 |
| DGatingDNN | 0.0327 | 0.2415 | 0.7119 | 0.6701 | 0.6606 | 0.7192 |
| Pearson | 0.1555 | 0.3404 | 0.4642 | 0.6384 | 0.7052 | 0.6887 |
| LassoNet | 0.1215 | 0.3212 | 0.6244 | 0.6354 | 0.6985 | 0.7046 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.7148
- 活跃阈值 0.01，最终活跃字段 21 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.7144
- 活跃阈值 0.5，最终活跃字段 19 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 365 个点，字段数从 0–40 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 16 / 21 | 76% |
| LassoNet | 12 / 21 | 57% |
| XGBoost | 15 / 21 | 71% |
| Lasso | 13 / 21 | 62% |
| Pearson | 10 / 21 | 48% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `gen_fuel_gas_pct` | 燃气占比 |
| 2 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 3 | `system_energy_price_da` | 日前能量价格 |
| 4 | `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 |
| 5 | `gen_fuel_nuclear_mw` | 核电出力 |
| 6 | `gross_sched_interchange_mw` | 总计划交换 |
| 7 | `gen_fuel_solar_mw` | 光伏出力 |
| 8 | `forecast_load_mw_latest_available` | 最新负荷预测 |
| 9 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 10 | `gen_fuel_coal_pct` | 燃煤占比 |
| 11 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 12 | `gen_fuel_oil_mw` | 燃油出力 |
| 13 | `marginal_loss_price_rt` | 实时边际损耗价 |
| 14 | `gen_fuel_hydro_pct` | 水电占比 |
| 15 | `total_pjm_loc_credit` | 机会成本信用 |
| 16 | `da_as_mcp_primary_reserve` | 日前主用备用出清价 |
| 17 | `total_pjm_assigned_reg` | 调频分配量 |
| 18 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 |
| 19 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 |
| 20 | `total_pjm_rmpcp_cr` | 调频性能信用额 |
| 21 | `system_energy_price_rt` | 实时能量价格 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
