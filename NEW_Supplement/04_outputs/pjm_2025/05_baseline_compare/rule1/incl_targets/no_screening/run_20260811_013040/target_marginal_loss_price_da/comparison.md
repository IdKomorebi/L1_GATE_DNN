# 各方法对比：日前边际损耗价

## 一、这次比的是什么

- 目标字段：`marginal_loss_price_da`（日前边际损耗价）
- 候选字段：57 个
- **统一预算 n = 22**，由本文方法的门控断崖自然切出来
- 全部 57 个字段的普通 DNN：R² = 0.9041（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| Lasso ⭐ | 线性稀疏（1996） | **0.9157** |
| XGBoost | 树模型特征重要性（2016） | **0.9150** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.9131** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.9052** |
| Pearson | 相关系数排序（只看两两关系） | **0.8937** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.8849** |

> 同预算下最好的是 **Lasso**（R² = 0.9157）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=11 | k=22 | k=27 | k=44 | k=57 |
|---|---|---|---|---|---|---|---|
| Lasso | 0.2082 | 0.7592 | 0.8774 | 0.9157 | 0.9193 | 0.9103 | 0.9062 |
| XGBoost | 0.6839 | 0.8014 | 0.8913 | 0.9150 | 0.9229 | 0.9224 | 0.9020 |
| DGatingDNN | 0.6839 | 0.7978 | 0.9068 | 0.9131 | 0.9072 | 0.9109 | 0.9058 |
| LassoNet | 0.6839 | 0.8097 | 0.8615 | 0.9052 | 0.9048 | 0.9134 | 0.8952 |
| Pearson | 0.6839 | 0.7254 | 0.7768 | 0.8937 | 0.8801 | 0.9039 | 0.9070 |
| STG | 0.0184 | 0.7703 | 0.8755 | 0.8849 | 0.8862 | 0.9055 | 0.9061 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.9111
- 活跃阈值 0.01，最终活跃字段 22 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.8975
- 活跃阈值 0.5，最终活跃字段 13 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 442 个点，字段数从 0–57 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 10 / 22 | 45% |
| LassoNet | 13 / 22 | 59% |
| XGBoost | 12 / 22 | 55% |
| Lasso | 11 / 22 | 50% |
| Pearson | 10 / 22 | 45% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `system_energy_price_da` | 日前能量价格 |
| 2 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 3 | `gen_fuel_gas_mw` | 燃气出力 |
| 4 | `total_losses` | 总网损 |
| 5 | `gen_fuel_oil_mw` | 燃油出力 |
| 6 | `congestion_price_da` | 日前阻塞价 |
| 7 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 8 | `gross_sched_interchange_mw` | 总计划交换 |
| 9 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 10 | `wind_generation_mw` | 风电出力 |
| 11 | `forecast_load_mw_latest_available` | 最新负荷预测 |
| 12 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 13 | `congestion_price_rt` | 实时阻塞价 |
| 14 | `gen_fuel_coal_pct` | 燃煤占比 |
| 15 | `net_actual_interchange_mw` | 净实际交换 |
| 16 | `da_as_mcp_primary_reserve` | 日前主用备用出清价 |
| 17 | `gen_fuel_hydro_pct` | 水电占比 |
| 18 | `total_pjm_reg_purchases` | 调频购买量 |
| 19 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 |
| 20 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 21 | `total_lmp_rt` | 实时总电价 |
| 22 | `rmpcp` | 调频性能出清价 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
