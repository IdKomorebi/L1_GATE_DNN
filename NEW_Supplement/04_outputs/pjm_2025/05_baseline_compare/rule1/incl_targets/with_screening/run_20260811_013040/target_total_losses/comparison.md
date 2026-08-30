# 各方法对比：总网损

## 一、这次比的是什么

- 目标字段：`total_losses`（总网损）
- 候选字段：46 个
- **统一预算 n = 21**，由本文方法的门控断崖自然切出来
- 全部 46 个字段的普通 DNN：R² = 0.9496（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| XGBoost ⭐ | 树模型特征重要性（2016） | **0.9543** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.9496** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.9486** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.9473** |
| Lasso | 线性稀疏（1996） | **0.9422** |
| Pearson | 相关系数排序（只看两两关系） | **0.9242** |

> 同预算下最好的是 **XGBoost**（R² = 0.9543）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=10 | k=21 | k=26 | k=42 | k=46 |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.5375 | 0.6155 | 0.9115 | 0.9543 | 0.9531 | 0.9529 | 0.9540 |
| DGatingDNN | 0.3356 | 0.7827 | 0.9396 | 0.9496 | 0.9514 | 0.9531 | 0.9531 |
| LassoNet | 0.5375 | 0.7659 | 0.9141 | 0.9486 | 0.9541 | 0.9538 | 0.9547 |
| STG | 0.0933 | 0.6272 | 0.9190 | 0.9473 | 0.9476 | 0.9551 | 0.9503 |
| Lasso | 0.4760 | 0.8771 | 0.9271 | 0.9422 | 0.9392 | 0.9559 | 0.9504 |
| Pearson | 0.5375 | 0.6170 | 0.9411 | 0.9242 | 0.9431 | 0.9448 | 0.9472 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.9504
- 活跃阈值 0.01，最终活跃字段 21 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.9468
- 活跃阈值 0.5，最终活跃字段 15 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 434 个点，字段数从 0–46 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 13 / 21 | 62% |
| LassoNet | 12 / 21 | 57% |
| XGBoost | 13 / 21 | 62% |
| Lasso | 11 / 21 | 52% |
| Pearson | 9 / 21 | 43% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `marginal_loss_price_rt` | 实时边际损耗价 |
| 2 | `total_lmp_rt` | 实时总电价 |
| 3 | `gen_fuel_nuclear_mw` | 核电出力 |
| 4 | `gen_fuel_coal_mw` | 燃煤出力 |
| 5 | `total_gen` | 总发电量 |
| 6 | `net_actual_interchange_mw` | 净实际交换 |
| 7 | `wind_generation_mw` | 风电出力 |
| 8 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 9 | `gross_actual_interchange_mw` | 总实际交换 |
| 10 | `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 |
| 11 | `marginal_loss_price_da` | 日前边际损耗价 |
| 12 | `gen_fuel_multiple_fuels_mw` | 多燃料出力 |
| 13 | `gross_sched_interchange_mw` | 总计划交换 |
| 14 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 15 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 16 | `gen_fuel_solar_pct` | 光伏占比 |
| 17 | `forecast_load_mw_latest_available` | 最新负荷预测 |
| 18 | `gen_fuel_hydro_pct` | 水电占比 |
| 19 | `congestion_price_rt` | 实时阻塞价 |
| 20 | `congestion_price_da` | 日前阻塞价 |
| 21 | `total_pjm_reg_purchases` | 调频购买量 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
