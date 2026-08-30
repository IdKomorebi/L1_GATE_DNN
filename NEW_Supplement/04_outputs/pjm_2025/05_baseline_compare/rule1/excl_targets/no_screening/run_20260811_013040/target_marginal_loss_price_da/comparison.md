# 各方法对比：日前边际损耗价

## 一、这次比的是什么

- 目标字段：`marginal_loss_price_da`（日前边际损耗价）
- 候选字段：49 个
- **统一预算 n = 17**，由本文方法的门控断崖自然切出来
- 全部 49 个字段的普通 DNN：R² = 0.8882（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| XGBoost ⭐ | 树模型特征重要性（2016） | **0.9074** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.9042** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.9000** |
| Lasso | 线性稀疏（1996） | **0.8962** |
| Pearson | 相关系数排序（只看两两关系） | **0.8736** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.8636** |

> 同预算下最好的是 **XGBoost**（R² = 0.9074）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=8 | k=17 | k=22 | k=34 | k=49 |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.6839 | 0.7294 | 0.8877 | 0.9074 | 0.9194 | 0.9085 | 0.8945 |
| LassoNet | 0.6839 | 0.7486 | 0.8678 | 0.9042 | 0.9094 | 0.9114 | 0.8876 |
| DGatingDNN | 0.6839 | 0.7910 | 0.8943 | 0.9000 | 0.9125 | 0.8803 | 0.8889 |
| Lasso | 0.2082 | 0.7592 | 0.8705 | 0.8962 | 0.9062 | 0.8772 | 0.8857 |
| Pearson | 0.6839 | 0.7254 | 0.7671 | 0.8736 | 0.8805 | 0.8835 | 0.8939 |
| STG | 0.0141 | 0.6172 | 0.8294 | 0.8636 | 0.8630 | 0.8717 | 0.8948 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.9028
- 活跃阈值 0.01，最终活跃字段 17 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.8948
- 活跃阈值 0.5，最终活跃字段 12 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 438 个点，字段数从 0–49 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 11 / 17 | 65% |
| LassoNet | 11 / 17 | 65% |
| XGBoost | 11 / 17 | 65% |
| Lasso | 10 / 17 | 59% |
| Pearson | 7 / 17 | 41% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `system_energy_price_da` | 日前能量价格 |
| 2 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 3 | `forecast_load_mw_latest_available` | 最新负荷预测 |
| 4 | `gen_fuel_gas_pct` | 燃气占比 |
| 5 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 6 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 7 | `gen_fuel_nuclear_mw` | 核电出力 |
| 8 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 9 | `gen_fuel_wind_mw` | 风电出力 |
| 10 | `gen_fuel_solar_mw` | 光伏出力 |
| 11 | `gross_sched_interchange_mw` | 总计划交换 |
| 12 | `gen_fuel_coal_mw` | 燃煤出力 |
| 13 | `gen_fuel_oil_pct` | 燃油占比 |
| 14 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 |
| 15 | `gen_fuel_hydro_mw` | 水电出力 |
| 16 | `marginal_loss_price_rt` | 实时边际损耗价 |
| 17 | `total_pjm_loc_credit` | 机会成本信用 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
