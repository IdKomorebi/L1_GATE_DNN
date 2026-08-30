# 各方法对比：净实际交换功率

## 一、这次比的是什么

- 目标字段：`net_actual_interchange_mw`（净实际交换功率）
- 候选字段：49 个
- **统一预算 n = 19**，由本文方法的门控断崖自然切出来
- 全部 49 个字段的普通 DNN：R² = 0.9674（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| DGatingDNN ⭐ | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.9697** |
| Lasso | 线性稀疏（1996） | **0.9692** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.9645** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.9610** |
| XGBoost | 树模型特征重要性（2016） | **0.9050** |
| Pearson | 相关系数排序（只看两两关系） | **0.8868** |

> 同预算下最好的是 **DGatingDNN**（R² = 0.9697）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=9 | k=19 | k=24 | k=38 | k=49 |
|---|---|---|---|---|---|---|---|
| DGatingDNN | 0.0424 | 0.3536 | 0.9389 | 0.9697 | 0.9694 | 0.9694 | 0.9656 |
| Lasso | 0.0424 | 0.4600 | 0.9433 | 0.9692 | 0.9692 | 0.9665 | 0.9667 |
| LassoNet | 0.2300 | 0.5777 | 0.8036 | 0.9645 | 0.9677 | 0.9712 | 0.9673 |
| STG | 0.0387 | 0.2130 | 0.7978 | 0.9610 | 0.9662 | 0.9642 | 0.9672 |
| XGBoost | 0.2953 | 0.4432 | 0.8195 | 0.9050 | 0.9621 | 0.9714 | 0.9667 |
| Pearson | 0.2300 | 0.5390 | 0.8002 | 0.8868 | 0.8857 | 0.9654 | 0.9676 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.9688
- 活跃阈值 0.01，最终活跃字段 19 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.9586
- 活跃阈值 0.5，最终活跃字段 18 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 421 个点，字段数从 0–49 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 10 / 19 | 53% |
| LassoNet | 11 / 19 | 58% |
| XGBoost | 8 / 19 | 42% |
| Lasso | 11 / 19 | 58% |
| Pearson | 11 / 19 | 58% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `prelim_load_avg_hourly` | 预估小时负荷 |
| 2 | `gen_fuel_gas_mw` | 燃气出力 |
| 3 | `total_pjm_rt_load_mwh` | 实时负荷电量 |
| 4 | `gen_fuel_coal_mw` | 燃煤出力 |
| 5 | `gen_fuel_nuclear_mw` | 核电出力 |
| 6 | `gen_fuel_solar_mw` | 光伏出力 |
| 7 | `gen_fuel_hydro_mw` | 水电出力 |
| 8 | `gen_fuel_nuclear_pct` | 核电占比 |
| 9 | `gen_fuel_wind_mw` | 风电出力 |
| 10 | `gross_sched_interchange_mw` | 总计划交换 |
| 11 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 |
| 12 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 13 | `gen_fuel_multiple_fuels_mw` | 多燃料出力 |
| 14 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 15 | `gen_fuel_oil_mw` | 燃油出力 |
| 16 | `da_as_as_mw_primary_reserve` | 日前主用备用实际量 |
| 17 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 18 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 |
| 19 | `total_pjm_reg_purchases` | 调频购买量 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
