# 各方法对比：净计划交换功率

## 一、这次比的是什么

- 目标字段：`net_sched_interchange_mw`（净计划交换功率）
- 候选字段：32 个
- **统一预算 n = 19**，由本文方法的门控断崖自然切出来
- 全部 32 个字段的普通 DNN：R² = 0.9057（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| DGatingDNN ⭐ | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.9066** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.9059** |
| XGBoost | 树模型特征重要性（2016） | **0.9020** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.8945** |
| Pearson | 相关系数排序（只看两两关系） | **0.8914** |
| Lasso | 线性稀疏（1996） | **0.8807** |

> 同预算下最好的是 **DGatingDNN**（R² = 0.9066）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=9 | k=19 | k=24 | k=32 |
|---|---|---|---|---|---|---|
| DGatingDNN | 0.2101 | 0.5441 | 0.8848 | 0.9066 | 0.9024 | 0.9000 |
| STG | 0.2278 | 0.4739 | 0.7978 | 0.9059 | 0.9012 | 0.9037 |
| XGBoost | 0.2923 | 0.5889 | 0.8233 | 0.9020 | 0.9106 | 0.9036 |
| LassoNet | 0.2278 | 0.5690 | 0.8143 | 0.8945 | 0.9083 | 0.9066 |
| Pearson | 0.2278 | 0.5533 | 0.7902 | 0.8914 | 0.9017 | 0.9036 |
| Lasso | 0.2278 | 0.6683 | 0.8348 | 0.8807 | 0.8912 | 0.9058 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.9161
- 活跃阈值 0.01，最终活跃字段 19 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.9120
- 活跃阈值 0.5，最终活跃字段 15 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 405 个点，字段数从 0–32 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 15 / 19 | 79% |
| LassoNet | 15 / 19 | 79% |
| XGBoost | 15 / 19 | 79% |
| Lasso | 16 / 19 | 84% |
| Pearson | 11 / 19 | 58% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `gen_fuel_coal_pct` | 燃煤占比 |
| 2 | `gen_fuel_gas_mw` | 燃气出力 |
| 3 | `gen_fuel_nuclear_mw` | 核电出力 |
| 4 | `system_energy_price_da` | 日前能量价格 |
| 5 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 6 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 7 | `marginal_loss_price_rt` | 实时边际损耗价 |
| 8 | `gross_sched_interchange_mw` | 总计划交换 |
| 9 | `gen_fuel_gas_pct` | 燃气占比 |
| 10 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 11 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 12 | `total_lmp_rt` | 实时总电价 |
| 13 | `gen_fuel_wind_mw` | 风电出力 |
| 14 | `gen_fuel_oil_mw` | 燃油出力 |
| 15 | `total_pjm_rmpcp_cr` | 调频性能信用额 |
| 16 | `total_pjm_assigned_reg` | 调频分配量 |
| 17 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 |
| 18 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 |
| 19 | `gen_fuel_storage_mw` | 储能出力 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
