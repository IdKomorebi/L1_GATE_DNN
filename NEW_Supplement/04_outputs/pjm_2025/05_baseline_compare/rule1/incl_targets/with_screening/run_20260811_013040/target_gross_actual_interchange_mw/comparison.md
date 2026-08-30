# 各方法对比：总实际交换功率

## 一、这次比的是什么

- 目标字段：`gross_actual_interchange_mw`（总实际交换功率）
- 候选字段：47 个
- **统一预算 n = 19**，由本文方法的门控断崖自然切出来
- 全部 47 个字段的普通 DNN：R² = 0.8841（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| Lasso ⭐ | 线性稀疏（1996） | **0.8918** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.8867** |
| XGBoost | 树模型特征重要性（2016） | **0.8862** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.8845** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.8778** |
| Pearson | 相关系数排序（只看两两关系） | **0.8525** |

> 同预算下最好的是 **Lasso**（R² = 0.8918）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=9 | k=19 | k=24 | k=38 | k=47 |
|---|---|---|---|---|---|---|---|
| Lasso | 0.6200 | 0.6988 | 0.8341 | 0.8918 | 0.8757 | 0.8821 | 0.8803 |
| LassoNet | 0.6200 | 0.7101 | 0.8424 | 0.8867 | 0.8867 | 0.8823 | 0.8829 |
| XGBoost | 0.6200 | 0.7087 | 0.8137 | 0.8862 | 0.8833 | 0.8825 | 0.8837 |
| DGatingDNN | 0.6200 | 0.7152 | 0.8702 | 0.8845 | 0.8833 | 0.8881 | 0.8879 |
| STG | 0.6200 | 0.7111 | 0.8331 | 0.8778 | 0.8822 | 0.8841 | 0.8869 |
| Pearson | 0.6200 | 0.6962 | 0.8238 | 0.8525 | 0.8673 | 0.8801 | 0.8895 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.8893
- 活跃阈值 0.01，最终活跃字段 19 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.8791
- 活跃阈值 0.5，最终活跃字段 13 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 438 个点，字段数从 0–47 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 12 / 19 | 63% |
| LassoNet | 11 / 19 | 58% |
| XGBoost | 8 / 19 | 42% |
| Lasso | 11 / 19 | 58% |
| Pearson | 8 / 19 | 42% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 2 | `gen_fuel_nuclear_mw` | 核电出力 |
| 3 | `gen_fuel_coal_pct` | 燃煤占比 |
| 4 | `gen_fuel_gas_pct` | 燃气占比 |
| 5 | `gross_sched_interchange_mw` | 总计划交换 |
| 6 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 7 | `net_actual_interchange_mw` | 净实际交换 |
| 8 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 9 | `metered_load_mw` | 计量负荷 |
| 10 | `total_losses` | 总网损 |
| 11 | `gen_fuel_oil_pct` | 燃油占比 |
| 12 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 |
| 13 | `gen_fuel_solar_pct` | 光伏占比 |
| 14 | `gen_fuel_hydro_mw` | 水电出力 |
| 15 | `marginal_loss_price_da` | 日前边际损耗价 |
| 16 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 |
| 17 | `congestion_price_da` | 日前阻塞价 |
| 18 | `total_pjm_reg_purchases` | 调频购买量 |
| 19 | `congestion_price_rt` | 实时阻塞价 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
