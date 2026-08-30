# 各方法对比：总实际交换功率

## 一、这次比的是什么

- 目标字段：`gross_actual_interchange_mw`（总实际交换功率）
- 候选字段：58 个
- **统一预算 n = 22**，由本文方法的门控断崖自然切出来
- 全部 58 个字段的普通 DNN：R² = 0.8739（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| STG ⭐ | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.8873** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.8814** |
| Lasso | 线性稀疏（1996） | **0.8777** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.8773** |
| XGBoost | 树模型特征重要性（2016） | **0.8755** |
| Pearson | 相关系数排序（只看两两关系） | **0.8562** |

> 同预算下最好的是 **STG**（R² = 0.8873）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=11 | k=22 | k=27 | k=44 | k=58 |
|---|---|---|---|---|---|---|---|
| STG | 0.6200 | 0.6935 | 0.8548 | 0.8873 | 0.8910 | 0.8871 | 0.8688 |
| DGatingDNN | 0.6200 | 0.6929 | 0.8707 | 0.8814 | 0.8815 | 0.8802 | 0.8710 |
| Lasso | 0.6200 | 0.7301 | 0.8629 | 0.8777 | 0.8796 | 0.8777 | 0.8781 |
| LassoNet | 0.6200 | 0.7101 | 0.8265 | 0.8773 | 0.8911 | 0.8938 | 0.8745 |
| XGBoost | 0.6200 | 0.7087 | 0.8490 | 0.8755 | 0.8971 | 0.8887 | 0.8781 |
| Pearson | 0.6200 | 0.6962 | 0.8295 | 0.8562 | 0.8786 | 0.8720 | 0.8798 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.8889
- 活跃阈值 0.01，最终活跃字段 22 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.8575
- 活跃阈值 0.0，最终活跃字段 12 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 438 个点，字段数从 0–58 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 13 / 22 | 59% |
| LassoNet | 14 / 22 | 64% |
| XGBoost | 13 / 22 | 59% |
| Lasso | 11 / 22 | 50% |
| Pearson | 11 / 22 | 50% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 2 | `gross_sched_interchange_mw` | 总计划交换 |
| 3 | `gen_fuel_multiple_fuels_mw` | 多燃料出力 |
| 4 | `gen_fuel_nuclear_mw` | 核电出力 |
| 5 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 6 | `net_actual_interchange_mw` | 净实际交换 |
| 7 | `gen_fuel_coal_pct` | 燃煤占比 |
| 8 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 9 | `gen_fuel_wind_pct` | 风电占比 |
| 10 | `gen_fuel_hydro_pct` | 水电占比 |
| 11 | `gen_fuel_solar_mw` | 光伏出力 |
| 12 | `total_losses` | 总网损 |
| 13 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 14 | `gen_fuel_gas_mw` | 燃气出力 |
| 15 | `total_lmp_da` | 日前总电价 |
| 16 | `gen_fuel_nuclear_pct` | 核电占比 |
| 17 | `congestion_price_da` | 日前阻塞价 |
| 18 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 |
| 19 | `total_pjm_reg_purchases` | 调频购买量 |
| 20 | `total_pjm_rmpcp_cr` | 调频性能信用额 |
| 21 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 |
| 22 | `total_pjm_assigned_reg` | 调频分配量 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
