# 各方法对比：净实际交换功率

## 一、这次比的是什么

- 目标字段：`net_actual_interchange_mw`（净实际交换功率）
- 候选字段：36 个
- **统一预算 n = 17**，由本文方法的门控断崖自然切出来
- 全部 36 个字段的普通 DNN：R² = 0.9189（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| DGatingDNN ⭐ | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.9276** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.9145** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.9139** |
| Lasso | 线性稀疏（1996） | **0.9095** |
| XGBoost | 树模型特征重要性（2016） | **0.9058** |
| Pearson | 相关系数排序（只看两两关系） | **0.8750** |

> 同预算下最好的是 **DGatingDNN**（R² = 0.9276）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=8 | k=17 | k=22 | k=34 | k=36 |
|---|---|---|---|---|---|---|---|
| DGatingDNN | 0.1129 | 0.3803 | 0.8764 | 0.9276 | 0.9214 | 0.9191 | 0.9237 |
| LassoNet | 0.2300 | 0.5744 | 0.8548 | 0.9145 | 0.9252 | 0.9124 | 0.9238 |
| STG | 0.0155 | 0.2837 | 0.6230 | 0.9139 | 0.9225 | 0.9254 | 0.9250 |
| Lasso | 0.1822 | 0.4855 | 0.8846 | 0.9095 | 0.9106 | 0.9229 | 0.9225 |
| XGBoost | 0.2953 | 0.4432 | 0.8521 | 0.9058 | 0.9233 | 0.9131 | 0.9158 |
| Pearson | 0.2300 | 0.5390 | 0.7752 | 0.8750 | 0.8810 | 0.9165 | 0.9195 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.9297
- 活跃阈值 0.01，最终活跃字段 17 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.9297
- 活跃阈值 0.5，最终活跃字段 19 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 411 个点，字段数从 0–36 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 12 / 17 | 71% |
| LassoNet | 13 / 17 | 76% |
| XGBoost | 11 / 17 | 65% |
| Lasso | 11 / 17 | 65% |
| Pearson | 10 / 17 | 59% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `gen_fuel_gas_mw` | 燃气出力 |
| 2 | `gen_fuel_coal_pct` | 燃煤占比 |
| 3 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 4 | `total_losses` | 总网损 |
| 5 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 6 | `gen_fuel_nuclear_mw` | 核电出力 |
| 7 | `gross_sched_interchange_mw` | 总计划交换 |
| 8 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 9 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 10 | `gen_fuel_gas_pct` | 燃气占比 |
| 11 | `wind_generation_mw` | 风电出力 |
| 12 | `system_energy_price_da` | 日前能量价格 |
| 13 | `marginal_loss_price_da` | 日前边际损耗价 |
| 14 | `total_lmp_da` | 日前总电价 |
| 15 | `total_pjm_assigned_reg` | 调频分配量 |
| 16 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 |
| 17 | `rmpcp` | 调频性能出清价 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
