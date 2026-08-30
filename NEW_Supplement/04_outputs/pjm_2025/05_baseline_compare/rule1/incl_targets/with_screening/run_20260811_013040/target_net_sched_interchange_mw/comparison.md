# 各方法对比：净计划交换功率

## 一、这次比的是什么

- 目标字段：`net_sched_interchange_mw`（净计划交换功率）
- 候选字段：34 个
- **统一预算 n = 17**，由本文方法的门控断崖自然切出来
- 全部 34 个字段的普通 DNN：R² = 0.9185（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| DGatingDNN ⭐ | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.9279** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.9254** |
| XGBoost | 树模型特征重要性（2016） | **0.9203** |
| Lasso | 线性稀疏（1996） | **0.9115** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.9084** |
| Pearson | 相关系数排序（只看两两关系） | **0.8847** |

> 同预算下最好的是 **DGatingDNN**（R² = 0.9279）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=8 | k=17 | k=22 | k=34 |
|---|---|---|---|---|---|---|
| DGatingDNN | 0.1082 | 0.4514 | 0.8971 | 0.9279 | 0.9218 | 0.9204 |
| STG | 0.0162 | 0.2868 | 0.6160 | 0.9254 | 0.9208 | 0.9182 |
| XGBoost | 0.2923 | 0.4441 | 0.8460 | 0.9203 | 0.9256 | 0.9175 |
| Lasso | 0.1796 | 0.4877 | 0.8628 | 0.9115 | 0.9129 | 0.9194 |
| LassoNet | 0.2278 | 0.5690 | 0.8446 | 0.9084 | 0.9239 | 0.9210 |
| Pearson | 0.2278 | 0.5778 | 0.7871 | 0.8847 | 0.8730 | 0.9166 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.9299
- 活跃阈值 0.01，最终活跃字段 17 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.9294
- 活跃阈值 0.5，最终活跃字段 18 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 410 个点，字段数从 0–34 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 13 / 17 | 76% |
| LassoNet | 11 / 17 | 65% |
| XGBoost | 12 / 17 | 71% |
| Lasso | 14 / 17 | 82% |
| Pearson | 10 / 17 | 59% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `gen_fuel_gas_mw` | 燃气出力 |
| 2 | `gen_fuel_coal_pct` | 燃煤占比 |
| 3 | `total_losses` | 总网损 |
| 4 | `gen_fuel_nuclear_mw` | 核电出力 |
| 5 | `gen_fuel_other_renewables_mw` | 其他可再生出力 |
| 6 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 7 | `gross_sched_interchange_mw` | 总计划交换 |
| 8 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 |
| 9 | `total_lmp_da` | 日前总电价 |
| 10 | `gen_fuel_gas_pct` | 燃气占比 |
| 11 | `gen_fuel_wind_mw` | 风电出力 |
| 12 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 13 | `marginal_loss_price_da` | 日前边际损耗价 |
| 14 | `gen_fuel_oil_mw` | 燃油出力 |
| 15 | `total_pjm_assigned_reg` | 调频分配量 |
| 16 | `gen_fuel_storage_mw` | 储能出力 |
| 17 | `total_pjm_rmpcp_cr` | 调频性能信用额 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
