# 各方法对比：日前边际损耗价

## 一、这次比的是什么

- 目标字段：`marginal_loss_price_da`（日前边际损耗价）
- 候选字段：34 个
- **统一预算 n = 15**，由本文方法的门控断崖自然切出来
- 全部 34 个字段的普通 DNN：R² = 0.8734（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| LassoNet ⭐ | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.8994** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.8918** |
| XGBoost | 树模型特征重要性（2016） | **0.8904** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.8632** |
| Pearson | 相关系数排序（只看两两关系） | **0.8614** |
| Lasso | 线性稀疏（1996） | **0.8461** |

> 同预算下最好的是 **LassoNet**（R² = 0.8994）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=7 | k=15 | k=20 | k=30 | k=34 |
|---|---|---|---|---|---|---|---|
| LassoNet | 0.6839 | 0.7486 | 0.8393 | 0.8994 | 0.8898 | 0.8796 | 0.8838 |
| DGatingDNN | 0.6839 | 0.7897 | 0.8832 | 0.8918 | 0.8913 | 0.8922 | 0.8835 |
| XGBoost | 0.6839 | 0.7294 | 0.8135 | 0.8904 | 0.9075 | 0.8889 | 0.8789 |
| STG | 0.1836 | 0.6392 | 0.7145 | 0.8632 | 0.8474 | 0.8767 | 0.8816 |
| Pearson | 0.6839 | 0.7254 | 0.7348 | 0.8614 | 0.8689 | 0.8796 | 0.8692 |
| Lasso | 0.6839 | 0.7430 | 0.8197 | 0.8461 | 0.9003 | 0.8790 | 0.8867 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.8922
- 活跃阈值 0.01，最终活跃字段 15 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.8608
- 活跃阈值 0.5，最终活跃字段 10 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 436 个点，字段数从 0–34 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 10 / 15 | 67% |
| LassoNet | 10 / 15 | 67% |
| XGBoost | 10 / 15 | 67% |
| Lasso | 9 / 15 | 60% |
| Pearson | 6 / 15 | 40% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `system_energy_price_da` | 日前能量价格 |
| 2 | `gen_fuel_gas_mw` | 燃气出力 |
| 3 | `gen_fuel_nuclear_mw` | 核电出力 |
| 4 | `forecast_load_mw_latest_available` | 最新负荷预测 |
| 5 | `gen_fuel_multiple_fuels_mw` | 多燃料出力 |
| 6 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 7 | `gen_fuel_solar_mw` | 光伏出力 |
| 8 | `gen_fuel_oil_mw` | 燃油出力 |
| 9 | `gen_fuel_coal_pct` | 燃煤占比 |
| 10 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 |
| 11 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 12 | `gen_fuel_hydro_pct` | 水电占比 |
| 13 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 |
| 14 | `marginal_loss_price_rt` | 实时边际损耗价 |
| 15 | `total_pjm_loc_credit` | 机会成本信用 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
