# 各方法对比：日前边际损耗价

## 一、这次比的是什么

- 目标字段：`marginal_loss_price_da`（日前边际损耗价）
- 候选字段：42 个
- **统一预算 n = 16**，由本文方法的门控断崖自然切出来
- 全部 42 个字段的普通 DNN：R² = 0.9024（能力上限参照）

每个方法按自己的排序取前 n 个字段，统一用同一个普通 DNN 训练测试。

## 二、同预算下的结果

| 方法 | 说明 | 测试 R² |
|---|---|---|
| XGBoost ⭐ | 树模型特征重要性（2016） | **0.9121** |
| DGatingDNN | 本文方法：可微稀疏门控（NeurIPS 2025） | **0.9002** |
| STG | 随机门控，高斯松弛逼近 L0（ICML 2020） | **0.8930** |
| LassoNet | 神经网络旁挂线性通路的特征级稀疏（JMLR 2021） | **0.8775** |
| Lasso | 线性稀疏（1996） | **0.8769** |
| Pearson | 相关系数排序（只看两两关系） | **0.8548** |

> 同预算下最好的是 **XGBoost**（R² = 0.9121）。

## 三、不同字段数下的完整曲线

只比一个预算容易碰巧，下表给出各方法在多个字段数下的表现（对应图 `fig_topk_curve.png`）：

| 方法 | k=1 | k=3 | k=8 | k=16 | k=21 | k=32 | k=42 |
|---|---|---|---|---|---|---|---|
| XGBoost | 0.6839 | 0.8014 | 0.8545 | 0.9121 | 0.9118 | 0.9018 | 0.8937 |
| DGatingDNN | 0.6839 | 0.8080 | 0.8823 | 0.9002 | 0.9042 | 0.8920 | 0.9034 |
| STG | 0.3283 | 0.6127 | 0.7848 | 0.8930 | 0.8736 | 0.8949 | 0.9070 |
| LassoNet | 0.6839 | 0.7715 | 0.8466 | 0.8775 | 0.8957 | 0.9059 | 0.8956 |
| Lasso | 0.6839 | 0.7980 | 0.8202 | 0.8769 | 0.8756 | 0.8888 | 0.8978 |
| Pearson | 0.6839 | 0.7254 | 0.7671 | 0.8548 | 0.8779 | 0.8869 | 0.9077 |

## 四、各方法的筛选过程

### DGatingDNN

- 门控模型自身的测试 R²：0.8915
- 活跃阈值 0.01，最终活跃字段 16 个
- 过程图 `DGatingDNN/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `DGatingDNN/gate_history.csv`

### STG

- 门控模型自身的测试 R²：0.8855
- 活跃阈值 0.5，最终活跃字段 12 个
- 过程图 `STG/fig_process.png`：每条曲线是一个字段的门控值随训练轮次的变化，左边线性刻度看保留字段的分化，右边对数刻度看被淘汰字段掉到多低
- 逐轮门控值记录在 `STG/gate_history.csv`

### LassoNet

- 它的筛选过程是一条 λ 路径：正则强度从小到大变化，字段被逐个剔除
- 路径共 443 个点，字段数从 0–42 变化
- 记录在 `LassoNet/lambda_path.csv`

### XGBoost / Lasso / Pearson

这三个不存在「逐轮收缩」的过程——它们一次拟合就给出全部字段的重要性排序，选多少个完全由外部指定的 k 决定。各字段得分见各自目录下的 `ranking.csv`。

## 五、各方法选中的字段重合情况

| 方法 | 与本文方法重合 | 重合率 |
|---|---|---|
| STG | 11 / 16 | 69% |
| LassoNet | 9 / 16 | 56% |
| XGBoost | 6 / 16 | 38% |
| Lasso | 9 / 16 | 56% |
| Pearson | 5 / 16 | 31% |

详见图 `fig_selection_matrix.png`。

## 六、本文方法选出的字段

| 排名 | 字段 | 业务含义 |
|---|---|---|
| 1 | `system_energy_price_da` | 日前能量价格 |
| 2 | `total_losses` | 总网损 |
| 3 | `gen_fuel_gas_mw` | 燃气出力 |
| 4 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 |
| 5 | `congestion_price_da` | 日前阻塞价 |
| 6 | `gen_fuel_oil_mw` | 燃油出力 |
| 7 | `gen_fuel_multiple_fuels_mw` | 多燃料出力 |
| 8 | `gen_fuel_coal_pct` | 燃煤占比 |
| 9 | `net_actual_interchange_mw` | 净实际交换 |
| 10 | `solar_generation_mw` | 光伏出力 |
| 11 | `gen_fuel_hydro_mw` | 水电出力 |
| 12 | `gross_actual_interchange_mw` | 总实际交换 |
| 13 | `da_as_mcp_primary_reserve` | 日前主用备用出清价 |
| 14 | `congestion_price_rt` | 实时阻塞价 |
| 15 | `gross_inadv_interchange_mw` | 总非计划交换 |
| 16 | `marginal_loss_price_rt` | 实时边际损耗价 |

## 七、本目录下的图

- `fig_topk_curve.png`　各方法在不同字段数下的表现（信息量最大）
- `fig_comparison.png`　统一预算下的横向对比
- `fig_selection_matrix.png`　各方法选中了哪些字段
- `<方法>/fig_process.png`　该方法的筛选过程（仅门控类方法有）
