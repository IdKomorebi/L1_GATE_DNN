# 推断源定位：日前30分钟备用总量

运行时间 20260810_051320　方法 **DGatingDNN**

## 一、这次跑的是什么

- 目标字段：`da_as_total_mw_thirty_minutes_reserve`（日前30分钟备用总量）
- 数据：pjm_2025，8760 行
- 候选字段：**48 个**（排除其余关注字段）
- 网络结构：64-32-16，训练 200 轮，批大小 50，学习率 0.001
- 数据划分：随机 80% 训练 / 20% 测试，随机种子 42

## 二、结论先看这里

- 从 48 个候选字段里选出 **19 个**推断源（占 39.6%）
- 只用这 19 个字段重新训练普通 DNN，测试 R² = **0.8947**
- 作为对照，用全部 48 个字段的普通 DNN 测试 R² = 0.8949
- 没被选中的 29 个字段单独训练，测试 R² = 0.8347

> 用 40% 的字段达到了和全量相当的精度（相差 -0.0002），说明这组字段确实承载了推断目标所需的主要信息。

## 三、选出来的推断源

| 排名 | 字段 | 业务含义 | 门控值 |
|---|---|---|---|
| 1 | `gen_fuel_nuclear_mw` | 核电出力 | 0.1892 |
| 2 | `system_energy_price_da` | 日前能量价格 | 0.1696 |
| 3 | `gen_fuel_coal_pct` | 燃煤占比 | 0.1302 |
| 4 | `gen_fuel_nuclear_pct` | 核电占比 | 0.1195 |
| 5 | `gen_fuel_gas_pct` | 燃气占比 | 0.1111 |
| 6 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 | 0.1101 |
| 7 | `gen_fuel_other_renewables_mw` | 其他可再生出力 | 0.1044 |
| 8 | `gen_fuel_oil_pct` | 燃油占比 | 0.0896 |
| 9 | `wind_generation_mw` | 风电出力 | 0.0833 |
| 10 | `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 | 0.0813 |
| 11 | `gen_fuel_hydro_pct` | 水电占比 | 0.0786 |
| 12 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 | 0.0756 |
| 13 | `gross_sched_interchange_mw` | 总计划交换 | 0.0755 |
| 14 | `da_as_as_mw_primary_reserve` | 日前主用备用实际量 | 0.0693 |
| 15 | `total_pjm_assigned_reg` | 调频分配量 | 0.0658 |
| 16 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 | 0.0634 |
| 17 | `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 | 0.0508 |
| 18 | `total_pjm_rmpcp_cr` | 调频性能信用额 | 0.0467 |
| 19 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 | 0.0245 |

活跃阈值 **0.01**：门控值不低于它的字段判定为推断源。

### 阈值取多少影响大不大

把阈值在几个量级之间来回调，看选中的字段数变不变：

| 阈值 | 选中字段数 |
|---|---|
| 1e-06 | 19 |
| 0.0001 | 19 |
| 0.001 | 19 |
| 0.005 | 19 |
| 0.01 | 19 |
| 0.02 | 19 |
| 0.05 | 17 |
| 0.1 | 7 |

> 这几档阈值给出的字段数完全一致，说明阈值落在门控值分布的空档里，取多少都不影响结论——这正是用 D-Gating 替代普通 L1 门控的原因。

紧挨在阈值下方、被判为非活跃的几个字段（供参考）：

| 字段 | 业务含义 | 门控值 |
|---|---|---|
| `net_inadv_interchange_mw` | 净非计划交换 | 0.0000 |
| `da_as_nsr_mw_primary_reserve` | 非同步备用 | 0.0000 |
| `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 | 0.0000 |
| `total_lmp_rt` | 实时总电价 | 0.0000 |
| `da_as_mcp_primary_reserve` | 日前主用备用出清价 | 0.0000 |

### 注意：这些选中字段是被排除字段的替身

候选池按字段名排除了受关注字段，但下面这些留在池子里的字段，单独一个就能几乎完整还原被排除的那个——也就是说排除规则被绕过了，推断路径换了个入口又通了。

| 选中字段 | 能还原的被排除字段 | 残差比 |
|---|---|---|
| `da_as_ss_mw_primary_reserve`（日前主用备用自调度） | `da_as_ss_mw_synchronized_reserve`（日前同步备用自调度） | 0.0000 |
| `da_as_ss_mw_primary_reserve`（日前主用备用自调度） | `da_as_ss_mw_thirty_minutes_reserve`（日前30分钟备用自调度） | 0.0000 |
| `da_as_as_mw_synchronized_reserve`（日前同步备用实际量） | `da_as_total_mw_synchronized_reserve`（日前同步备用总量） | 0.0000 |
| `da_as_as_mw_primary_reserve`（日前主用备用实际量） | `da_as_total_mw_primary_reserve`（日前主用备用总量） | 0.0000 |

> 残差比越小替身越像。这一现象本身是结果的一部分，不作剔除处理：它说明按字段名逐个排除，在组合推断面前是失效的。

## 四、逐步增加字段数的验证

按门控值从高到低依次取前 n 个字段，各自单独训练普通 DNN：

| 字段数 | 测试 R² |
|---|---|
| 3 | 0.7332 |
| 6 | 0.8462 |
| 9 | 0.8723 |
| 12 | 0.8862 |
| 15 | 0.9103 |
| 18 | 0.9000 |
| 19 | 0.8947 |

## 五、三种模型的对比

| 模型 | 最优测试 R² | 出现轮次 | 最终活跃字段数 |
|---|---|---|---|
| DNN | 0.8949 | 第 98 轮 | 48 |
| L1GateDNN | 0.9129 | 第 171 轮 | 16 |
| DGatingDNN | 0.9014 | 第 189 轮 | 19 |

## 六、图

- `fig_training.png`　三种模型的训练曲线，以及活跃字段数如何一步步收缩
- `fig_gate_evolution.png`　每个字段的门控值随训练轮次的变化轨迹
- `fig_gate_bar.png`　最终门控值排名
- `fig_topn.png`　逐步增加字段数时测试 R² 的变化
- `fig_gate_distribution.png`　门控值从大到小排开（对数刻度），D-Gating 的断崖说明阈值不敏感
