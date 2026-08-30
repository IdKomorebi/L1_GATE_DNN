# 推断源定位：日前主用备用总量

运行时间 20260811_010932　方法 **DGatingDNN**

## 一、这次跑的是什么

- 目标字段：`da_as_total_mw_primary_reserve`（日前主用备用总量）
- 数据：pjm_2025，8760 行
- 候选字段：**47 个**（排除其余关注字段；剥离第一类关系（1 个字段））
- 网络结构：64-32-16，训练 200 轮，批大小 50，学习率 0.001
- 数据划分：随机 80% 训练 / 20% 测试，随机种子 42

## 二、结论先看这里

- 从 47 个候选字段里选出 **17 个**推断源（占 36.2%）
- 只用这 17 个字段重新训练普通 DNN，测试 R² = **0.8531**
- 作为对照，用全部 47 个字段的普通 DNN 测试 R² = 0.8588
- 没被选中的 30 个字段单独训练，测试 R² = 0.7039

> 用 36% 的字段达到了和全量相当的精度（相差 -0.0057），说明这组字段确实承载了推断目标所需的主要信息。

## 三、选出来的推断源

| 排名 | 字段 | 业务含义 | 门控值 |
|---|---|---|---|
| 1 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 | 0.3849 |
| 2 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 | 0.3719 |
| 3 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 | 0.2616 |
| 4 | `gen_fuel_nuclear_mw` | 核电出力 | 0.1671 |
| 5 | `gen_fuel_hydro_pct` | 水电占比 | 0.1660 |
| 6 | `gen_fuel_coal_mw` | 燃煤出力 | 0.1518 |
| 7 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 | 0.1069 |
| 8 | `total_pjm_rt_load_mwh` | 实时负荷电量 | 0.1051 |
| 9 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 | 0.0904 |
| 10 | `gen_fuel_wind_mw` | 风电出力 | 0.0808 |
| 11 | `total_pjm_rmpcp_cr` | 调频性能信用额 | 0.0767 |
| 12 | `total_pjm_self_sched_reg` | 调频自调度量 | 0.0694 |
| 13 | `gen_fuel_solar_pct` | 光伏占比 | 0.0694 |
| 14 | `gross_inadv_interchange_mw` | 总非计划交换 | 0.0686 |
| 15 | `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 | 0.0556 |
| 16 | `total_pjm_assigned_reg` | 调频分配量 | 0.0391 |
| 17 | `total_pjm_reg_purchases` | 调频购买量 | 0.0290 |

活跃阈值 **0.01**：门控值不低于它的字段判定为推断源。

### 阈值取多少影响大不大

把阈值在几个量级之间来回调，看选中的字段数变不变：

| 阈值 | 选中字段数 |
|---|---|
| 1e-06 | 17 |
| 0.0001 | 17 |
| 0.001 | 17 |
| 0.005 | 17 |
| 0.01 | 17 |
| 0.02 | 17 |
| 0.05 | 15 |
| 0.1 | 8 |

> 这几档阈值给出的字段数完全一致，说明阈值落在门控值分布的空档里，取多少都不影响结论——这正是用 D-Gating 替代普通 L1 门控的原因。

紧挨在阈值下方、被判为非活跃的几个字段（供参考）：

| 字段 | 业务含义 | 门控值 |
|---|---|---|
| `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 | 0.0000 |
| `system_energy_price_da` | 日前能量价格 | 0.0000 |
| `system_energy_price_rt` | 实时能量价格 | 0.0000 |
| `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 | 0.0000 |
| `da_as_mcp_primary_reserve` | 日前主用备用出清价 | 0.0000 |

### 注意：这些选中字段是被排除字段的替身

候选池按字段名排除了受关注字段，但下面这些留在池子里的字段，单独一个就能几乎完整还原被排除的那个——也就是说排除规则被绕过了，推断路径换了个入口又通了。

| 选中字段 | 能还原的被排除字段 | 残差比 |
|---|---|---|
| `da_as_as_mw_thirty_minutes_reserve`（日前30分钟备用实际量） | `da_as_total_mw_thirty_minutes_reserve`（日前30分钟备用总量） | 0.0000 |
| `da_as_ss_mw_primary_reserve`（日前主用备用自调度） | `da_as_ss_mw_synchronized_reserve`（日前同步备用自调度） | 0.0000 |
| `da_as_ss_mw_primary_reserve`（日前主用备用自调度） | `da_as_ss_mw_thirty_minutes_reserve`（日前30分钟备用自调度） | 0.0000 |
| `da_as_as_mw_synchronized_reserve`（日前同步备用实际量） | `da_as_total_mw_synchronized_reserve`（日前同步备用总量） | 0.0000 |
| `total_pjm_rt_load_mwh`（实时负荷电量） | `metered_load_mw`（计量负荷） | 0.0141 |

> 残差比越小替身越像。这一现象本身是结果的一部分，不作剔除处理：它说明按字段名逐个排除，在组合推断面前是失效的。

## 四、逐步增加字段数的验证

按门控值从高到低依次取前 n 个字段，各自单独训练普通 DNN：

| 字段数 | 测试 R² |
|---|---|
| 2 | 0.7206 |
| 4 | 0.8736 |
| 6 | 0.8537 |
| 8 | 0.8751 |
| 10 | 0.8641 |
| 12 | 0.8665 |
| 14 | 0.8462 |
| 16 | 0.8498 |
| 17 | 0.8531 |

## 五、三种模型的对比

| 模型 | 最优测试 R² | 出现轮次 | 最终活跃字段数 |
|---|---|---|---|
| DNN | 0.8588 | 第 31 轮 | 47 |
| L1GateDNN | 0.8898 | 第 170 轮 | 22 |
| DGatingDNN | 0.8841 | 第 194 轮 | 17 |

## 六、图

- `fig_training.png`　三种模型的训练曲线，以及活跃字段数如何一步步收缩
- `fig_gate_evolution.png`　每个字段的门控值随训练轮次的变化轨迹
- `fig_gate_bar.png`　最终门控值排名
- `fig_topn.png`　逐步增加字段数时测试 R² 的变化
- `fig_gate_distribution.png`　门控值从大到小排开（对数刻度），D-Gating 的断崖说明阈值不敏感
