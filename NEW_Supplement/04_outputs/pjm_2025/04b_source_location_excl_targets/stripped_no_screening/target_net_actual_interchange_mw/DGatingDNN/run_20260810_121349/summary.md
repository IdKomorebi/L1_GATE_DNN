# 推断源定位：净实际交换功率

运行时间 20260810_121349　方法 **DGatingDNN**

## 一、这次跑的是什么

- 目标字段：`net_actual_interchange_mw`（净实际交换功率）
- 数据：pjm_2025，8760 行
- 候选字段：**49 个**（排除其余关注字段；剥离第一类关系（0 个字段））
- 网络结构：64-32-16，训练 200 轮，批大小 50，学习率 0.001
- 数据划分：随机 80% 训练 / 20% 测试，随机种子 42

## 二、结论先看这里

- 从 49 个候选字段里选出 **19 个**推断源（占 38.8%）
- 只用这 19 个字段重新训练普通 DNN，测试 R² = **0.9697**
- 作为对照，用全部 49 个字段的普通 DNN 测试 R² = 0.9674
- 没被选中的 30 个字段单独训练，测试 R² = 0.8204

> 用 39% 的字段达到了和全量相当的精度（相差 +0.0022），说明这组字段确实承载了推断目标所需的主要信息。

## 三、选出来的推断源

| 排名 | 字段 | 业务含义 | 门控值 |
|---|---|---|---|
| 1 | `prelim_load_avg_hourly` | 预估小时负荷 | 0.2703 |
| 2 | `gen_fuel_gas_mw` | 燃气出力 | 0.2396 |
| 3 | `total_pjm_rt_load_mwh` | 实时负荷电量 | 0.1715 |
| 4 | `gen_fuel_coal_mw` | 燃煤出力 | 0.1556 |
| 5 | `gen_fuel_nuclear_mw` | 核电出力 | 0.1069 |
| 6 | `gen_fuel_solar_mw` | 光伏出力 | 0.0972 |
| 7 | `gen_fuel_hydro_mw` | 水电出力 | 0.0758 |
| 8 | `gen_fuel_nuclear_pct` | 核电占比 | 0.0709 |
| 9 | `gen_fuel_wind_mw` | 风电出力 | 0.0622 |
| 10 | `gross_sched_interchange_mw` | 总计划交换 | 0.0542 |
| 11 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 | 0.0539 |
| 12 | `gen_fuel_other_renewables_mw` | 其他可再生出力 | 0.0525 |
| 13 | `gen_fuel_multiple_fuels_mw` | 多燃料出力 | 0.0284 |
| 14 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 | 0.0284 |
| 15 | `gen_fuel_oil_mw` | 燃油出力 | 0.0276 |
| 16 | `da_as_as_mw_primary_reserve` | 日前主用备用实际量 | 0.0272 |
| 17 | `gross_inadv_interchange_mw` | 总非计划交换 | 0.0243 |
| 18 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 | 0.0236 |
| 19 | `total_pjm_reg_purchases` | 调频购买量 | 0.0146 |

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
| 0.02 | 18 |
| 0.05 | 12 |
| 0.1 | 5 |

> 不同阈值给出的字段数不一致，说明门控值分布没有明显空档，结论对阈值敏感，需要谨慎解释。

紧挨在阈值下方、被判为非活跃的几个字段（供参考）：

| 字段 | 业务含义 | 门控值 |
|---|---|---|
| `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 | 0.0000 |
| `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 | 0.0000 |
| `da_as_nsr_mw_primary_reserve` | 非同步备用 | 0.0000 |
| `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 | 0.0000 |
| `da_as_mcp_primary_reserve` | 日前主用备用出清价 | 0.0000 |

### 注意：这些选中字段是被排除字段的替身

候选池按字段名排除了受关注字段，但下面这些留在池子里的字段，单独一个就能几乎完整还原被排除的那个——也就是说排除规则被绕过了，推断路径换了个入口又通了。

| 选中字段 | 能还原的被排除字段 | 残差比 |
|---|---|---|
| `da_as_as_mw_thirty_minutes_reserve`（日前30分钟备用实际量） | `da_as_total_mw_thirty_minutes_reserve`（日前30分钟备用总量） | 0.0000 |
| `da_as_as_mw_synchronized_reserve`（日前同步备用实际量） | `da_as_total_mw_synchronized_reserve`（日前同步备用总量） | 0.0000 |
| `da_as_as_mw_primary_reserve`（日前主用备用实际量） | `da_as_total_mw_primary_reserve`（日前主用备用总量） | 0.0000 |
| `prelim_load_avg_hourly`（预估小时负荷） | `metered_load_mw`（计量负荷） | 0.0071 |
| `total_pjm_rt_load_mwh`（实时负荷电量） | `metered_load_mw`（计量负荷） | 0.0141 |

> 残差比越小替身越像。这一现象本身是结果的一部分，不作剔除处理：它说明按字段名逐个排除，在组合推断面前是失效的。

## 四、逐步增加字段数的验证

按门控值从高到低依次取前 n 个字段，各自单独训练普通 DNN：

| 字段数 | 测试 R² |
|---|---|
| 3 | 0.3536 |
| 6 | 0.7318 |
| 9 | 0.9389 |
| 12 | 0.9605 |
| 15 | 0.9701 |
| 18 | 0.9717 |
| 19 | 0.9697 |

## 五、三种模型的对比

| 模型 | 最优测试 R² | 出现轮次 | 最终活跃字段数 |
|---|---|---|---|
| DNN | 0.9674 | 第 164 轮 | 49 |
| L1GateDNN | 0.9734 | 第 177 轮 | 11 |
| DGatingDNN | 0.9688 | 第 148 轮 | 19 |

## 六、图

- `fig_training.png`　三种模型的训练曲线，以及活跃字段数如何一步步收缩
- `fig_gate_evolution.png`　每个字段的门控值随训练轮次的变化轨迹
- `fig_gate_bar.png`　最终门控值排名
- `fig_topn.png`　逐步增加字段数时测试 R² 的变化
- `fig_gate_distribution.png`　门控值从大到小排开（对数刻度），D-Gating 的断崖说明阈值不敏感
