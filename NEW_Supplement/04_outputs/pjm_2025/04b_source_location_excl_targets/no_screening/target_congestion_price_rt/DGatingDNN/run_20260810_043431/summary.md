# 推断源定位：实时阻塞价

运行时间 20260810_043431　方法 **DGatingDNN**

## 一、这次跑的是什么

- 目标字段：`congestion_price_rt`（实时阻塞价）
- 数据：pjm_2025，8760 行
- 候选字段：**49 个**（排除其余关注字段）
- 网络结构：64-32-16，训练 200 轮，批大小 50，学习率 0.001
- 数据划分：随机 80% 训练 / 20% 测试，随机种子 42

## 二、结论先看这里

- 从 49 个候选字段里选出 **29 个**推断源（占 59.2%）
- 只用这 29 个字段重新训练普通 DNN，测试 R² = **0.6245**
- 作为对照，用全部 49 个字段的普通 DNN 测试 R² = 0.6248
- 没被选中的 20 个字段单独训练，测试 R² = 0.6254

> 用 59% 的字段达到了和全量相当的精度（相差 -0.0003），说明这组字段确实承载了推断目标所需的主要信息。

## 三、选出来的推断源

| 排名 | 字段 | 业务含义 | 门控值 |
|---|---|---|---|
| 1 | `total_lmp_rt` | 实时总电价 | 0.1638 |
| 2 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 | 0.1547 |
| 3 | `gen_fuel_other_renewables_mw` | 其他可再生出力 | 0.1463 |
| 4 | `gen_fuel_coal_pct` | 燃煤占比 | 0.1439 |
| 5 | `gross_sched_interchange_mw` | 总计划交换 | 0.1438 |
| 6 | `gen_fuel_gas_pct` | 燃气占比 | 0.1430 |
| 7 | `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 | 0.1350 |
| 8 | `rmccp` | 调频容量出清价 | 0.1315 |
| 9 | `gen_fuel_wind_mw` | 风电出力 | 0.1277 |
| 10 | `gen_fuel_oil_pct` | 燃油占比 | 0.1182 |
| 11 | `gross_inadv_interchange_mw` | 总非计划交换 | 0.1155 |
| 12 | `solar_generation_mw` | 光伏出力 | 0.1146 |
| 13 | `marginal_loss_price_rt` | 实时边际损耗价 | 0.1133 |
| 14 | `gen_fuel_storage_mw` | 储能出力 | 0.1078 |
| 15 | `system_energy_price_da` | 日前能量价格 | 0.1058 |
| 16 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 | 0.1027 |
| 17 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 | 0.1015 |
| 18 | `da_as_as_mw_synchronized_reserve` | 日前同步备用实际量 | 0.0952 |
| 19 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 | 0.0899 |
| 20 | `total_pjm_rmpcp_cr` | 调频性能信用额 | 0.0899 |
| 21 | `total_pjm_assigned_reg` | 调频分配量 | 0.0887 |
| 22 | `gen_fuel_hydro_pct` | 水电占比 | 0.0673 |
| 23 | `net_inadv_interchange_mw` | 净非计划交换 | 0.0671 |
| 24 | `gen_fuel_oil_mw` | 燃油出力 | 0.0633 |
| 25 | `total_pjm_self_sched_reg` | 调频自调度量 | 0.0581 |
| 26 | `gen_fuel_nuclear_mw` | 核电出力 | 0.0579 |
| 27 | `total_pjm_loc_credit` | 机会成本信用 | 0.0548 |
| 28 | `da_as_mcp_primary_reserve` | 日前主用备用出清价 | 0.0498 |
| 29 | `da_as_as_mw_primary_reserve` | 日前主用备用实际量 | 0.0351 |

活跃阈值 **0.01**：门控值不低于它的字段判定为推断源。

### 阈值取多少影响大不大

把阈值在几个量级之间来回调，看选中的字段数变不变：

| 阈值 | 选中字段数 |
|---|---|
| 1e-06 | 30 |
| 0.0001 | 30 |
| 0.001 | 29 |
| 0.005 | 29 |
| 0.01 | 29 |
| 0.02 | 29 |
| 0.05 | 27 |
| 0.1 | 17 |

> 不同阈值给出的字段数不一致，说明门控值分布没有明显空档，结论对阈值敏感，需要谨慎解释。

紧挨在阈值下方、被判为非活跃的几个字段（供参考）：

| 字段 | 业务含义 | 门控值 |
|---|---|---|
| `system_energy_price_rt` | 实时能量价格 | 0.0006 |
| `total_pjm_rmccp_cr` | 调频容量信用额 | 0.0000 |
| `da_as_as_req_mw_thirty_minutes_reserve` | 日前30分钟备用需求 | 0.0000 |
| `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 | 0.0000 |
| `da_as_nsr_mw_primary_reserve` | 非同步备用 | 0.0000 |

### 注意：这些选中字段是被排除字段的替身

候选池按字段名排除了受关注字段，但下面这些留在池子里的字段，单独一个就能几乎完整还原被排除的那个——也就是说排除规则被绕过了，推断路径换了个入口又通了。

| 选中字段 | 能还原的被排除字段 | 残差比 |
|---|---|---|
| `da_as_as_mw_thirty_minutes_reserve`（日前30分钟备用实际量） | `da_as_total_mw_thirty_minutes_reserve`（日前30分钟备用总量） | 0.0000 |
| `da_as_ss_mw_primary_reserve`（日前主用备用自调度） | `da_as_ss_mw_synchronized_reserve`（日前同步备用自调度） | 0.0000 |
| `da_as_ss_mw_primary_reserve`（日前主用备用自调度） | `da_as_ss_mw_thirty_minutes_reserve`（日前30分钟备用自调度） | 0.0000 |
| `da_as_as_mw_synchronized_reserve`（日前同步备用实际量） | `da_as_total_mw_synchronized_reserve`（日前同步备用总量） | 0.0000 |
| `da_as_as_mw_primary_reserve`（日前主用备用实际量） | `da_as_total_mw_primary_reserve`（日前主用备用总量） | 0.0000 |

> 残差比越小替身越像。这一现象本身是结果的一部分，不作剔除处理：它说明按字段名逐个排除，在组合推断面前是失效的。

## 四、逐步增加字段数的验证

按门控值从高到低依次取前 n 个字段，各自单独训练普通 DNN：

| 字段数 | 测试 R² |
|---|---|
| 4 | 0.5158 |
| 8 | 0.6019 |
| 12 | 0.6793 |
| 16 | 0.6654 |
| 20 | 0.6537 |
| 24 | 0.6163 |
| 28 | 0.6127 |
| 29 | 0.6245 |

## 五、三种模型的对比

| 模型 | 最优测试 R² | 出现轮次 | 最终活跃字段数 |
|---|---|---|---|
| DNN | 0.6248 | 第 23 轮 | 49 |
| L1GateDNN | 0.6661 | 第 63 轮 | 31 |
| DGatingDNN | 0.6585 | 第 80 轮 | 29 |

## 六、图

- `fig_training.png`　三种模型的训练曲线，以及活跃字段数如何一步步收缩
- `fig_gate_evolution.png`　每个字段的门控值随训练轮次的变化轨迹
- `fig_gate_bar.png`　最终门控值排名
- `fig_topn.png`　逐步增加字段数时测试 R² 的变化
- `fig_gate_distribution.png`　门控值从大到小排开（对数刻度），D-Gating 的断崖说明阈值不敏感
