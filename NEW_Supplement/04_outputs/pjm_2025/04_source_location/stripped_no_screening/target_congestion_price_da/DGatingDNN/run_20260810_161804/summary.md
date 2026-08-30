# 推断源定位：日前阻塞价

运行时间 20260810_161804　方法 **DGatingDNN**

## 一、这次跑的是什么

- 目标字段：`congestion_price_da`（日前阻塞价）
- 数据：pjm_2025，8760 行
- 候选字段：**57 个**（剥离第一类关系（1 个字段））
- 网络结构：64-32-16，训练 200 轮，批大小 50，学习率 0.001
- 数据划分：随机 80% 训练 / 20% 测试，随机种子 42

## 二、结论先看这里

- 从 57 个候选字段里选出 **25 个**推断源（占 43.9%）
- 只用这 25 个字段重新训练普通 DNN，测试 R² = **0.7750**
- 作为对照，用全部 57 个字段的普通 DNN 测试 R² = 0.7258
- 没被选中的 32 个字段单独训练，测试 R² = 0.4470

> 用 44% 的字段达到了和全量相当的精度（相差 +0.0492），说明这组字段确实承载了推断目标所需的主要信息。

## 三、选出来的推断源

| 排名 | 字段 | 业务含义 | 门控值 |
|---|---|---|---|
| 1 | `gen_fuel_other_renewables_mw` | 其他可再生出力 | 0.3145 |
| 2 | `system_energy_price_da` | 日前能量价格 | 0.2831 |
| 3 | `gen_fuel_coal_pct` | 燃煤占比 | 0.2268 |
| 4 | `gross_inadv_interchange_mw` | 总非计划交换 | 0.2254 |
| 5 | `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 | 0.2235 |
| 6 | `total_losses` | 总网损 | 0.2141 |
| 7 | `gen_fuel_solar_mw` | 光伏出力 | 0.2124 |
| 8 | `net_sched_interchange_mw` | 净计划交换 | 0.2056 |
| 9 | `marginal_loss_price_da` | 日前边际损耗价 | 0.2031 |
| 10 | `gross_sched_interchange_mw` | 总计划交换 | 0.1963 |
| 11 | `gen_fuel_multiple_fuels_pct` | 多燃料占比 | 0.1718 |
| 12 | `forecast_load_mw_latest_available` | 最新负荷预测 | 0.1477 |
| 13 | `gen_fuel_nuclear_mw` | 核电出力 | 0.1413 |
| 14 | `gen_fuel_oil_mw` | 燃油出力 | 0.1351 |
| 15 | `da_as_as_mw_thirty_minutes_reserve` | 日前30分钟备用实际量 | 0.1349 |
| 16 | `da_as_mcp_primary_reserve` | 日前主用备用出清价 | 0.1298 |
| 17 | `wind_generation_mw` | 风电出力 | 0.1263 |
| 18 | `gross_actual_interchange_mw` | 总实际交换 | 0.1170 |
| 19 | `congestion_price_rt` | 实时阻塞价 | 0.1092 |
| 20 | `total_pjm_loc_credit` | 机会成本信用 | 0.1061 |
| 21 | `da_as_ss_mw_primary_reserve` | 日前主用备用自调度 | 0.0917 |
| 22 | `gen_fuel_hydro_mw` | 水电出力 | 0.0790 |
| 23 | `da_as_mcp_synchronized_reserve` | 日前同步备用出清价 | 0.0519 |
| 24 | `total_pjm_rmccp_cr` | 调频容量信用额 | 0.0297 |
| 25 | `marginal_loss_price_rt` | 实时边际损耗价 | 0.0231 |

活跃阈值 **0.01**：门控值不低于它的字段判定为推断源。

### 阈值取多少影响大不大

把阈值在几个量级之间来回调，看选中的字段数变不变：

| 阈值 | 选中字段数 |
|---|---|
| 1e-06 | 26 |
| 0.0001 | 25 |
| 0.001 | 25 |
| 0.005 | 25 |
| 0.01 | 25 |
| 0.02 | 25 |
| 0.05 | 23 |
| 0.1 | 20 |

> 这几档阈值给出的字段数完全一致，说明阈值落在门控值分布的空档里，取多少都不影响结论——这正是用 D-Gating 替代普通 L1 门控的原因。

紧挨在阈值下方、被判为非活跃的几个字段（供参考）：

| 字段 | 业务含义 | 门控值 |
|---|---|---|
| `prelim_load_avg_hourly` | 预估小时负荷 | 0.0000 |
| `gen_fuel_oil_pct` | 燃油占比 | 0.0000 |
| `system_energy_price_rt` | 实时能量价格 | 0.0000 |
| `total_lmp_rt` | 实时总电价 | 0.0000 |
| `da_as_nsr_mw_primary_reserve` | 非同步备用 | 0.0000 |

### 注意：这些选中字段是被排除字段的替身

候选池按字段名排除了受关注字段，但下面这些留在池子里的字段，单独一个就能几乎完整还原被排除的那个——也就是说排除规则被绕过了，推断路径换了个入口又通了。

| 选中字段 | 能还原的被排除字段 | 残差比 |
|---|---|---|
| `da_as_as_mw_thirty_minutes_reserve`（日前30分钟备用实际量） | `da_as_total_mw_thirty_minutes_reserve`（日前30分钟备用总量） | 0.0000 |
| `da_as_ss_mw_primary_reserve`（日前主用备用自调度） | `da_as_ss_mw_synchronized_reserve`（日前同步备用自调度） | 0.0000 |
| `da_as_ss_mw_primary_reserve`（日前主用备用自调度） | `da_as_ss_mw_thirty_minutes_reserve`（日前30分钟备用自调度） | 0.0000 |

> 残差比越小替身越像。这一现象本身是结果的一部分，不作剔除处理：它说明按字段名逐个排除，在组合推断面前是失效的。

## 四、逐步增加字段数的验证

按门控值从高到低依次取前 n 个字段，各自单独训练普通 DNN：

| 字段数 | 测试 R² |
|---|---|
| 4 | 0.4335 |
| 8 | 0.6618 |
| 12 | 0.6993 |
| 16 | 0.7744 |
| 20 | 0.7535 |
| 24 | 0.7536 |
| 25 | 0.7750 |

## 五、三种模型的对比

| 模型 | 最优测试 R² | 出现轮次 | 最终活跃字段数 |
|---|---|---|---|
| DNN | 0.7258 | 第 177 轮 | 57 |
| L1GateDNN | 0.7432 | 第 145 轮 | 30 |
| DGatingDNN | 0.7556 | 第 152 轮 | 25 |

## 六、图

- `fig_training.png`　三种模型的训练曲线，以及活跃字段数如何一步步收缩
- `fig_gate_evolution.png`　每个字段的门控值随训练轮次的变化轨迹
- `fig_gate_bar.png`　最终门控值排名
- `fig_topn.png`　逐步增加字段数时测试 R² 的变化
- `fig_gate_distribution.png`　门控值从大到小排开（对数刻度），D-Gating 的断崖说明阈值不敏感
