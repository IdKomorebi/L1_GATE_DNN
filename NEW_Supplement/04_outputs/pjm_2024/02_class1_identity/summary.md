# 第一类关系检测汇总 — pjm_2024（main 版本）

数据规模：8784 行 × 69 列，跨年复验数据 pjm_2025

## 门槛扫描

| 门槛 | 关系条数 | 是否平台 |
|---|---|---|
| 1e-13 | 11 | 是 |
| 1e-12 | 11 | 是 |
| 1e-11 | 11 | 是 |
| 1e-10 | 11 | 是 |
| 1e-09 | 11 | 是 |
| 1e-08 | 12 | 是 |
| 1e-07 | 12 | 是 |
| 1e-06 | 13 |  |
| 1e-05 | 14 |  |
| 1e-04 | 15 |  |

采用求秩门槛 1e-05、残差门槛 1e-03。

## 关系分类

- constant：5 条
- duplicate：5 条
- 恒等式：3 条
- 近似关系：1 条

## 全部关系

| 空间 | 档次 | 残差比 | 跨年复验 | 系数整齐 | 关系 |
|---|---|---|---|---|---|
| linear | constant | 0.00e+00 | 0.00e+00 | 是 | `gen_fuel_storage_pct = 0` |
| linear | constant | 0.00e+00 | 0.00e+00 | 是 | `da_as_mcp_thirty_minutes_reserve = 0` |
| linear | duplicate | 1.95e-13 | 1.66e-13 | 是 | `da_as_total_mw_primary_reserve = +1*da_as_as_mw_primary_reserve` |
| linear | duplicate | 6.84e-14 | 6.01e-14 | 是 | `da_as_total_mw_synchronized_reserve = +1*da_as_as_mw_synchronized_reserve` |
| linear | duplicate | 2.36e-14 | 2.20e-14 | 是 | `da_as_total_mw_thirty_minutes_reserve = +1*da_as_as_mw_thirty_minutes_reserve` |
| linear | duplicate | 1.14e-14 | 1.19e-14 | 是 | `da_as_ss_mw_primary_reserve = +1*da_as_ss_mw_thirty_minutes_reserve` |
| linear | duplicate | 1.14e-14 | 1.19e-14 | 是 | `da_as_ss_mw_synchronized_reserve = +1*da_as_ss_mw_thirty_minutes_reserve` |
| linear | constant | 0.00e+00 | 0.00e+00 | 是 | `da_as_ircmwt2_primary_reserve = 0` |
| linear | constant | 0.00e+00 | 0.00e+00 | 是 | `da_as_ircmwt2_synchronized_reserve = 0` |
| linear | constant | 0.00e+00 | 0.00e+00 | 是 | `da_as_ircmwt2_thirty_minutes_reserve = 0` |
| linear | 恒等式 | 3.47e-15 | 3.17e-15 | 是 | `net_actual_interchange_mw = -1*net_inadv_interchange_mw +1*net_sched_interchange_mw` |
| linear | 恒等式 | 8.64e-06 | 2.26e-05 | 是 | `da_as_nsr_mw_primary_reserve = +1*da_as_total_mw_primary_reserve -1*da_as_total_mw_synchronized_reserve -0.00180767` |
| linear | 恒等式 | 2.60e-08 | 1.53e-08 | 是 | `system_energy_price_da = -1*congestion_price_da -1*marginal_loss_price_da +1*total_lmp_da +1.1132e-08` |
| linear | 近似关系 | 2.87e-04 | 2.21e-04 | 否 | `system_energy_price_rt = -1.0265*congestion_price_rt +0.999252*total_lmp_rt +0.00417477` |
| log | duplicate | 2.48e-12 | 2.14e-12 | 是 | `da_as_total_mw_primary_reserve = exp +1*da_as_as_mw_primary_reserve` |
| log | duplicate | 1.41e-12 | 1.27e-12 | 是 | `da_as_total_mw_synchronized_reserve = exp +1*da_as_as_mw_synchronized_reserve` |
| log | duplicate | 1.81e-13 | 1.62e-13 | 是 | `da_as_total_mw_thirty_minutes_reserve = exp +1*da_as_as_mw_thirty_minutes_reserve` |