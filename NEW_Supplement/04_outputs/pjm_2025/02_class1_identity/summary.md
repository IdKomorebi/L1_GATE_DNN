# 第一类关系检测汇总 — pjm_2025（main 版本）

数据规模：8760 行 × 69 列，跨年复验数据 pjm_2024

## 门槛扫描

| 门槛 | 关系条数 | 是否平台 |
|---|---|---|
| 1e-13 | 11 | 是 |
| 1e-12 | 11 | 是 |
| 1e-11 | 11 | 是 |
| 1e-10 | 11 | 是 |
| 1e-09 | 12 | 是 |
| 1e-08 | 12 | 是 |
| 1e-07 | 12 | 是 |
| 1e-06 | 13 |  |
| 1e-05 | 15 |  |
| 1e-04 | 16 |  |

采用求秩门槛 1e-05、残差门槛 1e-03。

## 关系分类

- constant：5 条
- duplicate：5 条
- 恒等式：3 条
- 规则型(跨年退化)：1 条
- 近似关系：1 条

## 全部关系

| 空间 | 档次 | 残差比 | 跨年复验 | 系数整齐 | 关系 |
|---|---|---|---|---|---|
| linear | constant | 0.00e+00 | 0.00e+00 | 是 | `gen_fuel_storage_pct = 0` |
| linear | constant | 0.00e+00 | 0.00e+00 | 是 | `da_as_mcp_thirty_minutes_reserve = 0` |
| linear | duplicate | 1.34e-13 | 1.57e-13 | 是 | `da_as_total_mw_primary_reserve = +1*da_as_as_mw_primary_reserve` |
| linear | duplicate | 7.42e-14 | 8.45e-14 | 是 | `da_as_total_mw_synchronized_reserve = +1*da_as_as_mw_synchronized_reserve` |
| linear | duplicate | 9.12e-15 | 9.88e-15 | 是 | `da_as_total_mw_thirty_minutes_reserve = +1*da_as_as_mw_thirty_minutes_reserve` |
| linear | duplicate | 1.29e-14 | 1.23e-14 | 是 | `da_as_ss_mw_primary_reserve = +1*da_as_ss_mw_thirty_minutes_reserve` |
| linear | duplicate | 1.29e-14 | 1.23e-14 | 是 | `da_as_ss_mw_synchronized_reserve = +1*da_as_ss_mw_thirty_minutes_reserve` |
| linear | constant | 0.00e+00 | 0.00e+00 | 是 | `da_as_ircmwt2_primary_reserve = 0` |
| linear | constant | 0.00e+00 | 0.00e+00 | 是 | `da_as_ircmwt2_synchronized_reserve = 0` |
| linear | constant | 0.00e+00 | 0.00e+00 | 是 | `da_as_ircmwt2_thirty_minutes_reserve = 0` |
| linear | 恒等式 | 2.01e-15 | 2.15e-15 | 是 | `net_actual_interchange_mw = -1*net_inadv_interchange_mw +1*net_sched_interchange_mw` |
| linear | 规则型(跨年退化) | 8.59e-05 | 3.93e-02 | 否 | `da_as_as_req_mw_primary_reserve = +1.49995*da_as_as_req_mw_synchronized_reserve -94.9004` |
| linear | 恒等式 | 2.24e-05 | 9.15e-06 | 否 | `da_as_nsr_mw_primary_reserve = +1*da_as_total_mw_primary_reserve -1*da_as_total_mw_synchronized_reserve +0.00367707` |
| linear | 恒等式 | 1.53e-08 | 2.60e-08 | 是 | `system_energy_price_da = -1*congestion_price_da -1*marginal_loss_price_da +1*total_lmp_da -6.29557e-09` |
| linear | 近似关系 | 1.85e-04 | 3.40e-04 | 否 | `system_energy_price_rt = -0.994154*congestion_price_rt +0.999242*total_lmp_rt +0.0049832` |
| log | duplicate | 1.62e-12 | 1.88e-12 | 是 | `da_as_total_mw_primary_reserve = exp +1*da_as_as_mw_primary_reserve` |
| log | duplicate | 3.22e-13 | 3.58e-13 | 是 | `da_as_total_mw_synchronized_reserve = exp +1*da_as_as_mw_synchronized_reserve` |
| log | duplicate | 1.05e-13 | 1.17e-13 | 是 | `da_as_total_mw_thirty_minutes_reserve = exp +1*da_as_as_mw_thirty_minutes_reserve` |
| log | 近似关系 | 5.01e-04 | 3.33e-02 | 否 | `da_as_as_req_mw_primary_reserve = exp +1.02948*da_as_as_req_mw_synchronized_reserve +0.149149` |