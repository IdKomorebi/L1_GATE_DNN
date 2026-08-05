# Stage05 Full-R 路径总结：prelim_load_avg_hourly

## 口径说明

- 中心字段 / target：`net_actual_interchange_mw`
- 被替代字段 xi：`prelim_load_avg_hourly`
- 固定上下文 `C_i`：8 个字段
- Stage04 活跃候选集 `A`：20 个字段，本阶段用它们作为 20 个 anchor/head
- Stage05 full-R 候选池：完整路径外字段 `R`，共 46 个字段
- 每个 head 固定包含自己的 anchor；其余 45 个 `R` 字段由 D-gating 压缩选择
- 本文件先列 20 个 anchor/head 的原始路径，再列全部已验证的去重路径

## 基准效果与目标

- `C_i` only R2：`0.770768`
- `C_i + R` R2：`0.966816`
- 本 xi 的动态目标：`max(0.95, C_i + R - 0.005) = 0.961816`

这里的目标不是固定所有 xi 都用同一个绝对标准。对当前 xi 来说，`C_i+R` 能到 `0.966816`，所以候选替代组需要接近这个上限；如果别的 xi 的 `C_i+R` 只有 `0.956`，它自己的动态目标就应接近 `0.951`，而不是硬套本 xi 的 `0.961816`。

## 20 条 anchor/head 原始路径

以下使用最终选中压缩 trial `trial_04_lam0p001` 的 `threshold_ratio=0.8` 读出。每一行对应一个 Stage04 活跃字段作为 anchor 的 head。R2 是把该路径 `Q` 与固定 `C_i` 合并后重新训练普通 DNN 的真实验证结果；不是残差网络内部的重构 R2。

| 序号 | anchor | |Q| | Q | 验证组 | test R2 | 状态 | anchor状态 |
|---:|---|---:|---|---|---:|---|---|
| 1 | `gen_fuel_other_renewables_mw` | 3 | `gen_fuel_other_renewables_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_012` | 0.957661 | 未达标 | `active_anchor` |
| 2 | `congestion_price_da` | 3 | `congestion_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_013` | 0.949848 | 未达标 | `inactive_or_dummy_anchor` |
| 3 | `total_losses` | 3 | `total_losses`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_003` | 0.954239 | 未达标 | `active_anchor` |
| 4 | `system_energy_price_da` | 3 | `system_energy_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_004` | 0.951400 | 未达标 | `active_anchor` |
| 5 | `total_pjm_self_sched_reg` | 3 | `total_pjm_self_sched_reg`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_011` | 0.947819 | 未达标 | `inactive_or_dummy_anchor` |
| 6 | `total_lmp_da` | 3 | `total_lmp_da`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_006` | 0.951133 | 未达标 | `active_anchor` |
| 7 | `total_pjm_loc_credit` | 3 | `total_pjm_loc_credit`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_020` | 0.949789 | 未达标 | `inactive_or_dummy_anchor` |
| 8 | `gross_inadv_interchange_mw` | 3 | `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_018` | 0.953530 | 未达标 | `active_anchor` |
| 9 | `gen_fuel_multiple_fuels_mw` | 3 | `gen_fuel_multiple_fuels_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_002` | 0.960193 | 未达标 | `active_anchor` |
| 10 | `gen_fuel_multiple_fuels_pct` | 3 | `gen_fuel_multiple_fuels_pct`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_005` | 0.957811 | 未达标 | `active_anchor` |
| 11 | `rmccp` | 3 | `rmccp`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_014` | 0.948669 | 未达标 | `inactive_or_dummy_anchor` |
| 12 | `congestion_price_rt` | 3 | `congestion_price_rt`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_019` | 0.947922 | 未达标 | `active_anchor` |
| 13 | `marginal_loss_price_rt` | 3 | `marginal_loss_price_rt`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_010` | 0.947816 | 未达标 | `active_anchor` |
| 14 | `da_as_total_mw_thirty_minutes_reserve` | 3 | `da_as_total_mw_thirty_minutes_reserve`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_007` | 0.952150 | 未达标 | `active_anchor` |
| 15 | `total_pjm_rmccp_cr` | 3 | `total_pjm_rmccp_cr`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_017` | 0.951072 | 未达标 | `inactive_or_dummy_anchor` |
| 16 | `marginal_loss_price_da` | 3 | `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_015` | 0.950873 | 未达标 | `active_anchor` |
| 17 | `net_inadv_interchange_mw` | 3 | `net_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_008` | 0.947790 | 未达标 | `active_anchor` |
| 18 | `gross_actual_interchange_mw` | 3 | `gross_actual_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_009` | 0.951404 | 未达标 | `active_anchor` |
| 19 | `da_as_mcp_primary_reserve` | 3 | `da_as_mcp_primary_reserve`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `group_016` | 0.950874 | 未达标 | `inactive_or_dummy_anchor` |
| 20 | `metered_load_mw` | 4 | `metered_load_mw`, `gen_fuel_multiple_fuels_pct`, `total_losses`, `da_as_nsr_mw_primary_reserve` | `group_021` | 0.963438 | 达标 | `active_anchor` |

## 去重后的已验证路径

- 已验证去重路径数：`64`
- 达到动态目标的路径数：`15`
- 下表按 `test R2` 从高到低排列；它展示的是不同候选替代组的实际准确率，不再只突出最小组。

| 序号 | group | |Q| | Q | source anchor | test R2 | 目标差值 | 状态 |
|---:|---|---:|---|---|---:|---:|---|
| 1 | `group_039` | 7 | `metered_load_mw`, `gen_fuel_other_renewables_mw`, `gen_fuel_multiple_fuels_pct`, `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `gross_actual_interchange_mw`, `total_losses` | `metered_load_mw` | 0.967547 | +0.005731 | 达标 |
| 2 | `group_061` | 8 | `gen_fuel_multiple_fuels_mw`, `gen_fuel_other_renewables_mw`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct`, `congestion_price_rt` | `gen_fuel_multiple_fuels_mw` | 0.966623 | +0.004808 | 达标 |
| 3 | `group_030` | 5 | `metered_load_mw`, `gen_fuel_multiple_fuels_pct`, `gross_inadv_interchange_mw`, `total_losses`, `da_as_nsr_mw_primary_reserve` | `metered_load_mw` | 0.966604 | +0.004788 | 达标 |
| 4 | `group_064` | 9 | `total_lmp_da`, `gen_fuel_other_renewables_mw`, `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gross_actual_interchange_mw`, `total_losses`, `gen_fuel_coal_pct`, `congestion_price_rt` | `total_lmp_da` | 0.965873 | +0.004057 | 达标 |
| 5 | `group_060` | 8 | `marginal_loss_price_da`, `gen_fuel_other_renewables_mw`, `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gross_actual_interchange_mw`, `total_losses`, `gen_fuel_coal_pct` | `marginal_loss_price_da` | 0.964848 | +0.003032 | 达标 |
| 6 | `group_048` | 7 | `da_as_total_mw_thirty_minutes_reserve`, `gen_fuel_other_renewables_mw`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `da_as_total_mw_thirty_minutes_reserve` | 0.964630 | +0.002814 | 达标 |
| 7 | `group_047` | 7 | `metered_load_mw`, `gen_fuel_other_renewables_mw`, `marginal_loss_price_da`, `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `total_losses`, `gen_fuel_coal_pct` | `metered_load_mw` | 0.964372 | +0.002557 | 达标 |
| 8 | `group_059` | 8 | `congestion_price_rt`, `gen_fuel_other_renewables_mw`, `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gross_actual_interchange_mw`, `total_losses`, `gen_fuel_coal_pct` | `congestion_price_rt` | 0.964189 | +0.002373 | 达标 |
| 9 | `group_041` | 7 | `total_losses`, `gen_fuel_other_renewables_mw`, `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gross_actual_interchange_mw`, `gen_fuel_coal_pct` | `total_losses` | 0.963693 | +0.001878 | 达标 |
| 10 | `group_021` | 4 | `metered_load_mw`, `gen_fuel_multiple_fuels_pct`, `total_losses`, `da_as_nsr_mw_primary_reserve` | `metered_load_mw` | 0.963438 | +0.001622 | 达标 |
| 11 | `group_050` | 7 | `marginal_loss_price_rt`, `gen_fuel_other_renewables_mw`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `marginal_loss_price_rt` | 0.963050 | +0.001235 | 达标 |
| 12 | `group_051` | 7 | `gross_inadv_interchange_mw`, `gen_fuel_other_renewables_mw`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `gross_inadv_interchange_mw` | 0.962757 | +0.000941 | 达标 |
| 13 | `group_042` | 7 | `gen_fuel_other_renewables_mw`, `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gross_actual_interchange_mw`, `total_losses`, `gen_fuel_coal_pct` | `gen_fuel_other_renewables_mw` | 0.962748 | +0.000933 | 达标 |
| 14 | `group_052` | 7 | `gen_fuel_multiple_fuels_pct`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct`, `congestion_price_rt` | `gen_fuel_multiple_fuels_pct` | 0.962730 | +0.000914 | 达标 |
| 15 | `group_058` | 8 | `system_energy_price_da`, `gen_fuel_other_renewables_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gross_actual_interchange_mw`, `total_losses`, `total_pjm_self_sched_reg`, `gen_fuel_coal_pct` | `system_energy_price_da` | 0.962014 | +0.000198 | 达标 |
| 16 | `group_044` | 7 | `marginal_loss_price_rt`, `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gross_actual_interchange_mw`, `total_losses`, `gen_fuel_coal_pct` | `marginal_loss_price_rt` | 0.961781 | -0.000034 | 未达标 |
| 17 | `group_037` | 6 | `gross_actual_interchange_mw`, `gen_fuel_other_renewables_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `gross_actual_interchange_mw` | 0.961515 | -0.000300 | 未达标 |
| 18 | `group_057` | 8 | `congestion_price_da`, `gen_fuel_other_renewables_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gross_actual_interchange_mw`, `total_losses`, `total_pjm_self_sched_reg`, `gen_fuel_coal_pct` | `congestion_price_da` | 0.961479 | -0.000336 | 未达标 |
| 19 | `group_046` | 7 | `total_pjm_self_sched_reg`, `gen_fuel_other_renewables_mw`, `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `total_pjm_self_sched_reg` | 0.961073 | -0.000742 | 未达标 |
| 20 | `group_027` | 4 | `gen_fuel_multiple_fuels_pct`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gen_fuel_coal_pct` | `gen_fuel_multiple_fuels_pct` | 0.960873 | -0.000943 | 未达标 |
| 21 | `group_062` | 8 | `total_pjm_loc_credit`, `gen_fuel_other_renewables_mw`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct`, `congestion_price_rt` | `total_pjm_loc_credit` | 0.960802 | -0.001014 | 未达标 |
| 22 | `group_054` | 7 | `congestion_price_rt`, `gen_fuel_other_renewables_mw`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `congestion_price_rt` | 0.960750 | -0.001065 | 未达标 |
| 23 | `group_002` | 3 | `gen_fuel_multiple_fuels_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `gen_fuel_multiple_fuels_mw` | 0.960193 | -0.001623 | 未达标 |
| 24 | `group_063` | 8 | `total_losses`, `gen_fuel_other_renewables_mw`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_pjm_self_sched_reg`, `gen_fuel_coal_pct`, `congestion_price_rt` | `total_losses` | 0.960095 | -0.001721 | 未达标 |
| 25 | `group_043` | 7 | `da_as_total_mw_thirty_minutes_reserve`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gross_actual_interchange_mw`, `total_losses`, `total_pjm_self_sched_reg`, `gen_fuel_coal_pct` | `da_as_total_mw_thirty_minutes_reserve` | 0.959891 | -0.001924 | 未达标 |
| 26 | `group_035` | 6 | `congestion_price_da`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `congestion_price_da` | 0.958991 | -0.002825 | 未达标 |
| 27 | `group_029` | 5 | `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `gross_inadv_interchange_mw` | 0.958707 | -0.003108 | 未达标 |
| 28 | `group_056` | 8 | `gross_actual_interchange_mw`, `gen_fuel_other_renewables_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `total_pjm_self_sched_reg`, `gen_fuel_coal_pct`, `congestion_price_rt` | `gross_actual_interchange_mw` | 0.958048 | -0.003768 | 未达标 |
| 29 | `group_038` | 6 | `gen_fuel_other_renewables_mw`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `gen_fuel_other_renewables_mw` | 0.958011 | -0.003805 | 未达标 |
| 30 | `group_005` | 3 | `gen_fuel_multiple_fuels_pct`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `gen_fuel_multiple_fuels_pct` | 0.957811 | -0.004005 | 未达标 |
| 31 | `group_012` | 3 | `gen_fuel_other_renewables_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `gen_fuel_other_renewables_mw` | 0.957661 | -0.004155 | 未达标 |
| 32 | `group_025` | 4 | `gen_fuel_other_renewables_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gen_fuel_coal_pct` | `gen_fuel_other_renewables_mw` | 0.956937 | -0.004879 | 未达标 |
| 33 | `group_036` | 6 | `total_pjm_self_sched_reg`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `total_pjm_self_sched_reg` | 0.956847 | -0.004968 | 未达标 |
| 34 | `group_055` | 7 | `rmccp`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct`, `congestion_price_rt` | `rmccp` | 0.956371 | -0.005445 | 未达标 |
| 35 | `group_053` | 7 | `system_energy_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_pjm_loc_credit`, `total_losses`, `gen_fuel_coal_pct`, `congestion_price_rt` | `system_energy_price_da` | 0.956330 | -0.005486 | 未达标 |
| 36 | `group_034` | 6 | `net_inadv_interchange_mw`, `gen_fuel_other_renewables_mw`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gen_fuel_coal_pct` | `net_inadv_interchange_mw` | 0.956080 | -0.005736 | 未达标 |
| 37 | `group_031` | 5 | `total_pjm_rmccp_cr`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `total_pjm_rmccp_cr` | 0.955842 | -0.005973 | 未达标 |
| 38 | `group_045` | 7 | `total_pjm_loc_credit`, `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `net_inadv_interchange_mw`, `gen_fuel_coal_pct` | `total_pjm_loc_credit` | 0.955556 | -0.006259 | 未达标 |
| 39 | `group_026` | 4 | `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gen_fuel_coal_pct` | `gross_inadv_interchange_mw` | 0.955347 | -0.006468 | 未达标 |
| 40 | `group_032` | 5 | `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `gen_fuel_coal_pct` | `marginal_loss_price_da` | 0.955124 | -0.006692 | 未达标 |
| 41 | `group_040` | 7 | `rmccp`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `net_inadv_interchange_mw`, `total_pjm_self_sched_reg`, `gen_fuel_coal_pct` | `rmccp` | 0.954796 | -0.007020 | 未达标 |
| 42 | `group_022` | 4 | `metered_load_mw`, `total_pjm_rt_load_mwh`, `total_losses`, `gen_fuel_coal_pct` | `metered_load_mw` | 0.954658 | -0.007158 | 未达标 |
| 43 | `group_028` | 4 | `system_energy_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gen_fuel_coal_pct` | `system_energy_price_da` | 0.954490 | -0.007326 | 未达标 |
| 44 | `group_003` | 3 | `total_losses`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `total_losses` | 0.954239 | -0.007576 | 未达标 |
| 45 | `group_018` | 3 | `gross_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `gross_inadv_interchange_mw` | 0.953530 | -0.008286 | 未达标 |
| 46 | `group_024` | 4 | `gross_actual_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gen_fuel_coal_pct` | `gross_actual_interchange_mw` | 0.952822 | -0.008994 | 未达标 |
| 47 | `group_049` | 7 | `da_as_mcp_primary_reserve`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `total_losses`, `total_pjm_self_sched_reg`, `gen_fuel_coal_pct` | `da_as_mcp_primary_reserve` | 0.952218 | -0.009597 | 未达标 |
| 48 | `group_007` | 3 | `da_as_total_mw_thirty_minutes_reserve`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `da_as_total_mw_thirty_minutes_reserve` | 0.952150 | -0.009666 | 未达标 |
| 49 | `group_009` | 3 | `gross_actual_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `gross_actual_interchange_mw` | 0.951404 | -0.010412 | 未达标 |
| 50 | `group_004` | 3 | `system_energy_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `system_energy_price_da` | 0.951400 | -0.010415 | 未达标 |
| 51 | `group_006` | 3 | `total_lmp_da`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `total_lmp_da` | 0.951133 | -0.010682 | 未达标 |
| 52 | `group_017` | 3 | `total_pjm_rmccp_cr`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `total_pjm_rmccp_cr` | 0.951072 | -0.010744 | 未达标 |
| 53 | `group_016` | 3 | `da_as_mcp_primary_reserve`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `da_as_mcp_primary_reserve` | 0.950874 | -0.010942 | 未达标 |
| 54 | `group_015` | 3 | `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `marginal_loss_price_da` | 0.950873 | -0.010942 | 未达标 |
| 55 | `group_033` | 5 | `da_as_mcp_primary_reserve`, `marginal_loss_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gen_fuel_coal_pct` | `da_as_mcp_primary_reserve` | 0.950604 | -0.011212 | 未达标 |
| 56 | `group_013` | 3 | `congestion_price_da`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `congestion_price_da` | 0.949848 | -0.011968 | 未达标 |
| 57 | `group_023` | 4 | `total_pjm_loc_credit`, `total_pjm_rt_load_mwh`, `metered_load_mw`, `gen_fuel_coal_pct` | `total_pjm_loc_credit` | 0.949825 | -0.011991 | 未达标 |
| 58 | `group_020` | 3 | `total_pjm_loc_credit`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `total_pjm_loc_credit` | 0.949789 | -0.012027 | 未达标 |
| 59 | `group_001` | 2 | `metered_load_mw`, `total_pjm_rt_load_mwh` | `metered_load_mw` | 0.949219 | -0.012596 | 未达标 |
| 60 | `group_014` | 3 | `rmccp`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `rmccp` | 0.948669 | -0.013147 | 未达标 |
| 61 | `group_019` | 3 | `congestion_price_rt`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `congestion_price_rt` | 0.947922 | -0.013893 | 未达标 |
| 62 | `group_011` | 3 | `total_pjm_self_sched_reg`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `total_pjm_self_sched_reg` | 0.947819 | -0.013996 | 未达标 |
| 63 | `group_010` | 3 | `marginal_loss_price_rt`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `marginal_loss_price_rt` | 0.947816 | -0.013999 | 未达标 |
| 64 | `group_008` | 3 | `net_inadv_interchange_mw`, `total_pjm_rt_load_mwh`, `metered_load_mw` | `net_inadv_interchange_mw` | 0.947790 | -0.014026 | 未达标 |

## 当前读法

- 这次不是固定 `|Q|=2/3/4`。`|Q|` 来自 D-gating 的有效范数读出：同一个 trial 下，阈值越高，读出的字段越少；阈值越低，读出的字段越多。
- 脚本最后的自动选择规则是：先找达到当前 xi 动态目标的候选组，再优先选择字段数最少的组；如果多个组字段数相同，再看 R2。
- 但从分析角度看，不应该只看最终最小组。更重要的是上面的去重路径表：它说明当前 xi 存在多少条不同替代路径，以及每条路径能恢复到什么准确率。
- 本轮最强信号仍然是 `metered_load_mw`，但 full-R 后也出现了 `total_pjm_rt_load_mwh`、`da_as_nsr_mw_primary_reserve`、`gen_fuel_coal_pct` 等 Stage04 活跃集之外或边缘字段进入路径，说明把完整 `R` 放回 Stage05 是有信息增益的。
