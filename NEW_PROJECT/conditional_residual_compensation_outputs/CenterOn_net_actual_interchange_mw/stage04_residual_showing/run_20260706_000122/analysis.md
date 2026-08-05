# 阶段 04：条件化残差补偿显影

## 目标

本阶段只针对阶段 03 已完成的一个字段执行：在固定 `C_i = P \ {x_i}` 的条件下，让路径外候选字段 `R` 去预测标准化残差 `residual_std`。

模型使用轻微 D-gating 作用在 `R` 的第一层有效权重上，同时只对 `R` 做随机 dropout。这里的目标不是最终替代认证，而是显影可能补偿 `x_i` 缺口的活跃候选集 `A`。

## 本次字段

- xi: `prelim_load_avg_hourly`
- sweep trial 数: `8`

## 选中的显影结果

- trial: `trial_02_lam1e-06_drop0p45`
- lambda_dgate: `1e-06`
- R dropout: `0.45`
- dgate_depth: `3`
- full-R validation MSE: `0.726074`
- full-R validation R2: `0.287308`
- active_count: `20`
- selected_topk MSE/full MSE: `1.131969`

## 活跃候选集 A

- `gen_fuel_other_renewables_mw`
- `congestion_price_da`
- `total_losses`
- `system_energy_price_da`
- `total_pjm_self_sched_reg`
- `total_lmp_da`
- `total_pjm_loc_credit`
- `gross_inadv_interchange_mw`
- `gen_fuel_multiple_fuels_mw`
- `gen_fuel_multiple_fuels_pct`
- `rmccp`
- `congestion_price_rt`
- `marginal_loss_price_rt`
- `da_as_total_mw_thirty_minutes_reserve`
- `total_pjm_rmccp_cr`
- `marginal_loss_price_da`
- `net_inadv_interchange_mw`
- `gross_actual_interchange_mw`
- `da_as_mcp_primary_reserve`
- `metered_load_mw`

## 产物

- `stage04_active_candidate_interface.json`: 给下一阶段使用的标准接口。
- `xi=<field>/01_data/data_summary.json`: 数据、字段、划分说明。
- `xi=<field>/02_sweep/sweep_summary.csv`: 多强度 D-gating 显影结果。
- `xi=<field>/02_sweep/trial_*/candidate_scores.csv`: 每个候选 R 字段的 gate、遮蔽敏感度和 combined score。
- `xi=<field>/02_sweep/trial_*/topk_compression.csv`: 按 combined score 逐步压缩 A 的验证。
- `xi=<field>/03_active_set/active_set.json`: 本阶段推荐的活跃候选集 A。
