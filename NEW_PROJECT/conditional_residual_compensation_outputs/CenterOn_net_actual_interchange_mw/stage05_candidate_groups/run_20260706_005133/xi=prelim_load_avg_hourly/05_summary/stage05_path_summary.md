# Stage05 路径总结：prelim_load_avg_hourly

## 口径说明

- 中心字段: `net_actual_interchange_mw`
- 被替代字段 xi: `prelim_load_avg_hourly`
- 固定上下文 `C_i`: 8 个字段
- Stage04 的完整路径外候选 `R`: 46 个字段
- Stage04 显影后的活跃候选集 `A`: 20 个字段
- Stage05 当前实现是在 `A` 内做锚定多头 D-gating，不是在完整 `R` 内做。
- 因此每个 head 固定 1 个 active anchor，D-gating 压缩其余 19 个 active 字段。

## 基准效果

- `C_i` only R2: `0.770768`
- `C_i + R` R2: `0.966816`
- 目标 R2: `0.950000`

## 为什么现在是 20 条原始路径

严格按锚头口径，确实应该先有 20 条路径：20 个活跃字段各自作为一次 anchor head，读出一个 `Q`。
之前我说 19 条，是因为我只统计了已经进入真实 DNN 验证表的路径，漏掉了 `metered_load_mw` 作为 anchor head 自己读出的路径。

`metered_load_mw` 的情况比较特殊：它在前 19 个 head 里几乎总是作为伴随字段被选中；但当它自己作为 anchor 时，读出的不是一个 2 字段小组，而是一个 19 字段大组。这个大组没有进入前 24 个候选组的真实 DNN 验证，所以它在旧版去重 JSON 里看起来像“没有路径”。更准确地说：它有路径，但这条路径未验证，而且不够紧凑。

- 原始 anchor 路径数: `20`
- 去重后路径数: `20`
- 已真实验证路径数: `19`
- 未验证路径数: `1`

## 可以替代 xi 的主要路径

- 路径 1（active 9, |Q|=2）: `gen_fuel_multiple_fuels_mw`, `metered_load_mw`; R2=`0.959036`，达标
- 路径 2（active 10, |Q|=2）: `gen_fuel_multiple_fuels_pct`, `metered_load_mw`; R2=`0.957324`，达标
- 路径 3（active 1, |Q|=2）: `gen_fuel_other_renewables_mw`, `metered_load_mw`; R2=`0.955877`，达标
- 路径 4（active 3, |Q|=2）: `total_losses`, `metered_load_mw`; R2=`0.955059`，达标
- 路径 5（active 8, |Q|=2）: `gross_inadv_interchange_mw`, `metered_load_mw`; R2=`0.953737`，达标
- 路径 6（active 4, |Q|=2）: `system_energy_price_da`, `metered_load_mw`; R2=`0.953560`，达标
- 路径 7（active 14, |Q|=2）: `da_as_total_mw_thirty_minutes_reserve`, `metered_load_mw`; R2=`0.953174`，达标
- 路径 8（active 2, |Q|=2）: `congestion_price_da`, `metered_load_mw`; R2=`0.952143`，达标
- 路径 9（active 6, |Q|=2）: `total_lmp_da`, `metered_load_mw`; R2=`0.952097`，达标
- 路径 10（active 16, |Q|=2）: `marginal_loss_price_da`, `metered_load_mw`; R2=`0.951896`，达标
- 路径 11（active 18, |Q|=2）: `gross_actual_interchange_mw`, `metered_load_mw`; R2=`0.950416`，达标
- 路径 12（active 13, |Q|=2）: `marginal_loss_price_rt`, `metered_load_mw`; R2=`0.950318`，达标
- 路径 13（active 15, |Q|=2）: `total_pjm_rmccp_cr`, `metered_load_mw`; R2=`0.950129`，达标
- 路径 14（active 19, |Q|=2）: `da_as_mcp_primary_reserve`, `metered_load_mw`; R2=`0.949193`，未达标
- 路径 15（active 7, |Q|=2）: `total_pjm_loc_credit`, `metered_load_mw`; R2=`0.948407`，未达标
- 路径 16（active 11, |Q|=2）: `rmccp`, `metered_load_mw`; R2=`0.947926`，未达标
- 路径 17（active 12, |Q|=2）: `congestion_price_rt`, `metered_load_mw`; R2=`0.947796`，未达标
- 路径 18（active 17, |Q|=2）: `net_inadv_interchange_mw`, `metered_load_mw`; R2=`0.947374`，未达标
- 路径 19（active 5, |Q|=2）: `total_pjm_self_sched_reg`, `metered_load_mw`; R2=`0.946441`，未达标
- 路径 20（active 20, |Q|=19）: `metered_load_mw`, `gen_fuel_other_renewables_mw`, `congestion_price_da`, `total_losses`, `system_energy_price_da`, `total_pjm_self_sched_reg`, `total_lmp_da`, `total_pjm_loc_credit`, `gross_inadv_interchange_mw`, `gen_fuel_multiple_fuels_mw`, `gen_fuel_multiple_fuels_pct`, `rmccp`, `congestion_price_rt`, `marginal_loss_price_rt`, `da_as_total_mw_thirty_minutes_reserve`, `marginal_loss_price_da`, `net_inadv_interchange_mw`, `gross_actual_interchange_mw`, `da_as_mcp_primary_reserve`; 未进入真实 DNN 验证表

## 当前最小且达标的推荐路径

- `Q = {gen_fuel_multiple_fuels_mw, metered_load_mw}`
- `C_i + Q` 总字段数: `10`
- 真实 DNN best test R2: `0.959036`

这个结果的含义是：在固定另外 8 个主路径字段 `C_i` 的情况下，用 `gen_fuel_multiple_fuels_mw` 和 `metered_load_mw` 两个路径外字段，可以把删除 `prelim_load_avg_hourly` 后的预测能力恢复到 0.95 以上。
