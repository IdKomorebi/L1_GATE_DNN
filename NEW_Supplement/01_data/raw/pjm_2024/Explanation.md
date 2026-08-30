# 2024 对齐且处理版本说明

本文件夹与 `data2025_Processed_V2` 使用相同的 72 列结构和字段顺序。数据按 `datetime_beginning_utc` 对齐到 2024 闰年全部 8784 个小时，并保留 PJM_RTO 汇总口径的 Tie Line 交换字段。

## 文件

- `pjm_rto_hourly_2024_aligned_processed_one_header.csv`：最终一行表头数据文件。
- `dataset_information_table_for_paper.md`：与 2025 V2 对应的数据字段说明。
- `source_manifest.json`：输入、官方接口、查询范围、校验结果和 SHA-256 校验值。

## 与 2025 V2 的对应关系

- 两个年度的最终 CSV 都是 72 列：2 个时间字段和 70 个数据字段。
- 2024 最终表头与 2025 V2 完全相同，字段顺序也完全相同。
- 2024 raw 原有 63 列按原值保留。
- 在 `metered_load_mw` 后加入与 2025 V2 相同的 9 个 `reg_zone_prelim_bill` 字段：

```text
rmccp
rmpcp
total_pjm_rt_load_mwh
total_pjm_loc_credit
total_pjm_reg_purchases
total_pjm_self_sched_reg
total_pjm_assigned_reg
total_pjm_rmccp_cr
total_pjm_rmpcp_cr
```

这 9 个字段来自 PJM 官方 Data Miner 2 的 `PJM Regulation Zone Preliminary Billing Data` feed：

- Definition: https://dataminer2.pjm.com/feed/reg_zone_prelim_bill/definition
- API: https://api.pjm.com/api/v1/reg_zone_prelim_bill
- UTC 查询范围：`2024-01-01 00:00` 至 `2024-12-31 23:00`
- 官方返回记录数：8784
- 新增字段缺失数：0

## Tie Line 字段处理

2024 raw 已经包含与 2025 V2 相同的 6 个 PJM_RTO 汇总交换字段，因此本次按 2025 V2 的做法原值保留，不重复对汇总结果进行二次求和：

```text
net_actual_interchange_mw
net_sched_interchange_mw
net_inadv_interchange_mw
gross_actual_interchange_mw
gross_sched_interchange_mw
gross_inadv_interchange_mw
```

字段含义为：

```text
net_*   = sum(flow over all tie lines)
gross_* = sum(abs(flow) over all tie lines)
```

`net_*` 表示净交换方向，`gross_*` 表示交换规模并避免正负方向互相抵消。

## 对齐与完整性检查

- 小时数据行数：8784。
- UTC 范围：`2024-01-01 00:00:00` 至 `2024-12-31 23:00:00`。
- UTC 重复时间戳：0。
- UTC 缺失小时：0。
- raw 与官方 API 的 UTC 一一对应：8784 / 8784。
- raw 与官方 API 的 EPT 小时不一致数量：0。
- CSV 每行字段数：72。

## 缺失值

本版本延续 2025 V2 的处理边界：负责多源对齐和汇总，不对原始业务字段插值，也不删除字段。

- `forecast_load_mw_latest_available`：缺失 12 / 8784。
- `forecast_load_mw_day_ahead`：缺失 2245 / 8784。
- 其余字段无缺失。
- 全表缺失单元格合计：2257。

如果用于要求无缺失输入的模型，应在训练预处理阶段明确补值策略，不能把这些空值直接解释为 0。

## 全零和重复字段提示

以下字段全年为 0：

- `gen_fuel_storage_pct`
- `da_as_mcp_thirty_minutes_reserve`
- `da_as_ircmwt2_primary_reserve`
- `da_as_ircmwt2_synchronized_reserve`
- `da_as_ircmwt2_thirty_minutes_reserve`

以下非全零字段存在完全重复关系：

- `da_as_total_mw_primary_reserve` = `da_as_as_mw_primary_reserve`
- `da_as_total_mw_synchronized_reserve` = `da_as_as_mw_synchronized_reserve`
- `da_as_total_mw_thirty_minutes_reserve` = `da_as_as_mw_thirty_minutes_reserve`
- `da_as_ss_mw_primary_reserve` = `da_as_ss_mw_synchronized_reserve` = `da_as_ss_mw_thirty_minutes_reserve`

正式建模时可按实验口径删除全零列，并避免让完全重复字段重复计权；最终 CSV 为了与 2025 V2 严格对应而全部保留。

## LMP 公式关系提示

- DA LMP：`total_lmp_da` 基本等于 `system_energy_price_da + congestion_price_da + marginal_loss_price_da`。
- RT LMP：`total_lmp_rt` 基本等于 `system_energy_price_rt + congestion_price_rt + marginal_loss_price_rt`。

如果预测总 LMP，不应同时把对应三个公式分量全部作为输入，否则可能产生公式型信息泄露。
