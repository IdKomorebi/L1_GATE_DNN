# 对齐且处理版本说明

这个文件夹里的 CSV 已经按 `datetime_beginning_utc` 对齐到 2025 全年 8760 个小时，并且已经把 Tie Line flow 汇总成 PJM_RTO 总交换字段。

## 文件

- `pjm_rto_hourly_2025_aligned_processed_one_header.csv`：一行表头。
- `pjm_rto_hourly_2025_aligned_processed_two_header.csv`：两行表头，第一行是来源 feed，第二行是字段名。

两个 CSV 的数据内容完全相同，区别只是 `two_header` 多了一行一级表头。

## Tie Line 字段如何处理

本版本对所有 Tie Line 在同一小时内做了汇总：

```text
net_actual_interchange_mw = sum(actual_flow over all tie lines)
net_sched_interchange_mw  = sum(sched_flow over all tie lines)
net_inadv_interchange_mw  = sum(inadv_flow over all tie lines)
```

为了避免正负抵消，还保留了交换规模字段：

```text
gross_actual_interchange_mw = sum(abs(actual_flow) over all tie lines)
gross_sched_interchange_mw  = sum(abs(sched_flow) over all tie lines)
gross_inadv_interchange_mw  = sum(abs(inadv_flow) over all tie lines)
```

## 与未处理版本的区别

未处理版本保留每条 Tie Line 的 flow 列，例如 `alte_actual_flow`。

处理版本删除单条 Tie Line 列，只保留 PJM_RTO 总交换口径字段，例如 `net_actual_interchange_mw`。

## 适用场景

这个版本更适合主实验，因为其他字段大多是 PJM_RTO / RTO / PJM-RTO 总体口径。把 Tie Line 汇总成总交换字段后，主体更一致。


## 内容质量检查与暂定处理意见

下面这些只是建模前的处理建议，本次没有直接修改 CSV 数据。

### 时间与对齐

- 已处理版本共有 `8760` 行小时数据。
- `datetime_beginning_utc` 重复时间戳数量：`0`。
- 因此这个版本可以作为 PJM_RTO 主体的小时级基表继续使用。

### 缺失值

- `forecast_load_mw_day_ahead` 缺失 `2202 / 8760 (25.14%)`。建议主实验优先使用 `forecast_load_mw_latest_available`；如果一定要使用 day-ahead 版本，需要先单独说明缺失处理策略。
- 20 个 `gen_fuel_*` 字段存在缺失，缺失数量为 3 小时；本次数据中主要对应 2025-07-02 01:00/02:00 UTC 和 2025-11-02 06:00 UTC。 建议暂时保留 NaN，不在这个整理步骤里插值。

### 全 0 或高度稀疏字段

全 0 字段：

- `da_as_mcp_thirty_minutes_reserve`：全 0，建议作为 drop 候选。
- `da_as_ircmwt2_primary_reserve`：全 0，建议作为 drop 候选。
- `da_as_ircmwt2_synchronized_reserve`：全 0，建议作为 drop 候选。
- `da_as_ircmwt2_thirty_minutes_reserve`：全 0，建议作为 drop 候选。

0 值比例较高但不一定错误的字段：

- `gen_fuel_storage_pct`：0 值 8757 / 8760 (99.97%)。
- `gen_fuel_storage_mw`：0 值 6569 / 8760 (74.99%)。
- `da_as_mcp_primary_reserve`：0 值 6063 / 8760 (69.21%)。
- `da_as_mcp_synchronized_reserve`：0 值 4577 / 8760 (52.25%)。

这些字段不建议直接作为 GAT 的核心节点；尤其全 0 字段没有信息量，应优先删除或只留在溯源表中。

### 完全重复字段

- `da_as_mcp_thirty_minutes_reserve` 与 `da_as_ircmwt2_primary_reserve` 完全相同。
- `da_as_mcp_thirty_minutes_reserve` 与 `da_as_ircmwt2_synchronized_reserve` 完全相同。
- `da_as_mcp_thirty_minutes_reserve` 与 `da_as_ircmwt2_thirty_minutes_reserve` 完全相同。
- `da_as_total_mw_primary_reserve` 与 `da_as_as_mw_primary_reserve` 完全相同。
- `da_as_total_mw_synchronized_reserve` 与 `da_as_as_mw_synchronized_reserve` 完全相同。
- `da_as_total_mw_thirty_minutes_reserve` 与 `da_as_as_mw_thirty_minutes_reserve` 完全相同。
- `da_as_ss_mw_primary_reserve` 与 `da_as_ss_mw_synchronized_reserve` 完全相同。
- `da_as_ss_mw_primary_reserve` 与 `da_as_ss_mw_thirty_minutes_reserve` 完全相同。
- `da_as_ss_mw_synchronized_reserve` 与 `da_as_ss_mw_thirty_minutes_reserve` 完全相同。
- `da_as_ircmwt2_primary_reserve` 与 `da_as_ircmwt2_synchronized_reserve` 完全相同。
- `da_as_ircmwt2_primary_reserve` 与 `da_as_ircmwt2_thirty_minutes_reserve` 完全相同。
- `da_as_ircmwt2_synchronized_reserve` 与 `da_as_ircmwt2_thirty_minutes_reserve` 完全相同。

这些字段进入同一个模型会重复计权。建议正式建模时每组只保留一个，并在论文或实验说明里说明删除原因。

### LMP 公式关系

- DA LMP：`total_lmp_da` 基本等于 `system_energy_price_da` + `congestion_price_da` + `marginal_loss_price_da`；最大绝对差约 1e-06，差值大于 1e-6 的小时数为 981。
- RT LMP：`total_lmp_rt` 基本等于 `system_energy_price_rt` + `congestion_price_rt` + `marginal_loss_price_rt`；最大绝对差约 0.005001，差值大于 1e-6 的小时数为 8121。

因此如果预测 `total_lmp_da` 或 `total_lmp_rt`，不要同时把对应的 system energy、congestion、marginal loss 三个分量作为输入，否则容易形成公式型泄露。

### Interchange 字段

- `net_actual_interchange_mw`、`net_sched_interchange_mw`、`net_inadv_interchange_mw` 是所有 Tie Line 在同一小时内的代数和，适合表示 PJM_RTO 总交换的净方向。
- `gross_actual_interchange_mw`、`gross_sched_interchange_mw`、`gross_inadv_interchange_mw` 是绝对值求和，适合表示交换规模，避免正负方向互相抵消。
- 如果后续实验关心系统总体风险，建议优先使用 net/gross 这些处理后的 PJM_RTO 字段；如果关心某条外部联络线的传播机制，再使用 `对齐但未处理` 文件夹里的单条 Tie Line 字段。

