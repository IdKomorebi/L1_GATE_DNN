# 阶段 01：主路径识别与剪枝验证

## 目标

本阶段为后续条件化残差补偿流程固定第一个标准接口：中心变量 `net_actual_interchange_mw` 的主路径字段集合 `P`。

逻辑顺序是：先用 D-gating 得到客观保留字段，再在这些字段内部做逐字段剪枝比较，最后对剪枝阶段选出的主路径做逐字段阻断验证。

## D-gating Compression

- model: DGatingDNN
- lambda_dgate: 0.03
- dgate_depth: 4
- dgate_normalize_lambda_by_depth: False
- active_threshold: 0.001
- best_test_r2: 0.9529446363449097
- best_epoch: 153
- selected_by_active_threshold_count: 11
- largest_log_gap_selected_count: 11

## 标准接口

后续阶段应读取 `stage01_main_path_interface.json`。

当前阶段选出的主路径字段：

1. `prelim_load_avg_hourly`
2. `gen_fuel_gas_mw`
3. `gen_fuel_coal_mw`
4. `gen_fuel_solar_mw`
5. `gen_fuel_hydro_mw`
6. `gen_fuel_wind_mw`
7. `gross_sched_interchange_mw`
8. `gen_fuel_nuclear_mw`
9. `da_as_nsr_mw_primary_reserve`
10. `gen_fuel_oil_mw`
11. `gen_fuel_multiple_fuels_mw`

## 结果分析

- 全量 55 字段 masked R2: 0.952945
- 阶段主路径 11 字段 masked R2: 0.952945
- 对应 D-gating best epoch: 153
- D-gating 客观保留 11 字段 masked R2: 0.952945
- 剪枝路径 11 字段 masked R2: 0.952945
- 剪枝路径 10 字段 masked R2: 0.938236
- 剪枝路径 9 字段 masked R2: 0.918666
- 03 验证中逐字段 drop-one R2 范围: -9.751678 到 0.938236

结论：如果剪掉任意一个字段后都低于阈值，说明 02 阶段选出的主路径在当前 D-gating 模型下已经不能继续安全压缩。

## 产物说明

- `stage01_main_path_interface.json`: 后续阶段读取的标准接口。
- `01_dgating/`: D-gating 模型、日志、门控历史和诊断图。
- `02_pruning/dgate11_compact_candidates.csv`: 从 D-gating 保留 11 字段开始的贪心剪枝路径摘要。
- `02_pruning/dgate11_pruning_trials.csv`: 每一步尝试 drop 每个字段后的完整准确率对比。
- `02_pruning/pruning_path.png`: 11 -> 10 -> 9 等剪枝路径准确率曲线。
- `03_validation/main_vs_full_validation.csv`: 最终主路径与全量字段的 masked 对比。
- `03_validation/drop_one_validation.csv` 和 `03_validation/drop_one_validation.png`: 对最终主路径逐字段阻断验证。
