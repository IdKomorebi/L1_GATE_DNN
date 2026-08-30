# 数据清洗报告

## pjm_2024

- 原始：8784 行 × 70 列
- 整列删除（缺失率 > 10%）：['forecast_load_mw_day_ahead']
- 插值补齐（缺失率 ≤ 1%）：共 1 列
    - `forecast_load_mw_latest_available`：补 12 个点
- **main 版本**（主实验用）：8784 行 × 69 列
- **allcols 版本**（对照用）：6539 行 × 70 列
- 样本增益：main 比 allcols 多 2245 行 （34.3%）

## pjm_2025

- 原始：8760 行 × 70 列
- 整列删除（缺失率 > 10%）：['forecast_load_mw_day_ahead']
- 插值补齐（缺失率 ≤ 1%）：共 20 列
    - `gen_fuel_coal_mw`：补 3 个点
    - `gen_fuel_coal_pct`：补 3 个点
    - `gen_fuel_gas_mw`：补 3 个点
    - `gen_fuel_gas_pct`：补 3 个点
    - `gen_fuel_hydro_mw`：补 3 个点
    - `gen_fuel_hydro_pct`：补 3 个点
    - `gen_fuel_multiple_fuels_mw`：补 3 个点
    - `gen_fuel_multiple_fuels_pct`：补 3 个点
    - `gen_fuel_nuclear_mw`：补 3 个点
    - `gen_fuel_nuclear_pct`：补 3 个点
    - `gen_fuel_oil_mw`：补 3 个点
    - `gen_fuel_oil_pct`：补 3 个点
    - `gen_fuel_other_renewables_mw`：补 3 个点
    - `gen_fuel_other_renewables_pct`：补 3 个点
    - `gen_fuel_solar_mw`：补 3 个点
    - `gen_fuel_solar_pct`：补 3 个点
    - `gen_fuel_storage_mw`：补 3 个点
    - `gen_fuel_storage_pct`：补 3 个点
    - `gen_fuel_wind_mw`：补 3 个点
    - `gen_fuel_wind_pct`：补 3 个点
- **main 版本**（主实验用）：8760 行 × 69 列
- **allcols 版本**（对照用）：6558 行 × 70 列
- 样本增益：main 比 allcols 多 2202 行 （33.6%）
