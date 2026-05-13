# data2025 Preprocessing Report V1

## Scope

本次只做数据预处理，不做 confidential/general 分类，不建模，不额外删列。

## Paths

- 输入文件路径：`/Users/haocun/Desktop/AllProjects/NEW_EPGAT/data/raw/data2025.csv`
- 输出文件夹路径：`/Users/haocun/Desktop/AllProjects/NEW_EPGAT/data/processed/data2025_Processed_V1`
- 输出特征表：`/Users/haocun/Desktop/AllProjects/NEW_EPGAT/data/processed/data2025_Processed_V1/data2025_processed.csv`
- 输出带时间索引表：`/Users/haocun/Desktop/AllProjects/NEW_EPGAT/data/processed/data2025_Processed_V1/data2025_processed_with_time_index.csv`

## Input Reading

- 读取方式：识别多行表头结构，第 2 行作为英文字段名，第 4 行开始作为小时级样本数据。
- 原始文件总行数：8763
- 原始数据行列数：8760 行 × 63 列
- 行长度异常：0 个
- 时间顺序是否保持：是
- `datetime_beginning_utc` 重复时间数量：0
- `datetime_beginning_utc` 解析异常数量：0

## Drop Fields

删除字段配置如下，便于复制到项目配置中：

```python
DROP_FIELDS_MAIN = [
    # 时间字段不作为图节点，保留为索引即可
    "datetime_beginning_utc",
    "datetime_beginning_ept",
    # 全 0 / 常数列
    "gen_fuel_storage_pct",
    "da_as_mcp_thirty_minutes_reserve",
    "da_as_ircmwt2_primary_reserve",
    "da_as_ircmwt2_synchronized_reserve",
    "da_as_ircmwt2_thirty_minutes_reserve",
    # 完全重复列
    "da_as_as_mw_primary_reserve",
    "da_as_as_mw_synchronized_reserve",
    "da_as_as_mw_thirty_minutes_reserve",
    "da_as_ss_mw_synchronized_reserve",
    "da_as_ss_mw_thirty_minutes_reserve",
]
```

实际删除字段数量：12

- 时间字段不作为图节点，保留为索引：`datetime_beginning_utc`, `datetime_beginning_ept`
- 全 0 / 常数列：`gen_fuel_storage_pct`, `da_as_mcp_thirty_minutes_reserve`, `da_as_ircmwt2_primary_reserve`, `da_as_ircmwt2_synchronized_reserve`, `da_as_ircmwt2_thirty_minutes_reserve`
- 完全重复列：`da_as_as_mw_primary_reserve`, `da_as_as_mw_synchronized_reserve`, `da_as_as_mw_thirty_minutes_reserve`, `da_as_ss_mw_synchronized_reserve`, `da_as_ss_mw_thirty_minutes_reserve`


未在原始数据中找到的指定删除字段：无

## Missing Value Filling

### forecast_load_mw_day_ahead

- 原本缺失数量：2202
- 补全数量：2202
- 方法：优先使用上下 3 小时、4-10 小时、11-30 小时非缺失均值，权重分别为 0.5、0.3、0.2；若某窗口无可用值则忽略并重归一化；若上下 30 小时无可用值，则扩大到 31-100 小时、101-200 小时；仍不可用则使用列中位数。
- 实际补全方法分布：
- `within_3h+within_4_10h+within_11_30h`: 2184
- `within_4_10h+within_11_30h`: 14
- `within_11_30h`: 4

### gen_fuel_* Fields

- 原本缺失总数：60
- 补全总数：60
- 方法：对所有 `gen_fuel_*` 字段按时间顺序使用前后最近非缺失小时做线性插值；边界缺失使用前向/后向补全；极端情况下使用列中位数。
- 实际补全方法分布：
- `linear_interpolation`: 60

各字段缺失与补全数量：

| 字段 | 原本缺失 | 补全数量 |
|---|---:|---:|
| `gen_fuel_coal_mw` | 3 | 3 |
| `gen_fuel_gas_mw` | 3 | 3 |
| `gen_fuel_hydro_mw` | 3 | 3 |
| `gen_fuel_multiple_fuels_mw` | 3 | 3 |
| `gen_fuel_nuclear_mw` | 3 | 3 |
| `gen_fuel_oil_mw` | 3 | 3 |
| `gen_fuel_other_renewables_mw` | 3 | 3 |
| `gen_fuel_solar_mw` | 3 | 3 |
| `gen_fuel_storage_mw` | 3 | 3 |
| `gen_fuel_wind_mw` | 3 | 3 |
| `gen_fuel_coal_pct` | 3 | 3 |
| `gen_fuel_gas_pct` | 3 | 3 |
| `gen_fuel_hydro_pct` | 3 | 3 |
| `gen_fuel_multiple_fuels_pct` | 3 | 3 |
| `gen_fuel_nuclear_pct` | 3 | 3 |
| `gen_fuel_oil_pct` | 3 | 3 |
| `gen_fuel_other_renewables_pct` | 3 | 3 |
| `gen_fuel_solar_pct` | 3 | 3 |
| `gen_fuel_storage_pct` | 3 | 3 |
| `gen_fuel_wind_pct` | 3 | 3 |


## Output Checks

- 处理后数据行列数：8760 行 × 51 列
- 处理后行数是否与原始样本行数一致：是
- `data2025_processed.csv` 是否删除时间字段：是
- `data2025_processed_with_time_index.csv` 是否使用 `datetime_beginning_utc` 作为首列索引：是
- 是否仍有缺失值：否

处理后未发现缺失值。

## Constant and Duplicate Column Checks

- 处理后是否仍有常数列：否
- 处理后是否仍有完全重复列：否

## Notes

- 除 `forecast_load_mw_day_ahead` 和 `gen_fuel_*` 字段外，本次未补全其他字段。
- 本次没有删除 `forecast_load_mw_day_ahead`、`wind_generation_mw`、`solar_generation_mw`。
- 本次没有做 confidential/general 分类。
