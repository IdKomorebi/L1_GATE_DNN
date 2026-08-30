# 推断源定位：跨区支路 AB3 负载率

- 分支：`with_screening`
- 剥离后/初筛后候选数：23
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.8239 ± 0.0111**。
- 共识推断源：**9 个**（`gen_fuel_coal_mw`, `gen_fuel_natural_gas_mw`, `area_1_load_day_ahead_mw`, `system_losses_mw`, `gen_fuel_wind_mw`, `gen_fuel_solar_mw`, `gen_fuel_hydro_mw`, `area_1_net_export_mw`, `area_2_net_export_mw`）。
- 只用共识字段重训普通 DNN：**0.7512 ± 0.0188**。
- 相对全候选精度变化：**-0.0727**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_coal_mw` | 0.329374 | 0.0184551 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.190755 | 0.0157079 | 3/3 |
| `area_1_load_day_ahead_mw` | 0.161641 | 0.00862656 | 3/3 |
| `system_losses_mw` | 0.117157 | 0.00567589 | 3/3 |
| `gen_fuel_wind_mw` | 0.10245 | 0.0165834 | 3/3 |
| `gen_fuel_solar_mw` | 0.0965295 | 0.0172789 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0809307 | 0.00908615 | 3/3 |
| `area_1_net_export_mw` | 0.048552 | 0.00782811 | 3/3 |
| `area_2_net_export_mw` | 0.0482416 | 0.00921263 | 3/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 11/11/11 |
| 0.0001 | 11/11/11 |
| 0.001 | 11/11/11 |
| 0.005 | 11/11/11 |
| 0.01 | 10/11/11 |
| 0.02 | 10/11/11 |
| 0.05 | 10/9/9 |
| 0.1 | 5/6/6 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.1155 | `gen_fuel_coal_mw` |
| 2 | 0.2851 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw` |
| 3 | 0.3350 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_1_load_day_ahead_mw` |
| 5 | 0.6785 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_1_load_day_ahead_mw|system_losses_mw|gen_fuel_wind_mw` |
| 8 | 0.7241 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_1_load_day_ahead_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_hydro_mw|area_1_net_export_mw` |
| 9 | 0.7265 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_1_load_day_ahead_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_hydro_mw|area_1_net_export_mw|area_2_net_export_mw` |
| 12 | 0.8209 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_1_load_day_ahead_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_hydro_mw|area_1_net_export_mw|area_2_net_export_mw|reserve_spin_r2_mw|area_2_load_day_ahead_mw|area_2_load_actual_mw` |
| 23 | 0.7976 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_1_load_day_ahead_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_hydro_mw|area_1_net_export_mw|area_2_net_export_mw|reserve_spin_r2_mw|area_2_load_day_ahead_mw|area_2_load_actual_mw|area_3_load_day_ahead_mw|reserve_spin_r3_mw|area_1_load_actual_mw|reserve_spin_r1_mw|system_generation_mw|system_load_day_ahead_mw|system_load_actual_mw|area_3_load_actual_mw|reserve_reg_down_mw|reserve_reg_up_mw|gen_fuel_nuclear_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
