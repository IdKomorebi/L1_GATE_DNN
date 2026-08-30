# 推断源定位：跨区支路 AB2 负载率

- 分支：`with_screening`
- 剥离后/初筛后候选数：23
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9725 ± 0.0017**。
- 共识推断源：**9 个**（`gen_fuel_natural_gas_mw`, `gen_fuel_coal_mw`, `gen_fuel_solar_mw`, `gen_fuel_wind_mw`, `area_1_net_export_mw`, `gen_fuel_hydro_mw`, `area_1_load_day_ahead_mw`, `system_losses_mw`, `area_2_net_export_mw`）。
- 只用共识字段重训普通 DNN：**0.9629 ± 0.0008**。
- 相对全候选精度变化：**-0.0096**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_natural_gas_mw` | 0.07363 | 0.00646448 | 3/3 |
| `gen_fuel_coal_mw` | 0.0603037 | 0.00649639 | 3/3 |
| `gen_fuel_solar_mw` | 0.0528531 | 0.00407367 | 3/3 |
| `gen_fuel_wind_mw` | 0.0384081 | 0.00459835 | 3/3 |
| `area_1_net_export_mw` | 0.0217137 | 0.000821709 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0210114 | 0.00122276 | 3/3 |
| `area_1_load_day_ahead_mw` | 0.0700409 | 0.0495267 | 2/3 |
| `system_losses_mw` | 0.0193418 | 0.0138986 | 2/3 |
| `area_2_net_export_mw` | 0.0072973 | 0.00518159 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 8/10/10 |
| 0.0001 | 8/10/10 |
| 0.001 | 8/10/10 |
| 0.005 | 8/10/10 |
| 0.01 | 7/9/9 |
| 0.02 | 7/7/8 |
| 0.05 | 4/4/3 |
| 0.1 | 1/1/1 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | -0.2265 | `gen_fuel_natural_gas_mw` |
| 2 | 0.3098 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw` |
| 3 | 0.5905 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw` |
| 5 | 0.8845 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_1_net_export_mw` |
| 8 | 0.9582 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_1_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw|system_losses_mw` |
| 9 | 0.9637 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_1_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw|system_losses_mw|area_2_net_export_mw` |
| 12 | 0.9716 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_1_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw|system_losses_mw|area_2_net_export_mw|reserve_spin_r1_mw|gen_fuel_nuclear_mw|area_1_load_actual_mw` |
| 23 | 0.9707 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_1_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw|system_losses_mw|area_2_net_export_mw|reserve_spin_r1_mw|gen_fuel_nuclear_mw|area_1_load_actual_mw|reserve_spin_r2_mw|area_2_load_actual_mw|system_generation_mw|system_load_day_ahead_mw|area_2_load_day_ahead_mw|system_load_actual_mw|reserve_reg_up_mw|reserve_reg_down_mw|area_3_load_day_ahead_mw|reserve_spin_r3_mw|area_3_load_actual_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
