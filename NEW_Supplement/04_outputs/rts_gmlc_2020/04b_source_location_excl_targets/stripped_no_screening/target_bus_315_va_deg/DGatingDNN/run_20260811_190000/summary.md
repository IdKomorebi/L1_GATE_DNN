# 推断源定位：母线 315 电压相角

- 分支：`no_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9631 ± 0.0011**。
- 共识推断源：**11 个**（`system_losses_mw`, `gen_fuel_natural_gas_mw`, `gen_fuel_wind_mw`, `gen_fuel_solar_mw`, `gen_fuel_hydro_mw`, `area_2_net_export_mw`, `area_3_net_export_mw`, `gen_fuel_nuclear_mw`, `area_1_load_day_ahead_mw`, `area_2_load_actual_mw`, `area_3_load_day_ahead_mw`）。
- 只用共识字段重训普通 DNN：**0.9587 ± 0.0039**。
- 相对全候选精度变化：**-0.0044**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `system_losses_mw` | 0.0951775 | 0.0132395 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.077507 | 0.0125368 | 3/3 |
| `gen_fuel_wind_mw` | 0.0713201 | 0.00664781 | 3/3 |
| `gen_fuel_solar_mw` | 0.0545231 | 0.0041964 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0410375 | 0.00453937 | 3/3 |
| `area_2_net_export_mw` | 0.039944 | 0.00467835 | 3/3 |
| `area_3_net_export_mw` | 0.0355444 | 0.00459231 | 3/3 |
| `gen_fuel_nuclear_mw` | 0.0206239 | 0.00335979 | 3/3 |
| `area_1_load_day_ahead_mw` | 0.0420014 | 0.0313989 | 2/3 |
| `area_2_load_actual_mw` | 0.0418813 | 0.0301943 | 2/3 |
| `area_3_load_day_ahead_mw` | 0.0271222 | 0.0192781 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 13/11/11 |
| 0.0001 | 11/11/11 |
| 0.001 | 11/11/11 |
| 0.005 | 11/11/11 |
| 0.01 | 11/11/11 |
| 0.02 | 11/10/11 |
| 0.05 | 6/6/5 |
| 0.1 | 1/0/0 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.2698 | `system_losses_mw` |
| 2 | 0.5316 | `system_losses_mw|gen_fuel_natural_gas_mw` |
| 3 | 0.8335 | `system_losses_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw` |
| 5 | 0.8583 | `system_losses_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_hydro_mw` |
| 8 | 0.9360 | `system_losses_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_hydro_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_nuclear_mw` |
| 11 | 0.9641 | `system_losses_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_hydro_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_nuclear_mw|area_1_load_day_ahead_mw|area_2_load_actual_mw|area_3_load_day_ahead_mw` |
| 12 | 0.9598 | `system_losses_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_hydro_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_nuclear_mw|area_1_load_day_ahead_mw|area_2_load_actual_mw|area_3_load_day_ahead_mw|area_1_load_actual_mw` |
| 24 | 0.9588 | `system_losses_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_hydro_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_nuclear_mw|area_1_load_day_ahead_mw|area_2_load_actual_mw|area_3_load_day_ahead_mw|area_1_load_actual_mw|area_2_load_day_ahead_mw|reserve_spin_r3_mw|reserve_spin_r1_mw|gen_fuel_coal_mw|reserve_reg_down_mw|area_1_net_export_mw|area_3_load_actual_mw|system_load_actual_mw|system_load_day_ahead_mw|system_generation_mw|reserve_reg_up_mw|reserve_spin_r2_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
