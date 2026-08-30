# 推断源定位：跨区支路 AB1 负载率

- 分支：`no_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9036 ± 0.0023**。
- 共识推断源：**11 个**（`system_losses_mw`, `gen_fuel_coal_mw`, `gen_fuel_natural_gas_mw`, `gen_fuel_wind_mw`, `area_2_net_export_mw`, `area_3_net_export_mw`, `gen_fuel_hydro_mw`, `area_1_load_actual_mw`, `area_2_load_day_ahead_mw`, `reserve_spin_r3_mw`, `gen_fuel_nuclear_mw`）。
- 只用共识字段重训普通 DNN：**0.8906 ± 0.0019**。
- 相对全候选精度变化：**-0.0130**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `system_losses_mw` | 0.123257 | 0.00850385 | 3/3 |
| `gen_fuel_coal_mw` | 0.118443 | 0.0111988 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.0824485 | 0.0103698 | 3/3 |
| `gen_fuel_wind_mw` | 0.0492991 | 0.00159348 | 3/3 |
| `area_2_net_export_mw` | 0.04884 | 0.00395627 | 3/3 |
| `area_3_net_export_mw` | 0.0447473 | 0.0020447 | 3/3 |
| `gen_fuel_hydro_mw` | 0.041824 | 0.00527811 | 3/3 |
| `area_1_load_actual_mw` | 0.123478 | 0.087388 | 2/3 |
| `area_2_load_day_ahead_mw` | 0.0404976 | 0.0435943 | 2/3 |
| `reserve_spin_r3_mw` | 0.0161978 | 0.0115135 | 2/3 |
| `gen_fuel_nuclear_mw` | 0.01067 | 0.000693566 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 13/14/12 |
| 0.0001 | 13/14/12 |
| 0.001 | 13/13/12 |
| 0.005 | 13/13/12 |
| 0.01 | 11/12/11 |
| 0.02 | 11/11/10 |
| 0.05 | 5/5/7 |
| 0.1 | 3/3/4 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.1626 | `system_losses_mw` |
| 2 | 0.5467 | `system_losses_mw|gen_fuel_coal_mw` |
| 3 | 0.6585 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw` |
| 5 | 0.6705 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|area_2_net_export_mw` |
| 8 | 0.8196 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_hydro_mw|area_1_load_actual_mw` |
| 11 | 0.8911 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_hydro_mw|area_1_load_actual_mw|area_2_load_day_ahead_mw|reserve_spin_r3_mw|gen_fuel_nuclear_mw` |
| 12 | 0.8917 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_hydro_mw|area_1_load_actual_mw|area_2_load_day_ahead_mw|reserve_spin_r3_mw|gen_fuel_nuclear_mw|reserve_spin_r1_mw` |
| 24 | 0.9037 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_hydro_mw|area_1_load_actual_mw|area_2_load_day_ahead_mw|reserve_spin_r3_mw|gen_fuel_nuclear_mw|reserve_spin_r1_mw|area_2_load_actual_mw|reserve_spin_r2_mw|area_1_load_day_ahead_mw|area_3_load_day_ahead_mw|reserve_reg_down_mw|area_3_load_actual_mw|system_load_actual_mw|system_load_day_ahead_mw|system_generation_mw|area_1_net_export_mw|reserve_reg_up_mw|gen_fuel_solar_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
