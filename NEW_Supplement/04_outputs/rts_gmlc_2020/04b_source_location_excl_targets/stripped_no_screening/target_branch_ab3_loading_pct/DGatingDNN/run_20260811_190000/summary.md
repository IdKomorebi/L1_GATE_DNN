# 推断源定位：跨区支路 AB3 负载率

- 分支：`no_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.8467 ± 0.0047**。
- 共识推断源：**11 个**（`gen_fuel_coal_mw`, `gen_fuel_natural_gas_mw`, `system_losses_mw`, `gen_fuel_wind_mw`, `gen_fuel_solar_mw`, `area_3_net_export_mw`, `area_2_net_export_mw`, `area_2_load_day_ahead_mw`, `area_1_load_actual_mw`, `gen_fuel_hydro_mw`, `area_3_load_actual_mw`）。
- 只用共识字段重训普通 DNN：**0.8337 ± 0.0102**。
- 相对全候选精度变化：**-0.0130**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_coal_mw` | 0.283957 | 0.0304243 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.164844 | 0.0240096 | 3/3 |
| `system_losses_mw` | 0.11895 | 0.0127873 | 3/3 |
| `gen_fuel_wind_mw` | 0.0946367 | 0.0088454 | 3/3 |
| `gen_fuel_solar_mw` | 0.0703975 | 0.0372019 | 3/3 |
| `area_3_net_export_mw` | 0.0573024 | 0.00311459 | 3/3 |
| `area_2_net_export_mw` | 0.0502435 | 0.00390314 | 3/3 |
| `area_2_load_day_ahead_mw` | 0.0992522 | 0.0703482 | 2/3 |
| `area_1_load_actual_mw` | 0.0632447 | 0.0646395 | 2/3 |
| `gen_fuel_hydro_mw` | 0.0520508 | 0.036856 | 2/3 |
| `area_3_load_actual_mw` | 0.038851 | 0.027477 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 11/11/11 |
| 0.0001 | 11/11/11 |
| 0.001 | 11/11/11 |
| 0.005 | 11/11/11 |
| 0.01 | 11/11/11 |
| 0.02 | 11/11/11 |
| 0.05 | 10/10/10 |
| 0.1 | 6/6/5 |

0.001–0.02 范围内活跃字段数不变，门控分布存在稳定空档。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.1155 | `gen_fuel_coal_mw` |
| 2 | 0.2851 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw` |
| 3 | 0.6418 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|system_losses_mw` |
| 5 | 0.6911 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw` |
| 8 | 0.7944 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|area_3_net_export_mw|area_2_net_export_mw|area_2_load_day_ahead_mw` |
| 11 | 0.8201 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|area_3_net_export_mw|area_2_net_export_mw|area_2_load_day_ahead_mw|area_1_load_actual_mw|gen_fuel_hydro_mw|area_3_load_actual_mw` |
| 12 | 0.8285 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|area_3_net_export_mw|area_2_net_export_mw|area_2_load_day_ahead_mw|area_1_load_actual_mw|gen_fuel_hydro_mw|area_3_load_actual_mw|area_1_load_day_ahead_mw` |
| 24 | 0.8393 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|area_3_net_export_mw|area_2_net_export_mw|area_2_load_day_ahead_mw|area_1_load_actual_mw|gen_fuel_hydro_mw|area_3_load_actual_mw|area_1_load_day_ahead_mw|reserve_spin_r1_mw|area_2_load_actual_mw|reserve_spin_r3_mw|reserve_spin_r2_mw|system_load_actual_mw|area_3_load_day_ahead_mw|system_load_day_ahead_mw|system_generation_mw|area_1_net_export_mw|reserve_reg_up_mw|reserve_reg_down_mw|gen_fuel_nuclear_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
