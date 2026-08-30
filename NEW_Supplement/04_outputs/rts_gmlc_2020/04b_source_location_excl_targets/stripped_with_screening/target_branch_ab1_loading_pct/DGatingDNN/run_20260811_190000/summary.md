# 推断源定位：跨区支路 AB1 负载率

- 分支：`with_screening`
- 剥离后/初筛后候选数：23
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.8890 ± 0.0072**。
- 共识推断源：**11 个**（`system_losses_mw`, `gen_fuel_coal_mw`, `gen_fuel_natural_gas_mw`, `area_2_net_export_mw`, `gen_fuel_wind_mw`, `area_1_net_export_mw`, `reserve_spin_r1_mw`, `area_1_load_actual_mw`, `area_2_load_actual_mw`, `reserve_spin_r3_mw`, `gen_fuel_nuclear_mw`）。
- 只用共识字段重训普通 DNN：**0.8507 ± 0.0017**。
- 相对全候选精度变化：**-0.0383**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `system_losses_mw` | 0.127238 | 0.00731509 | 3/3 |
| `gen_fuel_coal_mw` | 0.114195 | 0.00924343 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.0870625 | 0.00430858 | 3/3 |
| `area_2_net_export_mw` | 0.0743728 | 0.00422833 | 3/3 |
| `gen_fuel_wind_mw` | 0.0574563 | 0.000946755 | 3/3 |
| `area_1_net_export_mw` | 0.0543839 | 0.00255555 | 3/3 |
| `reserve_spin_r1_mw` | 0.125109 | 0.089581 | 2/3 |
| `area_1_load_actual_mw` | 0.0685994 | 0.0690351 | 2/3 |
| `area_2_load_actual_mw` | 0.0601085 | 0.0434332 | 2/3 |
| `reserve_spin_r3_mw` | 0.0217777 | 0.015505 | 2/3 |
| `gen_fuel_nuclear_mw` | 0.0103377 | 0.00757617 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 12/11/13 |
| 0.0001 | 11/11/13 |
| 0.001 | 11/11/13 |
| 0.005 | 11/11/11 |
| 0.01 | 10/10/11 |
| 0.02 | 10/9/10 |
| 0.05 | 8/8/8 |
| 0.1 | 3/3/4 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.1626 | `system_losses_mw` |
| 2 | 0.5467 | `system_losses_mw|gen_fuel_coal_mw` |
| 3 | 0.6585 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw` |
| 5 | 0.6931 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_2_net_export_mw|gen_fuel_wind_mw` |
| 8 | 0.7964 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_2_net_export_mw|gen_fuel_wind_mw|area_1_net_export_mw|reserve_spin_r1_mw|area_1_load_actual_mw` |
| 11 | 0.8531 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_2_net_export_mw|gen_fuel_wind_mw|area_1_net_export_mw|reserve_spin_r1_mw|area_1_load_actual_mw|area_2_load_actual_mw|reserve_spin_r3_mw|gen_fuel_nuclear_mw` |
| 12 | 0.8796 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_2_net_export_mw|gen_fuel_wind_mw|area_1_net_export_mw|reserve_spin_r1_mw|area_1_load_actual_mw|area_2_load_actual_mw|reserve_spin_r3_mw|gen_fuel_nuclear_mw|area_2_load_day_ahead_mw` |
| 23 | 0.8943 | `system_losses_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|area_2_net_export_mw|gen_fuel_wind_mw|area_1_net_export_mw|reserve_spin_r1_mw|area_1_load_actual_mw|area_2_load_actual_mw|reserve_spin_r3_mw|gen_fuel_nuclear_mw|area_2_load_day_ahead_mw|gen_fuel_hydro_mw|area_3_load_actual_mw|reserve_reg_down_mw|area_1_load_day_ahead_mw|reserve_spin_r2_mw|reserve_reg_up_mw|system_generation_mw|system_load_day_ahead_mw|area_3_load_day_ahead_mw|system_load_actual_mw|gen_fuel_solar_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
