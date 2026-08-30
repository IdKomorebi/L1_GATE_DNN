# 推断源定位：母线 215 电压相角

- 分支：`no_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9854 ± 0.0006**。
- 共识推断源：**7 个**（`gen_fuel_natural_gas_mw`, `gen_fuel_coal_mw`, `gen_fuel_solar_mw`, `gen_fuel_wind_mw`, `system_losses_mw`, `gen_fuel_hydro_mw`, `area_3_net_export_mw`）。
- 只用共识字段重训普通 DNN：**0.9570 ± 0.0022**。
- 相对全候选精度变化：**-0.0284**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_natural_gas_mw` | 0.0742364 | 0.00459167 | 3/3 |
| `gen_fuel_coal_mw` | 0.0648153 | 0.0055463 | 3/3 |
| `gen_fuel_solar_mw` | 0.0520281 | 0.00585771 | 3/3 |
| `gen_fuel_wind_mw` | 0.0388595 | 0.00539307 | 3/3 |
| `system_losses_mw` | 0.0322493 | 0.00338349 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0226596 | 0.00111246 | 3/3 |
| `area_3_net_export_mw` | 0.0216827 | 0.000438721 | 3/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 9/9/9 |
| 0.0001 | 9/9/9 |
| 0.001 | 9/9/9 |
| 0.005 | 9/9/9 |
| 0.01 | 8/8/8 |
| 0.02 | 8/8/8 |
| 0.05 | 4/3/4 |
| 0.1 | 0/0/0 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | -0.1770 | `gen_fuel_natural_gas_mw` |
| 2 | 0.2874 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw` |
| 3 | 0.6463 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw` |
| 5 | 0.9178 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|system_losses_mw` |
| 7 | 0.9546 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|system_losses_mw|gen_fuel_hydro_mw|area_3_net_export_mw` |
| 8 | 0.9689 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|system_losses_mw|gen_fuel_hydro_mw|area_3_net_export_mw|area_1_load_day_ahead_mw` |
| 12 | 0.9793 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|system_losses_mw|gen_fuel_hydro_mw|area_3_net_export_mw|area_1_load_day_ahead_mw|area_1_load_actual_mw|reserve_spin_r1_mw|gen_fuel_nuclear_mw|area_2_load_actual_mw` |
| 24 | 0.9855 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|system_losses_mw|gen_fuel_hydro_mw|area_3_net_export_mw|area_1_load_day_ahead_mw|area_1_load_actual_mw|reserve_spin_r1_mw|gen_fuel_nuclear_mw|area_2_load_actual_mw|area_3_load_actual_mw|system_load_actual_mw|area_2_load_day_ahead_mw|area_3_load_day_ahead_mw|system_load_day_ahead_mw|system_generation_mw|area_1_net_export_mw|area_2_net_export_mw|reserve_reg_up_mw|reserve_reg_down_mw|reserve_spin_r2_mw|reserve_spin_r3_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
