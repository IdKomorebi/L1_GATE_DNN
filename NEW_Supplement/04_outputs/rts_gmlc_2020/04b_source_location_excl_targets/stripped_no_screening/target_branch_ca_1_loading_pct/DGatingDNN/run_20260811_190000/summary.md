# 推断源定位：跨区支路 CA-1 负载率

- 分支：`no_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9914 ± 0.0007**。
- 共识推断源：**7 个**（`area_1_net_export_mw`, `area_2_net_export_mw`, `area_3_net_export_mw`, `gen_fuel_coal_mw`, `gen_fuel_natural_gas_mw`, `gen_fuel_wind_mw`, `gen_fuel_solar_mw`）。
- 只用共识字段重训普通 DNN：**0.9914 ± 0.0012**。
- 相对全候选精度变化：**-0.0000**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `area_1_net_export_mw` | 0.307126 | 0.00969406 | 3/3 |
| `area_2_net_export_mw` | 0.234114 | 0.0115717 | 3/3 |
| `area_3_net_export_mw` | 0.233868 | 0.00919739 | 3/3 |
| `gen_fuel_coal_mw` | 0.0435254 | 0.0103153 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.031612 | 0.0053367 | 3/3 |
| `gen_fuel_wind_mw` | 0.0175604 | 0.00676633 | 2/3 |
| `gen_fuel_solar_mw` | 0.0145445 | 0.010309 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 8/8/8 |
| 0.0001 | 8/8/8 |
| 0.001 | 8/8/8 |
| 0.005 | 7/7/8 |
| 0.01 | 7/7/7 |
| 0.02 | 7/6/6 |
| 0.05 | 3/3/4 |
| 0.1 | 3/3/3 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.1878 | `area_1_net_export_mw` |
| 2 | 0.4959 | `area_1_net_export_mw|area_2_net_export_mw` |
| 3 | 0.9723 | `area_1_net_export_mw|area_2_net_export_mw|area_3_net_export_mw` |
| 5 | 0.9912 | `area_1_net_export_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw` |
| 7 | 0.9906 | `area_1_net_export_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw` |
| 8 | 0.9956 | `area_1_net_export_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|reserve_spin_r1_mw` |
| 12 | 0.9930 | `area_1_net_export_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|reserve_spin_r1_mw|area_2_load_day_ahead_mw|gen_fuel_hydro_mw|area_3_load_day_ahead_mw|gen_fuel_nuclear_mw` |
| 24 | 0.9907 | `area_1_net_export_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|reserve_spin_r1_mw|area_2_load_day_ahead_mw|gen_fuel_hydro_mw|area_3_load_day_ahead_mw|gen_fuel_nuclear_mw|system_losses_mw|reserve_spin_r3_mw|area_1_load_day_ahead_mw|area_1_load_actual_mw|area_2_load_actual_mw|area_3_load_actual_mw|system_load_actual_mw|system_load_day_ahead_mw|system_generation_mw|reserve_reg_up_mw|reserve_reg_down_mw|reserve_spin_r2_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
