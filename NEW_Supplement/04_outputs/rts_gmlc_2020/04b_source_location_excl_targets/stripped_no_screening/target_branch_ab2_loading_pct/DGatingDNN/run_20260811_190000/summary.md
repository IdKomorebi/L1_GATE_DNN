# 推断源定位：跨区支路 AB2 负载率

- 分支：`no_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9881 ± 0.0008**。
- 共识推断源：**7 个**（`gen_fuel_natural_gas_mw`, `gen_fuel_coal_mw`, `gen_fuel_solar_mw`, `gen_fuel_wind_mw`, `system_losses_mw`, `area_3_net_export_mw`, `gen_fuel_hydro_mw`）。
- 只用共识字段重训普通 DNN：**0.9522 ± 0.0040**。
- 相对全候选精度变化：**-0.0359**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_natural_gas_mw` | 0.0729123 | 0.0043241 | 3/3 |
| `gen_fuel_coal_mw` | 0.0574661 | 0.0045988 | 3/3 |
| `gen_fuel_solar_mw` | 0.049678 | 0.00450568 | 3/3 |
| `gen_fuel_wind_mw` | 0.0358775 | 0.00538651 | 3/3 |
| `system_losses_mw` | 0.0355094 | 0.00540141 | 3/3 |
| `area_3_net_export_mw` | 0.0209625 | 0.00107626 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0204068 | 0.00115173 | 3/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 10/10/10 |
| 0.0001 | 10/10/10 |
| 0.001 | 10/10/10 |
| 0.005 | 10/9/10 |
| 0.01 | 9/8/8 |
| 0.02 | 7/7/8 |
| 0.05 | 4/3/4 |
| 0.1 | 1/0/1 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | -0.2265 | `gen_fuel_natural_gas_mw` |
| 2 | 0.3098 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw` |
| 3 | 0.5905 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw` |
| 5 | 0.9139 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|system_losses_mw` |
| 7 | 0.9477 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|system_losses_mw|area_3_net_export_mw|gen_fuel_hydro_mw` |
| 8 | 0.9627 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|system_losses_mw|area_3_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw` |
| 12 | 0.9737 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|system_losses_mw|area_3_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw|area_1_load_actual_mw|reserve_spin_r1_mw|gen_fuel_nuclear_mw|area_2_net_export_mw` |
| 24 | 0.9872 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|system_losses_mw|area_3_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw|area_1_load_actual_mw|reserve_spin_r1_mw|gen_fuel_nuclear_mw|area_2_net_export_mw|area_2_load_actual_mw|area_3_load_actual_mw|system_load_actual_mw|area_2_load_day_ahead_mw|area_3_load_day_ahead_mw|system_load_day_ahead_mw|system_generation_mw|area_1_net_export_mw|reserve_reg_up_mw|reserve_reg_down_mw|reserve_spin_r2_mw|reserve_spin_r3_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
