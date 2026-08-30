# 推断源定位：317_WIND_1 有功出力

- 分支：`no_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9168 ± 0.0134**。
- 共识推断源：**8 个**（`area_1_net_export_mw`, `area_3_net_export_mw`, `gen_fuel_wind_mw`, `area_2_net_export_mw`, `gen_fuel_hydro_mw`, `gen_fuel_solar_mw`, `gen_fuel_natural_gas_mw`, `system_losses_mw`）。
- 只用共识字段重训普通 DNN：**0.9176 ± 0.0018**。
- 相对全候选精度变化：**+0.0008**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `area_1_net_export_mw` | 0.191546 | 0.0316338 | 3/3 |
| `area_3_net_export_mw` | 0.170951 | 0.024293 | 3/3 |
| `gen_fuel_wind_mw` | 0.170496 | 0.0238414 | 3/3 |
| `area_2_net_export_mw` | 0.134173 | 0.0247405 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0649635 | 0.00600101 | 3/3 |
| `gen_fuel_solar_mw` | 0.0604085 | 0.00842602 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.0501779 | 0.035485 | 2/3 |
| `system_losses_mw` | 0.0273363 | 0.0193442 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 8/9/10 |
| 0.0001 | 8/9/10 |
| 0.001 | 8/9/10 |
| 0.005 | 8/9/10 |
| 0.01 | 7/9/9 |
| 0.02 | 7/8/9 |
| 0.05 | 7/6/7 |
| 0.1 | 4/4/4 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.4145 | `area_1_net_export_mw` |
| 2 | 0.5475 | `area_1_net_export_mw|area_3_net_export_mw` |
| 3 | 0.8727 | `area_1_net_export_mw|area_3_net_export_mw|gen_fuel_wind_mw` |
| 5 | 0.9006 | `area_1_net_export_mw|area_3_net_export_mw|gen_fuel_wind_mw|area_2_net_export_mw|gen_fuel_hydro_mw` |
| 8 | 0.9166 | `area_1_net_export_mw|area_3_net_export_mw|gen_fuel_wind_mw|area_2_net_export_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw|system_losses_mw` |
| 12 | 0.9209 | `area_1_net_export_mw|area_3_net_export_mw|gen_fuel_wind_mw|area_2_net_export_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw|system_losses_mw|reserve_spin_r3_mw|reserve_reg_down_mw|area_2_load_actual_mw|area_1_load_actual_mw` |
| 24 | 0.9234 | `area_1_net_export_mw|area_3_net_export_mw|gen_fuel_wind_mw|area_2_net_export_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw|system_losses_mw|reserve_spin_r3_mw|reserve_reg_down_mw|area_2_load_actual_mw|area_1_load_actual_mw|area_3_load_actual_mw|system_load_actual_mw|area_1_load_day_ahead_mw|area_2_load_day_ahead_mw|area_3_load_day_ahead_mw|system_load_day_ahead_mw|system_generation_mw|reserve_reg_up_mw|reserve_spin_r1_mw|reserve_spin_r2_mw|gen_fuel_coal_mw|gen_fuel_nuclear_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
