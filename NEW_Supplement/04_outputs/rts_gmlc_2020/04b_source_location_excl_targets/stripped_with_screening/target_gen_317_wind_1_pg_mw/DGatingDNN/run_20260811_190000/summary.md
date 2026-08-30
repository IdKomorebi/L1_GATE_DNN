# 推断源定位：317_WIND_1 有功出力

- 分支：`with_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9257 ± 0.0061**。
- 共识推断源：**8 个**（`area_1_net_export_mw`, `gen_fuel_wind_mw`, `area_3_net_export_mw`, `area_2_net_export_mw`, `gen_fuel_solar_mw`, `system_losses_mw`, `gen_fuel_hydro_mw`, `reserve_spin_r3_mw`）。
- 只用共识字段重训普通 DNN：**0.9206 ± 0.0009**。
- 相对全候选精度变化：**-0.0050**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `area_1_net_export_mw` | 0.171984 | 0.00501565 | 3/3 |
| `gen_fuel_wind_mw` | 0.170354 | 0.0167782 | 3/3 |
| `area_3_net_export_mw` | 0.153321 | 0.00260579 | 3/3 |
| `area_2_net_export_mw` | 0.126514 | 0.00282119 | 3/3 |
| `gen_fuel_solar_mw` | 0.0605172 | 0.00607559 | 3/3 |
| `system_losses_mw` | 0.0496598 | 0.00114329 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0456663 | 0.00198615 | 3/3 |
| `reserve_spin_r3_mw` | 0.0246305 | 0.0174387 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 10/10/10 |
| 0.0001 | 10/10/10 |
| 0.001 | 10/10/10 |
| 0.005 | 10/10/10 |
| 0.01 | 9/9/9 |
| 0.02 | 9/9/9 |
| 0.05 | 5/5/6 |
| 0.1 | 4/4/4 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.4145 | `area_1_net_export_mw` |
| 2 | 0.8542 | `area_1_net_export_mw|gen_fuel_wind_mw` |
| 3 | 0.8684 | `area_1_net_export_mw|gen_fuel_wind_mw|area_3_net_export_mw` |
| 5 | 0.8946 | `area_1_net_export_mw|gen_fuel_wind_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_solar_mw` |
| 8 | 0.9219 | `area_1_net_export_mw|gen_fuel_wind_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_solar_mw|system_losses_mw|gen_fuel_hydro_mw|reserve_spin_r3_mw` |
| 12 | 0.9293 | `area_1_net_export_mw|gen_fuel_wind_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_solar_mw|system_losses_mw|gen_fuel_hydro_mw|reserve_spin_r3_mw|gen_fuel_coal_mw|area_3_load_day_ahead_mw|area_2_load_actual_mw|area_2_load_day_ahead_mw` |
| 24 | 0.9252 | `area_1_net_export_mw|gen_fuel_wind_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_solar_mw|system_losses_mw|gen_fuel_hydro_mw|reserve_spin_r3_mw|gen_fuel_coal_mw|area_3_load_day_ahead_mw|area_2_load_actual_mw|area_2_load_day_ahead_mw|reserve_reg_down_mw|area_1_load_actual_mw|gen_fuel_nuclear_mw|gen_fuel_natural_gas_mw|reserve_spin_r2_mw|reserve_spin_r1_mw|reserve_reg_up_mw|system_generation_mw|system_load_day_ahead_mw|area_1_load_day_ahead_mw|system_load_actual_mw|area_3_load_actual_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
