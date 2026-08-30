# 推断源定位：母线 215 电压相角

- 分支：`with_screening`
- 剥离后/初筛后候选数：23
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9750 ± 0.0014**。
- 共识推断源：**7 个**（`gen_fuel_natural_gas_mw`, `gen_fuel_coal_mw`, `gen_fuel_solar_mw`, `gen_fuel_wind_mw`, `area_1_net_export_mw`, `gen_fuel_hydro_mw`, `area_1_load_day_ahead_mw`）。
- 只用共识字段重训普通 DNN：**0.9640 ± 0.0025**。
- 相对全候选精度变化：**-0.0110**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_natural_gas_mw` | 0.0700347 | 0.0129176 | 3/3 |
| `gen_fuel_coal_mw` | 0.0611341 | 0.00365063 | 3/3 |
| `gen_fuel_solar_mw` | 0.0530287 | 0.00522499 | 3/3 |
| `gen_fuel_wind_mw` | 0.0415322 | 0.00503088 | 3/3 |
| `area_1_net_export_mw` | 0.0226689 | 0.00163702 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0215239 | 0.0033304 | 3/3 |
| `area_1_load_day_ahead_mw` | 0.0610822 | 0.0439273 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 9/11/9 |
| 0.0001 | 9/11/9 |
| 0.001 | 9/11/9 |
| 0.005 | 9/11/9 |
| 0.01 | 7/10/7 |
| 0.02 | 7/8/7 |
| 0.05 | 4/3/4 |
| 0.1 | 0/0/1 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | -0.1770 | `gen_fuel_natural_gas_mw` |
| 2 | 0.2874 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw` |
| 3 | 0.6463 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw` |
| 5 | 0.8893 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_1_net_export_mw` |
| 7 | 0.9673 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_1_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw` |
| 8 | 0.9612 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_1_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw|reserve_spin_r1_mw` |
| 12 | 0.9728 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_1_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw|reserve_spin_r1_mw|system_losses_mw|area_2_net_export_mw|area_3_load_day_ahead_mw|gen_fuel_nuclear_mw` |
| 23 | 0.9739 | `gen_fuel_natural_gas_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_1_net_export_mw|gen_fuel_hydro_mw|area_1_load_day_ahead_mw|reserve_spin_r1_mw|system_losses_mw|area_2_net_export_mw|area_3_load_day_ahead_mw|gen_fuel_nuclear_mw|area_1_load_actual_mw|reserve_spin_r2_mw|area_2_load_actual_mw|system_generation_mw|system_load_day_ahead_mw|area_2_load_day_ahead_mw|system_load_actual_mw|reserve_reg_up_mw|reserve_reg_down_mw|reserve_spin_r3_mw|area_3_load_actual_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
