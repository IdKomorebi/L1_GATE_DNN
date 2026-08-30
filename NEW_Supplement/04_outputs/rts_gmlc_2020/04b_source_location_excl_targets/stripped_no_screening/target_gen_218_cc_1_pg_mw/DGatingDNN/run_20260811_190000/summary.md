# 推断源定位：218_CC_1 有功出力

- 分支：`no_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9202 ± 0.0089**。
- 共识推断源：**7 个**（`gen_fuel_natural_gas_mw`, `gen_fuel_hydro_mw`, `area_2_load_actual_mw`, `area_2_load_day_ahead_mw`, `gen_fuel_solar_mw`, `gen_fuel_wind_mw`, `reserve_spin_r3_mw`）。
- 只用共识字段重训普通 DNN：**0.9088 ± 0.0016**。
- 相对全候选精度变化：**-0.0114**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_natural_gas_mw` | 0.210677 | 0.0352643 | 3/3 |
| `gen_fuel_hydro_mw` | 0.072377 | 0.012409 | 3/3 |
| `area_2_load_actual_mw` | 0.0918783 | 0.0894223 | 2/3 |
| `area_2_load_day_ahead_mw` | 0.0647661 | 0.0464878 | 2/3 |
| `gen_fuel_solar_mw` | 0.0378687 | 0.0286899 | 2/3 |
| `gen_fuel_wind_mw` | 0.031563 | 0.0224694 | 2/3 |
| `reserve_spin_r3_mw` | 0.0229161 | 0.0168854 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 6/7/5 |
| 0.0001 | 6/7/5 |
| 0.001 | 6/6/5 |
| 0.005 | 6/6/5 |
| 0.01 | 6/6/5 |
| 0.02 | 6/6/5 |
| 0.05 | 5/3/5 |
| 0.1 | 1/2/2 |

0.001–0.02 范围内活跃字段数不变，门控分布存在稳定空档。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.5537 | `gen_fuel_natural_gas_mw` |
| 2 | 0.5923 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw` |
| 3 | 0.8057 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|area_2_load_actual_mw` |
| 5 | 0.9241 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|area_2_load_actual_mw|area_2_load_day_ahead_mw|gen_fuel_solar_mw` |
| 7 | 0.9104 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|area_2_load_actual_mw|area_2_load_day_ahead_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|reserve_spin_r3_mw` |
| 8 | 0.9059 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|area_2_load_actual_mw|area_2_load_day_ahead_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|reserve_spin_r3_mw|gen_fuel_coal_mw` |
| 12 | 0.9076 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|area_2_load_actual_mw|area_2_load_day_ahead_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|reserve_spin_r3_mw|gen_fuel_coal_mw|reserve_spin_r2_mw|area_1_load_actual_mw|area_3_load_actual_mw|system_load_actual_mw` |
| 24 | 0.9124 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|area_2_load_actual_mw|area_2_load_day_ahead_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|reserve_spin_r3_mw|gen_fuel_coal_mw|reserve_spin_r2_mw|area_1_load_actual_mw|area_3_load_actual_mw|system_load_actual_mw|area_1_load_day_ahead_mw|area_3_load_day_ahead_mw|system_load_day_ahead_mw|system_generation_mw|system_losses_mw|area_1_net_export_mw|area_2_net_export_mw|area_3_net_export_mw|reserve_reg_up_mw|reserve_reg_down_mw|reserve_spin_r1_mw|gen_fuel_nuclear_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
