# 推断源定位：321_CC_1 投入状态

- 分支：`with_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9685 ± 0.0013**。
- 共识推断源：**7 个**（`gen_fuel_natural_gas_mw`, `area_2_load_actual_mw`, `gen_fuel_hydro_mw`, `gen_fuel_solar_mw`, `gen_fuel_wind_mw`, `area_3_net_export_mw`, `reserve_spin_r1_mw`）。
- 只用共识字段重训普通 DNN：**0.9566 ± 0.0004**。
- 相对全候选精度变化：**-0.0119**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_natural_gas_mw` | 0.316872 | 0.00607533 | 3/3 |
| `area_2_load_actual_mw` | 0.133455 | 0.0347212 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0925045 | 0.00607033 | 3/3 |
| `gen_fuel_solar_mw` | 0.0789497 | 0.00809649 | 3/3 |
| `gen_fuel_wind_mw` | 0.0769699 | 0.00702342 | 3/3 |
| `area_3_net_export_mw` | 0.0439041 | 0.00436199 | 3/3 |
| `reserve_spin_r1_mw` | 0.0462283 | 0.0337392 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 10/9/9 |
| 0.0001 | 9/9/8 |
| 0.001 | 9/9/8 |
| 0.005 | 9/9/8 |
| 0.01 | 9/9/8 |
| 0.02 | 8/9/8 |
| 0.05 | 7/8/7 |
| 0.1 | 4/1/2 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.8570 | `gen_fuel_natural_gas_mw` |
| 2 | 0.9145 | `gen_fuel_natural_gas_mw|area_2_load_actual_mw` |
| 3 | 0.9369 | `gen_fuel_natural_gas_mw|area_2_load_actual_mw|gen_fuel_hydro_mw` |
| 5 | 0.9569 | `gen_fuel_natural_gas_mw|area_2_load_actual_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_wind_mw` |
| 7 | 0.9565 | `gen_fuel_natural_gas_mw|area_2_load_actual_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_3_net_export_mw|reserve_spin_r1_mw` |
| 8 | 0.9535 | `gen_fuel_natural_gas_mw|area_2_load_actual_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_3_net_export_mw|reserve_spin_r1_mw|reserve_spin_r3_mw` |
| 12 | 0.9623 | `gen_fuel_natural_gas_mw|area_2_load_actual_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_3_net_export_mw|reserve_spin_r1_mw|reserve_spin_r3_mw|area_3_load_actual_mw|reserve_spin_r2_mw|area_3_load_day_ahead_mw|system_losses_mw` |
| 24 | 0.9723 | `gen_fuel_natural_gas_mw|area_2_load_actual_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_wind_mw|area_3_net_export_mw|reserve_spin_r1_mw|reserve_spin_r3_mw|area_3_load_actual_mw|reserve_spin_r2_mw|area_3_load_day_ahead_mw|system_losses_mw|area_1_load_actual_mw|area_2_load_day_ahead_mw|gen_fuel_coal_mw|system_load_day_ahead_mw|gen_fuel_nuclear_mw|area_2_net_export_mw|area_1_net_export_mw|reserve_reg_up_mw|reserve_reg_down_mw|system_generation_mw|area_1_load_day_ahead_mw|system_load_actual_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
