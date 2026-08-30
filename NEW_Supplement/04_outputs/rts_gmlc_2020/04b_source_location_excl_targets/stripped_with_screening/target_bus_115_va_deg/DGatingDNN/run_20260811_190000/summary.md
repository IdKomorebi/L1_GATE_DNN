# 推断源定位：母线 115 电压相角

- 分支：`with_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9493 ± 0.0009**。
- 共识推断源：**8 个**（`system_losses_mw`, `gen_fuel_coal_mw`, `area_2_net_export_mw`, `area_3_net_export_mw`, `gen_fuel_hydro_mw`, `gen_fuel_nuclear_mw`, `reserve_spin_r1_mw`, `gen_fuel_wind_mw`）。
- 只用共识字段重训普通 DNN：**0.9399 ± 0.0030**。
- 相对全候选精度变化：**-0.0093**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `system_losses_mw` | 0.145191 | 0.0257131 | 3/3 |
| `gen_fuel_coal_mw` | 0.121096 | 0.0141887 | 3/3 |
| `area_2_net_export_mw` | 0.0481906 | 0.00918357 | 3/3 |
| `area_3_net_export_mw` | 0.0408736 | 0.00524288 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0379021 | 0.00133688 | 3/3 |
| `gen_fuel_nuclear_mw` | 0.0232299 | 0.00201633 | 3/3 |
| `reserve_spin_r1_mw` | 0.0617137 | 0.0450246 | 2/3 |
| `gen_fuel_wind_mw` | 0.0245179 | 0.0175827 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 11/10/10 |
| 0.0001 | 11/10/10 |
| 0.001 | 11/10/10 |
| 0.005 | 11/10/10 |
| 0.01 | 11/10/10 |
| 0.02 | 11/10/10 |
| 0.05 | 3/3/5 |
| 0.1 | 2/3/3 |

0.001–0.02 范围内活跃字段数不变，门控分布存在稳定空档。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.5822 | `system_losses_mw` |
| 2 | 0.7916 | `system_losses_mw|gen_fuel_coal_mw` |
| 3 | 0.7938 | `system_losses_mw|gen_fuel_coal_mw|area_2_net_export_mw` |
| 5 | 0.8667 | `system_losses_mw|gen_fuel_coal_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_hydro_mw` |
| 8 | 0.9406 | `system_losses_mw|gen_fuel_coal_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_hydro_mw|gen_fuel_nuclear_mw|reserve_spin_r1_mw|gen_fuel_wind_mw` |
| 12 | 0.9481 | `system_losses_mw|gen_fuel_coal_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_hydro_mw|gen_fuel_nuclear_mw|reserve_spin_r1_mw|gen_fuel_wind_mw|area_1_load_actual_mw|reserve_spin_r3_mw|area_3_load_actual_mw|area_3_load_day_ahead_mw` |
| 24 | 0.9490 | `system_losses_mw|gen_fuel_coal_mw|area_2_net_export_mw|area_3_net_export_mw|gen_fuel_hydro_mw|gen_fuel_nuclear_mw|reserve_spin_r1_mw|gen_fuel_wind_mw|area_1_load_actual_mw|reserve_spin_r3_mw|area_3_load_actual_mw|area_3_load_day_ahead_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw|area_2_load_day_ahead_mw|area_1_load_day_ahead_mw|area_2_load_actual_mw|area_1_net_export_mw|reserve_spin_r2_mw|reserve_reg_down_mw|reserve_reg_up_mw|system_generation_mw|system_load_day_ahead_mw|system_load_actual_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
