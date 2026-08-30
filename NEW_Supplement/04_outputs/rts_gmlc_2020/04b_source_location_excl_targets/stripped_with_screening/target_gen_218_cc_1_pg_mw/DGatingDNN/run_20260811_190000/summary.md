# 推断源定位：218_CC_1 有功出力

- 分支：`with_screening`
- 剥离后/初筛后候选数：20
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9141 ± 0.0060**。
- 共识推断源：**6 个**（`gen_fuel_natural_gas_mw`, `gen_fuel_hydro_mw`, `gen_fuel_solar_mw`, `gen_fuel_coal_mw`, `area_2_load_actual_mw`, `reserve_spin_r2_mw`）。
- 只用共识字段重训普通 DNN：**0.9058 ± 0.0008**。
- 相对全候选精度变化：**-0.0084**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_natural_gas_mw` | 0.154953 | 0.0231698 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0712962 | 0.00408007 | 3/3 |
| `gen_fuel_solar_mw` | 0.0617981 | 0.00512533 | 3/3 |
| `gen_fuel_coal_mw` | 0.0595202 | 0.00396294 | 3/3 |
| `area_2_load_actual_mw` | 0.0794115 | 0.0608457 | 2/3 |
| `reserve_spin_r2_mw` | 0.0734783 | 0.0812742 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 5/7/6 |
| 0.0001 | 5/7/5 |
| 0.001 | 5/7/5 |
| 0.005 | 5/7/5 |
| 0.01 | 5/7/5 |
| 0.02 | 5/7/5 |
| 0.05 | 5/5/5 |
| 0.1 | 2/1/2 |

0.001–0.02 范围内活跃字段数不变，门控分布存在稳定空档。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.5537 | `gen_fuel_natural_gas_mw` |
| 2 | 0.5923 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw` |
| 3 | 0.6273 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw` |
| 5 | 0.9078 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_coal_mw|area_2_load_actual_mw` |
| 6 | 0.9049 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_coal_mw|area_2_load_actual_mw|reserve_spin_r2_mw` |
| 8 | 0.9131 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_coal_mw|area_2_load_actual_mw|reserve_spin_r2_mw|area_2_load_day_ahead_mw|area_1_load_actual_mw` |
| 12 | 0.8993 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_coal_mw|area_2_load_actual_mw|reserve_spin_r2_mw|area_2_load_day_ahead_mw|area_1_load_actual_mw|reserve_spin_r1_mw|gen_fuel_nuclear_mw|system_load_actual_mw|area_1_load_day_ahead_mw` |
| 20 | 0.9070 | `gen_fuel_natural_gas_mw|gen_fuel_hydro_mw|gen_fuel_solar_mw|gen_fuel_coal_mw|area_2_load_actual_mw|reserve_spin_r2_mw|area_2_load_day_ahead_mw|area_1_load_actual_mw|reserve_spin_r1_mw|gen_fuel_nuclear_mw|system_load_actual_mw|area_1_load_day_ahead_mw|system_load_day_ahead_mw|system_generation_mw|system_losses_mw|area_1_net_export_mw|reserve_spin_r3_mw|area_2_net_export_mw|area_3_load_day_ahead_mw|area_3_load_actual_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
