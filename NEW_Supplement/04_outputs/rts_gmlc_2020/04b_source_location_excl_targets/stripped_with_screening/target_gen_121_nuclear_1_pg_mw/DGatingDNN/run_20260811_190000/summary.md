# 推断源定位：121_NUCLEAR_1 有功出力

- 分支：`with_screening`
- 剥离后/初筛后候选数：21
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9987 ± 0.0003**。
- 共识推断源：**8 个**（`gen_fuel_wind_mw`, `gen_fuel_solar_mw`, `gen_fuel_natural_gas_mw`, `gen_fuel_coal_mw`, `reserve_spin_r1_mw`, `gen_fuel_hydro_mw`, `area_2_load_day_ahead_mw`, `reserve_spin_r3_mw`）。
- 只用共识字段重训普通 DNN：**0.9982 ± 0.0002**。
- 相对全候选精度变化：**-0.0005**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_wind_mw` | 0.359327 | 0.00836507 | 3/3 |
| `gen_fuel_solar_mw` | 0.266411 | 0.0152273 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.239033 | 0.00908854 | 3/3 |
| `gen_fuel_coal_mw` | 0.15656 | 0.00907835 | 3/3 |
| `reserve_spin_r1_mw` | 0.154451 | 0.015702 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0763025 | 0.00321767 | 3/3 |
| `area_2_load_day_ahead_mw` | 0.0864618 | 0.0611403 | 2/3 |
| `reserve_spin_r3_mw` | 0.0656807 | 0.0464552 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 8/8/9 |
| 0.0001 | 8/8/9 |
| 0.001 | 8/8/9 |
| 0.005 | 8/8/9 |
| 0.01 | 8/8/9 |
| 0.02 | 8/8/8 |
| 0.05 | 8/8/8 |
| 0.1 | 6/6/6 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.2269 | `gen_fuel_wind_mw` |
| 2 | 0.4046 | `gen_fuel_wind_mw|gen_fuel_solar_mw` |
| 3 | 0.5327 | `gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw` |
| 5 | 0.8108 | `gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw|gen_fuel_coal_mw|reserve_spin_r1_mw` |
| 8 | 0.9980 | `gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw|gen_fuel_coal_mw|reserve_spin_r1_mw|gen_fuel_hydro_mw|area_2_load_day_ahead_mw|reserve_spin_r3_mw` |
| 12 | 0.9986 | `gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw|gen_fuel_coal_mw|reserve_spin_r1_mw|gen_fuel_hydro_mw|area_2_load_day_ahead_mw|reserve_spin_r3_mw|reserve_spin_r2_mw|area_3_load_day_ahead_mw|area_1_load_day_ahead_mw|area_2_load_actual_mw` |
| 21 | 0.9991 | `gen_fuel_wind_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw|gen_fuel_coal_mw|reserve_spin_r1_mw|gen_fuel_hydro_mw|area_2_load_day_ahead_mw|reserve_spin_r3_mw|reserve_spin_r2_mw|area_3_load_day_ahead_mw|area_1_load_day_ahead_mw|area_2_load_actual_mw|area_1_load_actual_mw|area_2_net_export_mw|system_load_actual_mw|area_1_net_export_mw|area_3_load_actual_mw|area_3_net_export_mw|system_load_day_ahead_mw|reserve_reg_up_mw|reserve_reg_down_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
