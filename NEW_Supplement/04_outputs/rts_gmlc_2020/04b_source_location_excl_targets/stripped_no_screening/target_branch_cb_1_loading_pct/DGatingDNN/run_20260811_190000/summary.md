# 推断源定位：跨区支路 CB-1 负载率

- 分支：`no_screening`
- 剥离后/初筛后候选数：24
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9914 ± 0.0003**。
- 共识推断源：**7 个**（`area_1_net_export_mw`, `area_3_net_export_mw`, `area_2_net_export_mw`, `gen_fuel_coal_mw`, `gen_fuel_natural_gas_mw`, `gen_fuel_wind_mw`, `gen_fuel_solar_mw`）。
- 只用共识字段重训普通 DNN：**0.9927 ± 0.0008**。
- 相对全候选精度变化：**+0.0013**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `area_1_net_export_mw` | 0.354748 | 0.00584361 | 3/3 |
| `area_3_net_export_mw` | 0.297955 | 0.00817739 | 3/3 |
| `area_2_net_export_mw` | 0.269746 | 0.0103806 | 3/3 |
| `gen_fuel_coal_mw` | 0.0458227 | 0.0147712 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.0315716 | 0.00640397 | 3/3 |
| `gen_fuel_wind_mw` | 0.0156512 | 0.0016062 | 3/3 |
| `gen_fuel_solar_mw` | 0.0149997 | 0.0106318 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 8/8/8 |
| 0.0001 | 8/8/8 |
| 0.001 | 8/8/8 |
| 0.005 | 7/7/7 |
| 0.01 | 7/7/7 |
| 0.02 | 6/6/6 |
| 0.05 | 4/3/4 |
| 0.1 | 3/3/3 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.3147 | `area_1_net_export_mw` |
| 2 | 0.4159 | `area_1_net_export_mw|area_3_net_export_mw` |
| 3 | 0.9801 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw` |
| 5 | 0.9904 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw` |
| 7 | 0.9938 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw` |
| 8 | 0.9939 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|reserve_spin_r1_mw` |
| 12 | 0.9927 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|reserve_spin_r1_mw|gen_fuel_hydro_mw|gen_fuel_nuclear_mw|area_3_load_day_ahead_mw|system_losses_mw` |
| 24 | 0.9919 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_coal_mw|gen_fuel_natural_gas_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|reserve_spin_r1_mw|gen_fuel_hydro_mw|gen_fuel_nuclear_mw|area_3_load_day_ahead_mw|system_losses_mw|area_1_load_day_ahead_mw|reserve_spin_r3_mw|area_1_load_actual_mw|area_2_load_actual_mw|area_3_load_actual_mw|system_load_actual_mw|area_2_load_day_ahead_mw|system_load_day_ahead_mw|system_generation_mw|reserve_reg_up_mw|reserve_reg_down_mw|reserve_spin_r2_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
