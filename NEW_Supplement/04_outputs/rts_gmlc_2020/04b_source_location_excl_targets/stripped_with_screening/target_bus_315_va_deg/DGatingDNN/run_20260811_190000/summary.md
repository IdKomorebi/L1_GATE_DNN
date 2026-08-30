# 推断源定位：母线 315 电压相角

- 分支：`with_screening`
- 剥离后/初筛后候选数：12
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9571 ± 0.0011**。
- 共识推断源：**9 个**（`gen_fuel_natural_gas_mw`, `system_losses_mw`, `gen_fuel_coal_mw`, `gen_fuel_wind_mw`, `gen_fuel_solar_mw`, `area_3_net_export_mw`, `area_2_net_export_mw`, `gen_fuel_hydro_mw`, `gen_fuel_nuclear_mw`）。
- 只用共识字段重训普通 DNN：**0.9579 ± 0.0012**。
- 相对全候选精度变化：**+0.0009**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_natural_gas_mw` | 0.0979458 | 0.0107492 | 3/3 |
| `system_losses_mw` | 0.084974 | 0.00407613 | 3/3 |
| `gen_fuel_coal_mw` | 0.0788549 | 0.0126563 | 3/3 |
| `gen_fuel_wind_mw` | 0.0713222 | 0.00677519 | 3/3 |
| `gen_fuel_solar_mw` | 0.0669056 | 0.00650523 | 3/3 |
| `area_3_net_export_mw` | 0.0592248 | 0.0182821 | 3/3 |
| `area_2_net_export_mw` | 0.0582151 | 0.0204538 | 3/3 |
| `gen_fuel_hydro_mw` | 0.0375938 | 0.0011093 | 3/3 |
| `gen_fuel_nuclear_mw` | 0.0136933 | 0.00108842 | 3/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 9/11/10 |
| 0.0001 | 9/11/10 |
| 0.001 | 9/11/10 |
| 0.005 | 9/10/9 |
| 0.01 | 9/10/9 |
| 0.02 | 8/9/8 |
| 0.05 | 5/8/5 |
| 0.1 | 1/0/1 |

0.001–0.02 范围内活跃字段数会变化，因此同时报告门控连续值和三种子频率，不把单一阈值结论绝对化。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | -0.0486 | `gen_fuel_natural_gas_mw` |
| 2 | 0.5302 | `gen_fuel_natural_gas_mw|system_losses_mw` |
| 3 | 0.7966 | `gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_coal_mw` |
| 5 | 0.9200 | `gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_coal_mw|gen_fuel_wind_mw|gen_fuel_solar_mw` |
| 8 | 0.9452 | `gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_coal_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_hydro_mw` |
| 9 | 0.9563 | `gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_coal_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_hydro_mw|gen_fuel_nuclear_mw` |
| 12 | 0.9629 | `gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_coal_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_hydro_mw|gen_fuel_nuclear_mw|area_1_net_export_mw|reserve_reg_down_mw|reserve_reg_up_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
