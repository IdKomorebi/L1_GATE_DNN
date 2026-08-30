# 推断源定位：跨区支路 CA-1 负载率

- 分支：`with_screening`
- 剥离后/初筛后候选数：10
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9528 ± 0.0014**。
- 共识推断源：**8 个**（`gen_fuel_coal_mw`, `gen_fuel_natural_gas_mw`, `system_losses_mw`, `gen_fuel_wind_mw`, `gen_fuel_solar_mw`, `area_2_net_export_mw`, `area_1_net_export_mw`, `gen_fuel_nuclear_mw`）。
- 只用共识字段重训普通 DNN：**0.9490 ± 0.0020**。
- 相对全候选精度变化：**-0.0038**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `gen_fuel_coal_mw` | 0.172883 | 0.00508421 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.144201 | 0.00501035 | 3/3 |
| `system_losses_mw` | 0.112328 | 0.00449271 | 3/3 |
| `gen_fuel_wind_mw` | 0.096953 | 0.00946203 | 3/3 |
| `gen_fuel_solar_mw` | 0.0945037 | 0.00474217 | 3/3 |
| `area_2_net_export_mw` | 0.0923287 | 0.00499948 | 3/3 |
| `area_1_net_export_mw` | 0.0742393 | 0.00214866 | 3/3 |
| `gen_fuel_nuclear_mw` | 0.0291218 | 0.000895574 | 3/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 8/8/8 |
| 0.0001 | 8/8/8 |
| 0.001 | 8/8/8 |
| 0.005 | 8/8/8 |
| 0.01 | 8/8/8 |
| 0.02 | 8/8/8 |
| 0.05 | 7/7/7 |
| 0.1 | 5/3/3 |

0.001–0.02 范围内活跃字段数不变，门控分布存在稳定空档。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.1170 | `gen_fuel_coal_mw` |
| 2 | 0.2836 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw` |
| 3 | 0.8021 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|system_losses_mw` |
| 5 | 0.9036 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw` |
| 8 | 0.9499 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|area_2_net_export_mw|area_1_net_export_mw|gen_fuel_nuclear_mw` |
| 10 | 0.9457 | `gen_fuel_coal_mw|gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_solar_mw|area_2_net_export_mw|area_1_net_export_mw|gen_fuel_nuclear_mw|reserve_reg_down_mw|reserve_reg_up_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
