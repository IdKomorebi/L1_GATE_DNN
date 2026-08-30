# 推断源定位：跨区支路 CB-1 负载率

- 分支：`with_screening`
- 剥离后/初筛后候选数：11
- D-Gating：深度 4，λ=0.005，报告阈值 0.01
- 三个随机种子中至少 2 次活跃才进入共识集合。
- 时间切分为 70% 训练、10% 验证、20% 测试；测试段不参与早停或选字段。

## 结论

- 全候选普通 DNN 测试 R²：**0.9897 ± 0.0004**。
- 共识推断源：**6 个**（`area_1_net_export_mw`, `area_3_net_export_mw`, `area_2_net_export_mw`, `gen_fuel_coal_mw`, `gen_fuel_solar_mw`, `gen_fuel_natural_gas_mw`）。
- 只用共识字段重训普通 DNN：**0.9916 ± 0.0017**。
- 相对全候选精度变化：**+0.0019**。

## 共识字段

| 字段 | 平均门控 | 门控标准差 | 入选种子数 |
|---|---:|---:|---:|
| `area_1_net_export_mw` | 0.467323 | 0.0162454 | 3/3 |
| `area_3_net_export_mw` | 0.402199 | 0.00233476 | 3/3 |
| `area_2_net_export_mw` | 0.355505 | 0.017142 | 3/3 |
| `gen_fuel_coal_mw` | 0.0804704 | 0.00388196 | 3/3 |
| `gen_fuel_solar_mw` | 0.0280973 | 0.00642737 | 3/3 |
| `gen_fuel_natural_gas_mw` | 0.0200153 | 0.0142211 | 2/3 |

## 阈值敏感性

| 阈值 | 三个种子的活跃数 |
|---:|---|
| 1e-06 | 6/7/7 |
| 0.0001 | 6/6/6 |
| 0.001 | 6/6/6 |
| 0.005 | 6/6/6 |
| 0.01 | 6/6/6 |
| 0.02 | 6/6/6 |
| 0.05 | 4/4/4 |
| 0.1 | 3/3/3 |

0.001–0.02 范围内活跃字段数不变，门控分布存在稳定空档。

## Top-n 累积验证

| 字段数 | 测试 R²（seed 42） | 字段 |
|---:|---:|---|
| 1 | 0.3147 | `area_1_net_export_mw` |
| 2 | 0.4159 | `area_1_net_export_mw|area_3_net_export_mw` |
| 3 | 0.9801 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw` |
| 5 | 0.9921 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_coal_mw|gen_fuel_solar_mw` |
| 6 | 0.9901 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw` |
| 8 | 0.9935 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_wind_mw` |
| 11 | 0.9897 | `area_1_net_export_mw|area_3_net_export_mw|area_2_net_export_mw|gen_fuel_coal_mw|gen_fuel_solar_mw|gen_fuel_natural_gas_mw|system_losses_mw|gen_fuel_wind_mw|gen_fuel_nuclear_mw|reserve_reg_down_mw|reserve_reg_up_mw` |

详细逐种子指标见 `metrics_by_seed.csv`，门控值见 `gates_by_seed.csv`。
