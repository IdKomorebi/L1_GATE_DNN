# 第一类关系剥离过程：计量负荷

候选池设定：含其余关注字段

## 一、这个目标一共有 12 层关系

按误差量级归档：

- 近似关系：7 层
- 高精度近似：5 层

档次的划分标准：精确公式（相对残差 < 1e-6）、含舍入的公式（1e-6 ~ 1e-3）、
高精度近似（1e-3 ~ 3e-2）、近似关系（3e-2 ~ 1e-1）。
相对残差 = 拟合剩下的误差 ÷ 该目标字段自身的波动幅度。

## 二、一层一层是怎么挖出来的

每一层的做法：用当时还剩下的全部字段拟合目标，再从全字段出发逐个删掉贡献最小的，缩到最小的一组；
然后只删掉这组里贡献最大的那**一个**字段，进入下一层看还能不能再找出关系。
只删一个而不是整组删掉，是为了让剩下的字段有机会自己凑出新路径，这样才挖得深。

### 第 1 层　[高精度近似]　相对残差 3.78e-03（相当于 R² = 1.0000）

```
metered_load_mw = +0.9998*total_gen +104.1*da_as_as_req_mw_primary_reserve -156.1*da_as_as_req_mw_synchronized_reserve +108.7*da_as_as_mw_primary_reserve -108.7*da_as_as_mw_synchronized_reserve -108.7*da_as_nsr_mw_primary_reserve +1*net_actual_interchange_mw +9927
```

- 这条关系用到 7 个字段
- **删掉贡献最大的 `da_as_nsr_mw_primary_reserve`（非同步备用）**
- 删完之后候选池剩 57 个字段，目标仍能被线性回归推到 R² = **0.9999**

### 第 2 层　[高精度近似]　相对残差 3.78e-03（相当于 R² = 1.0000）

```
metered_load_mw = +0.9999*total_gen +114.4*da_as_as_req_mw_primary_reserve -171.6*da_as_as_req_mw_synchronized_reserve +1*net_actual_interchange_mw +1.091e+04
```

- 这条关系用到 4 个字段
- **删掉贡献最大的 `da_as_as_req_mw_synchronized_reserve`（日前同步备用需求）**
- 删完之后候选池剩 56 个字段，目标仍能被线性回归推到 R² = **0.9999**

### 第 3 层　[高精度近似]　相对残差 3.78e-03（相当于 R² = 1.0000）

```
metered_load_mw = +0.9998*total_gen +1.001*net_actual_interchange_mw +40.99
```

- 这条关系用到 2 个字段
- **删掉贡献最大的 `total_gen`（总发电量）**
- 删完之后候选池剩 55 个字段，目标仍能被线性回归推到 R² = **0.9999**

### 第 4 层　[高精度近似]　相对残差 7.14e-03（相当于 R² = 0.9999）

```
metered_load_mw = +1*prelim_load_avg_hourly -489
```

- 这条关系用到 1 个字段
- **删掉贡献最大的 `prelim_load_avg_hourly`（预估小时负荷）**
- 删完之后候选池剩 54 个字段，目标仍能被线性回归推到 R² = **0.9999**

### 第 5 层　[高精度近似]　相对残差 1.41e-02（相当于 R² = 0.9998）

```
metered_load_mw = +1.026*total_pjm_rt_load_mwh -442.1
```

- 这条关系用到 1 个字段
- **删掉贡献最大的 `total_pjm_rt_load_mwh`（实时负荷电量）**
- 删完之后候选池剩 53 个字段，目标仍能被线性回归推到 R² = **0.9983**

### 第 6 层　[近似关系]　相对残差 8.72e-02（相当于 R² = 0.9924）

```
metered_load_mw = +0.9883*forecast_load_mw_latest_available -1.554e+06*system_energy_price_da +1.554e+06*total_lmp_da -1.554e+06*congestion_price_da -1.555e+06*marginal_loss_price_da -528.9*system_energy_price_rt +534.1*total_lmp_rt +270.1
```

- 这条关系用到 7 个字段
- **删掉贡献最大的 `total_lmp_da`（日前总电价）**
- 删完之后候选池剩 52 个字段，目标仍能被线性回归推到 R² = **0.9983**

### 第 7 层　[近似关系]　相对残差 8.75e-02（相当于 R² = 0.9923）

```
metered_load_mw = +0.9835*forecast_load_mw_latest_available -463.4*system_energy_price_rt +467*total_lmp_rt +650.2
```

- 这条关系用到 3 个字段
- **删掉贡献最大的 `total_lmp_rt`（实时总电价）**
- 删完之后候选池剩 51 个字段，目标仍能被线性回归推到 R² = **0.9983**

### 第 8 层　[近似关系]　相对残差 8.85e-02（相当于 R² = 0.9922）

```
metered_load_mw = +0.9896*forecast_load_mw_latest_available +302.6
```

- 这条关系用到 1 个字段
- **删掉贡献最大的 `forecast_load_mw_latest_available`（最新负荷预测）**
- 删完之后候选池剩 50 个字段，目标仍能被线性回归推到 R² = **0.9979**

### 第 9 层　[近似关系]　相对残差 8.50e-02（相当于 R² = 0.9928）

```
metered_load_mw = +0.8561*gen_fuel_coal_mw +1.79*gen_fuel_gas_mw +1.056*gen_fuel_solar_mw +2.929*gen_fuel_wind_mw -9.128e+04*gen_fuel_coal_pct -1.615e+05*gen_fuel_gas_pct -9.132e+04*gen_fuel_solar_pct -2.616e+05*gen_fuel_wind_pct +1.217*net_sched_interchange_mw +9.085e+04
```

- 这条关系用到 9 个字段
- **删掉贡献最大的 `gen_fuel_gas_mw`（燃气出力）**
- 删完之后候选池剩 49 个字段，目标仍能被线性回归推到 R² = **0.9964**

### 第 10 层　[近似关系]　相对残差 9.61e-02（相当于 R² = 0.9908）

```
metered_load_mw = +2.756*gen_fuel_coal_mw +1.679*gen_fuel_nuclear_mw +1.378*gen_fuel_solar_mw -2.985e+05*gen_fuel_coal_pct -1.877e+05*gen_fuel_nuclear_pct -1.354e+05*gen_fuel_solar_pct +1.235*net_sched_interchange_mw +1.092e+05
```

- 这条关系用到 7 个字段
- **删掉贡献最大的 `gen_fuel_coal_mw`（燃煤出力）**
- 删完之后候选池剩 48 个字段，目标仍能被线性回归推到 R² = **0.9928**

### 第 11 层　[近似关系]　相对残差 9.85e-02（相当于 R² = 0.9903）

```
metered_load_mw = +5.059*total_losses +0.9412*wind_generation_mw +3.081*gen_fuel_hydro_mw +4.725*gen_fuel_multiple_fuels_mw +2.421*gen_fuel_nuclear_mw +5.917*gen_fuel_oil_mw +1.649*gen_fuel_solar_mw -0.9929*gen_fuel_wind_mw +2.562e+05*gen_fuel_coal_pct +2.472e+05*gen_fuel_gas_pct -1.988e+05*gen_fuel_multiple_fuels_pct -2.794e+05*gen_fuel_oil_pct +1.035e+05*gen_fuel_solar_pct +2.48e+05*gen_fuel_wind_pct -0.7181*net_actual_interchange_mw +1.667*net_sched_interchange_mw -1.504e+05
```

- 这条关系用到 16 个字段
- **删掉贡献最大的 `gen_fuel_gas_pct`（燃气占比）**
- 删完之后候选池剩 47 个字段，目标仍能被线性回归推到 R² = **0.9927**

### 第 12 层　[近似关系]　相对残差 9.83e-02（相当于 R² = 0.9903）

```
metered_load_mw = +5.779*total_losses +0.8874*wind_generation_mw +3.422*gen_fuel_hydro_mw +4.402*gen_fuel_multiple_fuels_mw +2.437*gen_fuel_nuclear_mw +5.531*gen_fuel_oil_mw +1.616*gen_fuel_solar_mw -0.9479*gen_fuel_wind_mw -2.913e+05*gen_fuel_hydro_pct -4.26e+05*gen_fuel_multiple_fuels_pct -2.559e+05*gen_fuel_nuclear_pct -4.817e+05*gen_fuel_oil_pct -1.442e+05*gen_fuel_solar_pct -0.5476*net_actual_interchange_mw +1.506*net_sched_interchange_mw +9.837e+04
```

- 这条关系用到 15 个字段
- **删掉贡献最大的 `gen_fuel_nuclear_pct`（核电占比）**
- 删完之后候选池剩 46 个字段，目标仍能被线性回归推到 R² = **0.9733**

## 三、最终从候选池里剥掉了这些字段

| 层 | 字段 | 业务含义 | 剥离后线性 R² |
|---|---|---|---|
| 1 | `da_as_nsr_mw_primary_reserve` | 非同步备用 | 0.9999 |
| 2 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 | 0.9999 |
| 3 | `total_gen` | 总发电量 | 0.9999 |
| 4 | `prelim_load_avg_hourly` | 预估小时负荷 | 0.9999 |
| 5 | `total_pjm_rt_load_mwh` | 实时负荷电量 | 0.9983 |
| 6 | `total_lmp_da` | 日前总电价 | 0.9983 |
| 7 | `total_lmp_rt` | 实时总电价 | 0.9983 |
| 8 | `forecast_load_mw_latest_available` | 最新负荷预测 | 0.9979 |
| 9 | `gen_fuel_gas_mw` | 燃气出力 | 0.9964 |
| 10 | `gen_fuel_coal_mw` | 燃煤出力 | 0.9928 |
| 11 | `gen_fuel_gas_pct` | 燃气占比 | 0.9927 |
| 12 | `gen_fuel_nuclear_pct` | 核电占比 | 0.9733 |

候选池：**58 → 46 个字段**（实际剥掉 12 个）。

## 四、为什么必须先做这一步

这些关系有公式可循，属于已知的、可以靠规则枚举禁止的风险，不是本文要找的隐性推断源。
如果不剥离就直接跑门控模型，模型一定优先选中这些字段——因为它们推断得最准——
真正没有公式可循的推断路径就被完全盖住了。

剥离之后目标仍能被推到 R² = 0.9733，**这部分才是隐性推断的部分**。
