# 第一类关系剥离过程：总发电量

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

### 第 1 层　[高精度近似]　相对残差 3.14e-03（相当于 R² = 1.0000）

```
total_gen = +0.2785*prelim_load_avg_hourly +0.7218*metered_load_mw -113.7*da_as_as_mw_primary_reserve +113.7*da_as_as_mw_synchronized_reserve +113.7*da_as_nsr_mw_primary_reserve -1.001*net_actual_interchange_mw -171.6
```

- 这条关系用到 6 个字段
- **删掉贡献最大的 `da_as_nsr_mw_primary_reserve`（非同步备用）**
- 删完之后候选池剩 57 个字段，目标仍能被线性回归推到 R² = **1.0000**

### 第 2 层　[高精度近似]　相对残差 3.14e-03（相当于 R² = 1.0000）

```
total_gen = +0.2783*prelim_load_avg_hourly +0.722*metered_load_mw -1.001*net_actual_interchange_mw -176.7
```

- 这条关系用到 3 个字段
- **删掉贡献最大的 `metered_load_mw`（计量负荷）**
- 删完之后候选池剩 56 个字段，目标仍能被线性回归推到 R² = **1.0000**

### 第 3 层　[高精度近似]　相对残差 5.26e-03（相当于 R² = 1.0000）

```
total_gen = +0.8146*prelim_load_avg_hourly +0.191*total_pjm_rt_load_mwh +210.6*da_as_as_req_mw_primary_reserve -315.9*da_as_as_req_mw_synchronized_reserve -60.22*system_energy_price_rt +59.87*total_lmp_rt -0.9968*net_actual_interchange_mw +1.949e+04
```

- 这条关系用到 7 个字段
- **删掉贡献最大的 `da_as_as_req_mw_synchronized_reserve`（日前同步备用需求）**
- 删完之后候选池剩 55 个字段，目标仍能被线性回归推到 R² = **1.0000**

### 第 4 层　[高精度近似]　相对残差 5.28e-03（相当于 R² = 1.0000）

```
total_gen = +0.8178*prelim_load_avg_hourly +0.1878*total_pjm_rt_load_mwh -61.64*system_energy_price_rt +61.28*total_lmp_rt -0.9958*net_actual_interchange_mw -524.2
```

- 这条关系用到 5 个字段
- **删掉贡献最大的 `prelim_load_avg_hourly`（预估小时负荷）**
- 删完之后候选池剩 54 个字段，目标仍能被线性回归推到 R² = **1.0000**

### 第 5 层　[高精度近似]　相对残差 1.25e-02（相当于 R² = 0.9998）

```
total_gen = +1.026*total_pjm_rt_load_mwh -161.7*system_energy_price_rt +161.5*total_lmp_rt -0.9774*net_actual_interchange_mw -405
```

- 这条关系用到 4 个字段
- **删掉贡献最大的 `total_pjm_rt_load_mwh`（实时负荷电量）**
- 删完之后候选池剩 53 个字段，目标仍能被线性回归推到 R² = **0.9984**

### 第 6 层　[近似关系]　相对残差 8.35e-02（相当于 R² = 0.9930）

```
total_gen = +1.464*gen_fuel_gas_mw -1.342e+05*gen_fuel_gas_pct +0.3981*forecast_load_mw_latest_available +3.714e+07*system_energy_price_da -3.714e+07*total_lmp_da +3.714e+07*congestion_price_da +3.714e+07*marginal_loss_price_da +262*system_energy_price_rt -262.1*total_lmp_rt +5.554e+04
```

- 这条关系用到 9 个字段
- **删掉贡献最大的 `total_lmp_da`（日前总电价）**
- 删完之后候选池剩 52 个字段，目标仍能被线性回归推到 R² = **0.9984**

### 第 7 层　[近似关系]　相对残差 8.47e-02（相当于 R² = 0.9928）

```
total_gen = +1.474*gen_fuel_gas_mw -1.358e+05*gen_fuel_gas_pct +0.3884*forecast_load_mw_latest_available +411.7*system_energy_price_rt -413.6*total_lmp_rt +5.632e+04
```

- 这条关系用到 5 个字段
- **删掉贡献最大的 `total_lmp_rt`（实时总电价）**
- 删完之后候选池剩 51 个字段，目标仍能被线性回归推到 R² = **0.9984**

### 第 8 层　[近似关系]　相对残差 8.51e-02（相当于 R² = 0.9928）

```
total_gen = +1.479*gen_fuel_gas_mw -1.365e+05*gen_fuel_gas_pct +0.3824*forecast_load_mw_latest_available +5.679e+04
```

- 这条关系用到 3 个字段
- **删掉贡献最大的 `gen_fuel_gas_mw`（燃气出力）**
- 删完之后候选池剩 50 个字段，目标仍能被线性回归推到 R² = **0.9977**

### 第 9 层　[近似关系]　相对残差 7.68e-02（相当于 R² = 0.9941）

```
total_gen = +1.123*gen_fuel_coal_mw -1.141e+05*gen_fuel_coal_pct +0.8051*forecast_load_mw_latest_available -0.7138*net_actual_interchange_mw +1.881e+04
```

- 这条关系用到 4 个字段
- **删掉贡献最大的 `forecast_load_mw_latest_available`（最新负荷预测）**
- 删完之后候选池剩 49 个字段，目标仍能被线性回归推到 R² = **0.9967**

### 第 10 层　[近似关系]　相对残差 9.32e-02（相当于 R² = 0.9913）

```
total_gen = +2.757*gen_fuel_coal_mw +1.673*gen_fuel_nuclear_mw +1.377*gen_fuel_solar_mw -2.984e+05*gen_fuel_coal_pct -1.874e+05*gen_fuel_nuclear_pct -1.348e+05*gen_fuel_solar_pct -2.048*net_actual_interchange_mw +2.28*net_sched_interchange_mw +1.092e+05
```

- 这条关系用到 8 个字段
- **删掉贡献最大的 `gen_fuel_coal_mw`（燃煤出力）**
- 删完之后候选池剩 48 个字段，目标仍能被线性回归推到 R² = **0.9933**

### 第 11 层　[近似关系]　相对残差 9.83e-02（相当于 R² = 0.9903）

```
total_gen = +0.9343*wind_generation_mw +3.162*gen_fuel_hydro_mw +2.744*gen_fuel_multiple_fuels_mw +2.621*gen_fuel_nuclear_mw +6.937*gen_fuel_oil_mw +1.605*gen_fuel_solar_mw -0.7982*gen_fuel_wind_mw +2.693e+05*gen_fuel_coal_pct +2.572e+05*gen_fuel_gas_pct -3.785e+05*gen_fuel_oil_pct +1.158e+05*gen_fuel_solar_pct +2.478e+05*gen_fuel_wind_pct -1.5*net_actual_interchange_mw +1.528*net_sched_interchange_mw -1.615e+05
```

- 这条关系用到 14 个字段
- **删掉贡献最大的 `gen_fuel_gas_pct`（燃气占比）**
- 删完之后候选池剩 47 个字段，目标仍能被线性回归推到 R² = **0.9932**

### 第 12 层　[近似关系]　相对残差 9.84e-02（相当于 R² = 0.9903）

```
total_gen = +0.9989*wind_generation_mw +3.235*gen_fuel_hydro_mw +4.019*gen_fuel_multiple_fuels_mw +2.659*gen_fuel_nuclear_mw +6.393*gen_fuel_oil_mw +1.494*gen_fuel_solar_mw -0.9722*gen_fuel_wind_mw -2.81e+05*gen_fuel_hydro_pct -4.054e+05*gen_fuel_multiple_fuels_pct -2.691e+05*gen_fuel_nuclear_pct -5.776e+05*gen_fuel_oil_pct -1.37e+05*gen_fuel_solar_pct -1.329*net_actual_interchange_mw +1.403*net_sched_interchange_mw +9.865e+04
```

- 这条关系用到 14 个字段
- **删掉贡献最大的 `gen_fuel_nuclear_pct`（核电占比）**
- 删完之后候选池剩 46 个字段，目标仍能被线性回归推到 R² = **0.9748**

## 三、最终从候选池里剥掉了这些字段

| 层 | 字段 | 业务含义 | 剥离后线性 R² |
|---|---|---|---|
| 1 | `da_as_nsr_mw_primary_reserve` | 非同步备用 | 1.0000 |
| 2 | `metered_load_mw` | 计量负荷 | 1.0000 |
| 3 | `da_as_as_req_mw_synchronized_reserve` | 日前同步备用需求 | 1.0000 |
| 4 | `prelim_load_avg_hourly` | 预估小时负荷 | 1.0000 |
| 5 | `total_pjm_rt_load_mwh` | 实时负荷电量 | 0.9984 |
| 6 | `total_lmp_da` | 日前总电价 | 0.9984 |
| 7 | `total_lmp_rt` | 实时总电价 | 0.9984 |
| 8 | `gen_fuel_gas_mw` | 燃气出力 | 0.9977 |
| 9 | `forecast_load_mw_latest_available` | 最新负荷预测 | 0.9967 |
| 10 | `gen_fuel_coal_mw` | 燃煤出力 | 0.9933 |
| 11 | `gen_fuel_gas_pct` | 燃气占比 | 0.9932 |
| 12 | `gen_fuel_nuclear_pct` | 核电占比 | 0.9748 |

候选池：**58 → 46 个字段**（实际剥掉 12 个）。

## 四、为什么必须先做这一步

这些关系有公式可循，属于已知的、可以靠规则枚举禁止的风险，不是本文要找的隐性推断源。
如果不剥离就直接跑门控模型，模型一定优先选中这些字段——因为它们推断得最准——
真正没有公式可循的推断路径就被完全盖住了。

剥离之后目标仍能被推到 R² = 0.9748，**这部分才是隐性推断的部分**。
