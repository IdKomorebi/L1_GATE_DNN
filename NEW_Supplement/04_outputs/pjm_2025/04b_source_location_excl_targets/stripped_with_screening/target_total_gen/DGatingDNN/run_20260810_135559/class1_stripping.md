# 第一类关系剥离过程：总发电量

候选池设定：排除其余关注字段

## 一、这个目标一共有 10 层关系

按误差量级归档：

- 近似关系：10 层

档次的划分标准：精确公式（相对残差 < 1e-6）、含舍入的公式（1e-6 ~ 1e-3）、
高精度近似（1e-3 ~ 3e-2）、近似关系（3e-2 ~ 1e-1）。
相对残差 = 拟合剩下的误差 ÷ 该目标字段自身的波动幅度。

## 二、一层一层是怎么挖出来的

每一层的做法：用当时还剩下的全部字段拟合目标，再从全字段出发逐个删掉贡献最小的，缩到最小的一组；
然后只删掉这组里贡献最大的那**一个**字段，进入下一层看还能不能再找出关系。
只删一个而不是整组删掉，是为了让剩下的字段有机会自己凑出新路径，这样才挖得深。

### 第 1 层　[近似关系]　相对残差 8.44e-02（相当于 R² = 0.9929）

```
total_gen = +2.18*gen_fuel_coal_mw -2.245e+05*gen_fuel_coal_pct -1.899*prelim_load_avg_hourly +2.611*total_pjm_rt_load_mwh +358.6*da_as_as_req_mw_primary_reserve -537.4*da_as_as_req_mw_synchronized_reserve -1700*da_as_as_mw_primary_reserve +1701*da_as_as_mw_synchronized_reserve +1701*da_as_nsr_mw_primary_reserve +471.5*system_energy_price_rt -474.6*total_lmp_rt +6.817e+04
```

- 这条关系用到 11 个字段
- **删掉贡献最大的 `da_as_nsr_mw_primary_reserve`（非同步备用）**
- 删完之后候选池剩 48 个字段，目标仍能被线性回归推到 R² = **0.9986**

### 第 2 层　[近似关系]　相对残差 8.60e-02（相当于 R² = 0.9926）

```
total_gen = +2.168*gen_fuel_coal_mw -2.234e+05*gen_fuel_coal_pct -1.768*prelim_load_avg_hourly +2.482*total_pjm_rt_load_mwh +454.2*da_as_as_req_mw_primary_reserve -680.1*da_as_as_req_mw_synchronized_reserve +622.8*system_energy_price_rt -626.3*total_lmp_rt +7.719e+04
```

- 这条关系用到 8 个字段
- **删掉贡献最大的 `da_as_as_req_mw_primary_reserve`（日前主用备用需求）**
- 删完之后候选池剩 47 个字段，目标仍能被线性回归推到 R² = **0.9986**

### 第 3 层　[近似关系]　相对残差 8.79e-02（相当于 R² = 0.9923）

```
total_gen = +2.329*gen_fuel_coal_mw -2.415e+05*gen_fuel_coal_pct -1.654*prelim_load_avg_hourly +2.347*total_pjm_rt_load_mwh +794*system_energy_price_rt -797.9*total_lmp_rt +3.904e+04
```

- 这条关系用到 6 个字段
- **删掉贡献最大的 `total_lmp_rt`（实时总电价）**
- 删完之后候选池剩 46 个字段，目标仍能被线性回归推到 R² = **0.9986**

### 第 4 层　[近似关系]　相对残差 8.92e-02（相当于 R² = 0.9920）

```
total_gen = +2.352*gen_fuel_coal_mw -2.429e+05*gen_fuel_coal_pct -1.86*prelim_load_avg_hourly +2.545*total_pjm_rt_load_mwh +3.983e+04
```

- 这条关系用到 4 个字段
- **删掉贡献最大的 `total_pjm_rt_load_mwh`（实时负荷电量）**
- 删完之后候选池剩 45 个字段，目标仍能被线性回归推到 R² = **0.9985**

### 第 5 层　[近似关系]　相对残差 9.57e-02（相当于 R² = 0.9908）

```
total_gen = +2.385*gen_fuel_coal_mw -2.523e+05*gen_fuel_coal_pct +0.6219*prelim_load_avg_hourly +4.06e+04
```

- 这条关系用到 3 个字段
- **删掉贡献最大的 `gen_fuel_coal_mw`（燃煤出力）**
- 删完之后候选池剩 44 个字段，目标仍能被线性回归推到 R² = **0.9980**

### 第 6 层　[近似关系]　相对残差 8.08e-02（相当于 R² = 0.9935）

```
total_gen = +1.366*gen_fuel_gas_mw -1.269e+05*gen_fuel_gas_pct +0.4348*prelim_load_avg_hourly +5.264e+04
```

- 这条关系用到 3 个字段
- **删掉贡献最大的 `gen_fuel_gas_mw`（燃气出力）**
- 删完之后候选池剩 43 个字段，目标仍能被线性回归推到 R² = **0.9961**

### 第 7 层　[近似关系]　相对残差 8.30e-02（相当于 R² = 0.9931）

```
total_gen = +5.405*gen_fuel_multiple_fuels_mw +1.479*gen_fuel_nuclear_mw +5.559*gen_fuel_oil_mw -5.704e+05*gen_fuel_multiple_fuels_pct -1.466e+05*gen_fuel_nuclear_pct -5.918e+05*gen_fuel_oil_pct +0.496*prelim_load_avg_hourly +5.131e+04
```

- 这条关系用到 7 个字段
- **删掉贡献最大的 `prelim_load_avg_hourly`（预估小时负荷）**
- 删完之后候选池剩 42 个字段，目标仍能被线性回归推到 R² = **0.9949**

### 第 8 层　[近似关系]　相对残差 9.55e-02（相当于 R² = 0.9909）

```
total_gen = +8.036*gen_fuel_multiple_fuels_mw +1.617*gen_fuel_nuclear_mw +1.135*gen_fuel_oil_mw -8.436e+05*gen_fuel_multiple_fuels_pct -1.577e+05*gen_fuel_nuclear_pct +0.4487*forecast_load_mw_latest_available +5.46e+04
```

- 这条关系用到 6 个字段
- **删掉贡献最大的 `forecast_load_mw_latest_available`（最新负荷预测）**
- 删完之后候选池剩 41 个字段，目标仍能被线性回归推到 R² = **0.9929**

### 第 9 层　[近似关系]　相对残差 9.87e-02（相当于 R² = 0.9903）

```
total_gen = +1.158*wind_generation_mw +3.174*gen_fuel_hydro_mw +2.783*gen_fuel_multiple_fuels_mw +2.616*gen_fuel_nuclear_mw +6.838*gen_fuel_oil_mw +1.611*gen_fuel_solar_mw -1.013*gen_fuel_wind_mw +2.693e+05*gen_fuel_coal_pct +2.567e+05*gen_fuel_gas_pct -3.653e+05*gen_fuel_oil_pct +1.142e+05*gen_fuel_solar_pct +2.465e+05*gen_fuel_wind_pct -1.613e+05
```

- 这条关系用到 12 个字段
- **删掉贡献最大的 `gen_fuel_gas_pct`（燃气占比）**
- 删完之后候选池剩 40 个字段，目标仍能被线性回归推到 R² = **0.9928**

### 第 10 层　[近似关系]　相对残差 9.90e-02（相当于 R² = 0.9902）

```
total_gen = +1.206*wind_generation_mw +3.237*gen_fuel_hydro_mw +3.746*gen_fuel_multiple_fuels_mw +2.654*gen_fuel_nuclear_mw +6.28*gen_fuel_oil_mw +1.506*gen_fuel_solar_mw -1.17*gen_fuel_wind_mw -2.787e+05*gen_fuel_hydro_pct -3.674e+05*gen_fuel_multiple_fuels_pct -2.692e+05*gen_fuel_nuclear_pct -5.598e+05*gen_fuel_oil_pct -1.387e+05*gen_fuel_solar_pct +9.838e+04
```

- 这条关系用到 12 个字段
- **删掉贡献最大的 `gen_fuel_nuclear_pct`（核电占比）**
- 删完之后候选池剩 39 个字段，目标仍能被线性回归推到 R² = **0.9723**

## 三、最终从候选池里剥掉了这些字段

| 层 | 字段 | 业务含义 | 剥离后线性 R² |
|---|---|---|---|
| 1 | `da_as_nsr_mw_primary_reserve` | 非同步备用 | 0.9986 |
| 2 | `da_as_as_req_mw_primary_reserve` | 日前主用备用需求 | 0.9986 |
| 3 | `total_lmp_rt` | 实时总电价 | 0.9986 |
| 4 | `total_pjm_rt_load_mwh` | 实时负荷电量 | 0.9985 |
| 5 | `gen_fuel_coal_mw` | 燃煤出力 | 0.9980 |
| 6 | `gen_fuel_gas_mw` | 燃气出力 | 0.9961 |
| 7 | `prelim_load_avg_hourly` | 预估小时负荷 | 0.9949 |
| 8 | `forecast_load_mw_latest_available` | 最新负荷预测 | 0.9929 |
| 9 | `gen_fuel_gas_pct` | 燃气占比 | 0.9928 |
| 10 | `gen_fuel_nuclear_pct` | 核电占比 | 0.9723 |

候选池：**49 → 24 个字段**（实际剥掉 25 个）。

## 四、为什么必须先做这一步

这些关系有公式可循，属于已知的、可以靠规则枚举禁止的风险，不是本文要找的隐性推断源。
如果不剥离就直接跑门控模型，模型一定优先选中这些字段——因为它们推断得最准——
真正没有公式可循的推断路径就被完全盖住了。

剥离之后目标仍能被推到 R² = 0.9723，**这部分才是隐性推断的部分**。
