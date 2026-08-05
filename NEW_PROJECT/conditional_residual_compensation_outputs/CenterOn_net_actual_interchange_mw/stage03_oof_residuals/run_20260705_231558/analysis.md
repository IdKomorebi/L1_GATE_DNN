# 阶段 03：OOF 残差接口生成

## 目标

本阶段为每个可替代主路径字段 `x_i` 生成删除该字段后的 out-of-fold 残差目标。

固定主路径为剪枝后的 9 字段集合 `P`。对每个 `x_i`，上下文为 `C_i = P \ {x_i}`，先训练 `f_C(C_i) -> y`，再用未见过该样本的折外预测构造残差 `e_i = y - f_C(C_i)`。

## 本次执行范围

- 可替代字段总数: `9`
- 本次实际执行字段数: `1`
- 本次执行字段: `prelim_load_avg_hourly`

第一次只跑一个字段，用来确认目录结构、残差文件、标准化参数和后续接口是否完整。

## 核心结果

- OOF R2: `0.751795`
- OOF MSE: `1631867.000000`
- residual_mean: `-13.997056`
- residual_std_value: `1277.368774`
- gap_energy = Var(residual) / Var(y): `0.248175`
- f_C_full test R2: `0.765244`

## 产物

- `stage03_oof_residual_interface.json`: 给下一阶段使用的标准接口。
- `processed_xi_summary.csv`: 本 run 已执行字段的摘要。
- `xi=prelim_load_avg_hourly/oof_residuals.csv`: 训练集 OOF 残差，后续残差补偿网络应使用 `residual_std` 作为目标。
- `xi=prelim_load_avg_hourly/residual_stats.json`: 残差标准化参数与 gap energy。
- `xi=prelim_load_avg_hourly/f_C_full_model.pt`: 使用完整训练集训练的最终 `f_C_full`。
- `xi=prelim_load_avg_hourly/xi_interface.json`: 单字段接口。
