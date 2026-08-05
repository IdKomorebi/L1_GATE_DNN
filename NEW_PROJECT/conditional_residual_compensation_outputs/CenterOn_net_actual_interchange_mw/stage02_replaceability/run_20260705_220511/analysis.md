# 阶段 02：主路径字段可替代性预检

## 目标

对阶段 01 主路径中的每个字段 `x_i`，验证路径外字段 `R` 是否能在固定上下文 `C_i = P \ {x_i}` 下补偿它。

## 判定规则

- 阈值 tau: `0.95`
- 如果 `C_i` 单独达标，则该字段在当前主路径里是冗余字段。
- 如果 `C_i ∪ R` 不达标，则该字段在当前字段宇宙下不可替代。
- 如果 `R` 单独达标，则存在独立替代路径。
- 如果 `C_i` 不达标、`C_i ∪ R` 达标、`R` 不达标，则进入后续残差补偿阶段。

## 全局基准

- full all features R2: `0.968456`
- R only R2: `0.850134`

## 分类结果

- conditionally_replaceable: 7
- not_replaceable_under_current_universe: 0
- main_path_redundant: 4
- independent_alternative_path: 0

## 03 累计剪枝

对 `main_path_redundant` 字段按 `C_i_r2` 从高到低累计尝试删除；每一步重新训练当前主路径子集，只有 R2 仍达到 tau 才真正删除。

- 原始主路径字段数: `11`
- 累计删除字段数: `2`
- 删除字段: `da_as_nsr_mw_primary_reserve; gen_fuel_multiple_fuels_mw`
- 剪枝后主路径字段数: `9`
- 最后一步 R2: `0.944595`
- 最后一步决策: `stop_first_below_tau`

## 04 剪枝后重新预检

累计剪枝后，主路径 `P`、上下文 `C_i` 和路径外字段 `R` 都发生变化，因此重新执行阶段 02 的三项预检。

- conditionally_replaceable: 9
- not_replaceable_under_current_universe: 0
- main_path_redundant: 0
- independent_alternative_path: 0

## 产物

- `01_precheck/replaceability_precheck_cases.csv`: 所有子集模型训练结果。
- `02_summary/replaceability_summary.csv`: 每个主路径字段的三项预检和分类。
- `02_summary/replaceability_precheck.png`: `C_i` 与 `C_i ∪ R` 的 R2 对比图。
- `03_reprune/reprune_steps.csv`: 冗余候选的累计剪枝过程。
- `03_reprune/repruned_main_path.json`: 剪枝后的主路径接口片段。
- `03_reprune/reprune_steps.png`: 累计剪枝 R2 曲线。
- `04_recheck_after_reprune/replaceability_summary.csv`: 剪枝后主路径的重新可替代性预检。
- `04_recheck_after_reprune/replaceability_precheck.png`: 剪枝后 `C_i` 与 `C_i ∪ R` 的 R2 对比图。
- `stage02_replaceability_interface.json`: 后续阶段标准接口。
