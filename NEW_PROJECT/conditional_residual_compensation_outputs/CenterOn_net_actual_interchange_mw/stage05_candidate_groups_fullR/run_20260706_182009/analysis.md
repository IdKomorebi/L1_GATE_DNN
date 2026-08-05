# 阶段 05：锚定多头残差补偿门控

## 目标

本阶段以阶段 04 的活跃候选集 `A` 作为锚字段来源，但每个锚头的 D-gating 候选池扩大到完整路径外字段 `R`。

候选组随后用普通 DNN 在 `C_i + Q` 上重新训练验证，避免只依赖残差模型内部指标。

## 本次字段

- xi: `prelim_load_avg_hourly`
- anchor/head count: `20`
- full-R candidate count: `46`
- C_i 字段数: `8`

## 选中的压缩 trial

- trial: `trial_04_lam0p001`
- lambda_dgate: `0.001`
- threshold_ratio: `0.8`
- unique candidate groups: `64`

## 最小达标候选组

- group_id: `group_021`
- |Q|: `4`
- C_i + Q feature count: `12`
- best test R2: `0.963438`
- source anchors: `metered_load_mw`

## 产物

- `stage05_candidate_group_interface.json`: 给下一阶段使用的候选替代组接口。
- `xi=<field>/02_dgating_sweep/sweep_summary.csv`: D-gating 压缩强度 sweep。
- `xi=<field>/03_group_validation/group_validation.csv`: 候选组真实 `C_i + Q` 验证。
- `xi=<field>/04_candidate_groups/candidate_groups.csv`: 去重后的候选替代组。
