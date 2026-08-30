# RTS-GMLC 2020 实验输出

本目录对应 `01_data/raw/rts_gmlc_2020`，沿用 `04_outputs/pjm_2025` 的“阶段目录 + 可复现配置 + CSV 明细 + 人读总结”组织方式。

已完成阶段：

- `00_feasibility_check/`：普通神经网络前置可推断性检测、目标分级和 12 个核心目标选择。
- `01_preprocess/`：27 个候选发布字段质量检查、3 个恒定字段移除、时间切分记录和 D-Gating 验证集预校准。
- `02_class1_physical/`：两层确定性风险处理。
  - `layer1_documented_formula/`：官方资料与数据生成规则的公式闭合性审计；包含固定版本官方核电机组清单证据。
  - `layer2_empirical_identity/`：训练段线性/对数空间恒等与近恒等发现，并作五段时间复验。
  - `candidate_pools/`：每个目标剥离后的候选池。
- `03_screening/`：六类依赖指标和周块置换零分布阈值的宽松初筛。
- `04b_source_location_excl_targets/`：所有敏感目标均不进入 X 的正式推断源定位。
  - `stripped_no_screening/`：剥离后直接运行 D-Gating。
  - `stripped_with_screening/`：剥离后先初筛再运行 D-Gating。

根目录主要入口：

- `pipeline_summary.md`：12 个目标 × 两条分支的完整人读总结。
- `source_location_summary.csv`：逐目标汇总指标与共识推断源。
- `screening_impact.csv`：初筛相对未初筛的逐目标精度变化。
- `source_frequency.csv`：各字段被多少个目标反复选中。
- `pipeline_output_verification.csv`：24 个目标-分支目录的完整性检查。
- `fig_source_location_overview.png`：两条分支的选中字段重训 R² 总览。

正式流程只使用 27 个候选发布字段；`00` 中为了诊断而临时移入 X 的节点级字段没有进入后续候选池。第一层公式闭包只对核电目标发生：依次剥离直接燃料汇总、系统总发电和网损三个入口；第二层严格恒等发现没有继续剥离字段。

复现命令：

```bash
python3 \
  NEW_Supplement/03_scripts/08_rts_core_pipeline.py \
  --stage all --screening-draws 80 --epochs 220 --lambda-dgate 0.005
```

本轮按要求没有执行基线对比实验，因此本目录不包含 `05_baseline_compare/`。
