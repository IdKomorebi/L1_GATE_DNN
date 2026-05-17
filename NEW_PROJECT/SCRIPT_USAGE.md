# 脚本快速使用手册

在仓库根目录 `L1_GATE_DNN` 执行：

```bash
conda activate Pytorch310
```

默认配置由 `NEW_PROJECT/configs/active_config.yaml` 指定：

```yaml
active_config: data2025_v2.yaml
```

任意脚本都可临时加 `--config NEW_PROJECT/configs/data2025.yaml`。优先级：`--config` > 环境变量 `NEW_PROJECT_CONFIG` > `active_config.yaml`。

YAML 主要管：数据路径、输出路径、关系阈值、默认训练参数、模型参数、批量中心变量、全局字段组合 `column_combinations`、baseline 对比参数。命令行参数主要用于本次临时覆盖。

## 01_analyze_relations.py

```bash
python NEW_PROJECT/scripts/01_analyze_relations.py
```

对全部字段做两两关系分析，输出 `relationships.csv` 和知识图谱。

```bash
python NEW_PROJECT/scripts/01_analyze_relations.py --center total_lmp_rt
```

只分析一个中心变量与其他字段的关系。

```bash
python NEW_PROJECT/scripts/01_analyze_relations.py --center total_lmp_rt --no-graph --progress-every 200
```

不画图谱，并每 200 对字段输出一次进度。

## 02_train_center.py

```bash
python NEW_PROJECT/scripts/02_train_center.py --center total_lmp_rt --model L1GateDNN
```

训练一个中心变量的一个模型。

```bash
python NEW_PROJECT/scripts/02_train_center.py --center total_lmp_rt --model DNN L1GateDNN ImprovedL1GateDNN ImprovedL2GateDNN
```

同一个中心变量连续训练多个模型。

```bash
python NEW_PROJECT/scripts/02_train_center.py --combo 1 --model DNN L1GateDNN
```

使用 YAML 的 `column_combinations`，自动读取 `center` 和 `exclude_columns`；`--combo` 可传数字 ID 或组合名，输出 run 名会自动带 `combo...` 标志。

```bash
python NEW_PROJECT/scripts/02_train_center.py --combo total_lmp_rt_no_formula --model L1GateDNN --epochs 300 --lambda-l1 0.0001
```

临时覆盖训练轮数和 L1 正则强度。

```bash
python NEW_PROJECT/scripts/02_train_center.py --center total_lmp_rt --model L1GateDNN --top-n 10
```

先按 YAML 阈值筛选候选 X，再按 6 个关系指标绝对值求和取 Top-10 进入模型；不传 `--top-n` 则使用全部达阈值候选。

## 03_run_all.py

```bash
python NEW_PROJECT/scripts/03_run_all.py
```

按 YAML 的 `experiment.centers` 和 `experiment.models` 批量训练。

```bash
python NEW_PROJECT/scripts/03_run_all.py --centers total_lmp_rt total_lmp_da --models DNN L1GateDNN --epochs 100 --force-relations
```

临时指定中心变量、模型、轮数，并强制重建中心关系分析。

## 04_summarize_results.py

```bash
python NEW_PROJECT/scripts/04_summarize_results.py
```

汇总当前配置输出目录下所有 `metrics.json`。

```bash
python NEW_PROJECT/scripts/04_summarize_results.py --config NEW_PROJECT/configs/data2025.yaml
```

临时汇总 V1 配置对应的输出目录。

## 05_validate_l1_redundancy.py

```bash
python NEW_PROJECT/scripts/05_validate_l1_redundancy.py
```

使用 YAML 的 `validation.l1_redundancy.run_dir` 做 L1GateDNN 特征冗余验证。

```bash
python NEW_PROJECT/scripts/05_validate_l1_redundancy.py NEW_PROJECT/outputs/data2025_Processed_V2/CenterOn_total_lmp_rt/L1GateDNN/run_YYYYMMDD_HHMMSS
```

直接指定一个 L1GateDNN run 目录。

```bash
python NEW_PROJECT/scripts/05_validate_l1_redundancy.py NEW_PROJECT/outputs/data2025_Processed_V2/CenterOn_total_lmp_rt/L1GateDNN/run_YYYYMMDD_HHMMSS --top-n 5 --no-drop-one
```

只取最终 gate 绝对值最高的 5 个特征，并跳过逐个删除实验。脚本 5 会同步被验证 run 里的 `preprocessing.exclude_columns`。

## 06_run_baselines.py

```bash
python NEW_PROJECT/scripts/06_run_baselines.py --check-only
```

只检查 YAML、中心目标和数据列，不启动训练。

```bash
python NEW_PROJECT/scripts/06_run_baselines.py
```

按 YAML 的 `baseline_comparison` 跑完整 baseline 对比：先取每个中心目标的 L1GateDNN 有效变量数 `n_i`，再让 8 种方法各取 Top-`n_i` 特征，最后统一用普通 DNN 评估。

`baseline_comparison.centers` 里既可以写普通中心字段名，也可以写 `column_combinations` 的数字 ID 或组合名。

YAML 里 `min_features`/`max_features` 控制 Top-X 下限和上限；`fixed_feature_count: 5` 表示所有 Y 固定取 5 个 X，留空则按 L1GateDNN 的有效变量数并受 min/max 约束。

YAML 里 `reuse_l1_runs: false` 表示 baseline 每次都在 BaselineComparison 目录下重新训练 L1 来源模型，避免排除字段或参数不一致时误用旧 run。

```bash
python NEW_PROJECT/scripts/06_run_baselines.py --baselines L1GateDNN NMI Pearson --epochs 100
```

临时只跑部分 baseline，并覆盖 DNN 训练轮数。

```bash
python NEW_PROJECT/scripts/06_run_baselines.py --fixed-feature-count 5 --max-features 10
```

临时让所有中心目标固定取 5 个 X，并设置动态模式下的最大 X 数为 10。

输出位置：`NEW_PROJECT/outputs/{数据集名}/BaselineComparison/{run_name}/`，包含 `baseline_summary_long.csv`、`baseline_summary_wide.csv`、`baseline_test_r2.png`、`baseline_report.html` 和每个中心/方法的特征与训练结果。脚本会额外训练 `DNN_AllFeatures` 作为全量 X 的普通 DNN 对照；图中横坐标显示 `n=Top-X数量/all=候选X总数`；HTML 可切换单个 Y 的 loss/R² 曲线，也可切换多个 Y 之间的 loss/R² 对比。

以后新增或修改 baseline 脚本时，也同步修改本文件。
