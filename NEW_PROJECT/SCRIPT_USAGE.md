# 脚本快速使用手册

这个文件只记录 `NEW_PROJECT/scripts` 下脚本的常用执行命令。以后如果修改脚本参数、输出目录，或者新增 baseline 脚本，要同步修改这个文件。

建议在仓库根目录 `L1_GATE_DNN` 执行命令：

```bash
conda activate Pytorch310


## 参数来源

YAML 控制：数据路径、输出路径、关系指标和阈值、特征筛选规则、默认训练参数、模型超参数、批量训练中心变量和模型列表。

命令行控制：本次运行临时指定的中心变量、模型、训练轮数、学习率、正则强度、run 名称等。


## 01_analyze_relations.py

用途：生成关系分析表和知识图谱。

分析全部两两关系：

```bash
python NEW_PROJECT/scripts/01_analyze_relations.py
```

只分析一个中心变量：

```bash
python NEW_PROJECT/scripts/01_analyze_relations.py --center total_lmp_rt
```


输出：`RelationshipAnalysis/{threshold_tag}/relationships.csv` 或 `CenterOn_{center}/RelationshipAnalysis/{threshold_tag}/center_relationships.csv`。

## 02_train_center.py

用途：训练一个中心变量，可以一次训练一个或多个模型。

训练一个 模型比如L1GateDNN：

```bash
python NEW_PROJECT/scripts/02_train_center.py --center total_lmp_rt --model L1GateDNN
```

一次训练多个模型：

```bash
python NEW_PROJECT/scripts/02_train_center.py --center total_lmp_rt --model DNN L1GateDNN ImprovedL1GateDNN ImprovedL2GateDNN
```


输出：`CenterOn_{center}/DNN/`、`CenterOn_{center}/L1GateDNN/`、`CenterOn_{center}/ImprovedGateDNN/ImprovedL1GateDNN/` 或 `ImprovedL2GateDNN/`。

## 03_run_all.py

用途：批量训练多个中心变量和多个模型。

按 YAML 的 `experiment.centers` 和 `experiment.models` 批量训练：

```bash
python NEW_PROJECT/scripts/03_run_all.py
```




## 04_summarize_results.py

用途：汇总所有训练结果里的 `metrics.json`。

执行汇总：

```bash
python NEW_PROJECT/scripts/04_summarize_results.py
```

输出：

```text
NEW_PROJECT/outputs/data2025_Processed_V1/summary_all_centers.csv
NEW_PROJECT/outputs/data2025_Processed_V1/CenterOn_{center}/compare_models.csv
```

## 05_validate_l1_redundancy.py

用途：验证 L1GateDNN 选出来的特征是否冗余。

使用 YAML 里的 `validation.l1_redundancy.run_dir`：

```bash
python NEW_PROJECT/scripts/05_validate_l1_redundancy.py
```

直接指定一个 L1GateDNN run：

```bash
python NEW_PROJECT/scripts/05_validate_l1_redundancy.py NEW_PROJECT/outputs/data2025_Processed_V1/CenterOn_total_lmp_rt/L1GateDNN/run_20260515_160904
```

使用 gate 绝对值最高的前 N 个特征：

```bash
python NEW_PROJECT/scripts/05_validate_l1_redundancy.py NEW_PROJECT/outputs/data2025_Processed_V1/CenterOn_total_lmp_rt/L1GateDNN/run_20260515_160904 --top-n 5
```

不做逐个删除特征实验：

```bash
python NEW_PROJECT/scripts/05_validate_l1_redundancy.py NEW_PROJECT/outputs/data2025_Processed_V1/CenterOn_total_lmp_rt/L1GateDNN/run_20260515_160904 --no-drop-one
```

输出：默认写到 `{run_dir}/redundancy_validation/` 下。
