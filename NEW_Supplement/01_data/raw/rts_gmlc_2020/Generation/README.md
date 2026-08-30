# Generation

本目录保存 RTS-GMLC 2020 小时级 AC 潮流数据集的可复现生成材料。

## 主要文件

- `generate_rts_gmlc_dataset.py`：聚合官方 5 分钟时序、构造逐时调度、运行 AC 潮流、选择论文用实体并导出数据。
- `config.json`：固定源版本、噪声水平、随机种子和实体选择数量。
- `requirements.txt`：Python 依赖版本。
- `selected_entities.json`：基准场景运行后生成，记录选中的节点、线路、机组及选择规则。
- `source/rts_gmlc_official_inputs_3ece0d3.zip`：从官方仓库固定提交中提取的网络和时序输入。

## 复现

在本目录执行：

```bash
python generate_rts_gmlc_dataset.py
```

快速检查前 24 小时：

```bash
python generate_rts_gmlc_dataset.py --max-hours 24 --noise 0
```

脚本会把正式 CSV、字段字典和质量报告写入上一级 `rts_gmlc_2020` 目录。

## 计算边界

该流程进行的是逐时 AC power flow，而不是 AC OPF、SCUC 或动态暂态仿真。常规机组采用可审计的燃料成本 merit-order 调度，并遵守机组出力上下限；`status` 表示本生成流程中的逐时投入状态，不应描述为真实市场机组组合结果。电压幅值、相角和线路有功/无功潮流均来自 AC 潮流求解，不是添加到结果上的独立随机数。
