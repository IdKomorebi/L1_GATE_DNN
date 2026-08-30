# 第一层公式来源

- RTS-GMLC 官方仓库：<https://github.com/GridMod/RTS-GMLC>
- 固定提交：`3ece0d3725c844056132393ee252b3083dd4eab4`
- SourceData 字段说明：<https://github.com/GridMod/RTS-GMLC/blob/3ece0d3725c844056132393ee252b3083dd4eab4/RTS_Data/SourceData/README.md>
- 时序数据说明：<https://github.com/GridMod/RTS-GMLC/blob/3ece0d3725c844056132393ee252b3083dd4eab4/RTS_Data/timeseries_data_files/README.md>
- 本数据构造与字段计算：`01_data/raw/rts_gmlc_2020/Generation/generate_rts_gmlc_dataset.py`
- 数据集物理关系说明：`01_data/raw/rts_gmlc_2020/Explanation.md`
- 固定提交官方机组表中 Nuclear 行的审计副本：`official_nuclear_generator_inventory.csv`

官方资料提供网络、机组字段和时序构造定义；本项目生成脚本提供 27 个候选聚合字段的精确计算口径。
只有当公式所需信息全部位于当前 27 字段候选池内时，才允许据此剥离。
