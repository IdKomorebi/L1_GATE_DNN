# Dataset Information Table for Paper

说明：为和论文中的发布前风险流程对应，本表把 27 个区域/系统汇总量编号为 `D`（candidate disclosed variables），把 130 个细粒度目标编号为 `S`（security-sensitive target variables）。时间、场景、噪声和求解质量字段不计入 `D/S` 编号。逐列顺序与定义以 `field_dictionary.csv` 为准。

## Markdown Table

| Data No. | Dataset | Field scope | Data information |
|---|---|---:|---|
| D1-D4 | Real-time load aggregates | 4 | Hourly real-time load for Areas 1–3 and the whole system. |
| D5-D8 | Day-ahead load aggregates | 4 | Hourly day-ahead load for Areas 1–3 and the whole system. |
| D9-D13 | System balance and area interchange | 5 | Total generation, total AC losses, and net exports of Areas 1–3. |
| D14-D18 | Operating reserves | 5 | Regulation-up/down reserves and spinning reserves for regions R1–R3. |
| D19-D27 | Generation mix aggregates | 9 | System-level generation grouped by fuel/resource type, including synchronous condensers. |
| S1-S2 | Slack-bus balancing injections | 2 | Active and reactive balancing injections at named bus 113. |
| S3-S50 | Named-bus operating variables | 48 | Voltage magnitude, voltage angle, active injection, and reactive injection at 12 selected buses. |
| S51-S100 | Named-branch operating variables | 50 | Sending/receiving-end active and reactive flows and thermal loading for 10 selected branches. |
| S101-S130 | Named-generator operating variables | 30 | Active output, reactive output, and hourly in-service status for 10 selected generating units. |

## Entity and Variable Detail

| Object | Selected IDs | Variables per object |
|---|---|---|
| Bus | `118, 115, 218, 215, 318, 315, 107, 203, 113, 123, 217, 325` | `vm_pu`, `va_deg`, `p_injection_mw`, `q_injection_mvar` |
| Branch | `AB1, AB2, AB3, CA-1, CB-1, C6, C10, C2, A34, B10` | `p_from_mw`, `q_from_mvar`, `p_to_mw`, `q_to_mvar`, `loading_pct` |
| Generator | `121_NUCLEAR_1, 223_STEAM_3, 122_WIND_1, 313_CC_1, 317_WIND_1, 218_CC_1, 303_WIND_1, 216_STEAM_1, 321_CC_1, 123_STEAM_3` | `pg_mw`, `qg_mvar`, `status` |

## LaTeX Table

```latex
\begin{table*}[t]
\centering
\caption{Detailed information of the RTS-GMLC inference-risk dataset}
\label{tab:rts_gmlc_dataset_information}
\begin{tabularx}{\textwidth}{c>{\centering\arraybackslash}p{0.25\textwidth}cX}
\hline
\textbf{Data No.} & \textbf{Dataset} & \textbf{No. of Fields} & \textbf{Data Information} \\
\hline
D1--D4 & Real-time load aggregates & 4 & Hourly real-time loads for Areas 1--3 and the whole system. \\
D5--D8 & Day-ahead load aggregates & 4 & Hourly day-ahead loads for Areas 1--3 and the whole system. \\
D9--D13 & System balance and area interchange & 5 & Total generation, total AC losses, and net exports of Areas 1--3. \\
D14--D18 & Operating reserves & 5 & Regulation-up/down reserves and spinning reserves for regions R1--R3. \\
D19--D27 & Generation mix aggregates & 9 & System-level generation grouped by fuel/resource type. \\
S1--S2 & Slack-bus balancing injections & 2 & Active and reactive balancing injections at named bus 113. \\
S3--S50 & Named-bus operating variables & 48 & Voltage magnitude, voltage angle, active injection, and reactive injection at 12 buses. \\
S51--S100 & Named-branch operating variables & 50 & Two-ended active/reactive flows and thermal loading for 10 branches. \\
S101--S130 & Named-generator operating variables & 30 & Active output, reactive output, and hourly in-service status for 10 units. \\
\hline
\end{tabularx}
\end{table*}
```

## Recommended Paper Description

> To complement the PJM publication dataset with fine-grained physical ground truth, we constructed an hourly AC power-flow dataset using the public synthetic RTS-GMLC system. Official 2020 five-minute load and renewable profiles were aggregated to hourly resolution, followed by transparent area-balanced dispatch and AC power-flow simulation. The resulting dataset contains 8,784 hourly observations under a base case and three input-perturbation scenarios (1%, 3%, and 5%). We treat 27 system-, area-, and fuel-level aggregates as candidate disclosed variables and 130 named bus-, branch-, and unit-level operating quantities as security-sensitive target variable types. RTS-GMLC is itself public and synthetic; the experiment therefore evaluates recoverability of sensitive variable types under controlled physical ground truth rather than claiming disclosure of confidential real-grid records.

## Suggested Experimental Split

为避免时间泄露，建议按时间连续划分而不是随机打乱：前 70% 训练，中间 10% 验证，后 20% 测试。四个场景必须复用相同时间切分。若论文要检验跨扰动泛化，可以只用 `base` 训练，再分别在 1%、3%、5% 文件的同一测试时段评估；若只检验同分布鲁棒性，则每个场景独立按同一边界训练和测试，两种设置不要混写。
