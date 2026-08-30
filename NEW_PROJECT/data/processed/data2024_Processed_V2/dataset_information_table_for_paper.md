# Dataset Information Table for Paper

说明：`Data No.` 只统计数据字段，不包含 `datetime_beginning_utc` 和 `datetime_beginning_ept` 两个时间字段。字段顺序与 `pjm_rto_hourly_2025_aligned_processed_one_header.csv` 完全一致，当前表对应 `pjm_rto_hourly_2024_aligned_processed_one_header.csv`。

## Markdown Table

| Data No. | Dataset | Data Information |
|---|---|---|
| D1-D2 | Generation and Extra High Voltage Losses | This data contains the hourly PJM generation and extra high voltage (EHV) loss data for each day. This is informational data only. |
| D3 | Wind Generation | This feed provides the hourly wind generation amounts in PJM. |
| D4 | Solar Generation | This feed provides the hourly solar generation amounts in PJM. |
| D5-D24 | Generation by Fuel Type | This data shows the fuel mix of generation resources operating under PJM direction for each hour. |
| D25-D26 | Historical Load Forecasts | This feed provides forecasted load, grouped by the date and time the forecast was created. Forecasts are provided every six hours starting one day prior to the Effective Date. |
| D27 | Hourly Load: Preliminary | This feed contains preliminary integrated hourly loads that are calculated daily from raw telemetry data and are approximate for informational purposes only. |
| D28 | Hourly Load: Metered | This feed summarizes the MW-hour net energy for load as consumed by the service territories within the PJM RTO. |
| D29-D37 | PJM Regulation Zone Preliminary Billing Data | This feed contains hourly billing data for the PJM Regulation Market. |
| D38-D56 | Day-Ahead Ancillary Service Market Results | This feed provides the Day-Ahead Markets results for Ancillary Services including MW quantities and prices. |
| D57-D60 | Day-Ahead Hourly LMPs | This feed contains hourly Day-Ahead Energy Market locational marginal pricing (LMP) data for all bus locations, including aggregates. |
| D61-D64 | Real-Time Hourly LMPs | This feed contains hourly Real-Time Energy Market locational marginal pricing (LMP) data for all bus locations, including aggregates. |
| D65-D70 | Actual/Schedule Summary Report | Hourly actual, scheduled, and inadvertent flows by tie line for each hour. Positive Values represent an import into PJM and Negative Values represent an export from PJM. |

## LaTeX Table

```latex
\begin{table*}[t]
\centering
\caption{Detailed information in the dataset}
\label{tab:dataset_information}
\begin{tabularx}{\textwidth}{c>{\centering\arraybackslash}p{0.28\textwidth}X}
\hline
\textbf{\textit{Data No.}} & \textbf{\textit{Dataset}} & \textbf{\textit{Data Information}} \\
\hline
D1-D2 & Generation and Extra High Voltage Losses & This data contains the hourly PJM generation and extra high voltage (EHV) loss data for each day. This is informational data only. \\
D3 & Wind Generation & This feed provides the hourly wind generation amounts in PJM. \\
D4 & Solar Generation & This feed provides the hourly solar generation amounts in PJM. \\
D5-D24 & Generation by Fuel Type & This data shows the fuel mix of generation resources operating under PJM direction for each hour. \\
D25-D26 & Historical Load Forecasts & This feed provides forecasted load, grouped by the date and time the forecast was created. Forecasts are provided every six hours starting one day prior to the Effective Date. \\
D27 & Hourly Load: Preliminary & This feed contains preliminary integrated hourly loads that are calculated daily from raw telemetry data and are approximate for informational purposes only. \\
D28 & Hourly Load: Metered & This feed summarizes the MW-hour net energy for load as consumed by the service territories within the PJM RTO. \\
D29-D37 & PJM Regulation Zone Preliminary Billing Data & This feed contains hourly billing data for the PJM Regulation Market. \\
D38-D56 & Day-Ahead Ancillary Service Market Results & This feed provides the Day-Ahead Markets results for Ancillary Services including MW quantities and prices. \\
D57-D60 & Day-Ahead Hourly LMPs & This feed contains hourly Day-Ahead Energy Market locational marginal pricing (LMP) data for all bus locations, including aggregates. \\
D61-D64 & Real-Time Hourly LMPs & This feed contains hourly Real-Time Energy Market locational marginal pricing (LMP) data for all bus locations, including aggregates. \\
D65-D70 & Actual/Schedule Summary Report & Hourly actual, scheduled, and inadvertent flows by tie line for each hour. Positive Values represent an import into PJM and Negative Values represent an export from PJM. \\
\hline
\end{tabularx}
\end{table*}
```

## Source and Column Trace

| Data No. | Feed key | Definition URL | Column count | Columns used |
|---|---|---|---:|---|
| D1-D2 | `gen_ehv_losses` | https://dataminer2.pjm.com/feed/gen_ehv_losses/definition | 2 | `total_gen, total_losses` |
| D3 | `wind_gen` | https://dataminer2.pjm.com/feed/wind_gen/definition | 1 | `wind_generation_mw` |
| D4 | `solar_gen` | https://dataminer2.pjm.com/feed/solar_gen/definition | 1 | `solar_generation_mw` |
| D5-D24 | `gen_by_fuel` | https://dataminer2.pjm.com/feed/gen_by_fuel/definition | 20 | `gen_fuel_coal_mw, gen_fuel_gas_mw, gen_fuel_hydro_mw, gen_fuel_multiple_fuels_mw, gen_fuel_nuclear_mw, gen_fuel_oil_mw, gen_fuel_other_renewables_mw, gen_fuel_solar_mw, gen_fuel_storage_mw, gen_fuel_wind_mw, gen_fuel_coal_pct, gen_fuel_gas_pct, gen_fuel_hydro_pct, gen_fuel_multiple_fuels_pct, gen_fuel_nuclear_pct, gen_fuel_oil_pct, gen_fuel_other_renewables_pct, gen_fuel_solar_pct, gen_fuel_storage_pct, gen_fuel_wind_pct` |
| D25-D26 | `load_frcstd_hist` | https://dataminer2.pjm.com/feed/load_frcstd_hist/definition | 2 | `forecast_load_mw_latest_available, forecast_load_mw_day_ahead` |
| D27 | `hrl_load_prelim` | https://dataminer2.pjm.com/feed/hrl_load_prelim/definition | 1 | `prelim_load_avg_hourly` |
| D28 | `hrl_load_metered` | https://dataminer2.pjm.com/feed/hrl_load_metered/definition | 1 | `metered_load_mw` |
| D29-D37 | `reg_zone_prelim_bill` | https://dataminer2.pjm.com/feed/reg_zone_prelim_bill/definition | 9 | `rmccp, rmpcp, total_pjm_rt_load_mwh, total_pjm_loc_credit, total_pjm_reg_purchases, total_pjm_self_sched_reg, total_pjm_assigned_reg, total_pjm_rmccp_cr, total_pjm_rmpcp_cr` |
| D38-D56 | `da_reserve_market_results` | https://dataminer2.pjm.com/feed/da_reserve_market_results/definition | 19 | `da_as_mcp_primary_reserve, da_as_mcp_synchronized_reserve, da_as_mcp_thirty_minutes_reserve, da_as_as_req_mw_primary_reserve, da_as_as_req_mw_synchronized_reserve, da_as_as_req_mw_thirty_minutes_reserve, da_as_total_mw_primary_reserve, da_as_total_mw_synchronized_reserve, da_as_total_mw_thirty_minutes_reserve, da_as_as_mw_primary_reserve, da_as_as_mw_synchronized_reserve, da_as_as_mw_thirty_minutes_reserve, da_as_ss_mw_primary_reserve, da_as_ss_mw_synchronized_reserve, da_as_ss_mw_thirty_minutes_reserve, da_as_ircmwt2_primary_reserve, da_as_ircmwt2_synchronized_reserve, da_as_ircmwt2_thirty_minutes_reserve, da_as_nsr_mw_primary_reserve` |
| D57-D60 | `da_hrl_lmps` | https://dataminer2.pjm.com/feed/da_hrl_lmps/definition | 4 | `system_energy_price_da, total_lmp_da, congestion_price_da, marginal_loss_price_da` |
| D61-D64 | `rt_hrl_lmps` | https://dataminer2.pjm.com/feed/rt_hrl_lmps/definition | 4 | `system_energy_price_rt, total_lmp_rt, congestion_price_rt, marginal_loss_price_rt` |
| D65-D70 | `act_sch_interchange` | https://dataminer2.pjm.com/feed/act_sch_interchange/definition | 6 | `net_actual_interchange_mw, net_sched_interchange_mw, net_inadv_interchange_mw, gross_actual_interchange_mw, gross_sched_interchange_mw, gross_inadv_interchange_mw` |
