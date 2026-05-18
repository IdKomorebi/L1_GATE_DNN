# Manual exclude/n recommendations v2

Top-n DNN validation used epochs=100 and the L1GateDNN gate ranking for each chosen source run.

| center | n | validation full R2 | top-n R2 | exclude |
|---|---:|---:|---:|---|
| net_actual_interchange_mw | 13 | 0.966883 | 0.959721 | net_sched_interchange_mw, total_gen |
| gross_actual_interchange_mw | 12 | 0.864887 | 0.865801 | (none) |
| net_sched_interchange_mw | 12 | 0.964838 | 0.956095 | net_actual_interchange_mw, total_gen |
| total_gen | 15 | 0.920681 | 0.914915 | prelim_load_avg_hourly, metered_load_mw, total_pjm_rt_load_mwh, forecast_load_mw_latest_available, forecast_load_mw_day_ahead, wind_generation_mw, solar_generation_mw, gen_fuel_coal_mw, gen_fuel_gas_mw, gen_fuel_hydro_mw, gen_fuel_multiple_fuels_mw, gen_fuel_nuclear_mw, gen_fuel_oil_mw, gen_fuel_other_renewables_mw, gen_fuel_solar_mw, gen_fuel_storage_mw, gen_fuel_wind_mw, gen_fuel_coal_pct, gen_fuel_gas_pct, gen_fuel_hydro_pct, gen_fuel_multiple_fuels_pct, gen_fuel_nuclear_pct, gen_fuel_oil_pct, gen_fuel_other_renewables_pct, gen_fuel_solar_pct, gen_fuel_storage_pct, gen_fuel_wind_pct |
| metered_load_mw | 15 | 0.916783 | 0.910171 | total_gen, prelim_load_avg_hourly, total_pjm_rt_load_mwh, forecast_load_mw_latest_available, forecast_load_mw_day_ahead, wind_generation_mw, solar_generation_mw, gen_fuel_coal_mw, gen_fuel_gas_mw, gen_fuel_hydro_mw, gen_fuel_multiple_fuels_mw, gen_fuel_nuclear_mw, gen_fuel_oil_mw, gen_fuel_other_renewables_mw, gen_fuel_solar_mw, gen_fuel_storage_mw, gen_fuel_wind_mw, gen_fuel_coal_pct, gen_fuel_gas_pct, gen_fuel_hydro_pct, gen_fuel_multiple_fuels_pct, gen_fuel_nuclear_pct, gen_fuel_oil_pct, gen_fuel_other_renewables_pct, gen_fuel_solar_pct, gen_fuel_storage_pct, gen_fuel_wind_pct |
| total_losses | 16 | 0.943239 | 0.936411 | (none) |
| congestion_price_da | 2 | 0.936812 | 0.995009 | (none) |
| congestion_price_rt | 10 | 0.672150 | 0.676734 | (none) |
| marginal_loss_price_da | 12 | 0.897897 | 0.898318 | (none) |
| total_lmp_da | 12 | 0.957618 | 0.954802 | system_energy_price_da, congestion_price_da, marginal_loss_price_da |
| da_as_total_mw_primary_reserve | 8 | 0.849734 | 0.876784 | da_as_as_mw_primary_reserve, da_as_nsr_mw_primary_reserve |
| da_as_total_mw_thirty_minutes_reserve | 18 | 0.903935 | 0.896541 | da_as_as_mw_thirty_minutes_reserve |
