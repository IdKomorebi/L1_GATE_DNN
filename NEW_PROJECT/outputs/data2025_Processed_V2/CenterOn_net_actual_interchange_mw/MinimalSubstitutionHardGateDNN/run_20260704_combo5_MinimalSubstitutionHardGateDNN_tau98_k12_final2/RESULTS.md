# Minimal Substitution Hard Gate DNN

## Design

This run first trains a full-field MLP baseline, then sets `tau = 0.95 * R2_full`.
A straight-through binary gate searches for a sparse main inference path. The selected path is then retrained and greedily ablated until every remaining field is required to keep `R2 >= tau`.
The residual feature set is tested as a completely disjoint path. Finally, each main-path field is removed and the residual feature set is tested as a conditional replacement pool.

## Key Results

- Full-field R2: 0.969259
- Full-field MSE: 0.029123
- Tau: 0.949873
- Main path fields: 16
- Residual-only R2: 0.950717
- Residual reaches tau: True
- Replaceable main-path fields: 0/16

## Main Path

- gen_fuel_nuclear_mw
- da_as_as_req_mw_synchronized_reserve
- gen_fuel_wind_mw
- rmpcp
- total_pjm_reg_purchases
- da_as_total_mw_thirty_minutes_reserve
- gross_sched_interchange_mw
- congestion_price_rt
- da_as_ss_mw_primary_reserve
- gen_fuel_solar_mw
- net_inadv_interchange_mw
- total_pjm_rt_load_mwh
- system_energy_price_da
- gen_fuel_oil_mw
- rmccp
- gen_fuel_other_renewables_mw

## Output Files

- `full_metrics.csv`: full-field baseline metrics and tau.
- `main_path.csv`: main path selection, pruning, and final removal impact.
- `residual_test.csv`: residual-only and disjoint independent-path checks.
- `substitution_results.csv`: conditional replacement tests for each main-path field.
- `gate_values.csv`: final and historical gate values for the main path and substitution gates.
- `figures/`: training curves, gate bars, ablation impact, residual and substitution summaries.
