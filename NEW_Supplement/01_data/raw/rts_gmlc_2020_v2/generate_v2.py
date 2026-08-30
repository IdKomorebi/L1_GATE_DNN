#!/usr/bin/env python3
"""Generate physically consistent hourly RTS-GMLC AC power-flow datasets.

The script uses the official 2020 real-time load and renewable profiles, maps
them to the official pandapower network, performs a transparent hourly merit-
order dispatch for conventional units, and solves an AC power flow for every
hour.  Gaussian noise scenarios perturb *inputs before* the power-flow solve,
so voltage and branch-flow outputs remain physically linked.

This is an AC power-flow dataset, not an AC optimal-power-flow or unit-
commitment result.  Generator ``status`` means dispatched/in service in the
hourly construction; it does not claim to reproduce a market UC solution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pandapower as pp


HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent
DEFAULT_ARCHIVE = HERE / "source" / "rts_gmlc_official_inputs_3ece0d3.zip"
CONFIG_PATH = HERE / "config.json"

# v2 扩展用：燃料在源数据里的写法，以及输出字段名里用的写法
EXT_FUELS = [("Coal", "coal"), ("NG", "natural_gas"), ("Nuclear", "nuclear"),
             ("Oil", "oil"), ("Hydro", "hydro"), ("Wind", "wind"),
             ("Solar", "solar"), ("Storage", "storage")]


@dataclass(frozen=True)
class GeneratorElement:
    uid: str
    source_row: int
    bus_id: int
    element_type: str
    element_index: int


@dataclass(frozen=True)
class BranchElement:
    uid: str
    source_row: int
    from_bus_id: int
    to_bus_id: int
    element_type: str
    element_index: int


def sanitize_identifier(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def read_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def resolve_source_root(source_root: Path | None, source_archive: Path) -> Tuple[Path, tempfile.TemporaryDirectory | None]:
    if source_root is not None:
        candidate = source_root.resolve()
        if (candidate / "RTS_Data").is_dir():
            return candidate / "RTS_Data", None
        if (candidate / "SourceData").is_dir():
            return candidate, None
        raise FileNotFoundError(f"Cannot locate RTS_Data or SourceData below {candidate}")

    if not source_archive.is_file():
        raise FileNotFoundError(
            f"Official source archive not found: {source_archive}. "
            "Pass --source-root or place the pinned archive at the default path."
        )
    temp = tempfile.TemporaryDirectory(prefix="rts_gmlc_generation_")
    with zipfile.ZipFile(source_archive) as zf:
        zf.extractall(temp.name)
    root = Path(temp.name) / "RTS_Data"
    if not root.is_dir():
        raise RuntimeError("The source archive does not contain RTS_Data/ at its root")
    return root, temp


def add_timestamp(frame: pd.DataFrame, minutes_per_period: int) -> pd.DataFrame:
    base = pd.to_datetime(
        {
            "year": frame["Year"].astype(int),
            "month": frame["Month"].astype(int),
            "day": frame["Day"].astype(int),
        }
    )
    period_offset = pd.to_timedelta((frame["Period"].astype(int) - 1) * minutes_per_period, unit="min")
    result = frame.drop(columns=["Year", "Month", "Day", "Period"]).copy()
    result.index = pd.DatetimeIndex(base + period_offset, name="datetime_beginning")
    return result


def read_hourly_profile(path: Path, minutes_per_period: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = add_timestamp(frame, minutes_per_period)
    if minutes_per_period < 60:
        frame = frame.resample("1h").mean()
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate timestamps after hourly conversion: {path}")
    return frame.astype(float)


def read_daily_wide_hourly_profile(path: Path, minutes_per_period: int) -> pd.Series:
    """Read reserve files stored as one row per day and one column per period."""
    frame = pd.read_csv(path)
    id_columns = ["Year", "Month", "Day"]
    period_columns = [column for column in frame.columns if column not in id_columns]
    long = frame.melt(id_vars=id_columns, value_vars=period_columns, var_name="Period", value_name="value")
    long["Period"] = pd.to_numeric(long["Period"], errors="raise").astype(int)
    base = pd.to_datetime(
        {
            "year": long["Year"].astype(int),
            "month": long["Month"].astype(int),
            "day": long["Day"].astype(int),
        }
    )
    timestamp = base + pd.to_timedelta((long["Period"] - 1) * minutes_per_period, unit="min")
    series = pd.Series(pd.to_numeric(long["value"], errors="coerce").to_numpy(float), index=timestamp)
    series = series.sort_index()
    if minutes_per_period < 60:
        series = series.resample("1h").mean()
    series.index.name = "datetime_beginning"
    return series


def read_reserve_hourly_profile(path: Path) -> pd.Series:
    """Handle both long (Period column) and daily-wide reserve layouts."""
    columns = pd.read_csv(path, nrows=0).columns
    if "Period" in columns:
        profile = read_hourly_profile(path, 5)
        return profile.sum(axis=1)
    return read_daily_wide_hourly_profile(path, 5)


def load_hourly_inputs(source: Path) -> dict:
    ts = source / "timeseries_data_files"
    load_rt = read_hourly_profile(ts / "Load" / "REAL_TIME_regional_Load.csv", 5)
    load_da = read_hourly_profile(ts / "Load" / "DAY_AHEAD_regional_Load.csv", 60)

    renewable_parts = []
    for relative in [
        ("WIND/REAL_TIME_wind.csv", "wind"),
        ("PV/REAL_TIME_pv.csv", "pv"),
        ("RTPV/REAL_TIME_rtpv.csv", "rtpv"),
        ("Hydro/REAL_TIME_hydro.csv", "hydro"),
        ("CSP/REAL_TIME_Natural_Inflow.csv", "csp"),
    ]:
        part = read_hourly_profile(ts / relative[0], 5)
        renewable_parts.append(part)
    renewable = pd.concat(renewable_parts, axis=1)
    if renewable.columns.duplicated().any():
        duplicates = renewable.columns[renewable.columns.duplicated()].tolist()
        raise ValueError(f"Duplicated renewable generator UIDs: {duplicates}")

    reserve_files = {
        "reserve_reg_up_mw": "REAL_TIME_regional_Reg_Up.csv",
        "reserve_reg_down_mw": "REAL_TIME_regional_Reg_Down.csv",
        "reserve_spin_r1_mw": "REAL_TIME_regional_Spin_Up_R1.csv",
        "reserve_spin_r2_mw": "REAL_TIME_regional_Spin_Up_R2.csv",
        "reserve_spin_r3_mw": "REAL_TIME_regional_Spin_Up_R3.csv",
    }
    reserve = pd.DataFrame(index=load_rt.index)
    for output_name, file_name in reserve_files.items():
        reserve[output_name] = read_reserve_hourly_profile(ts / "Reserves" / file_name)

    common = load_rt.index.intersection(load_da.index).intersection(renewable.index).intersection(reserve.index)
    load_rt = load_rt.loc[common]
    load_da = load_da.loc[common]
    renewable = renewable.loc[common]
    reserve = reserve.loc[common]
    return {
        "load_rt": load_rt,
        "load_da": load_da,
        "renewable": renewable,
        "reserve": reserve,
    }


def build_generator_mapping(net, gen_source: pd.DataFrame, bus_source: pd.DataFrame) -> List[GeneratorElement]:
    bus_type = bus_source.set_index("Bus ID")["Bus Type"].astype(str).to_dict()
    pp_bus_by_id = {int(row.id): int(idx) for idx, row in net.bus.iterrows()}
    first_voltage_controller_seen: set[int] = set()
    gen_cursor = 0
    sgen_cursor = 0
    ext_cursor = 0
    mapping: List[GeneratorElement] = []

    for source_row, row in gen_source.iterrows():
        bus_id = int(row["Bus ID"])
        kind = bus_type[bus_id].strip().lower()
        if kind == "ref" and bus_id not in first_voltage_controller_seen:
            element_type, element_index = "ext_grid", ext_cursor
            ext_cursor += 1
            first_voltage_controller_seen.add(bus_id)
        elif kind == "pv" and bus_id not in first_voltage_controller_seen:
            element_type, element_index = "gen", gen_cursor
            gen_cursor += 1
            first_voltage_controller_seen.add(bus_id)
        else:
            element_type, element_index = "sgen", sgen_cursor
            sgen_cursor += 1

        mapping.append(
            GeneratorElement(
                uid=str(row["GEN UID"]),
                source_row=int(source_row),
                bus_id=bus_id,
                element_type=element_type,
                element_index=element_index,
            )
        )

        pp_bus = pp_bus_by_id[bus_id]
        if element_type == "gen":
            actual_bus = int(net.gen.at[element_index, "bus"])
        elif element_type == "sgen":
            actual_bus = int(net.sgen.at[element_index, "bus"])
        else:
            actual_bus = int(net.ext_grid.at[element_index, "bus"])
        if actual_bus != pp_bus:
            raise AssertionError(f"Generator mapping mismatch for {row['GEN UID']}: {actual_bus} != {pp_bus}")

    if gen_cursor != len(net.gen) or sgen_cursor != len(net.sgen) or ext_cursor != len(net.ext_grid):
        raise AssertionError(
            f"Generator mapping counts differ: gen={gen_cursor}/{len(net.gen)}, "
            f"sgen={sgen_cursor}/{len(net.sgen)}, ext_grid={ext_cursor}/{len(net.ext_grid)}"
        )
    return mapping


def build_branch_mapping(net, branch_source: pd.DataFrame) -> List[BranchElement]:
    is_transformer = pd.to_numeric(branch_source["Tr Ratio"], errors="coerce").fillna(0).ne(0)
    transformer_rows = branch_source.index[is_transformer].tolist()
    line_rows = branch_source.index[~is_transformer].tolist()
    if len(transformer_rows) != len(net.trafo):
        raise AssertionError(f"Transformer count mismatch: {len(transformer_rows)} != {len(net.trafo)}")
    if len(line_rows) != len(net.line):
        raise AssertionError(f"Line count mismatch: {len(line_rows)} != {len(net.line)}")
    trafo_index_by_source_row = {int(source_row): int(idx) for idx, source_row in enumerate(transformer_rows)}
    line_index_by_source_row = {int(source_row): int(idx) for idx, source_row in enumerate(line_rows)}

    mapping: List[BranchElement] = []
    for source_row, row in branch_source.iterrows():
        uid = str(row["UID"])
        if int(source_row) in trafo_index_by_source_row:
            element_type = "trafo"
            element_index = trafo_index_by_source_row[int(source_row)]
            net.trafo.at[element_index, "name"] = uid
        else:
            element_type = "line"
            element_index = line_index_by_source_row[int(source_row)]
            # The official converter duplicates the "-2" name for some parallel
            # circuits. Restore the source UID by row order, which is preserved.
            net.line.at[element_index, "name"] = uid
        mapping.append(
            BranchElement(
                uid=uid,
                source_row=int(source_row),
                from_bus_id=int(row["From Bus"]),
                to_bus_id=int(row["To Bus"]),
                element_type=element_type,
                element_index=element_index,
            )
        )
    return mapping


def compute_variable_costs(gen_source: pd.DataFrame) -> np.ndarray:
    fuel_price = pd.to_numeric(gen_source["Fuel Price $/MMBTU"], errors="coerce")
    heat_rate = pd.to_numeric(gen_source["HR_incr_1"], errors="coerce")
    heat_rate = heat_rate.fillna(pd.to_numeric(gen_source["HR_avg_0"], errors="coerce"))
    vom = pd.to_numeric(gen_source["VOM"], errors="coerce").fillna(0.0)
    cost = heat_rate.div(1000.0).mul(fuel_price).add(vom)
    defaults = {
        "Nuclear": 10.0,
        "Coal": 30.0,
        "NG": 45.0,
        "Oil": 120.0,
        "Storage": 80.0,
    }
    for fuel, default in defaults.items():
        cost = cost.mask(cost.isna() & gen_source["Fuel"].astype(str).eq(fuel), default)
    return cost.fillna(100.0).to_numpy(dtype=float)


def merit_dispatch(requirement_mw: float, indices: np.ndarray, gen_source: pd.DataFrame, costs: np.ndarray) -> np.ndarray:
    result = np.zeros(len(gen_source), dtype=float)
    if requirement_mw <= 0 or len(indices) == 0:
        return result

    pmax_all = pd.to_numeric(gen_source["PMax MW"], errors="coerce").fillna(0.0).to_numpy(float)
    pmin_all = pd.to_numeric(gen_source["PMin MW"], errors="coerce").fillna(0.0).to_numpy(float)
    order = indices[np.lexsort((indices, costs[indices]))]
    total_capacity = float(pmax_all[order].sum())
    requirement = min(float(requirement_mw), total_capacity)
    remaining = requirement
    online: List[int] = []

    for idx in order:
        if remaining <= 1e-9:
            break
        take = min(float(pmax_all[idx]), remaining)
        if take <= 0:
            continue
        result[idx] = take
        online.append(int(idx))
        remaining -= take

    if online:
        marginal = online[-1]
        required_min = float(pmin_all[marginal])
        if 0 < result[marginal] < required_min:
            delta = required_min - result[marginal]
            result[marginal] = required_min
            for donor in reversed(online[:-1]):
                available = max(0.0, result[donor] - float(pmin_all[donor]))
                transfer = min(delta, available)
                result[donor] -= transfer
                delta -= transfer
                if delta <= 1e-9:
                    break
            if delta > 1e-9:
                result[marginal] -= delta

    return result


def set_generator_state(net, mapping: Sequence[GeneratorElement], gen_source: pd.DataFrame, dispatch: np.ndarray) -> None:
    base_p = pd.to_numeric(gen_source["MW Inj"], errors="coerce").fillna(0.0).to_numpy(float)
    base_q = pd.to_numeric(gen_source["MVAR Inj"], errors="coerce").fillna(0.0).to_numpy(float)
    qmin = pd.to_numeric(gen_source["QMin MVAR"], errors="coerce").fillna(-1e9).to_numpy(float)
    qmax = pd.to_numeric(gen_source["QMax MVAR"], errors="coerce").fillna(1e9).to_numpy(float)
    unit_type = gen_source["Unit Type"].astype(str).to_numpy()

    for element in mapping:
        idx = element.source_row
        p_mw = max(0.0, float(dispatch[idx]))
        is_sync = unit_type[idx].upper() == "SYNC_COND"
        in_service = bool(p_mw > 1e-6 or is_sync or element.element_type == "ext_grid")
        # Non-voltage-controlling injections use unity power factor. Reactive
        # balance is solved by the voltage-controlled gen/ext_grid elements.
        q_mvar = 0.0

        if element.element_type == "gen":
            net.gen.at[element.element_index, "p_mw"] = p_mw
            # Retain the official PV-bus voltage controllers even when their
            # active dispatch is zero. This represents available voltage support
            # and avoids turning electrically supported buses into weak PQ buses.
            net.gen.at[element.element_index, "in_service"] = True
        elif element.element_type == "sgen":
            net.sgen.at[element.element_index, "p_mw"] = p_mw
            net.sgen.at[element.element_index, "q_mvar"] = q_mvar
            net.sgen.at[element.element_index, "in_service"] = in_service
        else:
            net.ext_grid.at[element.element_index, "in_service"] = True


def solve_power_flow(net, first: bool) -> bool:
    init = "dc" if first else "results"
    try:
        pp.runpp(
            net,
            algorithm="nr",
            calculate_voltage_angles=True,
            init=init,
            max_iteration=30,
            tolerance_mva=1e-7,
            enforce_q_lims=False,
            numba=False,
        )
        return bool(net.converged)
    except Exception:
        try:
            pp.runpp(
                net,
                algorithm="iwamoto_nr",
                calculate_voltage_angles=True,
                init="flat",
                max_iteration=80,
                tolerance_mva=1e-7,
                enforce_q_lims=False,
                numba=False,
            )
            return bool(net.converged)
        except Exception:
            return False


def extract_generator_results(net, mapping: Sequence[GeneratorElement], gen_source: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pg = np.zeros(len(mapping), dtype=float)
    qg = np.zeros(len(mapping), dtype=float)
    status = np.zeros(len(mapping), dtype=float)
    solved_q_by_bus: Dict[int, float] = {}
    for pos, element in enumerate(mapping):
        if element.element_type == "gen":
            row = net.res_gen.loc[element.element_index]
            pg[pos] = float(row.p_mw)
            solved_q_by_bus[element.bus_id] = solved_q_by_bus.get(element.bus_id, 0.0) + float(row.q_mvar)
            status[pos] = float(abs(pg[pos]) > 1e-6)
        elif element.element_type == "sgen":
            row = net.res_sgen.loc[element.element_index]
            pg[pos], qg[pos] = float(row.p_mw), float(row.q_mvar)
            status[pos] = float(bool(net.sgen.at[element.element_index, "in_service"]))
        else:
            row = net.res_ext_grid.loc[element.element_index]
            pg[pos], qg[pos] = float(row.p_mw), float(row.q_mvar)
            status[pos] = 1.0

    unit_type = gen_source["Unit Type"].astype(str).str.upper().to_numpy()
    qmin = pd.to_numeric(gen_source["QMin MVAR"], errors="coerce").fillna(0.0).to_numpy(float)
    qmax = pd.to_numeric(gen_source["QMax MVAR"], errors="coerce").fillna(0.0).to_numpy(float)
    for pos, element in enumerate(mapping):
        if unit_type[pos] == "SYNC_COND":
            status[pos] = 1.0

    # pandapower represents one voltage controller per PV bus. Allocate its
    # solved reactive output to the active synchronous units at that bus so an
    # offline unit is not reported with QG != 0 merely because it carries the
    # bus-level controller in the converted network.
    for bus_id, solved_q in solved_q_by_bus.items():
        candidates = [
            pos for pos, element in enumerate(mapping)
            if element.bus_id == bus_id and (status[pos] > 0 or unit_type[pos] == "SYNC_COND")
            and unit_type[pos] not in {"PV", "RTPV", "WIND"}
        ]
        if not candidates:
            continue
        weights = np.array([max(abs(qmin[pos]), abs(qmax[pos]), 1.0) for pos in candidates], dtype=float)
        weights /= weights.sum()
        for pos, weight in zip(candidates, weights):
            qg[pos] += solved_q * float(weight)
    return pg, qg, status


def extract_branch_results(net, mapping: Sequence[BranchElement]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pf = np.zeros(len(mapping), dtype=float)
    qf = np.zeros(len(mapping), dtype=float)
    pt = np.zeros(len(mapping), dtype=float)
    qt = np.zeros(len(mapping), dtype=float)
    loading = np.zeros(len(mapping), dtype=float)
    for pos, element in enumerate(mapping):
        if element.element_type == "line":
            row = net.res_line.loc[element.element_index]
            pf[pos] = float(row.p_from_mw)
            qf[pos] = float(row.q_from_mvar)
            pt[pos] = float(row.p_to_mw)
            qt[pos] = float(row.q_to_mvar)
        else:
            row = net.res_trafo.loc[element.element_index]
            hv_bus_idx = int(net.trafo.at[element.element_index, "hv_bus"])
            hv_bus_id = int(net.bus.at[hv_bus_idx, "id"])
            if hv_bus_id == element.from_bus_id:
                pf[pos], qf[pos] = float(row.p_hv_mw), float(row.q_hv_mvar)
                pt[pos], qt[pos] = float(row.p_lv_mw), float(row.q_lv_mvar)
            else:
                pf[pos], qf[pos] = float(row.p_lv_mw), float(row.q_lv_mvar)
                pt[pos], qt[pos] = float(row.p_hv_mw), float(row.q_hv_mvar)
        loading[pos] = float(row.loading_percent)
    return pf, qf, pt, qt, loading


def simulate_scenario(
    source: Path,
    hourly: Mapping[str, pd.DataFrame],
    noise_pct: float,
    seed: int,
    max_hours: int | None = None,
) -> dict:
    bus_source = pd.read_csv(source / "SourceData" / "bus.csv")
    branch_source = pd.read_csv(source / "SourceData" / "branch.csv")
    gen_source = pd.read_csv(source / "SourceData" / "gen.csv")
    net = pp.from_json(str(source / "FormattedData" / "pandapower" / "pandapower_net.json"))

    generator_mapping = build_generator_mapping(net, gen_source, bus_source)
    branch_mapping = build_branch_mapping(net, branch_source)
    pp_bus_by_id = {int(row.id): int(idx) for idx, row in net.bus.iterrows()}
    bus_ids = [int(value) for value in net.bus["id"].tolist()]
    branch_uids = [element.uid for element in branch_mapping]
    generator_uids = [element.uid for element in generator_mapping]

    index = hourly["load_rt"].index
    if max_hours is not None:
        index = index[:max_hours]
    n_hours = len(index)
    rng = np.random.default_rng(seed)

    load_rt = hourly["load_rt"].loc[index].copy()
    load_da = hourly["load_da"].loc[index].copy()
    renewable = hourly["renewable"].loc[index].copy()
    reserve = hourly["reserve"].loc[index].copy()
    if noise_pct > 0:
        sigma = noise_pct / 100.0
        load_rt.iloc[:, :] = np.maximum(0.0, load_rt.to_numpy() * (1.0 + rng.normal(0.0, sigma, load_rt.shape)))
        renewable.iloc[:, :] = np.maximum(0.0, renewable.to_numpy() * (1.0 + rng.normal(0.0, sigma, renewable.shape)))

    pmax = pd.to_numeric(gen_source["PMax MW"], errors="coerce").fillna(0.0).to_numpy(float)
    renewable_uids = set(renewable.columns.astype(str))
    uid_to_source_row = {str(uid): int(idx) for idx, uid in gen_source["GEN UID"].items()}
    missing_profiles = sorted(renewable_uids - set(uid_to_source_row))
    if missing_profiles:
        raise ValueError(f"Renewable time-series UIDs missing from gen.csv: {missing_profiles}")
    renewable_rows = np.array([uid_to_source_row[uid] for uid in renewable.columns], dtype=int)
    storage_rows = np.flatnonzero(gen_source["Unit Type"].astype(str).str.upper().eq("STORAGE").to_numpy())
    sync_rows = np.flatnonzero(gen_source["Unit Type"].astype(str).str.upper().eq("SYNC_COND").to_numpy())
    slack_rows = np.array([m.source_row for m in generator_mapping if m.element_type == "ext_grid"], dtype=int)
    excluded = set(renewable_rows.tolist()) | set(storage_rows.tolist()) | set(sync_rows.tolist()) | set(slack_rows.tolist())
    dispatchable_rows = np.array([i for i in range(len(gen_source)) if i not in excluded and pmax[i] > 0], dtype=int)
    variable_costs = compute_variable_costs(gen_source)

    bus_area = bus_source.set_index("Bus ID")["Area"].astype(int).to_dict()
    generator_area = np.array([bus_area[int(bus_id)] for bus_id in gen_source["Bus ID"]], dtype=int)
    load_bus_rows: Dict[int, List[int]] = {1: [], 2: [], 3: []}
    base_area_load = {1: 0.0, 2: 0.0, 3: 0.0}
    for load_idx, load_row in net.load.iterrows():
        bus_id = int(net.bus.at[int(load_row.bus), "id"])
        area = bus_area[bus_id]
        load_bus_rows[area].append(int(load_idx))
        base_area_load[area] += float(load_row.p_mw)
    base_load_p = net.load["p_mw"].to_numpy(float).copy()
    base_load_q = net.load["q_mvar"].to_numpy(float).copy()

    bus_vm = np.full((n_hours, len(bus_ids)), np.nan)
    bus_va = np.full((n_hours, len(bus_ids)), np.nan)
    bus_p_inj = np.full((n_hours, len(bus_ids)), np.nan)
    bus_q_inj = np.full((n_hours, len(bus_ids)), np.nan)
    branch_pf = np.full((n_hours, len(branch_uids)), np.nan)
    branch_qf = np.full((n_hours, len(branch_uids)), np.nan)
    branch_pt = np.full((n_hours, len(branch_uids)), np.nan)
    branch_qt = np.full((n_hours, len(branch_uids)), np.nan)
    branch_loading = np.full((n_hours, len(branch_uids)), np.nan)
    gen_pg = np.full((n_hours, len(generator_uids)), np.nan)
    gen_qg = np.full((n_hours, len(generator_uids)), np.nan)
    gen_status = np.full((n_hours, len(generator_uids)), np.nan)
    public_rows: List[dict] = []

    previous_loss = 150.0
    config = read_config()
    slack_target = float(config["slack_target_mw"])
    first_solve = True
    fuel_values = gen_source["Fuel"].astype(str).to_numpy()
    unit_values = gen_source["Unit Type"].astype(str).str.upper().to_numpy()

    for hour_pos, timestamp in enumerate(index):
        area_load = {area: float(load_rt.loc[timestamp, str(area)]) for area in [1, 2, 3]}
        for area, load_indices in load_bus_rows.items():
            scale = area_load[area] / base_area_load[area]
            net.load.loc[load_indices, "p_mw"] = base_load_p[load_indices] * scale
            net.load.loc[load_indices, "q_mvar"] = base_load_q[load_indices] * scale
        system_load = float(sum(area_load.values()))

        dispatch = np.zeros(len(gen_source), dtype=float)
        renewable_values = renewable.loc[timestamp].to_numpy(float)
        renewable_values = np.minimum(renewable_values, pmax[renewable_rows])
        dispatch[renewable_rows] = renewable_values
        desired_conventional_by_area: Dict[int, float] = {}
        for area in [1, 2, 3]:
            renewable_in_area = renewable_rows[generator_area[renewable_rows] == area]
            dispatchable_in_area = dispatchable_rows[generator_area[dispatchable_rows] == area]
            fixed_generation_area = float(dispatch[renewable_in_area].sum())
            loss_share = previous_loss * area_load[area] / system_load
            local_slack = slack_target if area == 1 else 0.0
            desired_local = area_load[area] + loss_share - local_slack - fixed_generation_area
            if desired_local < 0:
                allowed_fixed = max(0.0, area_load[area] + loss_share - local_slack)
                if fixed_generation_area > 0:
                    dispatch[renewable_in_area] *= allowed_fixed / fixed_generation_area
                desired_local = 0.0
            desired_conventional_by_area[area] = desired_local
            dispatch += merit_dispatch(desired_local, dispatchable_in_area, gen_source, variable_costs)
        dispatch[sync_rows] = 0.0
        dispatch[storage_rows] = 0.0
        set_generator_state(net, generator_mapping, gen_source, dispatch)

        converged = solve_power_flow(net, first_solve)
        first_solve = False
        if converged:
            slack_p = float(net.res_ext_grid["p_mw"].sum())
            if slack_p < 5.0 or slack_p > 55.0:
                area_1_dispatchable = dispatchable_rows[generator_area[dispatchable_rows] == 1]
                corrected_requirement = desired_conventional_by_area[1] + (slack_p - slack_target)
                dispatch[area_1_dispatchable] = 0.0
                dispatch += merit_dispatch(corrected_requirement, area_1_dispatchable, gen_source, variable_costs)
                set_generator_state(net, generator_mapping, gen_source, dispatch)
                converged = solve_power_flow(net, False)

        if converged:
            bus_vm[hour_pos, :] = net.res_bus.loc[net.bus.index, "vm_pu"].to_numpy(float)
            bus_va[hour_pos, :] = net.res_bus.loc[net.bus.index, "va_degree"].to_numpy(float)
            bus_p_inj[hour_pos, :] = -net.res_bus.loc[net.bus.index, "p_mw"].to_numpy(float)
            bus_q_inj[hour_pos, :] = -net.res_bus.loc[net.bus.index, "q_mvar"].to_numpy(float)
            pf, qf, pt, qt, loading = extract_branch_results(net, branch_mapping)
            branch_pf[hour_pos, :] = pf
            branch_qf[hour_pos, :] = qf
            branch_pt[hour_pos, :] = pt
            branch_qt[hour_pos, :] = qt
            branch_loading[hour_pos, :] = loading
            pg, qg, status = extract_generator_results(net, generator_mapping, gen_source)
            gen_pg[hour_pos, :] = pg
            gen_qg[hour_pos, :] = qg
            gen_status[hour_pos, :] = status
            system_loss = float(net.res_line["pl_mw"].sum() + net.res_trafo["pl_mw"].sum())
            previous_loss = system_loss
            slack_p = float(net.res_ext_grid["p_mw"].sum())
            slack_q = float(net.res_ext_grid["q_mvar"].sum())
        else:
            system_loss = math.nan
            slack_p = math.nan
            slack_q = math.nan

        area_net_export = {1: 0.0, 2: 0.0, 3: 0.0}
        if converged:
            for bpos, element in enumerate(branch_mapping):
                from_area = bus_area[element.from_bus_id]
                to_area = bus_area[element.to_bus_id]
                if from_area != to_area:
                    area_net_export[from_area] += branch_pf[hour_pos, bpos]
                    area_net_export[to_area] += branch_pt[hour_pos, bpos]

        row = {
            "datetime_beginning": timestamp,
            "scenario": "base" if noise_pct == 0 else f"noise_{int(noise_pct)}pct",
            "noise_std_pct": float(noise_pct),
            "area_1_load_actual_mw": area_load[1],
            "area_2_load_actual_mw": area_load[2],
            "area_3_load_actual_mw": area_load[3],
            "system_load_actual_mw": system_load,
            "area_1_load_day_ahead_mw": float(load_da.loc[timestamp, "1"]),
            "area_2_load_day_ahead_mw": float(load_da.loc[timestamp, "2"]),
            "area_3_load_day_ahead_mw": float(load_da.loc[timestamp, "3"]),
            "system_load_day_ahead_mw": float(load_da.loc[timestamp].sum()),
            "system_generation_mw": float(np.nansum(gen_pg[hour_pos, :])) if converged else math.nan,
            "system_losses_mw": system_loss,
            "area_1_net_export_mw": area_net_export[1] if converged else math.nan,
            "area_2_net_export_mw": area_net_export[2] if converged else math.nan,
            "area_3_net_export_mw": area_net_export[3] if converged else math.nan,
            "slack_bus_p_mw": slack_p,
            "slack_bus_q_mvar": slack_q,
            "ac_pf_converged": int(converged),
        }
        for reserve_name in reserve.columns:
            row[reserve_name] = float(reserve.loc[timestamp, reserve_name])
        if converged:
            for fuel_name, output_name in [
                ("Coal", "gen_fuel_coal_mw"),
                ("NG", "gen_fuel_natural_gas_mw"),
                ("Nuclear", "gen_fuel_nuclear_mw"),
                ("Oil", "gen_fuel_oil_mw"),
                ("Hydro", "gen_fuel_hydro_mw"),
                ("Wind", "gen_fuel_wind_mw"),
                ("Solar", "gen_fuel_solar_mw"),
                ("Storage", "gen_fuel_storage_mw"),
            ]:
                row[output_name] = float(np.nansum(gen_pg[hour_pos, fuel_values == fuel_name]))
            row["gen_sync_condenser_mw"] = float(np.nansum(gen_pg[hour_pos, unit_values == "SYNC_COND"]))
            # ---- 扩展字段（v2）：把机组出力按"分区 × 燃料"再汇总一层 ----
            # 原版只按燃料汇总到全系统，公开候选字段只有 27 个，对于一篇讲字段
            # 筛选的论文来说候选池太小、筛不筛差别不明显。这里补上真实电网普遍
            # 公布的更细口径：分区分燃料发电、分区发电总量、分区网损。
            # 这些全是对同一批机组出力的再汇总，不需要重新解潮流。
            for area in (1, 2, 3):
                in_area = generator_area == area
                for fuel_name, tag in EXT_FUELS:
                    row[f"area_{area}_fuel_{tag}_mw"] = float(
                        np.nansum(gen_pg[hour_pos, in_area & (fuel_values == fuel_name)]))
                row[f"area_{area}_generation_mw"] = float(np.nansum(gen_pg[hour_pos, in_area]))
                # 不加"分区网损"这个字段。真实电网并不公布分区网损——损耗按区分摊
                # 没有标准口径，只能自己定义。而一旦按"发电 − 负荷 − 净送出"反解
                # 定义出来，它就和这三项构成精确恒等式，意味着即使把区域净送出
                # 屏蔽掉，也能用发电、负荷、网损三者原样拼回来，第一层的剥离
                # 等于白做。造一个现实里不存在的字段、再让它制造出这种问题，
                # 对论文没有好处。
        else:
            for area in (1, 2, 3):
                for _, tag in EXT_FUELS:
                    row[f"area_{area}_fuel_{tag}_mw"] = math.nan
                row[f"area_{area}_generation_mw"] = math.nan
        # 时间特征：任何人看日历都知道，属于零成本可得的公开信息
        _ts = pd.Timestamp(timestamp)
        row["hour_of_day"] = int(_ts.hour)
        row["day_of_week"] = int(_ts.dayofweek)
        row["month_of_year"] = int(_ts.month)
        row["is_weekend"] = int(_ts.dayofweek >= 5)
        if not converged:
            for name in [
                "gen_fuel_coal_mw", "gen_fuel_natural_gas_mw", "gen_fuel_nuclear_mw",
                "gen_fuel_oil_mw", "gen_fuel_hydro_mw", "gen_fuel_wind_mw",
                "gen_fuel_solar_mw", "gen_fuel_storage_mw", "gen_sync_condenser_mw",
            ]:
                row[name] = math.nan
        public_rows.append(row)

        if (hour_pos + 1) % 500 == 0 or hour_pos + 1 == n_hours:
            print(f"scenario={row['scenario']} solved={hour_pos + 1}/{n_hours}", flush=True)

    public = pd.DataFrame(public_rows).set_index("datetime_beginning")
    return {
        "index": index,
        "public": public,
        "bus_ids": bus_ids,
        "branch_uids": branch_uids,
        "generator_uids": generator_uids,
        "bus_source": bus_source,
        "branch_source": branch_source,
        "gen_source": gen_source,
        "generator_mapping": generator_mapping,
        "branch_mapping": branch_mapping,
        "bus_vm": bus_vm,
        "bus_va": bus_va,
        "bus_p_inj": bus_p_inj,
        "bus_q_inj": bus_q_inj,
        "branch_pf": branch_pf,
        "branch_qf": branch_qf,
        "branch_pt": branch_pt,
        "branch_qt": branch_qt,
        "branch_loading": branch_loading,
        "gen_pg": gen_pg,
        "gen_qg": gen_qg,
        "gen_status": gen_status,
    }


def choose_entities(base: Mapping[str, object], config: Mapping[str, object]) -> dict:
    bus_source: pd.DataFrame = base["bus_source"]  # type: ignore[assignment]
    branch_source: pd.DataFrame = base["branch_source"]  # type: ignore[assignment]
    bus_ids: List[int] = base["bus_ids"]  # type: ignore[assignment]
    branch_uids: List[str] = base["branch_uids"]  # type: ignore[assignment]
    generator_uids: List[str] = base["generator_uids"]  # type: ignore[assignment]

    selected_buses: List[int] = []
    for area in sorted(bus_source["Area"].astype(int).unique()):
        candidates = bus_source.loc[bus_source["Area"].astype(int).eq(area)].nlargest(2, "MW Load")
        selected_buses.extend(candidates["Bus ID"].astype(int).tolist())

    mandatory = [uid for uid in config["mandatory_interarea_branches"] if uid in branch_uids]
    for uid in mandatory:
        row = branch_source.loc[branch_source["UID"].astype(str).eq(uid)].iloc[0]
        for bus_id in [int(row["From Bus"]), int(row["To Bus"])]:
            if bus_id not in selected_buses:
                selected_buses.append(bus_id)
    selected_buses = selected_buses[: int(config["selected_bus_count_target"])]

    loading: np.ndarray = base["branch_loading"]  # type: ignore[assignment]
    p95 = np.nanpercentile(loading, 95, axis=0)
    selected_branches = list(mandatory)
    for idx in np.argsort(np.nan_to_num(p95, nan=-np.inf))[::-1]:
        uid = branch_uids[int(idx)]
        if uid not in selected_branches:
            selected_branches.append(uid)
        if len(selected_branches) >= int(config["selected_branch_count_target"]):
            break

    pg: np.ndarray = base["gen_pg"]  # type: ignore[assignment]
    energy = np.nansum(np.maximum(pg, 0.0), axis=0)
    std = np.nanstd(pg, axis=0)
    selected_generators: List[str] = []
    for idx in np.argsort(energy)[::-1]:
        if energy[int(idx)] > 0:
            selected_generators.append(generator_uids[int(idx)])
        if len(selected_generators) >= 8:
            break
    for idx in np.argsort(std)[::-1]:
        uid = generator_uids[int(idx)]
        if uid not in selected_generators and std[int(idx)] > 0:
            selected_generators.append(uid)
        if len(selected_generators) >= int(config["selected_generator_count_target"]):
            break

    return {
        "buses": selected_buses,
        "branches": selected_branches,
        "generators": selected_generators,
        "selection_rule": {
            "buses": "Top two base-load buses per area, followed by endpoints of mandatory inter-area branches, capped at configured count.",
            "branches": "Mandatory inter-area branches plus highest base-scenario 95th-percentile AC loading.",
            "generators": "Eight largest annual-energy units plus highest-output-variability units, capped at configured count.",
        },
    }


def assemble_output(result: Mapping[str, object], selection: Mapping[str, Sequence[object]]) -> pd.DataFrame:
    output: pd.DataFrame = result["public"].copy()  # type: ignore[assignment]
    extra_columns: Dict[str, object] = {}
    bus_ids: List[int] = result["bus_ids"]  # type: ignore[assignment]
    branch_uids: List[str] = result["branch_uids"]  # type: ignore[assignment]
    generator_uids: List[str] = result["generator_uids"]  # type: ignore[assignment]

    for bus_id in selection["buses"]:
        idx = bus_ids.index(int(bus_id))
        extra_columns[f"bus_{bus_id}_vm_pu"] = result["bus_vm"][:, idx]  # type: ignore[index]
        extra_columns[f"bus_{bus_id}_va_deg"] = result["bus_va"][:, idx]  # type: ignore[index]
        extra_columns[f"bus_{bus_id}_p_injection_mw"] = result["bus_p_inj"][:, idx]  # type: ignore[index]
        extra_columns[f"bus_{bus_id}_q_injection_mvar"] = result["bus_q_inj"][:, idx]  # type: ignore[index]

    for uid_obj in selection["branches"]:
        uid = str(uid_obj)
        idx = branch_uids.index(uid)
        safe = sanitize_identifier(uid)
        extra_columns[f"branch_{safe}_p_from_mw"] = result["branch_pf"][:, idx]  # type: ignore[index]
        extra_columns[f"branch_{safe}_q_from_mvar"] = result["branch_qf"][:, idx]  # type: ignore[index]
        extra_columns[f"branch_{safe}_p_to_mw"] = result["branch_pt"][:, idx]  # type: ignore[index]
        extra_columns[f"branch_{safe}_q_to_mvar"] = result["branch_qt"][:, idx]  # type: ignore[index]
        extra_columns[f"branch_{safe}_loading_pct"] = result["branch_loading"][:, idx]  # type: ignore[index]

    for uid_obj in selection["generators"]:
        uid = str(uid_obj)
        idx = generator_uids.index(uid)
        safe = sanitize_identifier(uid)
        extra_columns[f"gen_{safe}_pg_mw"] = result["gen_pg"][:, idx]  # type: ignore[index]
        extra_columns[f"gen_{safe}_qg_mvar"] = result["gen_qg"][:, idx]  # type: ignore[index]
        status_values = result["gen_status"][:, idx]  # type: ignore[index]
        extra_columns[f"gen_{safe}_status"] = pd.array(status_values, dtype="Int64")

    output = pd.concat([output, pd.DataFrame(extra_columns, index=output.index)], axis=1)
    output = output.reset_index(names="datetime_beginning")
    output["datetime_beginning"] = pd.to_datetime(output["datetime_beginning"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    numeric_cols = output.select_dtypes(include=[np.number]).columns
    output[numeric_cols] = output[numeric_cols].round(6)
    return output


def scenario_quality(result: Mapping[str, object], output: pd.DataFrame) -> dict:
    public: pd.DataFrame = result["public"]  # type: ignore[assignment]
    vm: np.ndarray = result["bus_vm"]  # type: ignore[assignment]
    va: np.ndarray = result["bus_va"]  # type: ignore[assignment]
    loading: np.ndarray = result["branch_loading"]  # type: ignore[assignment]
    balance = public["system_generation_mw"] - public["system_load_actual_mw"] - public["system_losses_mw"]
    return {
        "rows": int(len(output)),
        "columns": int(output.shape[1]),
        "duplicate_timestamps": int(output["datetime_beginning"].duplicated().sum()),
        "converged_hours": int(public["ac_pf_converged"].sum()),
        "nonconverged_hours": int((1 - public["ac_pf_converged"]).sum()),
        "missing_cells": int(output.isna().sum().sum()),
        "max_abs_power_balance_error_mw": float(np.nanmax(np.abs(balance.to_numpy()))),
        "voltage_magnitude_min_pu": float(np.nanmin(vm)),
        "voltage_magnitude_max_pu": float(np.nanmax(vm)),
        "voltage_angle_min_deg": float(np.nanmin(va)),
        "voltage_angle_max_deg": float(np.nanmax(va)),
        "branch_loading_max_pct": float(np.nanmax(loading)),
        "hours_with_voltage_below_0_90_pu": int(np.sum(np.nanmin(vm, axis=1) < 0.90)),
        "hours_with_voltage_above_1_10_pu": int(np.sum(np.nanmax(vm, axis=1) > 1.10)),
        "hours_with_any_branch_above_100_pct": int(np.sum(np.nanmax(loading, axis=1) > 100.0)),
    }


def build_field_dictionary(output: pd.DataFrame, selection: Mapping[str, Sequence[object]], result: Mapping[str, object]) -> pd.DataFrame:
    public_definitions = {
        "datetime_beginning": ("identifier", "time", "", "Synthetic chronological timestamp supplied by RTS-GMLC profiles"),
        "scenario": ("identifier", "scenario", "", "Scenario label"),
        "noise_std_pct": ("identifier", "noise", "%", "Input perturbation standard deviation"),
        "ac_pf_converged": ("quality", "solver", "boolean", "pandapower AC power-flow convergence flag"),
    }
    rows = []
    selected_branch_map = {
        sanitize_identifier(str(uid)): str(uid) for uid in selection["branches"]
    }
    selected_gen_map = {
        sanitize_identifier(str(uid)): str(uid) for uid in selection["generators"]
    }
    for column in output.columns:
        role = "published_candidate"
        entity_type = "system_or_area"
        entity_id = ""
        unit = "MW" if column.endswith("_mw") else ""
        physical_quantity = column
        source_or_calculation = "Aggregation of RTS-GMLC inputs or solved AC power-flow outputs"
        mathematical_status = "aggregate_or_metadata"
        sensitivity_rationale = "Coarse system/area-level candidate release field"

        if column in public_definitions:
            role, entity_type, unit, source_or_calculation = public_definitions[column]
            mathematical_status = "metadata_or_quality"
            sensitivity_rationale = "Not an inference target"
        elif column in {"slack_bus_p_mw", "slack_bus_q_mvar"}:
            role, entity_type, entity_id = "sensitive_target", "bus", "113"
            unit = "MVAr" if column.endswith("_mvar") else "MW"
            source_or_calculation = "Solved active/reactive injection at the named AC slack bus"
            mathematical_status = "control_or_operational_state"
            sensitivity_rationale = "Fine-grained named-bus balancing injection"
        elif column.startswith("bus_"):
            match = re.match(r"bus_(\d+)_(.+)", column)
            entity_id = match.group(1) if match else ""
            quantity = match.group(2) if match else column
            role, entity_type = "sensitive_target", "bus"
            unit = "p.u." if quantity == "vm_pu" else "degree" if quantity == "va_deg" else "MVAr" if quantity.endswith("mvar") else "MW"
            physical_quantity = quantity
            source_or_calculation = "Solved nodal result from hourly AC power flow"
            mathematical_status = "strict_state_variable" if quantity in {"vm_pu", "va_deg"} else "derived_nodal_operating_quantity"
            sensitivity_rationale = "Fine-grained named-bus electrical state or injection"
        elif column.startswith("branch_"):
            match = re.match(r"branch_(.+)_(p_from_mw|q_from_mvar|p_to_mw|q_to_mvar|loading_pct)$", column)
            safe_uid = match.group(1) if match else ""
            entity_id = selected_branch_map.get(safe_uid, safe_uid)
            quantity = match.group(2) if match else column
            role, entity_type = "sensitive_target", "branch"
            unit = "%" if quantity == "loading_pct" else "MVAr" if "q_" in quantity else "MW"
            physical_quantity = quantity
            source_or_calculation = "Solved branch result from hourly AC power flow"
            mathematical_status = "derived_operational_state"
            sensitivity_rationale = "Fine-grained named transmission-corridor flow/loading"
        elif column.startswith("gen_") and not column.startswith("gen_fuel_") and column != "gen_sync_condenser_mw":
            match = re.match(r"gen_(.+)_(pg_mw|qg_mvar|status)$", column)
            safe_uid = match.group(1) if match else ""
            entity_id = selected_gen_map.get(safe_uid, safe_uid)
            quantity = match.group(2) if match else column
            role, entity_type = "sensitive_target", "generator"
            unit = "boolean" if quantity == "status" else "MVAr" if quantity == "qg_mvar" else "MW"
            physical_quantity = quantity
            source_or_calculation = "Hourly active dispatch; bus-level solved reactive output allocated across active synchronous units by reactive capability"
            mathematical_status = "control_or_operational_state"
            sensitivity_rationale = "Fine-grained named-unit dispatch, reactive output, or in-service state"
        elif column.endswith("_mvar"):
            unit = "MVAr"
        elif column.endswith("_pct"):
            unit = "%"

        rows.append({
            "column_name": column,
            "role": role,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "physical_quantity": physical_quantity,
            "unit": unit,
            "source_or_calculation": source_or_calculation,
            "mathematical_status": mathematical_status,
            "sensitivity_rationale": sensitivity_rationale,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=None, help="Path containing RTS_Data/ or its contents")
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--max-hours", type=int, default=None, help="Development smoke-test limit")
    parser.add_argument("--noise", type=float, nargs="*", default=None, help="Override configured noise scenarios")
    args = parser.parse_args()

    config = read_config()
    noise_values = list(args.noise) if args.noise is not None else [float(x) for x in config["noise_scenarios_pct"]]
    if not noise_values or noise_values[0] != 0:
        raise ValueError("The base (0% noise) scenario must be generated first")

    source, temp = resolve_source_root(args.source_root, args.source_archive)
    try:
        hourly = load_hourly_inputs(source)
        if args.max_hours is None and len(hourly["load_rt"]) != int(config["expected_hours"]):
            raise AssertionError(f"Expected {config['expected_hours']} hours, found {len(hourly['load_rt'])}")

        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        quality = {
            "dataset": config["dataset_name"],
            "source_commit": config["source_commit"],
            "scenarios": {},
        }

        base_result = simulate_scenario(
            source,
            hourly,
            noise_pct=0.0,
            seed=int(config["random_seed_base"]),
            max_hours=args.max_hours,
        )
        selection = choose_entities(base_result, config)
        (HERE / "selected_entities.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

        base_output = assemble_output(base_result, selection)
        base_name = "rts_gmlc_hourly_2020_acpf_base.csv" if args.max_hours is None else "smoke_rts_gmlc_base.csv"
        base_output.to_csv(output_dir / base_name, index=False)
        quality["scenarios"]["base"] = scenario_quality(base_result, base_output)

        dictionary = build_field_dictionary(base_output, selection, base_result)
        dictionary.to_csv(output_dir / "field_dictionary.csv", index=False)

        for noise_pct in noise_values[1:]:
            scenario_result = simulate_scenario(
                source,
                hourly,
                noise_pct=float(noise_pct),
                seed=int(config["random_seed_base"]) + int(round(noise_pct * 100)),
                max_hours=args.max_hours,
            )
            scenario_output = assemble_output(scenario_result, selection)
            if args.max_hours is None:
                file_name = f"rts_gmlc_hourly_2020_acpf_noise_{int(noise_pct)}pct.csv"
            else:
                file_name = f"smoke_rts_gmlc_noise_{int(noise_pct)}pct.csv"
            scenario_output.to_csv(output_dir / file_name, index=False)
            quality["scenarios"][f"noise_{int(noise_pct)}pct"] = scenario_quality(scenario_result, scenario_output)

        report_name = "quality_report.json" if args.max_hours is None else "smoke_quality_report.json"
        (output_dir / report_name).write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(quality, ensure_ascii=False, indent=2), flush=True)
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    main()
