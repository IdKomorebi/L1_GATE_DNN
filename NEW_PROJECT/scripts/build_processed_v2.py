#!/usr/bin/env python3
"""Build a yearly Processed_V2 CSV from the assembled raw CSV and PJM billing JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


ADDED_COLUMNS = [
    "rmccp",
    "rmpcp",
    "total_pjm_rt_load_mwh",
    "total_pjm_loc_credit",
    "total_pjm_reg_purchases",
    "total_pjm_self_sched_reg",
    "total_pjm_assigned_reg",
    "total_pjm_rmccp_cr",
    "total_pjm_rmpcp_cr",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_api_datetime(value: str) -> str:
    return value.replace("T", " ")[:19]


def format_api_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        raise ValueError("Boolean value found in a numeric PJM field")
    return str(value)


def expected_utc_hours(year: int) -> list[str]:
    current = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    result: list[str] = []
    while current < end:
        result.append(current.strftime("%Y-%m-%d %H:%M:%S"))
        current += timedelta(hours=1)
    return result


def read_raw(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            next(reader)  # bilingual source/feed header
            header = next(reader)  # canonical English field names
            next(reader)  # Chinese field descriptions
        except StopIteration as exc:
            raise ValueError(f"Raw CSV does not contain the expected three header rows: {path}") from exc

        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=4):
            if len(row) != len(header):
                raise ValueError(
                    f"Raw CSV row {line_number} has {len(row)} fields; expected {len(header)}"
                )
            rows.append(dict(zip(header, row)))
    return header, rows


def read_reference_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        try:
            return next(csv.reader(stream))
        except StopIteration as exc:
            raise ValueError(f"Reference CSV is empty: {path}") from exc


def read_api_items(path: Path) -> tuple[dict[str, dict[str, Any]], int | None]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("PJM response JSON does not contain an items list")

    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        utc = normalize_api_datetime(item["datetime_beginning_utc"])
        if utc in indexed:
            raise ValueError(f"Duplicate PJM API timestamp: {utc}")
        indexed[utc] = item
    return indexed, payload.get("totalRows")


def build(args: argparse.Namespace) -> dict[str, Any]:
    raw_path = args.raw.resolve()
    api_path = args.reg_json.resolve()
    reference_path = args.reference.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()

    raw_header, raw_rows = read_raw(raw_path)
    reference_header = read_reference_header(reference_path)
    api_by_utc, api_total_rows = read_api_items(api_path)
    expected_hours = expected_utc_hours(args.year)

    if len(reference_header) != 72:
        raise ValueError(f"Reference V2 must have 72 columns, found {len(reference_header)}")
    if len(raw_header) != 63:
        raise ValueError(f"Raw input must have 63 columns, found {len(raw_header)}")
    if [column for column in reference_header if column in raw_header] != raw_header:
        raise ValueError("Raw columns are not an order-preserving subset of the reference V2 header")
    added = [column for column in reference_header if column not in raw_header]
    if added != ADDED_COLUMNS:
        raise ValueError(f"Reference V2 added columns differ from the expected nine fields: {added}")

    raw_hours = [row["datetime_beginning_utc"] for row in raw_rows]
    if raw_hours != expected_hours:
        raise ValueError("Raw UTC timestamps are not the complete ordered hourly index for the requested year")
    if list(api_by_utc) != expected_hours:
        raise ValueError("PJM API UTC timestamps are not the complete ordered hourly index for the requested year")
    if api_total_rows is not None and api_total_rows != len(expected_hours):
        raise ValueError(f"PJM API totalRows={api_total_rows}; expected {len(expected_hours)}")

    ept_mismatches = 0
    missing_added_values = {column: 0 for column in ADDED_COLUMNS}
    for raw_row in raw_rows:
        utc = raw_row["datetime_beginning_utc"]
        api_row = api_by_utc[utc]
        api_ept = normalize_api_datetime(api_row["datetime_beginning_ept"])
        if raw_row["datetime_beginning_ept"][:19] != api_ept:
            ept_mismatches += 1
        for column in ADDED_COLUMNS:
            if api_row.get(column) is None:
                missing_added_values[column] += 1
    if ept_mismatches:
        raise ValueError(f"Found {ept_mismatches} EPT mismatches between raw and PJM API data")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=reference_header, lineterminator="\n")
        writer.writeheader()
        for raw_row in raw_rows:
            api_row = api_by_utc[raw_row["datetime_beginning_utc"]]
            output_row = dict(raw_row)
            for column in ADDED_COLUMNS:
                output_row[column] = format_api_value(api_row.get(column))
            writer.writerow(output_row)
    temporary_path.replace(output_path)

    raw_blank_cells = sum(value.strip() == "" for row in raw_rows for value in row.values())
    added_blank_cells = sum(missing_added_values.values())
    manifest = {
        "dataset": f"data{args.year}_Processed_V2",
        "year": args.year,
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source": {
            "raw_csv": str(raw_path),
            "raw_csv_sha256": sha256(raw_path),
            "pjm_feed": "reg_zone_prelim_bill",
            "pjm_feed_display_name": "PJM Regulation Zone Preliminary Billing Data",
            "pjm_api_url": "https://api.pjm.com/api/v1/reg_zone_prelim_bill",
            "pjm_definition_url": "https://dataminer2.pjm.com/feed/reg_zone_prelim_bill/definition",
            "query_filter_utc": (
                f"1/1/{args.year} 00:00to12/31/{args.year} 23:00"
            ),
            "api_response_sha256": sha256(api_path),
        },
        "reference": {
            "v2_csv": str(reference_path),
            "v2_header_sha256": hashlib.sha256(
                (",".join(reference_header) + "\n").encode("utf-8")
            ).hexdigest(),
        },
        "output": {
            "csv": str(output_path),
            "csv_sha256": sha256(output_path),
            "rows": len(raw_rows),
            "columns": len(reference_header),
            "first_utc": raw_hours[0],
            "last_utc": raw_hours[-1],
            "unique_utc": len(set(raw_hours)),
            "raw_blank_cells_preserved": raw_blank_cells,
            "added_blank_cells": added_blank_cells,
            "added_columns": ADDED_COLUMNS,
            "ept_mismatches": ept_mismatches,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--reg-json", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps(result["output"], ensure_ascii=False, indent=2))
