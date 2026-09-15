#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

INDEX_SERIES = ("cpi",)
RETURN_SERIES = ("eq_tr", "housing_tr", "bond_tr", "bill_rate")
SERIES = INDEX_SERIES + RETURN_SERIES
MISSING_VALUES = {"", "na", "n/a", "nan", "null", "none"}


def parse_number(value: str) -> Optional[float]:
    if value is None:
        return None

    text = value.strip()
    if not text or text.lower() in MISSING_VALUES:
        return None

    return float(text)


def parse_year(value: str) -> Optional[int]:
    number = parse_number(value)
    if number is None:
        return None
    return int(number)


def cagr_from_index(observations: Iterable[tuple[int, float]]) -> tuple[Optional[float], int]:
    cleaned = sorted(observations)
    length = len(cleaned)
    if length < 2:
        return None, length

    start_year, start_value = cleaned[0]
    end_year, end_value = cleaned[-1]
    periods = end_year - start_year
    if periods <= 0 or start_value <= 0 or end_value <= 0:
        return None, length

    return (end_value / start_value) ** (1 / periods) - 1, length


def cagr_from_returns(observations: Iterable[float]) -> tuple[Optional[float], int]:
    cleaned = list(observations)
    length = len(cleaned)
    if length == 0:
        return None, 0

    gross_return = 1.0
    for value in cleaned:
        if value < -1:
            raise ValueError(f"Return series contains a value below -100%: {value}")
        gross_return *= 1 + value

    return gross_return ** (1 / length) - 1, length


def summarize_country(rows: list[dict[str, str]], year_column: str) -> dict[str, tuple[Optional[float], int]]:
    summary: dict[str, tuple[Optional[float], int]] = {}

    cpi_observations = []
    for row in rows:
        year = parse_year(row.get(year_column, ""))
        value = parse_number(row.get("cpi", ""))
        if year is not None and value is not None:
            cpi_observations.append((year, value))
    summary["cpi"] = cagr_from_index(cpi_observations)

    for series in RETURN_SERIES:
        returns = []
        for row in rows:
            value = parse_number(row.get(series, ""))
            if value is not None:
                returns.append(value)
        summary[series] = cagr_from_returns(returns)

    return summary


def format_rate(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def build_output(rows_by_country: dict[str, list[dict[str, str]]], year_column: str) -> list[dict[str, str]]:
    output = []
    for country in sorted(rows_by_country):
        summary = summarize_country(rows_by_country[country], year_column)
        row = {"country": country}
        for series in SERIES:
            cagr, length = summary[series]
            row[f"{series}_cagr"] = format_rate(cagr)
            row[f"{series}_length"] = str(length)
        output.append(row)
    return output


def read_dataset(csv_path: Path, country_column: str, year_column: str) -> dict[str, list[dict[str, str]]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required_columns = {country_column, year_column, *SERIES}
        missing_columns = sorted(column for column in required_columns if column not in (reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")

        rows_by_country: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            country = (row.get(country_column) or "").strip()
            if country:
                rows_by_country[country].append(row)

    return rows_by_country


def write_csv(rows: list[dict[str, str]], output_path: Optional[Path]) -> None:
    fieldnames = ["country"]
    for series in SERIES:
        fieldnames.extend((f"{series}_cagr", f"{series}_length"))

    if output_path is None:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-country CAGR and data length for selected JST Macro History series."
    )
    parser.add_argument("csv_path", type=Path, help="Path to JSTdatasetR6.csv")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the output CSV. Defaults to stdout.",
    )
    parser.add_argument(
        "--country-column",
        default="country",
        help="Country column name in the input CSV (default: country)",
    )
    parser.add_argument(
        "--year-column",
        default="year",
        help="Year column name in the input CSV (default: year)",
    )
    args = parser.parse_args()

    rows_by_country = read_dataset(args.csv_path, args.country_column, args.year_column)
    output_rows = build_output(rows_by_country, args.year_column)
    write_csv(output_rows, args.output)


if __name__ == "__main__":
    main()
