"""Build country/asset CAGR tables and USD-denominated return correlations."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


JST_ASSETS = {
    "Equities (total return)": "eq_tr",
    "Housing (total return)": "housing_tr",
    "Government bonds (total return)": "bond_tr",
    "Bills": "bill_rate",
}
NYU_ASSETS = {
    "US Small cap (bottom decile)": "Small-cap equities",
    "Baa Corporate Bond": "Corporate bonds (Baa)",
    "Gold*": "Gold",
}
COUNTRY_START_YEARS = {"DEU": 1950}
GOLD_FIRST_YEAR = 1972
DEFAULT_MIN_CORRELATION_YEARS = 20
DEVELOPED_MARKET_COUNTRY = "Developed Market"
DEVELOPED_MARKET_ISO = "DM"
DEVELOPED_MARKET_PERIODS = {"Since 1900": 1900, "Since 1950": 1950, "Since 1975": 1975}


def read_jst_returns(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["year", "country", "iso", "cpi", "xrusd", *JST_ASSETS.values()]
    source = pd.read_csv(path, usecols=columns)
    source["year"] = pd.to_numeric(source["year"], errors="raise").astype(int)
    for column in ["cpi", "xrusd", *JST_ASSETS.values()]:
        source[column] = pd.to_numeric(source[column], errors="coerce")

    source = source.sort_values(["iso", "year"]).copy()
    grouped = source.groupby("iso", sort=False)
    source["previous_year"] = grouped["year"].shift()
    source["previous_xrusd"] = grouped["xrusd"].shift()
    source["usd_fx_factor"] = source["previous_xrusd"] / source["xrusd"]
    consecutive = source["year"].eq(source["previous_year"] + 1)
    valid_fx = source["xrusd"].gt(0) & source["previous_xrusd"].gt(0) & consecutive
    source.loc[~valid_fx, "usd_fx_factor"] = np.nan

    inflation = source[["year", "iso", "cpi"]].copy()
    inflation["previous_year"] = grouped["year"].shift()
    inflation["previous_cpi"] = grouped["cpi"].shift()
    inflation["inflation"] = inflation["cpi"] / inflation["previous_cpi"] - 1
    valid_cpi = (
        inflation["year"].eq(inflation["previous_year"] + 1)
        & inflation["cpi"].gt(0)
        & inflation["previous_cpi"].gt(0)
    )
    inflation.loc[~valid_cpi, "inflation"] = np.nan
    inflation = inflation[["year", "iso", "inflation"]]

    panel = source.melt(
        id_vars=["year", "country", "iso", "usd_fx_factor"],
        value_vars=list(JST_ASSETS.values()),
        var_name="source_series",
        value_name="annual_return",
    ).dropna(subset=["annual_return"])
    labels = {series: label for label, series in JST_ASSETS.items()}
    panel["asset_class"] = panel["source_series"].map(labels)
    panel["source"] = "JST"
    panel["return_currency"] = "local currency"
    panel["return_usd"] = (1 + panel["annual_return"]) * panel["usd_fx_factor"] - 1
    panel.loc[panel["iso"].eq("USA"), "return_usd"] = panel.loc[
        panel["iso"].eq("USA"), "annual_return"
    ]
    panel = panel.drop(columns=["source_series", "usd_fx_factor"])
    panel = apply_country_cutoffs(panel)
    return panel, inflation


def read_nyu_returns(path: str | Path) -> pd.DataFrame:
    source = pd.read_csv(path, header=1)
    source.columns = source.columns.str.strip()
    source["Year"] = pd.to_numeric(source["Year"], errors="coerce")
    rows = []
    for nyu_column, asset_class in NYU_ASSETS.items():
        values = source[nyu_column].astype("string").str.strip().str.replace("%", "", regex=False)
        annual_return = pd.to_numeric(values, errors="coerce") / 100
        rows.append(pd.DataFrame({
            "year": source["Year"].astype("Int64"),
            "country": "United States",
            "iso": "USA",
            "asset_class": asset_class,
            "annual_return": annual_return,
            "source": "NYU Stern",
            "return_currency": "USD",
            "return_usd": annual_return,
        }))

    panel = pd.concat(rows, ignore_index=True).dropna(subset=["year", "annual_return"])
    panel["year"] = panel["year"].astype(int)
    panel = panel.loc[~panel["asset_class"].eq("Gold") | panel["year"].ge(GOLD_FIRST_YEAR)]
    return panel


def apply_country_cutoffs(panel: pd.DataFrame) -> pd.DataFrame:
    first_year = panel["iso"].map(COUNTRY_START_YEARS).fillna(-np.inf)
    return panel.loc[panel["year"].ge(first_year)].copy()


def summarize_returns(panel: pd.DataFrame, value_column: str, basis: str) -> pd.DataFrame:
    group_columns = ["country", "iso", "asset_class", "source", "return_currency"]
    rows = []
    for keys, group in panel.groupby(group_columns, sort=True, dropna=False):
        values = group.dropna(subset=[value_column]).sort_values("year")
        returns = values[value_column].to_numpy(dtype=float)
        if len(returns) and np.all(1 + returns > 0):
            cagr = np.expm1(np.log1p(returns).mean())
        else:
            cagr = np.nan
        standard_deviation = values[value_column].std(ddof=1) if len(values) > 1 else np.nan
        rows.append({
            "country": keys[0],
            "iso": keys[1],
            "asset_class": keys[2],
            "source": keys[3],
            "return_currency": keys[4],
            "return_basis": basis,
            "cagr": cagr,
            "standard_deviation": standard_deviation,
            "first_year": int(values["year"].min()) if len(values) else pd.NA,
            "last_year": int(values["year"].max()) if len(values) else pd.NA,
            "observations": len(values),
            "countries_averaged": 1,
        })
    return pd.DataFrame(rows)


def add_developed_market_cagrs(summary: pd.DataFrame) -> pd.DataFrame:
    developed_rows = []
    jst_country_rows = summary.loc[
        summary["source"].eq("JST") & summary["iso"].ne(DEVELOPED_MARKET_ISO)
    ]
    for basis, basis_group in jst_country_rows.groupby("return_basis"):
        for asset_class in JST_ASSETS:
            contributors = basis_group.loc[
                basis_group["asset_class"].eq(asset_class) & basis_group["cagr"].notna()
            ]
            if contributors.empty:
                continue
            developed_rows.append({
                "country": DEVELOPED_MARKET_COUNTRY,
                "iso": DEVELOPED_MARKET_ISO,
                "asset_class": asset_class,
                "source": "JST country-CAGR average",
                "return_currency": "mixed country returns",
                "return_basis": basis,
                "cagr": contributors["cagr"].mean(),
                "standard_deviation": contributors["standard_deviation"].mean(),
                "first_year": contributors["first_year"].min(),
                "last_year": contributors["last_year"].max(),
                "observations": pd.NA,
                "countries_averaged": len(contributors),
            })
    return pd.concat([summary, pd.DataFrame(developed_rows)], ignore_index=True)


def developed_market_usd_returns(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    jst = panel.loc[panel["source"].eq("JST")]
    grouped = jst.groupby(["year", "asset_class"])["return_usd"]
    averages = grouped.mean().rename("return_usd").reset_index()
    counts = grouped.count().unstack("asset_class").sort_index(axis=1)
    averages["country"] = DEVELOPED_MARKET_COUNTRY
    averages["iso"] = DEVELOPED_MARKET_ISO
    averages["source"] = "JST cross-country mean"
    averages["return_currency"] = "USD"
    return averages, counts


def developed_market_period_summary(panel: pd.DataFrame) -> pd.DataFrame:
    jst = panel.loc[panel["source"].eq("JST") & panel["iso"].ne(DEVELOPED_MARKET_ISO)]
    rows = []
    for period, start_year in DEVELOPED_MARKET_PERIODS.items():
        for asset_class in JST_ASSETS:
            row = {"period": period, "period_start_year": start_year, "asset_class": asset_class}
            for basis, value_column in (("nominal", "annual_return"), ("real", "real_return")):
                country_summaries = []
                asset_data = jst.loc[jst["asset_class"].eq(asset_class)]
                for (iso, country), group in asset_data.groupby(["iso", "country"], sort=True):
                    country_start = max(start_year, COUNTRY_START_YEARS.get(iso, start_year))
                    values = group.loc[group["year"].ge(country_start)].dropna(subset=[value_column])
                    returns = values[value_column].to_numpy(dtype=float)
                    if len(returns) < 2 or not np.all(1 + returns > 0):
                        continue
                    country_summaries.append({
                        "cagr": np.expm1(np.log1p(returns).mean()),
                        "standard_deviation": values[value_column].std(ddof=1),
                        "first_year": int(values["year"].min()),
                        "last_year": int(values["year"].max()),
                        "observations": len(values),
                        "country": country,
                    })

                summaries = pd.DataFrame(country_summaries)
                prefix = f"{basis}_"
                if summaries.empty:
                    row.update({
                        f"{prefix}cagr": np.nan,
                        f"{prefix}standard_deviation": np.nan,
                        f"{prefix}countries_averaged": 0,
                        f"{prefix}mean_observations": np.nan,
                        f"{prefix}first_year": pd.NA,
                        f"{prefix}last_year": pd.NA,
                    })
                    continue
                row.update({
                    f"{prefix}cagr": summaries["cagr"].mean(),
                    f"{prefix}standard_deviation": summaries["standard_deviation"].mean(),
                    f"{prefix}countries_averaged": len(summaries),
                    f"{prefix}mean_observations": summaries["observations"].mean(),
                    f"{prefix}first_year": summaries["first_year"].min(),
                    f"{prefix}last_year": summaries["last_year"].max(),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def build_correlation_matrices(
    panel: pd.DataFrame, min_observations: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = panel.dropna(subset=["return_usd"]).copy()
    usable["series"] = (
        usable["country"] + " (" + usable["iso"] + ") | "
        + usable["asset_class"] + " [" + usable["source"] + "]"
    )
    series_order = (
        usable[["series", "asset_class", "country", "iso", "source"]]
        .drop_duplicates()
        .sort_values(["asset_class", "country", "iso", "source"])["series"]
        .tolist()
    )
    annual = usable.pivot(index="year", columns="series", values="return_usd")
    annual = annual.reindex(columns=series_order).sort_index()
    correlations = annual.corr(method="pearson", min_periods=min_observations)
    present = annual.notna().astype(int)
    observation_counts = present.T.dot(present)
    return correlations, observation_counts


def build_outputs(
    jst_path: str | Path,
    nyu_path: str | Path,
    output_dir: str | Path,
    min_correlation_years: int = DEFAULT_MIN_CORRELATION_YEARS,
) -> dict[str, pd.DataFrame]:
    jst_panel, inflation = read_jst_returns(jst_path)
    nyu_panel = read_nyu_returns(nyu_path)
    panel = pd.concat([jst_panel, nyu_panel], ignore_index=True)

    key = ["iso", "year", "asset_class", "source"]
    if panel.duplicated(key).any():
        raise ValueError("Return panel contains duplicate country/year/asset/source observations")

    panel = panel.merge(inflation, on=["iso", "year"], how="left", validate="many_to_one")
    panel["real_return"] = (1 + panel["annual_return"]) / (1 + panel["inflation"]) - 1
    nominal = add_developed_market_cagrs(summarize_returns(panel, "annual_return", "nominal"))
    real = add_developed_market_cagrs(summarize_returns(panel, "real_return", "domestic purchasing power"))
    period_summary = developed_market_period_summary(panel)
    developed_usd, developed_country_counts = developed_market_usd_returns(panel)
    correlation_panel = pd.concat([panel, developed_usd], ignore_index=True)
    correlations, correlation_counts = build_correlation_matrices(correlation_panel, min_correlation_years)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "nominal_total_returns_by_country.csv": nominal,
        "real_total_returns_by_country.csv": real,
        "usd_return_correlation_matrix.csv": correlations,
        "usd_return_correlation_observations.csv": correlation_counts,
        "developed_market_annual_country_counts.csv": developed_country_counts,
        "developed_market_period_summary.csv": period_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            output_dir / filename,
            index=(
                filename.startswith("usd_return_correlation_")
                or filename == "developed_market_annual_country_counts.csv"
            ),
        )
    return {
        "nominal": nominal,
        "real": real,
        "correlations": correlations,
        "correlation_observations": correlation_counts,
        "developed_market_annual_country_counts": developed_country_counts,
        "developed_market_period_summary": period_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JST/NYU country return tables and USD correlations.")
    parser.add_argument("--csv", default="JSTdatasetR6.csv", help="Path to the JST dataset")
    parser.add_argument(
        "--nyu-csv", default="NYU Stern Returns by year-Table 1.csv", help="Path to NYU Stern annual returns"
    )
    parser.add_argument("--out-dir", default=".", help="Directory for generated CSV tables")
    parser.add_argument(
        "--min-correlation-years", type=int, default=DEFAULT_MIN_CORRELATION_YEARS,
        help="Minimum pairwise annual observations required for a correlation",
    )
    args = parser.parse_args()
    if args.min_correlation_years < 2:
        parser.error("--min-correlation-years must be at least 2")

    tables = build_outputs(args.csv, args.nyu_csv, args.out_dir, args.min_correlation_years)
    for name, table in tables.items():
        print(f"{name}: {len(table)} rows")
    print(f"Saved output CSVs to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
