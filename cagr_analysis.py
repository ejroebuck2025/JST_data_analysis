"""
CAGR analysis by country for the JST Macro History dataset.

Computes, for each country:
  - CPI (inflation) CAGR: derived from the index growth between the first
    and last available year (index is 1990=100).
  - Equity/housing/bond total return and bill rate CAGR (nominal): geometric
    mean of the annual decimal return series.
  - Real (CPI-deflated) equivalents of the above, plus the real price-only
    return on housing (excludes rental yield, i.e. housing_capgain deflated
    by CPI), for comparing asset classes against inflation.

Each series is evaluated independently per country, since available data
length differs by country and by series. The number of years/observations
used for each series is reported alongside its CAGR.

Usage:
    python3 cagr_analysis.py [--csv JSTdatasetR6.csv] [--out cagr_by_country.csv]
"""

import argparse
import numpy as np
import pandas as pd


def cpi_cagr(group: pd.DataFrame) -> tuple[float, int]:
    """CAGR of the CPI index from first to last valid observation."""
    s = group[["year", "cpi"]].dropna()
    if len(s) < 2:
        return np.nan, len(s)
    s = s.sort_values("year")
    start_year, start_val = s.iloc[0]["year"], s.iloc[0]["cpi"]
    end_year, end_val = s.iloc[-1]["year"], s.iloc[-1]["cpi"]
    n_years = end_year - start_year
    if n_years <= 0 or start_val <= 0:
        return np.nan, len(s)
    cagr = (end_val / start_val) ** (1 / n_years) - 1
    return cagr, int(n_years)


def return_series_cagr(group: pd.DataFrame, column: str) -> tuple[float, int]:
    """CAGR of a decimal annual-return series via geometric compounding."""
    s = group[column].dropna()
    if len(s) == 0:
        return np.nan, 0
    growth = (1 + s).prod()
    if growth <= 0:
        return np.nan, len(s)
    cagr = growth ** (1 / len(s)) - 1
    return cagr, len(s)


def real_return_series_cagr(group: pd.DataFrame, column: str) -> tuple[float, int]:
    """CAGR of a nominal annual-return series deflated year-by-year by CPI inflation."""
    sub = group[["year", column, "cpi"]].dropna().sort_values("year")
    if len(sub) < 2:
        return np.nan, 0
    inflation = sub["cpi"].pct_change()
    real_return = (1 + sub[column]) / (1 + inflation) - 1
    real_return = real_return.dropna()
    if len(real_return) == 0:
        return np.nan, 0
    growth = (1 + real_return).prod()
    if growth <= 0:
        return np.nan, len(real_return)
    cagr = growth ** (1 / len(real_return)) - 1
    return cagr, len(real_return)


def main():
    parser = argparse.ArgumentParser(description="Compute per-country CAGR for JST dataset series.")
    parser.add_argument("--csv", default="JSTdatasetR6.csv", help="Path to JSTdatasetR6.csv")
    parser.add_argument("--out", default="cagr_by_country.csv", help="Path to write results CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    rows = []
    for country, group in df.groupby("country"):
        cpi_c, cpi_n = cpi_cagr(group)
        eq_c, eq_n = return_series_cagr(group, "eq_tr")
        housing_c, housing_n = return_series_cagr(group, "housing_tr")
        bond_c, bond_n = return_series_cagr(group, "bond_tr")
        bill_c, bill_n = return_series_cagr(group, "bill_rate")

        real_eq_c, real_eq_n = real_return_series_cagr(group, "eq_tr")
        real_housing_c, real_housing_n = real_return_series_cagr(group, "housing_tr")
        real_bond_c, real_bond_n = real_return_series_cagr(group, "bond_tr")
        real_bill_c, real_bill_n = real_return_series_cagr(group, "bill_rate")
        real_hprice_c, real_hprice_n = real_return_series_cagr(group, "housing_capgain")

        rows.append({
            "country": country,
            "cpi_cagr": cpi_c,
            "cpi_years": cpi_n,
            "eq_tr_cagr": eq_c,
            "eq_tr_years": eq_n,
            "housing_tr_cagr": housing_c,
            "housing_tr_years": housing_n,
            "bond_tr_cagr": bond_c,
            "bond_tr_years": bond_n,
            "bill_rate_cagr": bill_c,
            "bill_rate_years": bill_n,
            "real_eq_tr_cagr": real_eq_c,
            "real_eq_tr_years": real_eq_n,
            "real_housing_tr_cagr": real_housing_c,
            "real_housing_tr_years": real_housing_n,
            "real_bond_tr_cagr": real_bond_c,
            "real_bond_tr_years": real_bond_n,
            "real_bill_rate_cagr": real_bill_c,
            "real_bill_rate_years": real_bill_n,
            "real_housing_price_cagr": real_hprice_c,
            "real_housing_price_years": real_hprice_n,
        })

    result = pd.DataFrame(rows).sort_values("country").reset_index(drop=True)
    result.to_csv(args.out, index=False)

    pct_cols = [
        "cpi_cagr", "eq_tr_cagr", "housing_tr_cagr", "bond_tr_cagr", "bill_rate_cagr",
        "real_eq_tr_cagr", "real_housing_tr_cagr", "real_bond_tr_cagr", "real_bill_rate_cagr",
        "real_housing_price_cagr",
    ]
    display = result.copy()
    for c in pct_cols:
        display[c] = display[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "n/a")

    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(display.to_string(index=False))

    print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    main()
