"""
Real house price CAGR by country, sourced from the OECD Analytical House
Price Indicators database (via the OECD SDMX API).

This extends housing price-return coverage beyond the 18 JST countries to
everything OECD publishes a real house price index (RPI) for (~35+
countries/areas). Note this is PRICE return only (no rental yield), since
OECD does not publish rent-inclusive housing total-return series, nor
equity/bond/cash total-return series -- those remain best sourced from JST
(see cagr_analysis.py) or a paid vendor (DMS/GFD) for broader coverage.

Usage:
    python3 oecd_real_house_prices.py [--out oecd_real_house_prices.csv]
"""

import argparse
import io
import urllib.request

import numpy as np
import pandas as pd

OECD_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,1.0/"
    ".A.RPI./?format=csvfile"
)

# Aggregate/non-country areas to exclude from a per-country table.
NON_COUNTRY_AREAS = {"EA", "EA17", "EA19", "EA20", "OECD", "G7", "G20", "EU27_2020"}


def fetch_real_house_price_index() -> pd.DataFrame:
    request = urllib.request.Request(OECD_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    return pd.read_csv(io.StringIO(raw))


def cagr_from_index(sub: pd.DataFrame) -> tuple[float, int]:
    sub = sub.dropna(subset=["OBS_VALUE"]).sort_values("TIME_PERIOD")
    if len(sub) < 2:
        return np.nan, len(sub)
    start_year, start_val = int(sub.iloc[0]["TIME_PERIOD"]), sub.iloc[0]["OBS_VALUE"]
    end_year, end_val = int(sub.iloc[-1]["TIME_PERIOD"]), sub.iloc[-1]["OBS_VALUE"]
    n_years = end_year - start_year
    if n_years <= 0 or start_val <= 0:
        return np.nan, len(sub)
    cagr = (end_val / start_val) ** (1 / n_years) - 1
    return cagr, n_years


def main():
    parser = argparse.ArgumentParser(description="Real house price CAGR by country (OECD data).")
    parser.add_argument("--out", default="oecd_real_house_prices.csv", help="Path to write results CSV")
    args = parser.parse_args()

    df = fetch_real_house_price_index()
    df = df[~df["REF_AREA"].isin(NON_COUNTRY_AREAS)]

    rows = []
    for area, group in df.groupby("REF_AREA"):
        cagr, n_years = cagr_from_index(group)
        rows.append({"country": area, "real_housing_price_cagr": cagr, "years": n_years})

    result = pd.DataFrame(rows).sort_values("country").reset_index(drop=True)
    result.to_csv(args.out, index=False)

    display = result.copy()
    display["real_housing_price_cagr"] = display["real_housing_price_cagr"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else "n/a"
    )
    with pd.option_context("display.max_rows", None, "display.width", 100):
        print(display.to_string(index=False))

    print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    main()
