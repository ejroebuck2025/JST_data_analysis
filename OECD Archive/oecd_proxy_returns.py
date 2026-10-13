"""
OECD proxy CAGR for cash, bonds, and equities -- for countries NOT already
covered by usable JST total-return series.

Data sources (OECD SDMX API, DSD_KEI and DSD_PRICES dataflows):
  - IR3TIB : 3-month interbank rate (%, p.a.)      -> cash/bill proxy
  - IRLT   : 10yr government bond yield (%, p.a.)  -> bond proxy
  - SHARE  : share price index, YoY growth (%)     -> equity PRICE proxy
             (no dividends -- understates true total return)
  - CPI    : national all-items CPI index          -> deflator for real CAGR

Methodology notes:
  - Cash: annual return = average monthly IR3TIB rate for the year (as a
    decimal). CAGR = geometric mean of (1 + annual rate).
  - Bonds: IRLT is a YIELD, not a total return. Buy-at-par-and-hold-to-
    maturity would only capture the coupon and ignores the capital
    gain/loss from year-to-year yield changes -- which historically is a
    large share of realized bond total return (e.g. 2022's bond selloff
    came entirely from yield increases, not coupons). To avoid materially
    understating volatility/return, this script approximates annual total
    return using a constant-duration model:
        total_return_t ~= yield_{t-1} - duration * (yield_t - yield_{t-1})
    with a fixed duration of 7 years (typical modified duration for a
    10yr par bond). This is still an approximation, not a real bond index.
  - Equities: SHARE is a price index only (no dividend reinvestment), so
    its CAGR is a lower-bound proxy for true equity total return.

Country filters:
  - Only countries with more than MIN_YEARS (default 40) of data for the
    given series are included.
  - Countries already well covered by JST for that series are excluded,
    EXCEPT Canada and Ireland, whose JST eq_tr/housing_tr/bond_tr columns
    are entirely empty (unusable) despite being nominally "in" JST.

Usage:
    python3 oecd_proxy_returns.py [--min-years 40] [--duration 7]
"""

import argparse
import io
import urllib.request

import numpy as np
import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0"}

KEI_URL = (
    "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/"
    ".M.{measure}...../?startPeriod=1900&format=csvfile"
)
CPI_URL = (
    "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/"
    ".A.N.CPI.IX._T.N._Z/?startPeriod=1900&format=csvfile"
)

# JST countries with genuinely usable eq_tr/housing_tr/bond_tr series (i.e.
# exclude Canada and Ireland, whose columns are entirely null in JST).
JST_COVERED_ISO = {
    "AUS", "BEL", "DNK", "FIN", "FRA", "DEU", "ITA", "JPN",
    "NLD", "NOR", "PRT", "ESP", "SWE", "CHE", "GBR", "USA",
}


def fetch_csv(url: str) -> pd.DataFrame:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    return pd.read_csv(io.StringIO(raw))


def fetch_kei_measure(measure: str) -> pd.DataFrame:
    df = fetch_csv(KEI_URL.format(measure=measure))
    df = df[df["REF_AREA"] != "REF_AREA"]  # guard against stray header echo
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df["year"] = df["TIME_PERIOD"].str.slice(0, 4).astype(int)
    return df.dropna(subset=["value"])


def fetch_cpi() -> pd.DataFrame:
    df = fetch_csv(CPI_URL)
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df["year"] = df["TIME_PERIOD"].astype(int)
    return df.dropna(subset=["value"])[["REF_AREA", "year", "value"]].rename(columns={"value": "cpi"})


def cash_annual_returns(ir3tib: pd.DataFrame) -> pd.DataFrame:
    """Average monthly 3m rate per year, as a decimal annual return."""
    out = ir3tib.groupby(["REF_AREA", "year"])["value"].mean().reset_index()
    out["cash_return"] = out["value"] / 100
    return out[["REF_AREA", "year", "cash_return"]]


def bond_annual_returns(irlt: pd.DataFrame, duration: float) -> pd.DataFrame:
    """Constant-duration approximate total return from a 10yr par yield."""
    yearly = irlt.groupby(["REF_AREA", "year"])["value"].mean().reset_index()
    yearly = yearly.sort_values(["REF_AREA", "year"])
    yearly["yield_dec"] = yearly["value"] / 100
    yearly["prev_yield"] = yearly.groupby("REF_AREA")["yield_dec"].shift(1)
    yearly["bond_return"] = yearly["prev_yield"] - duration * (yearly["yield_dec"] - yearly["prev_yield"])
    return yearly.dropna(subset=["bond_return"])[["REF_AREA", "year", "bond_return"]]


def equity_annual_returns(share: pd.DataFrame, dividend_yield: float) -> pd.DataFrame:
    """December year-over-year %% growth of the share price index, as a decimal,
    plus an approximate total return using a constant dividend-yield add-on
    (compounded onto the price return, not simply added)."""
    dec = share[(share["TIME_PERIOD"].str.slice(5, 7) == "12") & (share["TRANSFORMATION"] == "GY")].copy()
    dec["eq_price_return"] = dec["value"] / 100
    dec["eq_tr_approx"] = (1 + dec["eq_price_return"]) * (1 + dividend_yield) - 1
    return dec[["REF_AREA", "year", "eq_price_return", "eq_tr_approx"]]


def series_cagr(df: pd.DataFrame, iso: str, col: str, cpi: pd.DataFrame | None = None):
    """Geometric-mean CAGR (and real version if cpi provided) for one country's series."""
    s = df[df["REF_AREA"] == iso].sort_values("year")
    if cpi is not None:
        s = s.merge(cpi[cpi["REF_AREA"] == iso][["year", "cpi"]], on="year", how="inner")
        s["inflation"] = s["cpi"].pct_change()
        s = s.dropna(subset=["inflation"])
        s["real_return"] = (1 + s[col]) / (1 + s["inflation"]) - 1
        real_growth = (1 + s["real_return"]).prod()
        real_cagr = real_growth ** (1 / len(s)) - 1 if len(s) > 0 and real_growth > 0 else np.nan
    else:
        real_cagr = np.nan

    n = len(s)
    if n == 0:
        return np.nan, real_cagr, 0
    nominal_growth = (1 + s[col]).prod()
    nominal_cagr = nominal_growth ** (1 / n) - 1 if nominal_growth > 0 else np.nan
    return nominal_cagr, real_cagr, n


def build_table(annual_returns: pd.DataFrame, col: str, cpi: pd.DataFrame, min_years: int) -> pd.DataFrame:
    rows = []
    for iso in sorted(annual_returns["REF_AREA"].unique()):
        if iso in JST_COVERED_ISO:
            continue  # JST already has a real total-return series for this country
        nominal_cagr, real_cagr, n = series_cagr(annual_returns, iso, col, cpi)
        if n < min_years:
            continue
        rows.append({"country": iso, "nominal_cagr": nominal_cagr, "real_cagr": real_cagr, "years": n})
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def print_table(title: str, table: pd.DataFrame):
    print(f"\n=== {title} ===")
    if table.empty:
        print("(no countries met the >40yr / non-JST-overlap criteria)")
        return
    display = table.copy()
    for c in ["nominal_cagr", "real_cagr"]:
        display[c] = display[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "n/a")
    print(display.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="OECD proxy CAGR for cash/bonds/equities.")
    parser.add_argument("--min-years", type=int, default=40, help="Minimum years of data required")
    parser.add_argument("--duration", type=float, default=7.0, help="Assumed bond duration for the total-return model")
    parser.add_argument("--dividend-yield", type=float, default=0.0407,
                        help="Constant dividend yield add-on for the equity total-return approximation "
                             "(default 4.07%%, the median eq_dp across JST countries)")
    parser.add_argument("--out-prefix", default="oecd_proxy", help="Prefix for output CSV files")
    args = parser.parse_args()

    print("Fetching OECD data...")
    ir3tib = fetch_kei_measure("IR3TIB")
    irlt = fetch_kei_measure("IRLT")
    share = fetch_kei_measure("SHARE")
    cpi = fetch_cpi()

    cash = cash_annual_returns(ir3tib)
    bonds = bond_annual_returns(irlt, args.duration)
    equities = equity_annual_returns(share, args.dividend_yield)

    cash_table = build_table(cash, "cash_return", cpi, args.min_years)
    bond_table = build_table(bonds, "bond_return", cpi, args.min_years)
    equity_price_table = build_table(equities, "eq_price_return", cpi, args.min_years)
    equity_tr_table = build_table(equities, "eq_tr_approx", cpi, args.min_years)

    print_table("Cash / bill proxy (IR3TIB, average annual rate)", cash_table)
    print_table(f"Bond proxy (IRLT, duration={args.duration} total-return model)", bond_table)
    print_table("Equity price proxy (SHARE index, NO dividends)", equity_price_table)
    print_table(f"Equity approx. total return (SHARE + {args.dividend_yield:.2%} constant dividend yield)", equity_tr_table)

    cash_table.to_csv(f"{args.out_prefix}_cash.csv", index=False)
    bond_table.to_csv(f"{args.out_prefix}_bonds.csv", index=False)
    equity_price_table.to_csv(f"{args.out_prefix}_equities_price.csv", index=False)
    equity_tr_table.to_csv(f"{args.out_prefix}_equities_tr_approx.csv", index=False)
    print(
        f"\nSaved {args.out_prefix}_cash.csv, {args.out_prefix}_bonds.csv, "
        f"{args.out_prefix}_equities_price.csv, {args.out_prefix}_equities_tr_approx.csv"
    )


if __name__ == "__main__":
    main()
