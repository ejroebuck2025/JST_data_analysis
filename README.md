# JST Data Analysis

Build country-by-asset return summaries and a cross-country correlation matrix from the JST Macrohistory dataset, supplemented with non-overlapping annual returns from the NYU Stern historical returns table.

## Run

Requires Python 3, pandas, and NumPy.

```sh
python3 cagr_analysis.py
```

Optional arguments:

```sh
python3 cagr_analysis.py --csv JSTdatasetR6.csv \
	--nyu-csv "NYU Stern Returns by year-Table 1.csv" \
	--out-dir results --min-correlation-years 20
```

## Outputs

- `nominal_total_returns_by_country.csv`: geometric mean annual nominal total returns and sample standard deviations by country and asset, with source, currency basis, first/last year, and observation count.
- `real_total_returns_by_country.csv`: geometric mean annual CPI-adjusted returns and sample standard deviations after deflating each annual return by that country's inflation.
- `usd_return_correlation_matrix.csv`: pairwise Pearson correlations across each country/asset annual return series, expressed in USD.
- `usd_return_correlation_observations.csv`: paired observation counts for each correlation coefficient.
- `developed_market_annual_country_counts.csv`: number of JST countries contributing to each annual Developed Market USD return, by asset class.
- `developed_market_period_summary.csv`: nominal and real average country CAGR and standard deviation for Developed Market JST asset classes since 1900, 1950, and 1975.

The existing `cagr_by_country.csv` is a legacy output and is not refreshed by this workflow.

CAGRs are calculated as the geometric mean of valid annual returns. `standard_deviation` is the sample standard deviation (`ddof=1`) of the same annual return observations used for that row's CAGR; it is blank when fewer than two observations are available. Values are decimals (for example, `0.10` means 10%). The reported first/last year and observation count make missing years visible; they are not imputed. Real returns are calculated as `(1 + nominal return) / (1 + CPI inflation) - 1` and use the CPI observation for the return year and its prior calendar year. Developed Market rows in the return files are the arithmetic mean of the country CAGRs and country standard deviations for that asset and return basis. `countries_averaged` gives the number of country summaries used; `observations` is blank because the aggregate averages country summaries rather than representing one common annual return series.

The period-summary CSV contains one row per start period and JST asset class. For each period, country series are restricted to years at or after its start year (and each country's configured cutoff); the file then reports the arithmetic mean of country CAGRs and country sample standard deviations. The country counts, mean observation counts, and earliest/latest observation years expose differences in coverage. Real returns use domestic CPI.

## Sources And Coverage

JST contributes equities (`eq_tr`), housing (`housing_tr`), government bonds (`bond_tr`), and bills (`bill_rate`) for each country where the series is available. NYU Stern contributes U.S. small-cap stocks, Baa corporate bonds, and gold. NYU S&P 500, Treasury bonds, and bills are excluded because they duplicate JST asset classes; NYU real estate is excluded as a duplicate housing category. Gold is included beginning in 1972, after 1971. Germany's JST observations begin in 1950; other JST histories use their available data.

For each JST asset class, the Developed Market correlation series is the equal-weight arithmetic mean of the available member-country annual returns after conversion to USD. The country membership can vary by year; `developed_market_annual_country_counts.csv` reports the number of contributing countries. The 16 JST countries with usable core return data contribute; Canada and Ireland are absent because their JST asset-return series are empty. NYU U.S. small-cap, Baa corporate-bond, and gold series remain separate and are not folded into the Developed Market averages. Correlation columns and rows are ordered by asset class, then country name A-Z.

JST returns are nominal local-currency returns. For USD correlations, the script uses the documented `xrusd` quote (local currency per USD):

```text
USD gross return = local gross return * previous xrusd / current xrusd
```

The U.S. JST series and the NYU U.S. series are already in USD. JST documentation identifies `xrusd` as the local-currency-per-USD exchange rate, but source notes show a mix of annual-average and end-of-period rates. The USD conversion is therefore an annual-frequency approximation. Correlations use pairwise available years; coefficients with fewer than the configured minimum (20 by default) are left blank. The companion count matrix shows the sample size for every pair.

No outlier returns are winsorized or removed. Country-specific cutoff years are explicit in `cagr_analysis.py`; add a cutoff there when an earlier historical period is unsuitable for comparison. NYU returns are read from the annual-return columns in the local CSV; the file's first row is a title row, followed by the data headers. Third-party series from the archived OECD data are not included in these outputs.