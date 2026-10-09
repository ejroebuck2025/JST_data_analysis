# JST_data_analysis

`/home/runner/work/JST_data_analysis/JST_data_analysis/analyze_jst_cagr.py` reads the JST Macro History CSV and outputs per-country CAGR plus the observation count used for each requested series:

- `cpi` (inflation index, annualized from first and last available observation using the year span)
- `eq_tr`
- `housing_tr`
- `bond_tr`
- `bill_rate`

Usage:

```bash
python /home/runner/work/JST_data_analysis/JST_data_analysis/analyze_jst_cagr.py /absolute/path/to/JSTdatasetR6.csv --output /absolute/path/to/jst_cagr_by_country.csv
```

If `--output` is omitted, the result is written to stdout as CSV.
