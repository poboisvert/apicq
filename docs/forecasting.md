# Forecasting

Ridge forecast of Île de Montréal **Copropriété** and **Plex (2-5)** from APCIQ monthly figures plus Québec / Canada macro series. Default horizon is **9 months**. The script is still named `forecast_six_months.py` from an earlier default.

This is not an official APCIQ forecast.

The live model does **not** fit on an expanding 24-month sample. For the September 2026 origin (last actual = août 2026) it fits two calendar windows:

1. **Année précédente + 3 mois** — all of 2025, plus juin–août 2026.
2. **Année précédente + 9 mois de l’année** — all of 2025, plus the last 9 months of 2026 that exist (janv.–août 2026).

Lags 1 / 2 / 3 / 12 still read earlier APCIQ months. Only the Ridge *targets* are windowed. The 24-month rule lives in the [backtest](backtesting.md): it is burn-in before the first origin, not the training set.

## Files

| Path | Role |
|---|---|
| `scripts/forecast_six_months.py` | Fit both windows and write the live 9-month forecast |
| `scripts/backtest_forecast.py` | Walk-forward test of those same windows vs later APCIQ actuals |
| `data/ile-montreal.json` | Housing history (input) |
| `data/forecast-data.json` | Latest forecast (both windows) |
| `data/backtest.json` | Walk-forward metrics per window, overlay series, scored points |
| `data/backtest-points.csv` | Same points as a table (`window`, origin, target, predicted, actual, error) |
| `ile-montreal.html` | Dashboard. Inline `FORECAST` and `BACKTEST` blobs so `file://` works |

Refresh:

```bash
source .venv/bin/activate
python scripts/forecast_six_months.py --horizon 9
python scripts/backtest_forecast.py --min-train 24 --horizon 9
```

Both need network (StatCan WDS and Bank of Canada Valet). Each run patches `ile-montreal.html` between `// FORECAST_START` / `// FORECAST_END` and `// BACKTEST_START` / `// BACKTEST_END`.

## Train windows

`window_train_keys(housing, origin_key, window)` returns the target months used to fit. At origin *T* in year *Y*:

| `window` | Targets |
|---|---|
| `prior_plus_3` | Every month of *Y*−1, plus the last 3 months of *Y* that are ≤ *T* |
| `prior_plus_9` | Every month of *Y*−1, plus the last 9 months of *Y* that are ≤ *T* |
| `expanding` | Every month ≤ *T* (optional backtest only) |

September 2026 example (origin août 2026, selected year 2026):

| Window | Caption | *n* targets |
|---|---|---|
| `prior_plus_3` | janv.–déc. 2025 et juin–août 2026 | 15 |
| `prior_plus_9` | janv.–déc. 2025 et janv.–août 2026 | 20 |

If the selected year is incomplete, “last 9 months” means however many months of that year exist up to the origin.

## What the model does

One `RidgeCV` + `StandardScaler` pipeline **per window × category × series** (`ventes`, `inscriptions`, `prix`, `jours`). Recursive: each predicted month is fed back as a lag for the next.

Features for month *t*:

- seasonality: `sin` / `cos` of the calendar month, plus a time trend
- lags 1, 2, 3, and 12 of the housing series
- Québec employment, unemployment, CPI, housing starts; Canada monthly GDP (t−1)
- overnight rate, 5-year posted mortgage, 5-year GoC bond (t−1; rates persist, other macros are projected)
- April/May year-over-year ventes and prix (slowing-market flag)

After the raw Ridge value:

1. **Spring bias** — if April or May ventes were lower than the year before, cap a ventes rebound and do not let prix keep rising when volume is leading.
2. **Clip** — inscriptions 82–122 % of last actual; prix 85–112 % of last actual; ventes 55–145 % of last actual (min 40); jours 20–140.

Décembre and janvier stay in the model and in the monthly PDF table. They are dropped from charts, the forecast table, and backtest scores (listings expire at year-end and renew in February).

## `scripts/forecast_six_months.py`

Entry: `main()` → `build_forecast()` → `run_forecast()` twice (once per live window) → `patch_html()`.

### Data

| Function | What it does |
|---|---|
| `load_housing` | Read `data/ile-montreal.json` into month rows with a `period_key` |
| `fetch_statcan` | Québec employment / unemployment, CPI, housing starts, Canada monthly GDP |
| `fetch_boc` | Overnight, 5-year mortgage, 5-year bond; last observation of each month |
| `ffill_series` | Forward-fill a macro series onto a calendar |
| `forecast_econ_series` | Project macros beyond the origin. Rates persist; others get a small Ridge on seasonality + lag 1 / 12 |

### Time keys

`period_key(year, month)` is `year * 12 + month`. `key_to_year_month` reverses it. `add_months` / `month_label` build the `09/26` labels used in JSON and the dashboard.

### Features and fit

| Function | What it does |
|---|---|
| `month_features` | sin, cos, trend |
| `target_history` | `{period_key: value}` for one category and field |
| `window_train_keys` | Target months for `prior_plus_3` / `prior_plus_9` / `expanding` |
| `window_meta` | Caption, *n*, and month list for the dashboard |
| `design_matrix` | Rows with lags + macros + spring flags. Training requires the target; a one-step predict does not |
| `ridge_pipeline` | `StandardScaler` then `RidgeCV(alphas=0.1, 1, 10, 100, 1000)` |
| `cv_mae` | Time-series CV MAE on the training window |
| `clip_forecast` | Bound a raw prediction to the last actual |

If lags or macros are missing (for example around mars 2023), `design_matrix` returns no rows for a predict step and `run_forecast` falls back to the last actual.

### Spring pulse

| Function | What it does |
|---|---|
| `build_spring_years` | April/May ventes and prix vs the same months a year earlier |
| `latest_spring` | Most recent comparable spring |
| `spring_features` | Four flags on the design matrix (`slow`, ventes YoY, prix YoY, `price_lag`) |
| `apply_spring_bias` | Post-model cap on ventes / prix / jours when spring was weak |
| `spring_reading` | French sentence for the dashboard callout |

`price_lag` is true when spring is slow, ventes YoY is negative, and prix YoY is still about flat or up. Spring flags use all months known at the origin, not only the train window.

### Forecast run

| Function | What it does |
|---|---|
| `run_forecast` | Truncate housing and macros at `origin_key`, project the horizon, fit each series on `window`, recurse |
| `window_forecast_block` | Slice of one window for `FORECAST.windows` |
| `build_forecast` | Live run: origin = last APCIQ month, both live windows |
| `patch_html` | Replace `const FORECAST` in `ile-montreal.html` |

`run_forecast` is also the backtest engine: the backtest calls it at every origin with only data known at that date, so future GDP or CPI is never used as a feature.

## `scripts/backtest_forecast.py`

Walk-forward test of both Ridge windows vs later APCIQ actuals. Full walk, functions, JSON shape, and dashboard hooks: [backtesting.md](backtesting.md).

## Dashboard (`ile-montreal.html`)

| Marker / function | What it does |
|---|---|
| `const FORECAST` | Live 9-month path for both windows; top-level `forecast` is the default window |
| `renderForecastCompare` | Side-by-side +3 vs +9 table (both windows) |
| `forecastLine` | Dotted Chart.js series from last actual through the horizon |
| `chartForecastLabels` | Horizon labels with déc. / janv. removed |
| `renderForecastCompare` | Side-by-side +3 vs +9 table |
| `renderForecastTables` | Spring pulse + monthly forecast rows + CV MAE |
| `const BACKTEST` | Overlay APCIQ vs 3-month vs 9-month, accuracy scores |
| `renderBacktest` | KPIs and charts for the selected window; compare table for both |

Buy zones on the inscriptions chart (`condoInventoryPath`, souple ≥ 90 implied days, ferme ≥ 120) are a separate path after the 9-month Ridge. They are not part of the sklearn forecast. They follow the selected window’s inscriptions path.

## Macro series IDs

StatCan WDS `getDataFromVectorByReferencePeriodRange`, 2018-01-01 → 2027-12-31:

| Key | Vector | Table |
|---|---|---|
| `qc_employment` | 2063756 | 14-10-0287-01 |
| `qc_unemp_rate` | 2063760 | 14-10-0287-01 |
| `ca_gdp` | 65201210 | 36-10-0434-01 |
| `qc_cpi` | 41691783 | 18-10-0004-01 |
| `qc_starts` | 52300163 | 34-10-0158-01 |

Bank of Canada Valet: `V39079` (overnight), `V80691335` (5-year posted mortgage), `BD.CDN.5YR.DQ.YLD` (5-year bond).
