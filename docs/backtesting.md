# Backtesting

Walk-forward test of the Île de Montréal Ridge forecast against later APCIQ (Centris) actuals. The script is `scripts/backtest_forecast.py`. It does **not** invent a second model: at each origin it calls `run_forecast()` from `scripts/forecast_six_months.py` with only data known at that date.

See [forecasting.md](forecasting.md) for the Ridge features, spring bias, clips, and the two train windows.

## What it tests

From **sept. 2019**, wait until there are **at least 24 months** of **inscriptions** (inventory) and **prix médian** for both Copropriété and Plex. That wait is **burn-in**, so lag-12 and a prior calendar year exist. It is not the Ridge training sample. First origin is **sept. 2021**. Then, month by month through **juil. 2026**:

1. Pretend the latest APCIQ PDF is that origin month.
2. Refit the Ridge twice: **prior year + last 3 months** of the origin year, and **prior year + last 9 months** of the origin year.
3. Forecast 1–9 months ahead (macros beyond the origin are projected, not peeked).
4. Compare predicted **inscriptions** and **prix** to the later Centris figure.

Ventes and jours are fitted inside `run_forecast` (same as production) but are **not scored** here. Décembre and janvier are predicted, then dropped from MAE / MAPE and from the overlay charts (same rule as the rest of the dashboard).

Baselines on the same points:

- **persist** — last actual at the origin, held constant
- **seasonal naïve** — same calendar month a year earlier (`key - 12`), else persist

## Run

```bash
source .venv/bin/activate
python scripts/backtest_forecast.py
python scripts/backtest_forecast.py --min-train 24 --horizon 9
python scripts/backtest_forecast.py --windows prior_plus_3,prior_plus_9
python scripts/backtest_forecast.py --skip-html
```

Needs network (StatCan and Bank of Canada). Writes:

| File | Contents |
|---|---|
| `data/backtest.json` | Per-window metrics, overlay series, `compare`, and every scored point |
| `data/backtest-points.csv` | One row per window × origin × target month × category × field × horizon |
| `ile-montreal.html` | `const BACKTEST` between `// BACKTEST_START` and `// BACKTEST_END` (without `points`) |

The dashboard default window is the one with the higher copro inscriptions 3-month accuracy.

Last run used **58 origins** (09/21 → 07/26). Copro inscriptions exactitude vs APCIQ: **92 % à 3 mois**, **89–90 % à 9 mois**. Copro prix: **97 % à 3 mois**, **96 % à 9 mois**.

## How `main()` walks forward

```
load_housing()
collect_origins(min_train=24)          # burn-in only
fetch_statcan() + fetch_boc()           # once; truncated per origin inside run_forecast
for each window in prior_plus_3, prior_plus_9:
    for each origin:
        run_forecast(..., window=window)
        for copropriete / plex × inscriptions / prix:
            for step 0..8:
                predicted vs actual; persist and seasonal naïve
    summarize by horizon and by year (h=1)
pick default window from copro inscriptions h=1 MAE
write JSON, CSV, HTML
```

A missing actual (avril 2020, mars 2023) is skipped. If `run_forecast` raises (not enough overlapping lags around a gap), that origin is skipped.

## Functions

All live in `scripts/backtest_forecast.py` unless noted.

| Function | What it does |
|---|---|
| `main` | CLI (`--min-train`, `--horizon`, `--windows`, `--skip-html`), loop, files |
| `coverage_ok` | True when **both** categories have ≥ `min_train` non-null inscriptions **and** prix |
| `collect_origins` | Months that pass `coverage_ok` and still have at least one later actual to score |
| `walk_window` | One train window: origins, `run_forecast`, metrics, overlay series |
| `compare_windows` | Side-by-side h=1 / h=9 MAE for the dashboard compare table |
| `pick_default_window` | Lower copro inscriptions h=1 MAE |
| `actual_map` | `{period_key: value}` via `forecast_six_months.target_history` |
| `run_forecast` | Imported. Truncates housing/macros at `origin_key`, fits the requested window |
| `naive_seasonal` | Same month last year, else last origin value |
| `skip_chart_month` | `True` for January and December |
| `summarize` | n, MAE, RMSE, MAPE, bias on `(predicted, actual)` pairs |
| `align_series` | Align a `{label: value}` map to the chart timeline (null if missing) |
| `dashboard_payload` | Copy of the JSON **without** `points` (keeps the HTML blob small) |
| `patch_html` | Replace `const BACKTEST = ...` in `ile-montreal.html` |

### `coverage_ok(rows, min_train)`

Counts months in `rows` with a non-null `inscriptions` and a non-null `prix` for copropriété **and** plex. Both must be ≥ `min_train` (default 24). Prix starts août 2019; inscriptions start sept. 2019. With the avril 2020 gap, the 24th month that has both series is sept. 2021.

### `collect_origins(housing, min_train)`

Walks the housing list in order. At index `i`, the history to date is `housing[:i+1]`. If that history is not ready for burn-in, skip. If no later month exists, skip (nothing to score). Remaining months become origins.

### `run_forecast` at an origin

Imported from `forecast_six_months`. Housing rows with `key > origin_key` are dropped. Macro series are cut at the origin, then projected for the next 9 months (rates persist; GDP / CPI / employment / starts get a small Ridge). Spring flags use only Aprils and Mays already in the window. Ridge *targets* are the prior calendar year plus the last 3 or 9 months of the origin year.

The recursive path can fall back to the last actual when lags are missing (mars 2023 gap: May’s lag-2 is March).

### `summarize(pairs)`

For each `(predicted, actual)`:

- `error` = predicted − actual
- **MAE** — mean absolute error
- **RMSE** — root mean square error
- **MAPE** — mean `|pred − actual| / |actual|` × 100 (skips actual = 0)
- **bias** — mean error (negative = forecast too low)

Only pairs whose target month is not décembre/janvier go into `scored`, `naive_persist`, `naive_yoy`, and `by_year`.

### `naive_seasonal(history, key, last)`

Looks up `history[key - 12]`. If that month is missing, uses `last` (the origin’s actual). Persist is always the origin’s actual, for every horizon.

## JSON shape (`data/backtest.json`)

- `default_window` — `prior_plus_3` or `prior_plus_9`
- `compare[category][field][window]` — h=1 / h=9 MAE and MAPE for the dashboard compare table
- `windows[id].headline` — h=1 / h=9 MAE and MAPE per category and field
- `windows[id].metrics[category][field].by_horizon["1".."9"]` — Ridge plus naïve MAE/MAPE
- `windows[id].metrics[category][field].by_year_h1` — h=1 only, by calendar year
- `windows[id].series.labels` — chart months from the first origin onward, déc./janv. dropped
- `windows[id].series[category][field].actual` / `h1` / `h9` — overlay arrays
- `points` — every predicted vs actual pair, including `window` and déc./janv. (`chart_month: false`)

Top-level `headline` / `metrics` / `series` copy the default window so older dashboard code still reads.

CSV columns match `points`: `window`, `origin`, `target`, `year`, `month`, `category`, `field`, `horizon`, `predicted`, `actual`, `error`, `chart_month`.

## Dashboard

`ile-montreal.html` section *Exactitude vs APCIQ*.

| Function | What it does |
|---|---|
| `activeBacktest` | `BACKTEST.windows[selected]` or the flattened default |
| `backtestFieldSeries` | Read `actual\|h3\|h9` |
| `upsertBacktestChart` | Chart.js: solid APCIQ, dashed 3 months, dotted 9 months |
| `renderBacktest` | Two accuracy scores (3 mois / 9 mois) plus one table |

Accuracy = `100 − MAPE` vs the later APCIQ (Centris) figure. Inscriptions and prix only. No h=1, naïve, or year table on the dashboard.
