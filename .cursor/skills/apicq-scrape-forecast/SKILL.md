---
name: apicq-scrape-forecast
description: >-
  Scrapes APCIQ monthly Centris PDFs for Île de Montréal (Copropriété and Plex),
  extracts ventes / inscriptions / prix médian / jours, trains a sklearn Ridge
  forecast with Québec macro series, and updates ile-montreal.html. Use when
  the user mentions APCIQ, baromètre, statistiques mensuelles, scrape, extract
  PDFs, forecast, Ridge, inscriptions, or refreshing ile-montreal data.
---

# APCIQ scrape and forecast

Work in the repo root. Use `.venv` and `pip install -r requirements.txt` if needed.

## Workflow

```
- [ ] Scrape / extract monthly PDFs
- [ ] Check coverage and known gaps
- [ ] Retrain the 9-month forecast
- [ ] Sync ile-montreal.html (FORECAST blob + chart arrays)
- [ ] Respect chart rules (no décembre / janvier)
- [ ] Verify the dashboard
```

## Scrape

Do not rewrite the parser unless a new PDF layout fails. Run:

```bash
python scripts/fetch_monthly_stats.py
python scripts/fetch_monthly_stats.py --skip-download
python scripts/fetch_monthly_stats.py --years 2024 2025 2026
```

Outputs:

- `data/ile-montreal.json`
- `data/ile-montreal.csv`
- `data/pdfs/` (gitignored)

Sources:

- https://apciq.ca/barometre-residentiel/statistiques-mensuelles/
- https://apciq.ca/barometre-residentiel/statistiques-mensuelles-archive/

Region: **Île de Montréal** only (not RMR). Categories: **Copropriété** and **Plex (2-5 logements)**. Metrics: ventes totales, inscriptions en vigueur, prix médian, moyenne de jours sur le marché.

Known gaps: janv.–juil. 2019 (RMR only), avril 2020 (COVID, empty tables), mars 2023 (dead archive URL). Août 2019 has ventes only — omit from charts.

## Forecast

```bash
python scripts/forecast_six_months.py --horizon 9
```

That writes `data/forecast-6m.json` and patches `const FORECAST` between `// FORECAST_START` and `// FORECAST_END` in `ile-montreal.html`.

Model: `RidgeCV` + `StandardScaler`. Features: month seasonality, trend, lags 1/2/3/12, Québec employment and unemployment, Québec CPI, Québec housing starts, Canada monthly GDP (no official Québec monthly GDP), Bank of Canada overnight / 5-year mortgage / 5-year bond (rates persist). Horizon default: **9 months**.

Spring rule: if April or May ventes are lower than the same month a year earlier, treat it as a slowing market. Compare that volume drop with median price. When sales fall and prices still hold, volume is leading — the forecast damps ventes and does not keep lifting prices.

After a scrape, also refresh the inline `TIMELINE` / `COPRO` / `PLEX` arrays in `ile-montreal.html` so charts match JSON.

## Chart rules

- Dotted lines = official sklearn forecast.
- Ventes chart: a dashed line connects each year’s highest month (pics annuels), Dec/Jan excluded.
- **Drop décembre and janvier** from every chart and from the forecast table. Listings expire in December and are renewed in February; those months look like a crash. Keep them in the monthly PDF table only.
- Lines connect novembre → février.
- Inscriptions chart: Ridge for the 9-month horizon; seasonal inventory path only after that, for buy zones.
- Condo buy zones (inscriptions only): souple ≥ 90 estimated days, ferme ≥ 120. Do not use the old 7-month-inventory + 55-day rule.
- Do not claim condo prices “doubled” since 2019; 357k → 475k is about **+33 %**.

## Publish

Live report: https://poboisvert.github.io/apicq/ile-montreal.html  
Repo: https://github.com/poboisvert/apicq  
Pages deploys from `.github/workflows/pages.yml` on push to `main`.

## Extra detail

Parser layouts, StatCan / Valet series IDs, and HTML markers: [reference.md](reference.md)
