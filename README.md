# Île de Montréal — 2019 à 2026

Monthly Centris figures for **Copropriété** and **Plex (2-5 logements)** on the Island of Montreal, extracted from [APCIQ](https://apciq.ca/barometre-residentiel/statistiques-mensuelles/) monthly PDFs.

Four series: ventes totales, inscriptions en vigueur, prix médian, moyenne de jours sur le marché.

![Dashboard Île de Montréal 2019–2026](docs/ile-montreal.png)

Open the report: [ile-montreal.html](https://poboisvert.github.io/apicq/ile-montreal.html)

Filters: Les deux / Copropriété / Plex, and Chronologie / Fév–août. Chronology charts carry a dotted sklearn forecast through mai 2027 (Québec employment, unemployment, CPI and housing starts, Canada monthly GDP, Bank of Canada rates).

## Chart conventions

- **Décembre and janvier are omitted** from every chart and from the forecast table. Listings expire at year-end and are renewed in February, so those months look like a crash. They stay in the monthly PDF table. Lines connect novembre → février.
- Dotted lines are the 9-month Ridge forecast (septembre 2026 – mai 2027).
- On **Inscriptions en vigueur**, green buy zones apply to condos only: souple ≥ 90 estimated days, ferme ≥ 120. After mai 2027 the inventory path is seasonal, not Ridge.
- Condo median price 357k → 475k is **+33 %**, not a doubling.

## Coverage

82 complete months, septembre 2019 – août 2026.

Missing months:

- Janvier–juillet 2019 — PDFs cover RMR Montréal only, not Île
- Avril 2020 — COVID pause, empty Île tables
- Mars 2023 — archive PDF link is dead

Août 2019 has ventes only and is omitted from the charts.

## Refresh the data

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_monthly_stats.py
```

Reuse already-downloaded PDFs:

```bash
python scripts/fetch_monthly_stats.py --skip-download
```

Limit years:

```bash
python scripts/fetch_monthly_stats.py --years 2024 2025 2026
```

Refresh the 9-month forecast (needs network for StatCan and the Bank of Canada):

```bash
python scripts/forecast_six_months.py --horizon 9
```

Outputs:

- `data/ile-montreal.json`
- `data/ile-montreal.csv`
- `data/forecast-6m.json`
- `data/pdfs/` (gitignored cache)

The forecast is a Ridge model (`sklearn.linear_model.RidgeCV`) on seasonality, lags 1/2/3/12 of each housing series, and these macro series (lagged one month):

- Québec employment and unemployment rate ([StatCan 14-10-0287-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410028701))
- Canada monthly GDP ([36-10-0434-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3610043401)); Québec does not publish monthly GDP
- Québec CPI ([18-10-0004-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000401))
- Québec housing starts ([34-10-0158-01](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3410015801))
- Overnight target, 5-year posted mortgage, 5-year GoC bond ([Bank of Canada Valet](https://www.bankofcanada.ca/valet-api-how-to/))

Sources: [statistiques mensuelles](https://apciq.ca/barometre-residentiel/statistiques-mensuelles/) and the [archive](https://apciq.ca/barometre-residentiel/statistiques-mensuelles-archive/).
