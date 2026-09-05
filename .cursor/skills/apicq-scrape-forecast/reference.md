# APCIQ scrape / forecast reference

## Parser

`scripts/fetch_monthly_stats.py` handles two PDF layouts:

- Wide 2025–26 booklets
- Letter-landscape 2019–24

Column detection uses year headers, then a text fallback, then a compact 2019 table fallback. Prefer the candidate with the most filled fields.

Strip comma thousands (`4,072`). Plex “2 à 5 logements” can be read as ventes=2 — skip plex sales below 50.

## Forecast script

`scripts/forecast_six_months.py` despite the name. `--horizon` defaults to 9.

StatCan WDS (GET):

```
https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorByReferencePeriodRange?vectorIds=...&startRefPeriod=2018-01-01&endReferencePeriod=2027-12-31
```

| Key | Vector | Table |
|---|---|---|
| qc_employment | 2063756 | 14-10-0287-01 |
| qc_unemp_rate | 2063760 | 14-10-0287-01 |
| ca_gdp | 65201210 | 36-10-0434-01 |
| qc_cpi | 41691783 | 18-10-0004-01 |
| qc_starts | 52300163 | 34-10-0158-01 |

Bank of Canada Valet: `V39079` (overnight), `V80691335` (5-year posted mortgage), `BD.CDN.5YR.DQ.YLD` (5-year bond).

Inscriptions clip: 82–122 % of last actual. Prix clip: 85–112 % of last actual.

## HTML

`ile-montreal.html` is self-contained (Chart.js CDN). Data is inline so `file://` works.

- Chart months: `CHART_TIMELINE` / `skipChartMonth` drop `12/` and `01/`
- Forecast blob: `// FORECAST_START` … `// FORECAST_END`
- Buy zones: `condoInventoryPath()`, souple 90 / ferme 120, inscriptions chart only
- GitHub Pages: `.nojekyll`, `index.html` redirect, workflow `.github/workflows/pages.yml`
