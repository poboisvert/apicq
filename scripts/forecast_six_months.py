#!/usr/bin/env python3
"""Île de Montréal forecast using sklearn and Québec macro series.

Joins APCIQ monthly housing figures to StatCan / Bank of Canada indicators
(employment, unemployment, CPI, housing starts, Canada monthly GDP, policy
and mortgage rates), fits a Ridge model per series, and writes a dotted-line
forecast for the HTML dashboard (default: 9 months).

Usage:
    python scripts/forecast_six_months.py
    python scripts/forecast_six_months.py --horizon 9
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HTML_PATH = ROOT / "ile-montreal.html"
USER_AGENT = "apicq-forecast/1.0 (https://github.com/poboisvert/apicq_statistique)"

HORIZON = 9
START_ECON = "2018-01-01"
END_ECON = "2027-12-31"

STATCAN_VECTORS = {
    "qc_employment": 2063756,  # Québec employment, SA, 15+
    "qc_unemp_rate": 2063760,  # Québec unemployment rate, SA
    "ca_gdp": 65201210,  # Canada monthly GDP, chained 2017$, all industries
    "qc_cpi": 41691783,  # Québec CPI all-items
    "qc_starts": 52300163,  # Québec housing starts, SAAR
}

BOC_SERIES = {
    "overnight": "V39079",
    "mortgage_5y": "V80691335",
    "bond_5y": "BD.CDN.5YR.DQ.YLD",
}

ECON_KEYS = [
    "qc_employment",
    "qc_unemp_rate",
    "ca_gdp",
    "qc_cpi",
    "qc_starts",
    "overnight",
    "mortgage_5y",
    "bond_5y",
]

TARGETS = ("ventes", "inscriptions", "prix", "jours")
CATEGORIES = ("copropriete", "plex")
CATEGORY_LABELS = {"copropriete": "Copropriété", "plex": "Plex (2-5)"}
LAGS = (1, 2, 3, 12)
SPRING_MONTHS = ((4, "april"), (5, "may"))


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return resp.read()


def add_months(year: int, month: int, n: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + n
    return total // 12, total % 12 + 1


def month_label(year: int, month: int) -> str:
    return f"{month:02d}/{year % 100:02d}"


def period_key(year: int, month: int) -> int:
    return year * 12 + month


def fetch_statcan() -> dict[str, dict[int, float]]:
    ids = ",".join(str(vector_id) for vector_id in STATCAN_VECTORS.values())
    url = (
        "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorByReferencePeriodRange"
        f"?vectorIds={ids}&startRefPeriod={START_ECON}&endReferencePeriod={END_ECON}"
    )
    raw = json.loads(http_get(url).decode())
    out: dict[str, dict[int, float]] = {name: {} for name in STATCAN_VECTORS}
    id_to_name = {vid: name for name, vid in STATCAN_VECTORS.items()}
    for item in raw:
        obj = item.get("object") or {}
        name = id_to_name.get(obj.get("vectorId"))
        if not name:
            continue
        for point in obj.get("vectorDataPoint") or []:
            ref = str(point.get("refPer") or "")
            value = point.get("value")
            if len(ref) < 7 or value is None:
                continue
            year, month = int(ref[:4]), int(ref[5:7])
            out[name][period_key(year, month)] = float(value)
    missing = [name for name, series in out.items() if not series]
    if missing:
        raise RuntimeError(f"StatCan returned no points for: {', '.join(missing)}")
    return out


def fetch_boc() -> dict[str, dict[int, float]]:
    names = ",".join(BOC_SERIES.values())
    url = (
        "https://www.bankofcanada.ca/valet/observations/"
        f"{names}/json?start_date=2018-01-01"
    )
    data = json.loads(http_get(url).decode())
    monthly: dict[str, dict[int, list[float]]] = {name: {} for name in BOC_SERIES}
    code_to_name = {code: name for name, code in BOC_SERIES.items()}
    for obs in data.get("observations") or []:
        day = str(obs.get("d") or "")
        if len(day) < 7:
            continue
        key = period_key(int(day[:4]), int(day[5:7]))
        for code, name in code_to_name.items():
            cell = obs.get(code) or {}
            value = cell.get("v")
            if value in (None, ""):
                continue
            monthly[name].setdefault(key, []).append(float(value))
    return {
        name: {key: values[-1] for key, values in series.items()}
        for name, series in monthly.items()
    }


def load_housing() -> list[dict]:
    payload = json.loads((DATA / "ile-montreal.json").read_text())
    rows = []
    for item in payload["months"]:
        rows.append(
            {
                "year": item["year"],
                "month": item["month"],
                "key": period_key(item["year"], item["month"]),
                "copropriete": item["copropriete"],
                "plex": item["plex"],
            }
        )
    return sorted(rows, key=lambda row: row["key"])


def ffill_series(series: dict[int, float], keys: list[int]) -> dict[int, float]:
    filled: dict[int, float] = {}
    last = None
    for key in keys:
        if key in series:
            last = series[key]
        if last is not None:
            filled[key] = last
    return filled


def calendar_keys(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


def month_features(year: int, month: int) -> list[float]:
    angle = 2 * math.pi * month / 12
    return [math.sin(angle), math.cos(angle), year + (month - 1) / 12]


def ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))),
        ]
    )


def target_history(rows: list[dict], category: str, field: str) -> dict[int, float]:
    history: dict[int, float] = {}
    for row in rows:
        value = row[category].get(field)
        if value is not None:
            history[row["key"]] = float(value)
    return history


def month_field(rows: list[dict], year: int, month: int, category: str, field: str) -> float | None:
    for row in rows:
        if row["year"] == year and row["month"] == month:
            value = row[category].get(field)
            return None if value is None else float(value)
    return None


def ratio_yoy(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return current / previous - 1.0


def build_spring_years(rows: list[dict], category: str) -> dict[int, dict]:
    """April/May vs the same months a year earlier. A drop is a slowing-market flag."""
    years = sorted({row["year"] for row in rows})
    out: dict[int, dict] = {}
    for year in years:
        months: dict[str, dict] = {}
        lowers: list[bool] = []
        ventes_yoys: list[float] = []
        prix_yoys: list[float] = []
        for month_num, key in SPRING_MONTHS:
            ventes = month_field(rows, year, month_num, category, "ventes")
            ventes_prev = month_field(rows, year - 1, month_num, category, "ventes")
            prix = month_field(rows, year, month_num, category, "prix")
            prix_prev = month_field(rows, year - 1, month_num, category, "prix")
            ventes_yoy = ratio_yoy(ventes, ventes_prev)
            prix_yoy = ratio_yoy(prix, prix_prev)
            comparable = ventes is not None and ventes_prev is not None
            lower = comparable and ventes < ventes_prev
            if ventes_yoy is not None:
                ventes_yoys.append(ventes_yoy)
            if prix_yoy is not None:
                prix_yoys.append(prix_yoy)
            if comparable:
                lowers.append(lower)
            months[key] = {
                "ventes": ventes,
                "ventes_prev": ventes_prev,
                "ventes_yoy": None if ventes_yoy is None else round(ventes_yoy, 4),
                "prix": prix,
                "prix_prev": prix_prev,
                "prix_yoy": None if prix_yoy is None else round(prix_yoy, 4),
                "lower": lower if comparable else None,
            }
        ventes_mean = sum(ventes_yoys) / len(ventes_yoys) if ventes_yoys else None
        prix_mean = sum(prix_yoys) / len(prix_yoys) if prix_yoys else None
        slow = bool(lowers) and any(lowers)
        out[year] = {
            "comparable": bool(lowers),
            "slow": slow,
            "april_lower": months["april"]["lower"],
            "may_lower": months["may"]["lower"],
            "ventes_yoy": None if ventes_mean is None else round(ventes_mean, 4),
            "prix_yoy": None if prix_mean is None else round(prix_mean, 4),
            "price_lag": bool(
                slow
                and ventes_mean is not None
                and ventes_mean < 0
                and (prix_mean is None or prix_mean >= -0.005)
            ),
            "months": months,
        }
    return out


def latest_spring(spring_years: dict[int, dict]) -> tuple[int | None, dict | None]:
    comparable = [year for year, item in spring_years.items() if item.get("comparable")]
    if not comparable:
        return None, None
    year = max(comparable)
    return year, spring_years[year]


def spring_reading(label: str, year: int, prior: int, signal: dict) -> str:
    if not signal.get("comparable"):
        return f"{label} : avril–mai {year} incomparable (mois manquant)."
    if not signal.get("slow"):
        return (
            f"{label} : avril et mai {year} ne sont pas plus faibles qu’en {prior}. "
            "Pas de signal de ralentissement printanier."
        )
    if signal.get("price_lag"):
        price = (
            "Le prix médian tient encore ou monte : le volume précède le prix, "
            "mauvais signe pour la suite."
        )
    elif signal.get("prix_yoy") is not None and signal["prix_yoy"] < -0.02:
        price = "Le prix médian recule aussi : le ralentissement est confirmé."
    else:
        price = (
            "Le prix médian est à peu près stable : le ralentissement se lit "
            "d’abord dans les ventes."
        )
    return (
        f"{label} : avril ou mai {year} plus faible qu’en {prior} — "
        f"mauvais signe, le marché ralentit. {price}"
    )


def spring_features(
    year: int,
    month: int,
    spring_years: dict[int, dict],
    last_spring_year: int | None,
) -> list[float]:
    if last_spring_year is None:
        return [0.0, 0.0, 0.0, 0.0]
    ref = min(year if month >= 5 else year - 1, last_spring_year)
    signal = spring_years.get(ref)
    if not signal or not signal.get("comparable"):
        return [0.0, 0.0, 0.0, 0.0]
    return [
        1.0 if signal.get("slow") else 0.0,
        float(signal.get("ventes_yoy") or 0.0),
        float(signal.get("prix_yoy") or 0.0),
        1.0 if signal.get("price_lag") else 0.0,
    ]


def apply_spring_bias(
    field: str,
    pred: float,
    last: float,
    month: int,
    step: int,
    horizon: int,
    signal: dict | None,
) -> float:
    """When last spring was weaker, cap a rebound and do not let prices keep rising."""
    if not signal or not signal.get("slow"):
        return pred
    ventes_yoy = signal.get("ventes_yoy")
    if ventes_yoy is None:
        return pred
    if field == "ventes":
        pred = pred * (1 + 0.4 * min(0.0, ventes_yoy))
        name = {4: "april", 5: "may"}.get(month)
        same = (signal.get("months") or {}).get(name or "")
        if same and same.get("ventes") is not None:
            pred = min(pred, float(same["ventes"]))
        return pred
    if field == "prix":
        progress = (step + 1) / max(1, horizon)
        prix_yoy = signal.get("prix_yoy")
        if prix_yoy is None or prix_yoy >= -0.005:
            ceiling = last * (1 + 0.25 * min(0.0, ventes_yoy) * progress)
            return min(pred, last, ceiling)
        return min(pred, last)
    if field == "jours":
        return pred * (1 - 0.15 * min(0.0, ventes_yoy))
    return pred


def design_matrix(
    keys: list[int],
    target: dict[int, float],
    econ: dict[str, dict[int, float]],
    *,
    require_target: bool,
    spring_years: dict[int, dict] | None = None,
    last_spring_year: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    used: list[int] = []
    for key in keys:
        if require_target and key not in target:
            continue
        year, month = divmod(key, 12)
        if month == 0:
            year -= 1
            month = 12
        features = month_features(year, month)
        missing_lag = False
        for lag in LAGS:
            prev = key - lag
            if prev not in target:
                missing_lag = True
                break
            features.append(target[prev])
        if missing_lag:
            continue
        skip = False
        for name in ECON_KEYS:
            value = econ[name].get(key - 1)
            if value is None:
                skip = True
                break
            features.append(value)
        if skip:
            continue
        features.extend(spring_features(year, month, spring_years or {}, last_spring_year))
        X_rows.append(features)
        y_rows.append(target.get(key, 0.0))
        used.append(key)
    if not X_rows:
        raise RuntimeError("Not enough overlapping housing and economic history to fit.")
    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float), used


def cv_mae(X: np.ndarray, y: np.ndarray) -> float:
    splits = min(5, max(2, len(y) // 12))
    if len(y) < 16:
        return float("nan")
    folder = TimeSeriesSplit(n_splits=splits)
    errors = []
    for train_idx, test_idx in folder.split(X):
        model = ridge_pipeline()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        errors.append(float(np.mean(np.abs(pred - y[test_idx]))))
    return float(np.mean(errors))


RATE_KEYS = {"overnight", "mortgage_5y", "bond_5y"}


def forecast_econ_series(
    series: dict[int, float], future_keys: list[int], *, persist: bool
) -> dict[int, float]:
    keys = sorted(series)
    if not keys:
        return {}
    history = dict(series)
    for key in future_keys:
        year, month = divmod(key, 12)
        if month == 0:
            year -= 1
            month = 12
        lag1 = history.get(key - 1)
        lag12 = history.get(key - 12)
        if lag1 is None:
            continue
        if persist or lag12 is None:
            history[key] = lag1
            continue
        X_train, y_train = [], []
        for past in keys:
            if past - 1 not in history or past - 12 not in history:
                continue
            py, pm = divmod(past, 12)
            if pm == 0:
                py -= 1
                pm = 12
            X_train.append(month_features(py, pm) + [history[past - 1], history[past - 12]])
            y_train.append(history[past])
        if len(X_train) < 18:
            history[key] = lag1
            continue
        model = ridge_pipeline()
        model.fit(np.asarray(X_train), np.asarray(y_train))
        pred = float(
            model.predict(
                [month_features(year, month) + [lag1, lag12 if lag12 is not None else lag1]]
            )[0]
        )
        history[key] = pred
    return {key: history[key] for key in future_keys if key in history}


def clip_forecast(field: str, value: float, last: float) -> float:
    if field == "ventes":
        return float(max(40.0, round(value)))
    if field == "inscriptions":
        low, high = last * 0.82, last * 1.22
        return float(max(200.0, min(high, max(low, round(value)))))
    if field == "prix":
        low, high = last * 0.85, last * 1.12
        return float(max(low, min(high, round(value / 500.0) * 500.0)))
    return float(max(20.0, min(140.0, round(value))))


def build_forecast() -> dict:
    housing = load_housing()
    print("Fetching StatCan Québec / Canada series", flush=True)
    statcan = fetch_statcan()
    print("Fetching Bank of Canada rates", flush=True)
    boc = fetch_boc()
    econ_raw = {**statcan, **boc}
    last_housing = housing[-1]["key"]
    future_keys = [add_months(housing[-1]["year"], housing[-1]["month"], step + 1) for step in range(HORIZON)]
    future_keys = [period_key(year, month) for year, month in future_keys]
    all_keys = calendar_keys(period_key(2018, 1), future_keys[-1])
    econ = {name: ffill_series(series, all_keys) for name, series in econ_raw.items()}
    for name in ECON_KEYS:
        econ[name].update(
            forecast_econ_series(econ[name], future_keys, persist=name in RATE_KEYS)
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon_months": HORIZON,
        "model": "RidgeCV + StandardScaler (sklearn)",
        "last_actual": month_label(housing[-1]["year"], housing[-1]["month"]),
        "labels": [month_label(*divmod(key if key % 12 else key, 12)[:2]) for key in future_keys],
        "cv_mae": {},
        "forecast": {category: {field: [] for field in TARGETS} for category in CATEGORIES},
        "last_values": {category: {} for category in CATEGORIES},
        "economy": {"latest": {}, "path": []},
        "sources": [
            "StatCan 14-10-0287-01 — Québec employment and unemployment, seasonally adjusted",
            "StatCan 36-10-0434-01 — Canada monthly GDP (Québec has no official monthly GDP)",
            "StatCan 18-10-0004-01 — Québec CPI, all-items",
            "StatCan 34-10-0158-01 — Québec housing starts, SAAR",
            "Bank of Canada Valet — overnight target, 5-year posted mortgage, 5-year GoC bond",
        ],
        "features": [
            "month seasonality",
            "trend",
            "lags 1/2/3/12 of the housing series",
            "Québec employment and unemployment rate",
            "Canada monthly GDP",
            "Québec CPI",
            "Québec housing starts",
            "overnight rate, 5-year mortgage, 5-year bond (t-1)",
            "April/May year-over-year ventes (slowing if either month is lower)",
            "April/May year-over-year median price (volume vs price)",
        ],
        "spring": {
            "rule": (
                "If April or May ventes are lower than the same month a year earlier, "
                "the market is slowing. Compare that volume signal with median price: "
                "when sales fall and prices still hold, volume is leading."
            ),
            "last_year": None,
            "prior_year": None,
            "categories": {},
        },
    }
    # Fix labels properly
    labels = []
    for key in future_keys:
        year, month = divmod(key, 12)
        if month == 0:
            year -= 1
            month = 12
        labels.append(month_label(year, month))
    result["labels"] = labels

    latest_econ_key = max(k for k in econ["qc_employment"] if k <= last_housing)
    result["economy"]["latest"] = {
        name: round(econ[name][latest_econ_key], 3)
        for name in ECON_KEYS
        if latest_econ_key in econ[name]
    }
    for key in future_keys:
        year, month = divmod(key, 12)
        if month == 0:
            year -= 1
            month = 12
        result["economy"]["path"].append(
            {
                "label": month_label(year, month),
                **{
                    name: round(econ[name][key], 3)
                    for name in ECON_KEYS
                    if key in econ[name]
                },
            }
        )

    train_keys = [row["key"] for row in housing]
    for category in CATEGORIES:
        spring_years = build_spring_years(housing, category)
        last_year, latest = latest_spring(spring_years)
        latest = latest or None
        prior_year = None if last_year is None else last_year - 1
        if last_year is not None and latest is not None:
            result["spring"]["last_year"] = last_year
            result["spring"]["prior_year"] = prior_year
            payload = {
                **latest,
                "reading": spring_reading(
                    CATEGORY_LABELS[category], last_year, prior_year, latest
                ),
            }
            result["spring"]["categories"][category] = payload
            print(
                f"  spring {category}: {last_year} vs {prior_year} "
                f"slow={latest['slow']} ventes_yoy={latest['ventes_yoy']} "
                f"prix_yoy={latest['prix_yoy']}",
                flush=True,
            )
        result["cv_mae"][category] = {}
        for field in TARGETS:
            history = target_history(housing, category, field)
            X, y, _used = design_matrix(
                train_keys,
                history,
                econ,
                require_target=True,
                spring_years=spring_years,
                last_spring_year=last_year,
            )
            mae = cv_mae(X, y)
            result["cv_mae"][category][field] = None if math.isnan(mae) else round(mae, 1)
            model = ridge_pipeline()
            model.fit(X, y)
            last_actual = history[max(history)]
            result["last_values"][category][field] = last_actual
            rolling = dict(history)
            preds = []
            for step, key in enumerate(future_keys):
                year, month = divmod(key, 12)
                if month == 0:
                    year -= 1
                    month = 12
                X_one, _, used = design_matrix(
                    [key],
                    rolling,
                    econ,
                    require_target=False,
                    spring_years=spring_years,
                    last_spring_year=last_year,
                )
                if not used:
                    pred = last_actual
                else:
                    pred = float(model.predict(X_one)[0])
                pred = apply_spring_bias(field, pred, last_actual, month, step, HORIZON, latest)
                pred = clip_forecast(field, pred, last_actual)
                rolling[key] = pred
                preds.append(pred)
            result["forecast"][category][field] = preds
            print(f"  {category} {field}: CV MAE={result['cv_mae'][category][field]} → {preds}")
    return result


def patch_html(forecast: dict) -> None:
    html = HTML_PATH.read_text()
    blob = json.dumps(forecast, ensure_ascii=False)
    pattern = re.compile(
        r"    // FORECAST_START\n    const FORECAST = .*?\n    // FORECAST_END",
        re.S,
    )
    replacement = f"    // FORECAST_START\n    const FORECAST = {blob};\n    // FORECAST_END"
    if not pattern.search(html):
        raise SystemExit("ile-montreal.html is missing FORECAST_START/END markers")
    HTML_PATH.write_text(pattern.sub(replacement, html, count=1))


def main() -> int:
    global HORIZON
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=HORIZON, help="Months ahead (default: 9)")
    args = parser.parse_args()
    HORIZON = args.horizon
    forecast = build_forecast()
    DATA.mkdir(exist_ok=True)
    out = DATA / "forecast-6m.json"
    out.write_text(json.dumps(forecast, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {out}")
    if HTML_PATH.exists() and "// FORECAST_START" in HTML_PATH.read_text():
        patch_html(forecast)
        print(f"Updated {HTML_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
