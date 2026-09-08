#!/usr/bin/env python3
"""Walk-forward backtest of the Île de Montréal Ridge forecast vs APCIQ actuals.

At each origin, refits the same sklearn model as the dashboard on one of the
live train windows (last calendar year + last 3 or 9 months of that year) and
scores 1–9 month forecasts against later Centris figures.

The 24-month rule is only the burn-in before the first origin (so lag-12 and
a prior year exist). It is not the Ridge training sample.

Usage:
    python scripts/backtest_forecast.py
    python scripts/backtest_forecast.py --min-train 24 --horizon 9
    python scripts/backtest_forecast.py --windows prior_plus_3,prior_plus_9
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import forecast_six_months as fc  # noqa: E402

DATA = ROOT / "data"
HTML_PATH = ROOT / "ile-montreal.html"
BACKTEST_FIELDS = ("inscriptions", "prix")
CHART_HORIZONS = (3, 9)
DEFAULT_WINDOWS = ",".join(fc.LIVE_WINDOWS)


def skip_chart_month(month: int) -> bool:
    return month in (1, 12)


def coverage_ok(rows: list[dict], min_train: int) -> bool:
    for category in fc.CATEGORIES:
        inscriptions = sum(1 for row in rows if row[category].get("inscriptions") is not None)
        prix = sum(1 for row in rows if row[category].get("prix") is not None)
        if inscriptions < min_train or prix < min_train:
            return False
    return True


def actual_map(housing: list[dict], category: str, field: str) -> dict[int, float]:
    return fc.target_history(housing, category, field)


def summarize(pairs: list[tuple[float, float]]) -> dict | None:
    if not pairs:
        return None
    errors = [pred - actual for pred, actual in pairs]
    abs_err = [abs(value) for value in errors]
    pct = [abs(pred - actual) / abs(actual) for pred, actual in pairs if actual != 0]
    rmse = math.sqrt(sum(value * value for value in errors) / len(errors))
    return {
        "n": len(pairs),
        "mae": round(sum(abs_err) / len(abs_err), 1),
        "rmse": round(rmse, 1),
        "mape": None if not pct else round(100.0 * sum(pct) / len(pct), 2),
        "bias": round(sum(errors) / len(errors), 1),
    }


def accuracy_from_mape(mape: float | None) -> float | None:
    if mape is None:
        return None
    return round(max(0.0, 100.0 - mape), 1)


def naive_seasonal(history: dict[int, float], key: int, last: float | None) -> float | None:
    seasonal = history.get(key - 12)
    if seasonal is not None:
        return seasonal
    return last


def collect_origins(housing: list[dict], min_train: int) -> list[dict]:
    origins = []
    for idx, row in enumerate(housing):
        window = housing[: idx + 1]
        if not coverage_ok(window, min_train):
            continue
        remaining = [item for item in housing if item["key"] > row["key"]]
        if not remaining:
            continue
        origins.append(row)
    return origins


def align_series(labels: list[str], by_label: dict[str, float | None]) -> list[float | None]:
    return [by_label.get(label) for label in labels]


def patch_html(payload: dict) -> None:
    html = HTML_PATH.read_text()
    blob = json.dumps(payload, ensure_ascii=False)
    pattern = re.compile(
        r"    // BACKTEST_START\n    const BACKTEST = .*?\n    // BACKTEST_END",
        re.S,
    )
    replacement = f"    // BACKTEST_START\n    const BACKTEST = {blob};\n    // BACKTEST_END"
    if not pattern.search(html):
        raise SystemExit("ile-montreal.html is missing BACKTEST_START/END markers")
    HTML_PATH.write_text(pattern.sub(replacement, html, count=1))


def dashboard_payload(full: dict) -> dict:
    """Drop origin-level rows from the HTML blob; keep metrics and chart series."""
    keep = dict(full)
    keep.pop("points", None)
    windows = {}
    for window_id, block in (full.get("windows") or {}).items():
        slim = dict(block)
        slim.pop("points", None)
        windows[window_id] = slim
    keep["windows"] = windows
    return keep


def parse_windows(raw: str) -> list[str]:
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    known = set(fc.WINDOW_LABELS)
    for window_id in ids:
        if window_id not in known:
            raise SystemExit(f"Unknown window {window_id}. Choose from: {', '.join(sorted(known))}")
    if not ids:
        raise SystemExit("Need at least one train window.")
    return ids


def walk_window(
    housing: list[dict],
    econ_raw: dict[str, dict[int, float]],
    origins: list[dict],
    horizon: int,
    window: str,
) -> dict:
    points: list[dict] = []
    scored = defaultdict(list)
    naive_persist = defaultdict(list)
    naive_yoy = defaultdict(list)
    by_year = defaultdict(list)
    chart_actual: dict[str, dict[str, dict[str, float]]] = {
        category: {field: {} for field in BACKTEST_FIELDS} for category in fc.CATEGORIES
    }
    chart_h: dict[int, dict[str, dict[str, dict[str, float]]]] = {
        step: {
            category: {field: {} for field in BACKTEST_FIELDS} for category in fc.CATEGORIES
        }
        for step in CHART_HORIZONS
    }

    for category in fc.CATEGORIES:
        for field in BACKTEST_FIELDS:
            history = actual_map(housing, category, field)
            for key, value in history.items():
                year, month = fc.key_to_year_month(key)
                chart_actual[category][field][fc.month_label(year, month)] = value

    n_ok = 0
    for index, origin in enumerate(origins, start=1):
        origin_key = origin["key"]
        print(
            f"[{window} {index}/{len(origins)}] origin "
            f"{fc.month_label(origin['year'], origin['month'])}",
            flush=True,
        )
        try:
            fitted = fc.run_forecast(
                housing,
                econ_raw,
                origin_key,
                horizon,
                quiet=True,
                window=window,
            )
        except RuntimeError as exc:
            print(f"  skip origin: {exc}", flush=True)
            continue
        n_ok += 1
        for category in fc.CATEGORIES:
            for field in BACKTEST_FIELDS:
                history = actual_map(housing, category, field)
                last = history.get(origin_key)
                preds = fitted["forecast"][category][field]
                for step, label in enumerate(fitted["labels"]):
                    pred = preds[step] if step < len(preds) else None
                    if pred is None:
                        continue
                    year, month = fc.add_months(origin["year"], origin["month"], step + 1)
                    key = fc.period_key(year, month)
                    actual = history.get(key)
                    if actual is None:
                        continue
                    horizon_n = step + 1
                    err = pred - actual
                    point = {
                        "window": window,
                        "origin": fitted["origin"],
                        "target": label,
                        "year": year,
                        "month": month,
                        "category": category,
                        "field": field,
                        "horizon": horizon_n,
                        "predicted": pred,
                        "actual": actual,
                        "error": round(err, 1),
                        "chart_month": not skip_chart_month(month),
                    }
                    points.append(point)
                    persist = last
                    seasonal = naive_seasonal(history, key, last)
                    if not skip_chart_month(month):
                        scored[(category, field, horizon_n)].append((pred, actual))
                        if persist is not None:
                            naive_persist[(category, field, horizon_n)].append((persist, actual))
                        if seasonal is not None:
                            naive_yoy[(category, field, horizon_n)].append((seasonal, actual))
                        if horizon_n == 1:
                            by_year[(category, field, year)].append((pred, actual))
                    if horizon_n in chart_h:
                        chart_h[horizon_n][category][field][label] = pred

    labels = []
    for row in housing:
        year, month = row["year"], row["month"]
        if skip_chart_month(month):
            continue
        if row["key"] < origins[0]["key"]:
            continue
        labels.append(fc.month_label(year, month))

    metrics: dict = {}
    for category in fc.CATEGORIES:
        metrics[category] = {}
        for field in BACKTEST_FIELDS:
            by_horizon = {}
            for step in range(1, horizon + 1):
                ridge = summarize(scored[(category, field, step)])
                persist = summarize(naive_persist[(category, field, step)])
                seasonal = summarize(naive_yoy[(category, field, step)])
                if ridge is None:
                    continue
                by_horizon[str(step)] = {
                    **ridge,
                    "naive_persist_mae": None if persist is None else persist["mae"],
                    "naive_seasonal_mae": None if seasonal is None else seasonal["mae"],
                    "naive_seasonal_mape": None if seasonal is None else seasonal["mape"],
                }
            years = {}
            for (cat, fld, year), pairs in sorted(by_year.items()):
                if cat != category or fld != field:
                    continue
                years[str(year)] = summarize(pairs)
            metrics[category][field] = {"by_horizon": by_horizon, "by_year_h1": years}

    series = {"labels": labels, "copropriete": {}, "plex": {}}
    for category in fc.CATEGORIES:
        for field in BACKTEST_FIELDS:
            series[category][field] = {
                "actual": align_series(labels, chart_actual[category][field]),
                "h3": align_series(labels, chart_h[3][category][field]),
                "h9": align_series(labels, chart_h[9][category][field]),
            }

    headline = {}
    for category in fc.CATEGORIES:
        headline[category] = {}
        for field in BACKTEST_FIELDS:
            h3 = metrics[category][field]["by_horizon"].get("3") or {}
            h9 = metrics[category][field]["by_horizon"].get("9") or {}
            headline[category][field] = {
                "h3_mae": h3.get("mae"),
                "h3_mape": h3.get("mape"),
                "h3_accuracy": accuracy_from_mape(h3.get("mape")),
                "h9_mae": h9.get("mae"),
                "h9_mape": h9.get("mape"),
                "h9_accuracy": accuracy_from_mape(h9.get("mape")),
            }

    origin_meta = {
        "id": window,
        "label": fc.WINDOW_LABELS.get(window, window),
        "first_origin": fc.month_label(origins[0]["year"], origins[0]["month"]),
        "last_origin": fc.month_label(origins[-1]["year"], origins[-1]["month"]),
        "n_origins": n_ok,
        "n_points": len([item for item in points if item["chart_month"]]),
        "headline": headline,
        "metrics": metrics,
        "series": series,
        "points": points,
    }
    return origin_meta


def compare_windows(windows: dict[str, dict]) -> dict:
    """3-month and 9-month accuracy vs APCIQ for the dashboard."""
    out = {}
    for category in fc.CATEGORIES:
        out[category] = {}
        for field in BACKTEST_FIELDS:
            row = {}
            for window_id, block in windows.items():
                headline = ((block.get("headline") or {}).get(category) or {}).get(field) or {}
                row[window_id] = {
                    "h3_mae": headline.get("h3_mae"),
                    "h3_mape": headline.get("h3_mape"),
                    "h3_accuracy": headline.get("h3_accuracy"),
                    "h9_mae": headline.get("h9_mae"),
                    "h9_mape": headline.get("h9_mape"),
                    "h9_accuracy": headline.get("h9_accuracy"),
                }
            out[category][field] = row
    return out


def pick_default_window(windows: dict[str, dict]) -> str:
    """Prefer the window with the higher copro inscriptions 3-month accuracy."""
    best_id = next(iter(windows))
    best_acc = None
    for window_id, block in windows.items():
        acc = (
            ((block.get("headline") or {}).get("copropriete") or {})
            .get("inscriptions", {})
            .get("h3_accuracy")
        )
        if acc is None:
            continue
        if best_acc is None or acc > best_acc:
            best_acc = acc
            best_id = window_id
    return best_id


def flatten_window(block: dict, *, generated_at: str, min_train: int, horizon: int) -> dict:
    return {
        "generated_at": generated_at,
        "model": "RidgeCV + StandardScaler (sklearn)",
        "train_rule": (
            "Last calendar year plus the last 3 or 9 months of the origin year "
            "(not an expanding 24-month sample)"
        ),
        "min_train_months": min_train,
        "horizon_months": horizon,
        "score_rule": "Décembre et janvier exclus (même règle que les graphiques)",
        "default_window": block["id"],
        "window": {
            "id": block["id"],
            "label": block["label"],
        },
        "first_origin": block["first_origin"],
        "last_origin": block["last_origin"],
        "n_origins": block["n_origins"],
        "n_points": block["n_points"],
        "headline": block["headline"],
        "metrics": block["metrics"],
        "series": block["series"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-train", type=int, default=24, help="Burn-in months of inventory and median price")
    parser.add_argument("--horizon", type=int, default=9, help="Months ahead (default: 9)")
    parser.add_argument(
        "--windows",
        default=DEFAULT_WINDOWS,
        help="Comma-separated train windows (default: prior_plus_3,prior_plus_9)",
    )
    parser.add_argument("--skip-html", action="store_true")
    args = parser.parse_args()
    window_ids = parse_windows(args.windows)

    housing = fc.load_housing()
    origins = collect_origins(housing, args.min_train)
    if not origins:
        raise SystemExit("Not enough months of inscriptions and prix médian to start a backtest.")

    print("Fetching StatCan Québec / Canada series", flush=True)
    statcan = fc.fetch_statcan()
    print("Fetching Bank of Canada rates", flush=True)
    boc = fc.fetch_boc()
    econ_raw = {**statcan, **boc}

    print(
        f"Walk-forward {len(origins)} origins · "
        f"{fc.month_label(origins[0]['year'], origins[0]['month'])} → "
        f"{fc.month_label(origins[-1]['year'], origins[-1]['month'])} · "
        f"burn-in {args.min_train} months · windows {', '.join(window_ids)}",
        flush=True,
    )

    windows: dict[str, dict] = {}
    all_points: list[dict] = []
    for window_id in window_ids:
        block = walk_window(housing, econ_raw, origins, args.horizon, window_id)
        windows[window_id] = block
        all_points.extend(block["points"])
        for category in fc.CATEGORIES:
            for field in BACKTEST_FIELDS:
                h = block["headline"][category][field]
                print(
                    f"  {window_id} {category} {field}: "
                    f"3 mois exactitude {h['h3_accuracy']}% (MAPE {h['h3_mape']}%) · "
                    f"9 mois exactitude {h['h9_accuracy']}% (MAPE {h['h9_mape']}%)",
                    flush=True,
                )

    generated_at = datetime.now(timezone.utc).isoformat()
    default = pick_default_window(windows)
    primary = windows[default]
    payload = flatten_window(
        primary, generated_at=generated_at, min_train=args.min_train, horizon=args.horizon
    )
    payload["windows"] = {
        window_id: {
            "id": block["id"],
            "label": block["label"],
            "first_origin": block["first_origin"],
            "last_origin": block["last_origin"],
            "n_origins": block["n_origins"],
            "n_points": block["n_points"],
            "headline": block["headline"],
            "metrics": block["metrics"],
            "series": block["series"],
            "points": block["points"],
        }
        for window_id, block in windows.items()
    }
    payload["compare"] = compare_windows(windows)
    payload["points"] = all_points

    DATA.mkdir(exist_ok=True)
    json_path = DATA / "backtest.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {json_path} · default window {default}")

    csv_path = DATA / "backtest-points.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "window",
                "origin",
                "target",
                "year",
                "month",
                "category",
                "field",
                "horizon",
                "predicted",
                "actual",
                "error",
                "chart_month",
            ],
        )
        writer.writeheader()
        writer.writerows(all_points)
    print(f"Wrote {csv_path}")

    html_blob = dashboard_payload(payload)
    if not args.skip_html and HTML_PATH.exists() and "// BACKTEST_START" in HTML_PATH.read_text():
        patch_html(html_blob)
        print(f"Updated {HTML_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
