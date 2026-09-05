#!/usr/bin/env python3
"""Scrape APCIQ monthly statistic PDFs and extract Île de Montréal series.

Pulls monthly PDF links from the current barometer page and the archive:

  https://apciq.ca/barometre-residentiel/statistiques-mensuelles/
  https://apciq.ca/barometre-residentiel/statistiques-mensuelles-archive/

Then extracts Copropriété + Plex (2-5 logements) figures for Île de Montréal:

  - Ventes totales
  - Inscriptions en vigueur
  - Prix médian
  - Moyenne de jours sur le marché

Usage:
    python scripts/fetch_monthly_stats.py
    python scripts/fetch_monthly_stats.py --years 2019 2020 2021
    python scripts/fetch_monthly_stats.py --skip-download
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyMuPDF is required. Run: pip install -r requirements.txt") from exc

SOURCE_URL = "https://apciq.ca/barometre-residentiel/statistiques-mensuelles/"
ARCHIVE_URL = "https://apciq.ca/barometre-residentiel/statistiques-mensuelles-archive/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

MONTH_FR = {
    1: "Janvier",
    2: "Février",
    3: "Mars",
    4: "Avril",
    5: "Mai",
    6: "Juin",
    7: "Juillet",
    8: "Août",
    9: "Septembre",
    10: "Octobre",
    11: "Novembre",
    12: "Décembre",
}

MONTH_LOOKUP = {
    "janvier": 1,
    "fevrier": 2,
    "février": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "août": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
    "décembre": 12,
}


@dataclass
class MonthlyLink:
    year: int
    month: int
    label: str
    url: str


@dataclass
class CategoryMetrics:
    ventes: int | None
    inscriptions: int | None
    prix: int | None
    jours: int | None
    ventes_yoy: float | None
    inscriptions_yoy: float | None
    prix_yoy: float | None
    jours_delta: float | None
    ventes_ytd: int | None
    inscriptions_ytd: int | None
    prix_ytd: int | None
    jours_ytd: int | None


@dataclass
class MonthlyRow:
    year: int
    month: int
    label: str
    url: str
    pdf: str
    region: str
    copropriete: CategoryMetrics
    plex: CategoryMetrics


class StatsPageParser(HTMLParser):
    """Collect PDF hrefs that sit under a 2024 / 2025 / 2026 heading."""

    def __init__(self) -> None:
        super().__init__()
        self.current_year: int | None = None
        self._capture_heading = False
        self._heading_buf: list[str] = []
        self.links: list[tuple[int, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = dict(attrs)
        if tag in {"h2", "h3"}:
            self._capture_heading = True
            self._heading_buf = []
        if tag == "a" and self.current_year and attrs_d.get("href"):
            href = attrs_d["href"]
            self.links.append((self.current_year, href, ""))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h2", "h3"} and self._capture_heading:
            text = "".join(self._heading_buf).strip()
            match = re.fullmatch(r"20\d{2}", text)
            self.current_year = int(text) if match else None
            self._capture_heading = False
            self._heading_buf = []

    def handle_data(self, data: str) -> None:
        if self._capture_heading:
            self._heading_buf.append(data)
        if self.links and self.links[-1][2] == "" and data.strip():
            year, href, _ = self.links[-1]
            self.links[-1] = (year, href, data.strip())


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def normalize_month_label(text: str) -> tuple[int, int] | None:
    text = re.sub(r"\s+", " ", text).strip().lower()
    match = re.search(
        r"(janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[eé]cembre)\s*(20\d{2})",
        text,
    )
    if not match:
        return None
    month = MONTH_LOOKUP[match.group(1)]
    year = int(match.group(2))
    return year, month


def parse_href_date(href: str) -> tuple[int, int] | None:
    """Best-effort year/month from an APCIQ PDF filename."""
    name = href.rsplit("/", 1)[-1].lower()
    patterns = [
        r"(20\d{2})[-_]?0?([1-9]|1[0-2])(?!\d)",
        r"(20\d{2})(0[1-9]|1[0-2])",
        r"(0[1-9]|1[0-2])[-_](20\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            a, b = match.group(1), match.group(2)
            if len(a) == 4:
                return int(a), int(b)
            return int(b), int(a)
    month_name = re.search(
        r"(janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|aout|septembre|octobre|novembre|d[eé]cembre).*?(20\d{2})",
        name,
    )
    if month_name:
        key = month_name.group(1).replace("aout", "août")
        return int(month_name.group(2)), MONTH_LOOKUP[key]
    return None


def _record_link(
    found: dict[tuple[int, int], MonthlyLink],
    year: int,
    month: int,
    href: str,
    base: str,
) -> None:
    found[(year, month)] = MonthlyLink(
        year=year,
        month=month,
        label=f"{MONTH_FR[month]} {year}",
        url=urljoin(base, href),
    )


def discover_links(
    html: str,
    years: Iterable[int],
    base: str = SOURCE_URL,
) -> list[MonthlyLink]:
    wanted = set(years)
    found: dict[tuple[int, int], MonthlyLink] = {}

    parser = StatsPageParser()
    parser.feed(html)
    for year, href, label in parser.links:
        if not href.lower().split("?")[0].endswith(".pdf"):
            continue
        parsed = normalize_month_label(label) or parse_href_date(href)
        if parsed is None:
            continue
        year, month = parsed
        if year in wanted:
            _record_link(found, year, month, href, base)

    for href, label in re.findall(
        r'href="([^"]+\.pdf)"[^>]*>([^<]*)',
        html,
        flags=re.IGNORECASE,
    ):
        parsed = normalize_month_label(label) or parse_href_date(href)
        if parsed is None:
            continue
        year, month = parsed
        if year in wanted:
            _record_link(found, year, month, href, base)

    return [found[key] for key in sorted(found)]


def merge_links(*groups: list[MonthlyLink]) -> list[MonthlyLink]:
    found: dict[tuple[int, int], MonthlyLink] = {}
    for group in groups:
        for link in group:
            found[(link.year, link.month)] = link
    return [found[key] for key in sorted(found)]


def _candidate_urls(url: str) -> list[str]:
    urls = [url]
    if url.startswith("http://"):
        urls.append("https://" + url[len("http://") :])
    return urls


def download_pdf(link: MonthlyLink, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{link.year}{link.month:02d}.pdf"
    if path.exists() and path.stat().st_size > 10_000:
        if path.read_bytes()[:4] == b"%PDF":
            return path
        path.unlink()
    last_error = "download failed"
    for url in _candidate_urls(link.url):
        try:
            data = http_get(url)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            continue
        if data[:4] == b"%PDF":
            path.write_bytes(data)
            link.url = url
            return path
        last_error = f"Not a PDF: {url}"
    raise RuntimeError(last_error)


def find_ile_page(doc: fitz.Document) -> int | None:
    needles = (
        "Île de Montréal",
        "Ile de Montréal",
        "Île de Montreal",
        "Ile de Montreal",
    )
    fallback = None
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if not any(n in text for n in needles):
            continue
        if len(text) > 400:
            return i
        fallback = i
    return fallback


YEAR_TOKEN = re.compile(r"^20\d{2}$")
SKIP_TOKENS = {"⬆", "⬇", "⬌", "", "", "D", "$", "%", "Variation"}


def _join_lines(words: list, y_tol: float = 12.0) -> list[list]:
    words = sorted(words, key=lambda w: (w[1], w[0]))
    lines: list[list] = []
    current: list = []
    current_y: float | None = None
    for word in words:
        if current_y is None or abs(word[1] - current_y) <= y_tol:
            current.append(word)
            if current_y is None:
                current_y = word[1]
        else:
            lines.append(sorted(current, key=lambda item: item[0]))
            current = [word]
            current_y = word[1]
    if current:
        lines.append(sorted(current, key=lambda item: item[0]))
    return lines


def _column_anchors(words: list) -> list[float]:
    """X positions of the six value columns from the category header row."""
    for line in _join_lines(words, y_tol=8):
        texts = [w[4] for w in line]
        years = [w[0] for w in line if YEAR_TOKEN.match(w[4])]
        variations = [w[0] for w in line if w[4] == "Variation"]
        if len(years) >= 4 and len(variations) >= 2:
            return [
                years[0],
                years[1],
                variations[0],
                years[2],
                years[3],
                variations[1],
            ]
        if len(years) >= 2 and len(variations) >= 1:
            # Some early pages only expose the monthly trio clearly.
            return [years[0], years[1], variations[0]]
    return []


def _assign_columns(line: list, anchors: list[float]) -> list[list[str]]:
    buckets = [[] for _ in anchors]
    if not anchors:
        return buckets
    bounds = []
    for i, x in enumerate(anchors):
        left = (anchors[i - 1] + x) / 2 if i else x - 80
        right = (x + anchors[i + 1]) / 2 if i + 1 < len(anchors) else x + 120
        bounds.append((left, right))
    for word in line:
        token = word[4]
        if token in SKIP_TOKENS or YEAR_TOKEN.match(token):
            continue
        if not re.search(r"[\d*]", token):
            continue
        x = word[0]
        best = min(range(len(anchors)), key=lambda i: abs(x - anchors[i]))
        left, right = bounds[best]
        if left - 15 <= x <= right + 15:
            buckets[best].append(token)
    return buckets


def _to_number(parts: list[str], kind: str) -> float | int | None:
    cleaned = [p for p in parts if p not in SKIP_TOKENS]
    raw = (
        "".join(cleaned)
        .replace("\xa0", "")
        .replace(" ", "")
        .replace(",", "")
        .replace("−", "-")
    )
    if not raw or raw == "**":
        return None
    raw = raw.replace("%", "").replace("$", "")
    try:
        if kind in {"prix", "ventes", "inscriptions", "jours"}:
            if "." in raw:
                value = float(raw)
                return int(value) if value.is_integer() else value
            return int(raw)
        return float(raw)
    except ValueError:
        return None


def _metric_name(texts: list[str]) -> str | None:
    joined = " ".join(texts).lower()
    if "ventes" in joined:
        return "ventes"
    if "inscriptions" in joined:
        return "inscriptions"
    if "prix" in joined:
        return "prix"
    if "jours" in joined or "délai" in joined or "delai" in joined or "moy." in joined:
        return "jours"
    return None


def _parse_section(words: list) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    anchors = _column_anchors(words)
    for line in _join_lines(words):
        texts = [w[4] for w in line]
        metric = _metric_name(texts)
        if not metric:
            continue
        buckets = _assign_columns(line, anchors)
        month = _to_number(buckets[0], metric) if buckets else None
        month_prior = _to_number(buckets[1], metric) if len(buckets) > 1 else None
        var = _to_number(
            buckets[2],
            "jours" if metric == "jours" else metric,
        ) if len(buckets) > 2 else None
        ytd = _to_number(buckets[3], metric) if len(buckets) > 3 else None
        metrics[metric] = {
            "month": month,
            "month_prior": month_prior,
            "var": var,
            "ytd": ytd,
        }
    return metrics


def _metrics_from_section(section: dict[str, dict]) -> CategoryMetrics:
    ventes = section.get("ventes", {})
    insc = section.get("inscriptions", {})
    prix = section.get("prix", {})
    jours = section.get("jours", {})
    return CategoryMetrics(
        ventes=_as_int(ventes.get("month")),
        inscriptions=_as_int(insc.get("month")),
        prix=_as_int(prix.get("month")),
        jours=_as_int(jours.get("month")),
        ventes_yoy=_as_float(ventes.get("var")),
        inscriptions_yoy=_as_float(insc.get("var")),
        prix_yoy=_as_float(prix.get("var")),
        jours_delta=_as_float(jours.get("var")),
        ventes_ytd=_as_int(ventes.get("ytd")),
        inscriptions_ytd=_as_int(insc.get("ytd")),
        prix_ytd=_as_int(prix.get("ytd")),
        jours_ytd=_as_int(jours.get("ytd")),
    )


def _as_int(value: float | int | None) -> int | None:
    if value is None:
        return None
    return int(round(float(value)))


def _as_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


YEAR_VALUES = set(range(2014, 2028))


def _slice_category(text: str, start_label: str, end_label: str) -> str:
    start = text.find(start_label)
    if start < 0:
        return ""
    end = text.find(end_label, start + len(start_label))
    return text[start:end] if end > 0 else text[start:]


def _numbers_after(block: str, label: str, limit: int = 8) -> list[int]:
    idx = block.lower().find(label.lower())
    if idx < 0:
        return []
    tail = block[idx + len(label) :]
    values: list[int] = []
    for match in re.finditer(r"\*{2}|-?\d[\d \t,\u00a0]*", tail):
        token = match.group().strip()
        if token.startswith("*"):
            continue
        raw = token.replace("\xa0", "").replace(" ", "").replace(",", "").replace("−", "-")
        if not re.fullmatch(r"-?\d+", raw):
            continue
        number = int(raw)
        if number in YEAR_VALUES:
            continue
        values.append(number)
        if len(values) >= limit:
            break
    return values


def _metrics_from_text_block(block: str) -> CategoryMetrics | None:
    ventes_label = "Ventes totales" if "Ventes totales" in block else "Ventes"
    jours_label = (
        "Moyenne de jours sur le marché"
        if "Moyenne de jours" in block
        else "Moy. jours sur le marché"
        if "Moy. jours" in block
        else "Délai de vente moyen"
    )
    ventes = _numbers_after(block, ventes_label)
    insc = _numbers_after(block, "Inscriptions en vigueur")
    prix = [n for n in _numbers_after(block, "Prix médian") if n >= 50_000]
    jours = [n for n in _numbers_after(block, jours_label) if 5 <= abs(n) <= 200]
    if not ventes:
        return None
    return CategoryMetrics(
        ventes=_as_int(ventes[0]),
        inscriptions=_as_int(insc[0] if insc else None),
        prix=_as_int(prix[0] if prix else None),
        jours=_as_int(jours[0] if jours else None),
        ventes_yoy=_as_float(ventes[2] if len(ventes) > 2 and abs(ventes[2]) <= 200 else None),
        inscriptions_yoy=_as_float(insc[2] if len(insc) > 2 and abs(insc[2]) <= 200 else None),
        prix_yoy=_as_float(prix[2] if len(prix) > 2 and abs(prix[2]) <= 200 else None),
        jours_delta=_as_float(jours[2] if len(jours) > 2 else None),
        ventes_ytd=_as_int(next((n for n in ventes[1:] if n > ventes[0] * 1.2), None)),
        inscriptions_ytd=_as_int(insc[3] if len(insc) > 3 else None),
        prix_ytd=_as_int(prix[3] if len(prix) > 3 else None),
        jours_ytd=_as_int(jours[3] if len(jours) > 3 else None),
    )


def extract_from_text(text: str) -> tuple[CategoryMetrics, CategoryMetrics] | None:
    """Fallback when word coordinates do not line up (older tableaux-web PDFs)."""
    copro_block = _slice_category(text, "Copropriété", "Plex")
    plex_idx = text.find("Plex")
    source_idx = text.find("Source", plex_idx if plex_idx > 0 else 0)
    plex_block = text[plex_idx:source_idx] if plex_idx >= 0 else ""
    if not copro_block or not plex_block:
        return None
    copro = _metrics_from_text_block(copro_block)
    plex = _metrics_from_text_block(plex_block)
    if copro is None or plex is None or copro.ventes is None or plex.ventes is None:
        return None
    return copro, plex


def extract_compact_ile_table(text: str) -> tuple[CategoryMetrics, CategoryMetrics] | None:
    """2019-style one-page Île table: category sales and prices only."""
    start = text.find("Île de Montréal")
    if start < 0:
        start = text.find("Ile de Montréal")
    if start < 0:
        return None
    end = text.find("Laval", start)
    block = text[start:end if end > start else start + 1500]
    if "Copropriété" not in block or "Plex" not in block:
        return None
    price_at = block.find("Prix médian")
    ventes_part = block[:price_at] if price_at > 0 else block
    price_part = block[price_at:] if price_at > 0 else ""
    copro_sales = [n for n in _numbers_after(ventes_part, "Copropriété") if n >= 50]
    plex_sales = [n for n in _numbers_after(ventes_part, "Plex") if n >= 50]
    copro_price = [n for n in _numbers_after(price_part, "Copropriété") if n >= 150_000]
    plex_price = [n for n in _numbers_after(price_part, "Plex") if n >= 150_000]
    if not copro_sales or not plex_sales:
        return None
    copro_v, plex_v = copro_sales[0], plex_sales[0]
    copro = CategoryMetrics(
        ventes=copro_v,
        inscriptions=None,
        prix=copro_price[0] if copro_price else None,
        jours=None,
        ventes_yoy=None,
        inscriptions_yoy=None,
        prix_yoy=None,
        jours_delta=None,
        ventes_ytd=None,
        inscriptions_ytd=None,
        prix_ytd=None,
        jours_ytd=None,
    )
    plex = CategoryMetrics(
        ventes=plex_v,
        inscriptions=None,
        prix=plex_price[0] if plex_price else None,
        jours=None,
        ventes_yoy=None,
        inscriptions_yoy=None,
        prix_yoy=None,
        jours_delta=None,
        ventes_ytd=None,
        inscriptions_ytd=None,
        prix_ytd=None,
        jours_ytd=None,
    )
    return copro, plex


def _looks_plausible(copro: CategoryMetrics, plex: CategoryMetrics) -> bool:
    if copro.ventes is None or plex.ventes is None:
        return False
    if not (200 <= copro.ventes <= 2500):
        return False
    if not (40 <= plex.ventes <= 600):
        return False
    if copro.prix is not None and not (150_000 <= copro.prix <= 800_000):
        return False
    if plex.prix is not None and not (300_000 <= plex.prix <= 1_500_000):
        return False
    return True


def extract_ile_montreal(pdf_path: Path) -> tuple[CategoryMetrics, CategoryMetrics]:
    doc = fitz.open(pdf_path)
    page_index = find_ile_page(doc)
    candidates: list[tuple[CategoryMetrics, CategoryMetrics]] = []
    if page_index is not None:
        pages = [doc[page_index]]
    else:
        pages = [
            page
            for page in doc
            if "Île de Montréal" in (page.get_text() or "")
            or "Ile de Montréal" in (page.get_text() or "")
        ]

    for page in pages:
        text = page.get_text() or ""
        words = page.get_text("words")
        headers: dict[str, float] = {}
        for word in words:
            if word[4] in {"Copropriété", "Plex", "Unifamiliale"}:
                headers[word[4]] = word[1]
        if "Copropriété" in headers and "Plex" in headers:
            copro_y = headers["Copropriété"]
            plex_y = headers["Plex"]
            end_y = max(word[1] for word in words)
            for word in words:
                if word[4] == "Source" and word[1] > plex_y:
                    end_y = word[1]
                    break

            def between(y0: float, y1: float) -> list:
                return [w for w in words if y0 - 5 <= w[1] < y1 - 5]

            copro = _metrics_from_section(_parse_section(between(copro_y, plex_y)))
            plex = _metrics_from_section(_parse_section(between(plex_y, end_y)))
            if copro.ventes is not None and plex.ventes is not None:
                candidates.append((copro, plex))
        try:
            text_pair = extract_from_text(text)
            if text_pair:
                candidates.append(text_pair)
        except Exception:
            pass
        try:
            compact = extract_compact_ile_table(text)
            if compact:
                candidates.append(compact)
        except Exception:
            pass

    if not candidates and page_index is None:
        raise RuntimeError(f"Île de Montréal page not found in {pdf_path.name}")

    def score(pair: tuple[CategoryMetrics, CategoryMetrics]) -> int:
        copro, plex = pair
        filled = [
            copro.ventes,
            copro.inscriptions,
            copro.prix,
            copro.jours,
            plex.ventes,
            plex.inscriptions,
            plex.prix,
            plex.jours,
        ]
        return sum(1 for value in filled if value is not None)

    ranked = sorted(candidates, key=score, reverse=True)
    for pair in ranked:
        if _looks_plausible(*pair):
            return pair
    if ranked:
        return ranked[0]
    raise RuntimeError(f"Could not parse sales figures in {pdf_path.name}")


def write_outputs(rows: list[MonthlyRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": SOURCE_URL,
        "archive": ARCHIVE_URL,
        "region": "Île de Montréal",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "months": [
            {
                "year": row.year,
                "month": row.month,
                "label": row.label,
                "url": row.url,
                "pdf": row.pdf,
                "copropriete": asdict(row.copropriete),
                "plex": asdict(row.plex),
            }
            for row in rows
        ],
    }
    (out_dir / "ile-montreal.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = out_dir / "ile-montreal.csv"
    fieldnames = [
        "year",
        "month",
        "label",
        "category",
        "ventes",
        "ventes_yoy",
        "inscriptions",
        "inscriptions_yoy",
        "prix",
        "prix_yoy",
        "jours",
        "jours_delta",
        "ventes_ytd",
        "url",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for category, metrics in (("copropriete", row.copropriete), ("plex", row.plex)):
                writer.writerow(
                    {
                        "year": row.year,
                        "month": row.month,
                        "label": row.label,
                        "category": category,
                        "ventes": metrics.ventes,
                        "ventes_yoy": metrics.ventes_yoy,
                        "inscriptions": metrics.inscriptions,
                        "inscriptions_yoy": metrics.inscriptions_yoy,
                        "prix": metrics.prix,
                        "prix_yoy": metrics.prix_yoy,
                        "jours": metrics.jours,
                        "jours_delta": metrics.jours_delta,
                        "ventes_ytd": metrics.ventes_ytd,
                        "url": row.url,
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2019, 2027)),
        help="Calendar years to scrape (default: 2019–2026)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse PDFs already in data/pdfs/",
    )
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_dir = args.data_dir / "pdfs"
    print(f"Fetching {SOURCE_URL}", flush=True)
    current_html = http_get(SOURCE_URL).decode("utf-8", errors="replace")
    print(f"Fetching {ARCHIVE_URL}", flush=True)
    archive_html = http_get(ARCHIVE_URL).decode("utf-8", errors="replace")
    links = merge_links(
        discover_links(archive_html, args.years, ARCHIVE_URL),
        discover_links(current_html, args.years, SOURCE_URL),
    )
    if not links:
        print("No monthly PDF links found.", file=sys.stderr)
        return 1
    print(f"Found {len(links)} monthly PDFs", flush=True)

    rows: list[MonthlyRow] = []
    failures: list[str] = []
    for link in links:
        print(f"  {link.label}: {link.url}", flush=True)
        try:
            pdf_path = (
                pdf_dir / f"{link.year}{link.month:02d}.pdf"
                if args.skip_download
                else download_pdf(link, pdf_dir)
            )
            if args.skip_download and not pdf_path.exists():
                raise FileNotFoundError(pdf_path)
            copro, plex = extract_ile_montreal(pdf_path)
            rows.append(
                MonthlyRow(
                    year=link.year,
                    month=link.month,
                    label=link.label,
                    url=link.url,
                    pdf=pdf_path.name,
                    region="Île de Montréal",
                    copropriete=copro,
                    plex=plex,
                )
            )
            print(
                f"    copro ventes={copro.ventes}  plex ventes={plex.ventes}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 — keep going through the year
            failures.append(f"{link.label}: {exc}")
            print(f"    ERROR {exc}", flush=True)

    if rows:
        write_outputs(rows, args.data_dir)
        print(f"Wrote {args.data_dir / 'ile-montreal.json'} and ile-montreal.csv")
    if failures:
        print(f"\n{len(failures)} file(s) failed:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
