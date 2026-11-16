#!/usr/bin/env python3
"""Scarica livelli idrometrici (e opzionalmente pioggia) storici ARPAE.

Fonte: https://dati-simc.arpae.it/opendata/osservati/meteo/storico/YYYY-MM.json.gz
Variabili:
  - B13215 livello idrometrico istantaneo, timerange [254,0,0]
  - B13011 precipitazione cumulata 15′, timerange [1,0,900] (pluviometri in config.rainfall)

Scrive CSV e/o Parquet `time,value` in --out (livelli) e --out/rain (pioggia).
I file .json.gz grezzi restano in --cache (default ./storico/raw) per riuso.
Il CSV resta per i tool JS (backtest); il Parquet è preferito dai fit Python via DuckDB.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore

BASE_URL = "https://dati-simc.arpae.it/opendata/osservati/meteo/storico"
LEVEL_CODE = "B13215"
INSTANT_TR = [254, 0, 0]
RAIN_CODE = "B13011"
RAIN_TR = [1, 0, 900]
USER_AGENT = "livelliReno-fetch-storico/1.0 (+https://github.com/ComuneDiCento/livelliReno)"


def month_range(start: date, end: date) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def parse_ym(text: str) -> date:
    year, month = text.split("-", 1)
    return date(int(year), int(month), 1)


def download_month(year: int, month: int, cache_dir: Path, force: bool) -> Path:
    name = f"{year:04d}-{month:02d}.json.gz"
    dest = cache_dir / name
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  cache hit {dest.name}", flush=True)
        return dest
    url = f"{BASE_URL}/{name}"
    print(f"  download {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(f"mese non disponibile: {name}") from exc
        raise
    dest.write_bytes(data)
    print(f"  wrote {dest.name} ({len(data) / 1e6:.1f} MB)", flush=True)
    return dest


def extract_month(
    gz_path: Path,
    level_names: set[str],
    rain_names: set[str],
    level_code: str = LEVEL_CODE,
    level_tr: list[int] | None = None,
    rain_code: str = RAIN_CODE,
    rain_tr: list[int] | None = None,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, list[tuple[str, float]]]]:
    """One pass over the monthly file for levels and rainfall."""
    level_tr = INSTANT_TR if level_tr is None else level_tr
    rain_tr = RAIN_TR if rain_tr is None else rain_tr
    levels: dict[str, list[tuple[str, float]]] = {n: [] for n in level_names}
    rains: dict[str, list[tuple[str, float]]] = {n: [] for n in rain_names}
    wanted = level_names | rain_names
    if not wanted:
        return levels, rains
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            data = obj.get("data") or []
            if not data:
                continue
            meta = (data[0].get("vars") or {})
            name = (meta.get("B01019") or {}).get("v")
            if name not in wanted:
                continue
            stamp = obj.get("date")
            if not stamp:
                continue
            want_level = name in level_names
            want_rain = name in rain_names
            for block in data[1:]:
                tr = block.get("timerange")
                payload_vars = block.get("vars") or {}
                if want_level and tr == level_tr:
                    payload = payload_vars.get(level_code)
                    if payload and isinstance(payload.get("v"), (int, float)):
                        levels[name].append((stamp, float(payload["v"])))
                if want_rain and tr == rain_tr:
                    payload = payload_vars.get(rain_code)
                    if payload and isinstance(payload.get("v"), (int, float)):
                        rains[name].append((stamp, float(payload["v"])))
    return levels, rains


def merge_series(
    buckets: dict[str, dict[str, float]],
    chunk: dict[str, list[tuple[str, float]]],
) -> None:
    for name, points in chunk.items():
        series = buckets.setdefault(name, {})
        for stamp, value in points:
            series[stamp] = value  # last write wins on duplicates


def write_csv(path: Path, series: dict[str, float]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(series.items(), key=lambda p: p[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "value"])
        for stamp, value in rows:
            writer.writerow([stamp, value])
    return len(rows)


def write_parquet(path: Path, series: dict[str, float]) -> int:
    """Compact columnar store; preferred by Python loaders that use DuckDB."""
    if duckdb is None:
        raise SystemExit("duckdb non disponibile: uv sync, oppure --format csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(series.items(), key=lambda p: p[0])
    con = duckdb.connect()
    dest = str(path).replace("'", "''")
    if not rows:
        con.execute(
            "COPY (SELECT CAST(NULL AS VARCHAR) AS time, CAST(NULL AS DOUBLE) AS value "
            f"WHERE FALSE) TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        return 0
    # Keep time as ISO VARCHAR (same as CSV) so loaders can CAST without pytz.
    times = [t for t, _ in rows]
    values = [v for _, v in rows]
    con.execute(
        """
        CREATE TEMP TABLE series AS
        SELECT time, CAST(value AS DOUBLE) AS value
        FROM (SELECT UNNEST(?::VARCHAR[]) AS time, UNNEST(?::DOUBLE[]) AS value)
        ORDER BY 1
        """,
        [times, values],
    )
    con.execute(f"COPY series TO '{dest}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    return len(rows)


def write_series(out_dir: Path, name: str, series: dict[str, float], fmt: str) -> int:
    n = 0
    if fmt in ("csv", "both"):
        n = write_csv(out_dir / f"{name}.csv", series)
    if fmt in ("parquet", "both"):
        n = write_parquet(out_dir / f"{name}.parquet", series)
    return n


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Scarica storico livelli/pioggia ARPAE per il dashboard")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--out", type=Path, default=root / "storico", help="Directory CSV/Parquet per stazione")
    parser.add_argument("--cache", type=Path, default=root / "storico" / "raw", help="Cache file .json.gz mensili")
    parser.add_argument("--from", dest="date_from", type=str, default=None, help="YYYY-MM (incluso)")
    parser.add_argument("--to", dest="date_to", type=str, default=None, help="YYYY-MM (incluso)")
    parser.add_argument(
        "--months",
        type=int,
        default=24,
        help="Se --from assente: ultimi N mesi fino a --to/oggi (default 24: serve variabilità idrologica, non solo magra)",
    )
    parser.add_argument("--force", action="store_true", help="Riscarica i .json.gz anche se in cache")
    parser.add_argument("--no-rain", action="store_true", help="Non estrarre i pluviometri")
    parser.add_argument(
        "--format",
        choices=("csv", "parquet", "both"),
        default="both",
        help="Formato di scrittura (default both: CSV per JS + Parquet per fit Python)",
    )
    args = parser.parse_args()
    if args.format in ("parquet", "both") and duckdb is None:
        print("duckdb richiesto per --format parquet/both", file=sys.stderr)
        return 1

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stations = set(config.get("sensors", {}).keys())
    raw_gauges = [] if args.no_rain else list((config.get("rainfall") or {}).get("gauges") or [])
    rain_gauges = set()
    for entry in raw_gauges:
        if isinstance(entry, str):
            rain_gauges.add(entry)
        elif isinstance(entry, dict) and entry.get("name"):
            rain_gauges.add(str(entry["name"]))
    if not stations:
        print("Nessuna stazione in config.json", file=sys.stderr)
        return 1

    today = date.today()
    end = parse_ym(args.date_to) if args.date_to else date(today.year, today.month, 1)
    if args.date_from:
        start = parse_ym(args.date_from)
    else:
        y, m = end.year, end.month - (args.months - 1)
        while m <= 0:
            m += 12
            y -= 1
        start = date(y, m, 1)

    months = month_range(start, end)
    print(
        f"Stazioni livello: {len(stations)} | pioggia: {len(rain_gauges)} | "
        f"mesi: {months[0][0]:04d}-{months[0][1]:02d} … {months[-1][0]:04d}-{months[-1][1]:02d}"
    )
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    level_buckets: dict[str, dict[str, float]] = {n: {} for n in stations}
    rain_buckets: dict[str, dict[str, float]] = {n: {} for n in rain_gauges}
    n_months = len(months)
    for idx, (year, month) in enumerate(months, start=1):
        try:
            gz_path = download_month(year, month, args.cache, force=args.force)
        except FileNotFoundError as exc:
            print(f"  skip: {exc}", flush=True)
            continue
        print(f"  [{idx}/{n_months}] extract {gz_path.name} …", flush=True)
        chunk, rain_chunk = extract_month(gz_path, stations, rain_gauges)
        counts = {k: len(v) for k, v in chunk.items() if v}
        print(f"  livelli: {sum(counts.values())} punti su {len(counts)} stazioni", flush=True)
        merge_series(level_buckets, chunk)
        if rain_gauges:
            rcounts = {k: len(v) for k, v in rain_chunk.items() if v}
            print(f"  pioggia: {sum(rcounts.values())} punti su {len(rcounts)} pluviometri", flush=True)
            merge_series(rain_buckets, rain_chunk)

    print(f"Scrittura livelli ({args.format})…", flush=True)
    total = 0
    for name in sorted(stations):
        n = write_series(args.out, name, level_buckets.get(name, {}), args.format)
        total += n
        print(f"  {name}: {n} punti")

    rain_total = 0
    rain_dir = args.out / "rain"
    if rain_gauges:
        print(f"Scrittura pioggia ({args.format})…", flush=True)
        for name in sorted(rain_gauges):
            n = write_series(rain_dir, name, rain_buckets.get(name, {}), args.format)
            rain_total += n
            print(f"  rain/{name}: {n} punti")

    meta = {
        "source": BASE_URL,
        "variable": LEVEL_CODE,
        "timerange": INSTANT_TR,
        "rainfallVariable": RAIN_CODE,
        "rainfallTimerange": RAIN_TR,
        "from": f"{months[0][0]:04d}-{months[0][1]:02d}",
        "to": f"{months[-1][0]:04d}-{months[-1][1]:02d}",
        "stations": sorted(stations),
        "rainfallGauges": sorted(rain_gauges),
        "points": total,
        "rainfallPoints": rain_total,
        "format": args.format,
        "fetchedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Totale livelli {total} + pioggia {rain_total} → {args.out}")
    print("Poi: node tools/build-event-catalog.js && uv run python tools/fit-ml-cento.py …")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
