#!/usr/bin/env python3
"""Scarica livelli idrometrici (e opzionalmente pioggia) storici ARPAE.

Fonte: https://dati-simc.arpae.it/opendata/osservati/meteo/storico/YYYY-MM.json.gz
Variabili:
  - B13215 livello idrometrico istantaneo, timerange [254,0,0]
  - B13011 precipitazione cumulata 15′, timerange [1,0,900] (pluviometri in config.rainfall)

Scrive CSV `time,value` in --out (livelli) e --out/rain (pioggia).
I file .json.gz grezzi restano in --cache (default ./storico/raw) per riuso.
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


def extract_var(
    gz_path: Path,
    station_names: set[str],
    var_code: str,
    timerange: list[int],
) -> dict[str, list[tuple[str, float]]]:
    """Return {station: [(iso_utc, value), ...]} for a BUFR variable/timerange."""
    out: dict[str, list[tuple[str, float]]] = {n: [] for n in station_names}
    if not station_names:
        return out
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
            if name not in station_names:
                continue
            stamp = obj.get("date")
            if not stamp:
                continue
            for block in data[1:]:
                if block.get("timerange") != timerange:
                    continue
                payload = (block.get("vars") or {}).get(var_code)
                if not payload:
                    continue
                value = payload.get("v")
                if isinstance(value, (int, float)):
                    out[name].append((stamp, float(value)))
    return out


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


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Scarica storico livelli/pioggia ARPAE per il dashboard")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--out", type=Path, default=root / "storico", help="Directory CSV per stazione")
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
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stations = set(config.get("sensors", {}).keys())
    rain_gauges = set() if args.no_rain else set((config.get("rainfall") or {}).get("gauges") or [])
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
    for year, month in months:
        try:
            gz_path = download_month(year, month, args.cache, force=args.force)
        except FileNotFoundError as exc:
            print(f"  skip: {exc}", flush=True)
            continue
        print(f"  extract {gz_path.name} …", flush=True)
        chunk = extract_var(gz_path, stations, LEVEL_CODE, INSTANT_TR)
        counts = {k: len(v) for k, v in chunk.items() if v}
        print(f"  livelli: {sum(counts.values())} punti su {len(counts)} stazioni", flush=True)
        merge_series(level_buckets, chunk)
        if rain_gauges:
            rain_chunk = extract_var(gz_path, rain_gauges, RAIN_CODE, RAIN_TR)
            rcounts = {k: len(v) for k, v in rain_chunk.items() if v}
            print(f"  pioggia: {sum(rcounts.values())} punti su {len(rcounts)} pluviometri", flush=True)
            merge_series(rain_buckets, rain_chunk)

    print("Scrittura CSV livelli…", flush=True)
    total = 0
    for name in sorted(stations):
        n = write_csv(args.out / f"{name}.csv", level_buckets.get(name, {}))
        total += n
        print(f"  {name}: {n} punti")

    rain_total = 0
    rain_dir = args.out / "rain"
    if rain_gauges:
        print("Scrittura CSV pioggia…", flush=True)
        for name in sorted(rain_gauges):
            n = write_csv(rain_dir / f"{name}.csv", rain_buckets.get(name, {}))
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
        "fetchedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Totale livelli {total} + pioggia {rain_total} → {args.out}")
    print("Poi: uv run python tools/fit-transfer.py --data ./storico --out models/transfer-coeffs.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
