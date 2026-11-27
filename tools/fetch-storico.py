#!/usr/bin/env python3
"""Scarica livelli idrometrici (e opzionalmente pioggia) storici ARPAE.

Fonte: https://dati-simc.arpae.it/opendata/osservati/meteo/storico/YYYY-MM.json.gz
Variabili:
  - B13215 livello idrometrico istantaneo, timerange [254,0,0]
  - B13011 precipitazione cumulata 15′, timerange [1,0,900] (pluviometri in config.rainfall)

Scrive Parquet `time,value` (zstd, ordinato per tempo) in --out (livelli) e
--out/rain (pioggia); `--format csv|both` resta disponibile per compatibilità.
La cache in --cache (default ./storico/raw) è Parquet flatten mensile: i .json.gz
scaricati vengono convertiti al volo e rimossi. Per migrare una cache già presente
senza perdere i gz finché non c'è parità: tools/verify-storico-cache.py.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
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
ROW_GROUP_SIZE = 1_000_000

# Decodifica Bufr usata in code_desc (ignoti → NULL in SQL).
CODE_DESC: dict[str, str] = {
    "B01019": "nome stazione",
    "B01194": "rete",
    "B04001": "anno",
    "B04002": "mese",
    "B04003": "giorno",
    "B04004": "ora",
    "B04005": "minuto",
    "B04006": "secondo",
    "B05001": "latitudine",
    "B06001": "longitudine",
    "B07030": "altezza stazione",
    "B10004": "pressione",
    "B11001": "direzione vento",
    "B11002": "velocità vento",
    "B12101": "temperatura",
    "B13003": "umidità relativa",
    "B13011": "precipitazione",
    "B13215": "livello idrometrico",
    "B22001": "direzione onde",
    "B22021": "altezza onde",
    "B22043": "temperatura mare",
    "B33007": "percentuale confidenza",
}


def _sql_quote(path: Path) -> str:
    return str(path).replace("'", "''")


def _code_desc_sql(code_expr: str = "v.key") -> str:
    parts = [f"WHEN '{code}' THEN '{desc.replace(chr(39), chr(39)+chr(39))}'" for code, desc in CODE_DESC.items()]
    return f"CASE {code_expr} {' '.join(parts)} ELSE NULL END"

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


def month_stem(cache_dir: Path, year: int, month: int) -> Path:
    return cache_dir / f"{year:04d}-{month:02d}"


def json_gz_to_parquet(gz_path: Path, out_path: Path) -> None:
    """Convert monthly .json.gz to flat typed Parquet (zstd). Does not delete the gz."""
    if duckdb is None:
        raise SystemExit("duckdb non disponibile: uv sync")
    src = _sql_quote(gz_path)
    dest = _sql_quote(out_path)
    code_desc = _code_desc_sql("v.key")
    # Atomic replace: write beside target then rename.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_sql = _sql_quote(tmp)
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
          WITH base AS (
            SELECT
              date AS stamp,
              version,
              network,
              ident,
              lon,
              lat,
              data,
              json_extract_string(data, '$[0].vars.B01019.v') AS station_b01019
            FROM read_json(
              '{src}',
              columns={{
                'date': 'VARCHAR',
                'version': 'VARCHAR',
                'network': 'VARCHAR',
                'ident': 'VARCHAR',
                'lon': 'INTEGER',
                'lat': 'INTEGER',
                'data': 'JSON'
              }},
              format='newline_delimited',
              maximum_object_size=8000000
            )
          ),
          blocks AS (
            SELECT
              stamp, station_b01019, version, network, ident, lon, lat,
              CAST(je.key AS INTEGER)::SMALLINT AS block_idx,
              je.value AS block
            FROM base, json_each(data) AS je
          ),
          exploded AS (
            SELECT
              b.stamp,
              b.station_b01019,
              b.version,
              b.network,
              b.ident,
              b.lon,
              b.lat,
              b.block_idx,
              TRY_CAST(json_extract(b.block, '$.timerange[0]') AS INTEGER) AS timerange_0,
              TRY_CAST(json_extract(b.block, '$.timerange[1]') AS INTEGER) AS timerange_1,
              TRY_CAST(json_extract(b.block, '$.timerange[2]') AS INTEGER) AS timerange_2,
              TRY_CAST(json_extract(b.block, '$.level[0]') AS INTEGER) AS level_0,
              TRY_CAST(json_extract(b.block, '$.level[1]') AS INTEGER) AS level_1,
              TRY_CAST(json_extract(b.block, '$.level[2]') AS INTEGER) AS level_2,
              TRY_CAST(json_extract(b.block, '$.level[3]') AS INTEGER) AS level_3,
              v.key AS code,
              {code_desc} AS code_desc,
              TRY_CAST(json_extract(v.value, '$.v') AS DOUBLE) AS value_num,
              json_extract(v.value, '$.a') AS attrs
            FROM blocks b, json_each(json_extract(b.block, '$.vars')) AS v
          )
          SELECT *
          FROM exploded
          ORDER BY
            version, network, ident,
            level_0, level_1, level_2, level_3,
            timerange_0, timerange_1, timerange_2,
            block_idx, code, station_b01019, lat, lon, stamp
        ) TO '{tmp_sql}' (
          FORMAT PARQUET,
          COMPRESSION ZSTD,
          ROW_GROUP_SIZE {ROW_GROUP_SIZE}
        )
        """
    )
    tmp.replace(out_path)


def download_month(year: int, month: int, cache_dir: Path, force: bool) -> Path:
    """Return the monthly cache file: flat .parquet preferred, .json.gz fallback."""
    stem = month_stem(cache_dir, year, month)
    parquet = stem.with_suffix(".parquet")
    gz = stem.with_suffix(".json.gz")
    if parquet.exists() and parquet.stat().st_size > 0 and not force:
        print(f"  cache hit {parquet.name}", flush=True)
        return parquet
    if gz.exists() and gz.stat().st_size > 0 and not force:
        if duckdb is not None:
            print(f"  convert cache {gz.name} -> parquet …", flush=True)
            json_gz_to_parquet(gz, parquet)
            # Keep .json.gz until tools/verify-storico-cache.py --delete-json
            return parquet
        print(f"  cache hit {gz.name}", flush=True)
        return gz
    name = gz.name
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
    gz.write_bytes(data)
    print(f"  wrote {gz.name} ({len(data) / 1e6:.1f} MB)", flush=True)
    if duckdb is not None:
        json_gz_to_parquet(gz, parquet)
        # Keep .json.gz until tools/verify-storico-cache.py --delete-json
        print(f"  cache parquet {parquet.name} ({parquet.stat().st_size / 1e6:.1f} MB)", flush=True)
        return parquet
    return gz


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
    if gz_path.suffix == ".parquet":
        return _extract_month_parquet(
            gz_path, level_names, rain_names, level_code, level_tr, rain_code, rain_tr
        )
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


def _extract_month_parquet(
    parquet_path: Path,
    level_names: set[str],
    rain_names: set[str],
    level_code: str = LEVEL_CODE,
    level_tr: list[int] | None = None,
    rain_code: str = RAIN_CODE,
    rain_tr: list[int] | None = None,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, list[tuple[str, float]]]]:
    """Same extraction as extract_month but reading the flat monthly Parquet cache."""
    level_tr = INSTANT_TR if level_tr is None else level_tr
    rain_tr = RAIN_TR if rain_tr is None else rain_tr
    levels: dict[str, list[tuple[str, float]]] = {n: [] for n in level_names}
    rains: dict[str, list[tuple[str, float]]] = {n: [] for n in rain_names}
    wanted = level_names | rain_names
    if not wanted:
        return levels, rains
    lvl_tr, rn_tr = tuple(level_tr), tuple(rain_tr)
    rows = duckdb.connect().execute(
        """
        SELECT stamp, station_b01019, timerange_0, timerange_1, timerange_2,
               code, value_num
        FROM read_parquet(?)
        WHERE station_b01019 = ANY(?::VARCHAR[])
          AND code = ANY(?::VARCHAR[])
          AND value_num IS NOT NULL
        ORDER BY station_b01019, stamp, block_idx
        """,
        [str(parquet_path), sorted(wanted), [level_code, rain_code]],
    ).fetchall()
    for stamp, station, tr0, tr1, tr2, code, value in rows:
        if station is None or value is None or not math.isfinite(float(value)):
            continue
        tr_t = (tr0, tr1, tr2)
        stamp_s = str(stamp)
        val = float(value)
        if station in level_names and code == level_code and tr_t == lvl_tr:
            levels[station].append((stamp_s, val))
        if station in rain_names and code == rain_code and tr_t == rn_tr:
            rains[station].append((stamp_s, val))
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


def merge_existing(out_dir: Path, name: str, series: dict[str, float]) -> dict[str, float]:
    """Unisce le serie già presenti su disco: gli aggiornamenti incrementali non
    cancellano l'archivio lungo. I valori nuovi vincono sullo stesso timestamp."""
    merged: dict[str, float] = {}
    csv_path = out_dir / f"{name}.csv"
    parquet_path = out_dir / f"{name}.parquet"
    if duckdb is not None and parquet_path.exists():
        src = str(parquet_path).replace("'", "''")
        for stamp, value in duckdb.sql(f"SELECT time, value FROM '{src}'").fetchall():
            if stamp is not None:
                merged[str(stamp)] = value
    elif csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("time"):
                    merged[row["time"]] = float(row["value"]) if row.get("value") not in (None, "") else None
    merged.update(series)
    return merged


def write_series(out_dir: Path, name: str, series: dict[str, float], fmt: str) -> int:
    series = merge_existing(out_dir, name, series)
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
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "storico" / "raw",
        help="Cache mensile parquet flatten (json.gz convertiti al volo)",
    )
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
        default="parquet",
        help="Formato di scrittura (default parquet: compatto, zstd; 'both' aggiunge i CSV)",
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
