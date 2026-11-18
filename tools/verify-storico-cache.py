#!/usr/bin/env python3
"""Verifica parità cache mensile flatten vs .json.gz; unico gate per cancellare i gz.

Esempi:
  uv run python tools/verify-storico-cache.py --convert
  uv run python tools/verify-storico-cache.py
  uv run python tools/verify-storico-cache.py --delete-json
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fetch_storico", ROOT / "tools" / "fetch-storico.py"
)
assert _spec and _spec.loader
_fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fetch)

LEVEL_CODE = _fetch.LEVEL_CODE
RAIN_CODE = _fetch.RAIN_CODE
extract_month = _fetch.extract_month
json_gz_to_parquet = _fetch.json_gz_to_parquet
_sql_quote = _fetch._sql_quote

EXPECTED_COLS = {
    "stamp",
    "station_b01019",
    "version",
    "network",
    "ident",
    "lon",
    "lat",
    "block_idx",
    "timerange_0",
    "timerange_1",
    "timerange_2",
    "level_0",
    "level_1",
    "level_2",
    "level_3",
    "code",
    "code_desc",
    "value_num",
    "attrs",
}


def load_names(config_path: Path) -> tuple[set[str], set[str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stations = set(config.get("sensors", {}).keys())
    rain: set[str] = set()
    for entry in (config.get("rainfall") or {}).get("gauges") or []:
        if isinstance(entry, str):
            rain.add(entry)
        elif isinstance(entry, dict) and entry.get("name"):
            rain.add(str(entry["name"]))
    return stations, rain


def list_months(cache: Path, date_from: str | None, date_to: str | None) -> list[str]:
    stems = {p.name[:7] for p in cache.glob("????-??.json.gz")}
    stems |= {p.name[:7] for p in cache.glob("????-??.parquet")}
    out = sorted(stems)
    if date_from:
        out = [s for s in out if s >= date_from]
    if date_to:
        out = [s for s in out if s <= date_to]
    return out


def series_fingerprint(
    chunk: dict[str, list[tuple[str, float]]],
) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    for name, points in chunk.items():
        cleaned = [(str(t), float(v)) for t, v in points]
        cleaned.sort(key=lambda p: p[0])
        out[name] = cleaned
    return out


def diff_series(
    label: str,
    left: dict[str, list[tuple[str, float]]],
    right: dict[str, list[tuple[str, float]]],
) -> list[str]:
    errors: list[str] = []
    for name in sorted(set(left) | set(right)):
        a = left.get(name) or []
        b = right.get(name) or []
        if len(a) != len(b):
            errors.append(f"{label}/{name}: count gz={len(a)} pq={len(b)}")
            continue
        for i, (pa, pb) in enumerate(zip(a, b)):
            if pa[0] != pb[0] or pa[1] != pb[1]:
                errors.append(f"{label}/{name}[{i}]: gz={pa!r} pq={pb!r}")
                break
    return errors


def check_gz_timerange_level(gz_path: Path, sample_lines: int = 2000) -> list[str]:
    errors: list[str] = []
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if idx >= sample_lines:
                break
            obj = json.loads(line)
            for block in obj.get("data") or []:
                tr = block.get("timerange")
                lv = block.get("level")
                if tr is not None and len(tr) != 3:
                    errors.append(f"{gz_path.name}: timerange len={len(tr)} (atteso 3)")
                    return errors
                if lv is not None and len(lv) != 4:
                    errors.append(f"{gz_path.name}: level len={len(lv)} (atteso 4)")
                    return errors
    return errors


def check_parquet_schema(parquet_path: Path) -> list[str]:
    errors: list[str] = []
    con = duckdb.connect()
    cols = {
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_sql_quote(parquet_path)}')"
        ).fetchall()
    }
    missing = EXPECTED_COLS - cols
    if missing:
        errors.append(f"{parquet_path.name}: colonne mancanti {sorted(missing)}")
        return errors
    null_meas = con.execute(
        f"""
        SELECT count(*) FROM read_parquet('{_sql_quote(parquet_path)}')
        WHERE code IN ('{LEVEL_CODE}', '{RAIN_CODE}')
          AND station_b01019 IS NULL
        """
    ).fetchone()[0]
    if null_meas:
        errors.append(
            f"{parquet_path.name}: {null_meas} misure con station_b01019 NULL"
        )
    return errors


def verify_month(
    stem: str,
    cache: Path,
    level_names: set[str],
    rain_names: set[str],
) -> list[str]:
    gz = cache / f"{stem}.json.gz"
    pq = cache / f"{stem}.parquet"
    if not gz.exists():
        return [f"{stem}: manca {gz.name}"]
    if not pq.exists():
        return [f"{stem}: manca {pq.name} (usa --convert)"]
    errors = check_gz_timerange_level(gz)
    errors.extend(check_parquet_schema(pq))
    if errors:
        return errors
    gz_levels, gz_rain = extract_month(gz, level_names, rain_names)
    pq_levels, pq_rain = extract_month(pq, level_names, rain_names)
    errors.extend(
        diff_series("livelli", series_fingerprint(gz_levels), series_fingerprint(pq_levels))
    )
    errors.extend(
        diff_series("pioggia", series_fingerprint(gz_rain), series_fingerprint(pq_rain))
    )
    if not errors:
        n_lvl = sum(len(v) for v in gz_levels.values())
        n_rain = sum(len(v) for v in gz_rain.values())
        print(f"  ok {stem}: livelli={n_lvl} pioggia={n_rain}", flush=True)
    return errors


def _fmt_mb(n: int | float) -> str:
    return f"{n / 1e6:.2f} MB"


def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "?"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def convert_months(stems: list[str], cache: Path, force: bool) -> int:
    todo = []
    hits = 0
    skip = 0
    for stem in stems:
        gz = cache / f"{stem}.json.gz"
        pq = cache / f"{stem}.parquet"
        if not gz.exists():
            skip += 1
            continue
        if pq.exists() and pq.stat().st_size > 0 and not force:
            hits += 1
            continue
        todo.append(stem)

    gz_total = sum((cache / f"{s}.json.gz").stat().st_size for s in stems if (cache / f"{s}.json.gz").exists())
    pq_existing = sum(
        (cache / f"{s}.parquet").stat().st_size
        for s in stems
        if (cache / f"{s}.parquet").exists() and (cache / f"{s}.parquet").stat().st_size > 0
    )
    print(
        f"  piano: {len(todo)} da convertire, {hits} già ok, {skip} senza gz | "
        f"gz totali {_fmt_mb(gz_total)} | parquet già presenti {_fmt_mb(pq_existing)}",
        flush=True,
    )
    if not todo:
        return 0

    converted = 0
    gz_done = 0
    pq_done = 0
    t_batch = time.time()
    durations: list[float] = []

    for i, stem in enumerate(todo, 1):
        gz = cache / f"{stem}.json.gz"
        pq = cache / f"{stem}.parquet"
        gz_sz = gz.stat().st_size
        avg = sum(durations) / len(durations) if durations else 0.0
        eta = avg * (len(todo) - i + 1) if avg else 0.0
        print(
            f"  [{i}/{len(todo)}] {stem}: {_fmt_mb(gz_sz)} gz → parquet …"
            f"  (eta {_fmt_eta(eta)})",
            flush=True,
        )
        t0 = time.time()
        json_gz_to_parquet(gz, pq)
        dt = time.time() - t0
        durations.append(dt)
        pq_sz = pq.stat().st_size
        ratio = (pq_sz / gz_sz) if gz_sz else 0.0
        saved = gz_sz - pq_sz
        gz_done += gz_sz
        pq_done += pq_sz
        converted += 1
        print(
            f"    → {_fmt_mb(pq_sz)} in {dt:.1f}s | "
            f"ratio {ratio:.2%} | risparmiati {_fmt_mb(saved)} | "
            f"cumulato gz {_fmt_mb(gz_done)} → pq {_fmt_mb(pq_done)} "
            f"({(pq_done / gz_done) if gz_done else 0:.2%})",
            flush=True,
        )

    elapsed = time.time() - t_batch
    print(
        f"  fatto: {converted} convertiti in {_fmt_eta(elapsed)} "
        f"(media {sum(durations)/len(durations):.1f}s/mese) | "
        f"gz {_fmt_mb(gz_done)} → parquet {_fmt_mb(pq_done)} "
        f"(ratio {(pq_done / gz_done) if gz_done else 0:.2%}, "
        f"risparmiati {_fmt_mb(gz_done - pq_done)})",
        flush=True,
    )
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parità cache parquet flatten vs json.gz; gate per --delete-json"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "storico" / "raw")
    parser.add_argument("--from", dest="date_from", type=str, default=None)
    parser.add_argument("--to", dest="date_to", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Max mesi da processare")
    parser.add_argument("--convert", action="store_true", help="Converte gz→parquet (gz restano)")
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Riconverte anche se il parquet esiste già",
    )
    parser.add_argument(
        "--delete-json",
        action="store_true",
        help="Dopo verifica ok, cancella i .json.gz dei mesi verificati",
    )
    args = parser.parse_args()

    if not args.cache.is_dir():
        print(f"cache assente: {args.cache}", file=sys.stderr)
        return 1

    level_names, rain_names = load_names(args.config)
    if not level_names:
        print("Nessuna stazione in config.json", file=sys.stderr)
        return 1

    stems = list_months(args.cache, args.date_from, args.date_to)
    if args.limit is not None:
        stems = stems[: args.limit]
    if not stems:
        print("Nessun mese in cache", file=sys.stderr)
        return 1

    convert_only = (args.convert or args.force_convert) and not args.delete_json
    # Bare --convert: only convert (verify is a separate step). --delete-json always re-verifies.
    do_convert = args.convert or args.force_convert
    do_verify = args.delete_json or not do_convert

    if do_convert:
        print(f"Conversione ({len(stems)} mesi)…", flush=True)
        convert_months(stems, args.cache, force=bool(args.force_convert))
        if convert_only:
            return 0

    print(f"Verifica parità ({len(stems)} mesi)…", flush=True)
    failed: list[tuple[str, list[str]]] = []
    ok_stems: list[str] = []
    for stem in stems:
        errs = verify_month(stem, args.cache, level_names, rain_names)
        if errs:
            failed.append((stem, errs))
            for e in errs[:5]:
                print(f"  FAIL {e}", flush=True)
            if len(errs) > 5:
                print(f"  … +{len(errs) - 5} errori", flush=True)
        else:
            ok_stems.append(stem)

    print(f"Esito: {len(ok_stems)} ok, {len(failed)} fail su {len(stems)}", flush=True)
    if failed:
        print("Niente cancellazione: fix i mesi in fail e rilancia.", file=sys.stderr)
        return 1

    if args.delete_json:
        freed = 0
        for stem in ok_stems:
            gz = args.cache / f"{stem}.json.gz"
            if gz.exists():
                freed += gz.stat().st_size
                gz.unlink()
                print(f"  deleted {gz.name}", flush=True)
        print(f"Rimossi {len(ok_stems)} json.gz ({freed / 1e9:.2f} GB)", flush=True)
    elif do_verify:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
