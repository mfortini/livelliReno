#!/usr/bin/env python3
"""Inventory storico coverage and Cento Muskingum usable-sample accounting.

Writes models/storico-cento-inventory.json with per-station intervals, gaps,
month-block coverage, usable train/val/holdout counts, exclusion reasons,
and loss from the 12_000 train cap. Does not train.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOOKBACK = 48
MAX_H = 8
EVENT_ABS = 0.10
TRAIN_CAP = 12000
CENTO_STATIONS = ["Cento", "Forcelli", "Sostegno Reno", "Bonconvento"]
ACCEPTORS = ["Dosso", "Opera Reno Panfilia", "Secondo Salto", "Gallo"]


def _load_ft():
    path = Path(__file__).resolve().parent / "fit-transfer.py"
    spec = importlib.util.spec_from_file_location("fit_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fit_transfer"] = mod
    spec.loader.exec_module(mod)
    return mod


def raw_station_stats(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    df = pd.read_csv(path, parse_dates=["time"]).sort_values("time")
    df = df[~df.time.duplicated(keep="last")]
    v = df["value"].astype(float).to_numpy()
    fin = np.isfinite(v)
    dt = df.time.diff().dt.total_seconds() / 60.0
    return {
        "exists": True,
        "n": int(len(df)),
        "finite": int(fin.sum()),
        "from": df.time.min().isoformat(),
        "to": df.time.max().isoformat(),
        "medianDtMin": None if len(dt) < 2 else float(np.nanmedian(dt)),
        "gapsGt20min": int((dt > 20).sum()),
        "gapsGt2h": int((dt > 120).sum()),
        "missingPct": round(float(100 * (1 - fin.mean())), 3),
        "min": None if not fin.any() else float(v[fin].min()),
        "max": None if not fin.any() else float(v[fin].max()),
    }


def collect_indices(h_c, h_f, h_s, mask, lookback=LOOKBACK, max_h=MAX_H):
    out = []
    n = len(h_c)
    for t in range(lookback, n - max_h):
        if not mask[t] or not np.isfinite(h_c[t]):
            continue
        if not (
            np.isfinite(h_f[t - lookback + 1 : t + max_h + 1]).all()
            and np.isfinite(h_s[t - lookback + 1 : t + max_h + 1]).all()
            and np.isfinite(h_c[t + 1 : t + max_h + 1]).all()
        ):
            continue
        out.append(int(t))
    return out


def exclusion_breakdown(h_c, h_f, h_s_raw, h_s_fill, mask):
    cand = np.where(mask)[0]
    cand = cand[(cand >= LOOKBACK) & (cand < len(h_c) - MAX_H)]
    counts = {
        "candidateSlots": int(len(cand)),
        "ok": 0,
        "nanCentoT": 0,
        "nanCentoFuture": 0,
        "nanForcelliWindow": 0,
        "nanSostegnoEvenFilled": 0,
        "recoveredByBonconvento": 0,
    }
    for t in cand:
        if not np.isfinite(h_c[t]):
            counts["nanCentoT"] += 1
            continue
        if not np.isfinite(h_c[t + 1 : t + MAX_H + 1]).all():
            counts["nanCentoFuture"] += 1
            continue
        if not np.isfinite(h_f[t - LOOKBACK + 1 : t + MAX_H + 1]).all():
            counts["nanForcelliWindow"] += 1
            continue
        raw_ok = np.isfinite(h_s_raw[t - LOOKBACK + 1 : t + MAX_H + 1]).all()
        fill_ok = np.isfinite(h_s_fill[t - LOOKBACK + 1 : t + MAX_H + 1]).all()
        if not fill_ok:
            counts["nanSostegnoEvenFilled"] += 1
            continue
        if not raw_ok:
            counts["recoveredByBonconvento"] += 1
        counts["ok"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory storico for Cento hybrids")
    parser.add_argument("--data", type=Path, default=ROOT / "storico")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "storico-cento-inventory.json")
    parser.add_argument("--train-cap", type=int, default=TRAIN_CAP)
    args = parser.parse_args()

    ft = _load_ft()
    config = json.loads(args.config.read_text())
    station_names = [s.get("name") or s.get("id") for s in (config.get("sensors") or {}).values()]
    # Prefer CSV names from manifest when present
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    stations = list(manifest.get("stations") or station_names)
    rain_gauges = list(manifest.get("rainfallGauges") or [])

    raw = {name: raw_station_stats(args.data / f"{name}.csv") for name in stations}
    rain = {name: raw_station_stats(args.data / "rain" / f"{name}.csv") for name in rain_gauges}

    wide_stations = CENTO_STATIONS + [a for a in ACCEPTORS if (args.data / f"{a}.csv").exists()]
    wide = ft.load_wide(args.data, wide_stations)
    wide, spike_counts = ft.filter_spikes(wide)
    blocks = ft.month_blocks(wide.index)
    naive = wide.index.tz_convert("UTC").tz_localize(None) if wide.index.tz is not None else wide.index
    months = naive.to_period("M")
    uniq = months.unique().sort_values()
    hold = [str(m) for m in (uniq[-ft.HOLDOUT_MONTHS:] if len(uniq) > ft.HOLDOUT_MONTHS else [])]
    remaining = [m for m in uniq if str(m) not in hold]
    train_months = [str(m) for i, m in enumerate(remaining) if i % 2 == 0]
    val_months = [str(m) for i, m in enumerate(remaining) if i % 2 == 1]

    month_cov = []
    for m in uniq:
        sl = wide.loc[months == m]
        row = {
            "month": str(m),
            "block": int(blocks[months == m][0]),
            "cento": round(float(sl["Cento"].notna().mean()), 4),
            "forcelli": round(float(sl["Forcelli"].notna().mean()), 4),
            "sostegno": round(float(sl["Sostegno Reno"].notna().mean()), 4),
            "bonconvento": round(float(sl["Bonconvento"].notna().mean()), 4),
            "nCento": int(sl["Cento"].notna().sum()),
            "nSostegno": int(sl["Sostegno Reno"].notna().sum()),
        }
        for a in ACCEPTORS:
            if a in sl.columns:
                key = a.replace(" ", "_")
                row[key] = round(float(sl[a].notna().mean()), 4)
                row[f"n_{key}"] = int(sl[a].notna().sum())
        month_cov.append(row)

    h_c = wide["Cento"].to_numpy(float)
    h_f = wide["Forcelli"].to_numpy(float)
    h_s = wide["Sostegno Reno"].to_numpy(float)
    h_a = wide["Bonconvento"].to_numpy(float)
    h_s_fill = h_s.copy()
    miss = ~np.isfinite(h_s_fill) & np.isfinite(h_a)
    h_s_fill[miss] = h_a[miss]

    train_m = blocks == 0
    val_m = blocks == 1
    hold_m = blocks == 2
    train_idx = collect_indices(h_c, h_f, h_s_fill, train_m)
    val_idx = collect_indices(h_c, h_f, h_s_fill, val_m)
    hold_idx = collect_indices(h_c, h_f, h_s_fill, hold_m)
    train_nofill = collect_indices(h_c, h_f, h_s, train_m)
    exclude = exclusion_breakdown(h_c, h_f, h_s, h_s_fill, train_m)

    n_train = len(train_idx)
    cap = max(0, int(args.train_cap))
    lost = max(0, n_train - cap) if cap else 0
    event_n = 0
    event_stats = None
    if train_idx:
        deltas = np.abs(h_c[np.asarray(train_idx) + 4] - h_c[np.asarray(train_idx)])
        event_n = int((deltas >= EVENT_ABS).sum())
        event_stats = {
            "eventAbsM": EVENT_ABS,
            "nEventAtH4": event_n,
            "deltaP50": round(float(np.median(deltas)), 6),
            "deltaP90": round(float(np.quantile(deltas, 0.9)), 6),
            "deltaP99": round(float(np.quantile(deltas, 0.99)), 6),
        }

    out = {
        "version": 1,
        "kind": "storico_cento_inventory",
        "disclaimer": "Diagnostica dati: non allertistica.",
        "manifest": {
            "from": manifest.get("from"),
            "to": manifest.get("to"),
            "points": manifest.get("points"),
            "rainfallPoints": manifest.get("rainfallPoints"),
            "fetchedAt": manifest.get("fetchedAt"),
        },
        "rawTotals": {
            "hydrometricPoints": int(sum(v.get("n", 0) for v in raw.values() if v.get("exists"))),
            "rainfallPoints": int(sum(v.get("n", 0) for v in rain.values() if v.get("exists"))),
        },
        "stations": raw,
        "rainfall": rain,
        "wide30min": {
            "steps": int(len(wide)),
            "from": str(wide.index.min()),
            "to": str(wide.index.max()),
            "stepMinutes": ft.STEP_MINUTES,
            "spikeCounts": spike_counts,
            "blockCounts": {
                "train": int(train_m.sum()),
                "validate": int(val_m.sum()),
                "holdout": int(hold_m.sum()),
            },
            "trainMonths": train_months,
            "validateMonths": val_months,
            "holdoutMonths": hold,
            "monthCoverage": month_cov,
        },
        "muskingumUsable": {
            "lookbackSteps": LOOKBACK,
            "maxHorizonSteps": MAX_H,
            "stations": CENTO_STATIONS,
            "trainAll": n_train,
            "trainWithoutBonconventoFill": len(train_nofill),
            "validate": len(val_idx),
            "holdout": len(hold_idx),
            "trainCap": cap,
            "trainAfterCap": min(n_train, cap) if cap else n_train,
            "lostToCap": lost,
            "fracKept": round(min(n_train, cap) / max(1, n_train), 4) if cap else 1.0,
            "trainExclusions": exclude,
            "trainEvents": event_stats,
            "note": (
                "Sostegno Reno has long zero-coverage months (2025-09..2026-02); "
                "Bonconvento fill recovers most windows. Cap discards surplus train windows."
            ),
        },
        "acceptors": {
            "names": ACCEPTORS,
            "role": "valle backwater (not chase inputs)",
            "coverage": {
                a: {
                    "inWide": a in wide.columns,
                    "finiteFrac": None if a not in wide.columns else round(float(wide[a].notna().mean()), 4),
                    "n": None if a not in wide.columns else int(wide[a].notna().sum()),
                }
                for a in ACCEPTORS
            },
        },
        "inventoriedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "hydrometricPoints": out["rawTotals"]["hydrometricPoints"],
        "trainAll": n_train,
        "lostToCap": lost,
        "val": len(val_idx),
        "hold": len(hold_idx),
        "sostegnoZeroMonths": [m["month"] for m in month_cov if m["sostegno"] == 0.0],
    }, indent=2))


if __name__ == "__main__":
    main()
