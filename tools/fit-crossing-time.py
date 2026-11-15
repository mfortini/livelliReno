#!/usr/bin/env python3
"""M9: flood-only crossing-time quantiles for Cento.

At each issue t in a catalog flood window, predict minutes until first crossing
of relative thresholds y(t)+{0.5,1.0,1.5} m (censored at 6 h). Uses the same
feature stack as fit-ml-cento (optional Casalecchio chiusa).

Uso:
  uv run python tools/fit-crossing-time.py --data ./storico --flood-only \\
    --export models/cento-crossing-time.json
  uv run python tools/fit-crossing-time.py --no-chiusa --export models/cento-crossing-time-nochiusa.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_M = (0.5, 1.0, 1.5)
TAU_QUANTILES = (0.1, 0.5, 0.9)
CENSOR_MIN = 360  # 6 h
STEP_MIN = 30


def _load_ml():
    path = ROOT / "tools" / "fit-ml-cento.py"
    spec = importlib.util.spec_from_file_location("fit_ml_cento", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["fit_ml_cento"] = mod
    spec.loader.exec_module(mod)
    return mod


def crossing_tau_minutes(level: pd.Series, offsets: tuple[float, ...], censor_min: int) -> pd.DataFrame:
    """For each t, minutes to first cross of y(t)+offset (forward search on lattice)."""
    vals = level.to_numpy(dtype=float)
    idx = level.index
    n = len(vals)
    max_steps = censor_min // STEP_MIN
    cols = {}
    for off in offsets:
        out = np.full(n, float(censor_min), dtype=float)
        crossed = np.zeros(n, dtype=bool)
        for i in range(n):
            y0 = vals[i]
            if not np.isfinite(y0):
                out[i] = np.nan
                continue
            thr = y0 + off
            hi = min(n, i + max_steps + 1)
            for j in range(i + 1, hi):
                v = vals[j]
                if not np.isfinite(v):
                    continue
                if v >= thr:
                    out[i] = (j - i) * STEP_MIN
                    crossed[i] = True
                    break
        cols[f"tau_{off:g}"] = out
        cols[f"hit_{off:g}"] = crossed.astype(float)
    return pd.DataFrame(cols, index=idx)


def main() -> None:
    ap = argparse.ArgumentParser(description="Flood-only crossing-time MLP for Cento")
    ap.add_argument("--data", type=Path, default=ROOT / "storico")
    ap.add_argument("--events", type=Path, default=ROOT / "models" / "events.json")
    ap.add_argument("--export", type=Path, default=ROOT / "models" / "cento-crossing-time.json")
    ap.add_argument("--no-chiusa", action="store_true")
    ap.add_argument("--flood-only", action="store_true", default=True)
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--spike-abs", type=float, default=None)
    ap.add_argument("--split", choices=("alt", "block"), default="block")
    args = ap.parse_args()

    ml = _load_ml()
    ft = ml.ft
    lookback = int(args.lookback or ml.LOOKBACK)
    spike = float(args.spike_abs if args.spike_abs is not None else ft.SPIKE_ABS)
    ups = list(ml.UPSTREAMS_NO_CHIUSA if args.no_chiusa else ml.UPSTREAMS)
    needed = [ml.DOWNSTREAM, *ups]

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    wide = ft.load_wide(args.data, list(config["sensors"].keys()))
    wide, _ = ft.filter_spikes(wide, spike_abs=spike)
    gauges = []
    for g in (config.get("rainfall") or {}).get("gauges") or []:
        name = g if isinstance(g, str) else (g or {}).get("name")
        if name:
            gauges.append(name)
    rain = ft.load_rain_basin(args.data / "rain", gauges, wide.index) if gauges else None
    wide = wide[needed].copy()

    feats, feature_cols = ml.build_feature_matrix(
        wide, rain, lookback=lookback, stations=needed, poly_points=ml.POLY_NAIVE_POINTS
    )
    tau_df = crossing_tau_minutes(wide[ml.DOWNSTREAM], THRESHOLDS_M, CENSOR_MIN)

    # Align: need finite y0 and at least one tau
    frame = feats[feature_cols].copy()
    frame["y0"] = wide[ml.DOWNSTREAM]
    for off in THRESHOLDS_M:
        frame[f"tau_{off:g}"] = tau_df[f"tau_{off:g}"]
        frame[f"hit_{off:g}"] = tau_df[f"hit_{off:g}"]
    miss_cols = [c for c in feature_cols if c.startswith("miss_")]
    ok = frame["y0"].notna()
    if miss_cols:
        ok &= frame[miss_cols].notna().all(axis=1)
    for off in THRESHOLDS_M:
        ok &= frame[f"tau_{off:g}"].notna()
    frame = frame.loc[ok]
    index = frame.index
    X = frame[feature_cols]
    train_m, val_m, hold_m = ml.split_masks(index, mode=args.split)

    events = ml.load_catalog_events(args.events, ml.DOWNSTREAM)
    if args.flood_only:
        flood_m = ml.flood_window_mask(index, events)
        train_m = train_m & flood_m
        # Validate also on flood for timing skill (calm has no meaningful τ).
        val_m = val_m & flood_m
        hold_m = hold_m & flood_m
        print(f"flood-only: train={int(train_m.sum())} val={int(val_m.sum())} hold={int(hold_m.sum())}")

    if int(train_m.sum()) < ml.WEAK_POINTS:
        raise SystemExit(f"Too few flood train samples: {int(train_m.sum())}")

    X_std, stats = ml.standardize_features(X, feature_cols, train_m)
    mean = [stats[c][0] for c in feature_cols]
    std = [stats[c][1] for c in feature_cols]

    thresholds_out: dict = {}
    for off in THRESHOLDS_M:
        y = frame[f"tau_{off:g}"].to_numpy(dtype=float)
        q_models: dict = {}
        for q in TAU_QUANTILES:
            model, pred, hist = ml.train_dnn(
                X_std, y, train_m, val_m, quantile=float(q), event_weight=1.0
            )
            # Clamp predictions to [STEP_MIN, CENSOR_MIN]
            pred = np.clip(pred, STEP_MIN, CENSOR_MIN)
            mae_val = float(np.mean(np.abs(pred[val_m] - y[val_m]))) if val_m.any() else None
            print(f"τ+{off:g} q{q}: best_epoch={hist['best_epoch']} mae_val={mae_val}")
            q_models[str(q)] = {
                "layers": ml.export_mlp_layers(model),
                "validateMae": mae_val,
                "earlyStopping": hist,
            }
        thresholds_out[str(off)] = {
            "offsetM": off,
            "censorMin": CENSOR_MIN,
            "quantiles": list(TAU_QUANTILES),
            "models": q_models,
        }

    use_rain = rain is not None and {"rain6h", "rain24h"}.issubset(rain.columns)
    payload = {
        "version": 1,
        "kind": "cento_crossing_time",
        "downstream": ml.DOWNSTREAM,
        "upstreams": ups,
        "stations": needed,
        "lookbackSteps": lookback,
        "stepMinutes": STEP_MIN,
        "featureNames": feature_cols,
        "featureRecipe": {
            "usePolyDerivatives": True,
            "polyDerivPoints": ml.POLY_NAIVE_POINTS,
            "polyDerivDegree": ml.POLY_NAIVE_DEGREE,
        },
        "mean": [round(m, 8) for m in mean],
        "std": [round(s, 8) for s in std],
        "useRain": use_rain,
        "floodOnly": bool(args.flood_only),
        "noChiusa": bool(args.no_chiusa),
        "adopt": False,
        "thresholds": thresholds_out,
        "primaryOffsetM": 1.0,
        "disclaimer": "Sperimentale: tempi di attraversamento empirici, non allertistici.",
        "trainedAt": pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z"),
    }
    args.export.parent.mkdir(parents=True, exist_ok=True)
    args.export.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.export}")


if __name__ == "__main__":
    main()
