#!/usr/bin/env python3
"""Dump Cento MLP parity fixtures: Python features + forward vs exported weights.

Uso:
  uv run python tools/dump-mlp-parity.py \\
    --model models/cento-dnn-novalle-6h.json \\
    --data ./storico \\
    --out tests/fixtures/cento-mlp-parity.json \\
    --from 2026-01-01 --n 64 --horizon 120
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import importlib.util


def _load(name: str, filename: str):
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# fit-ml-cento loads fit-transfer as sys.modules['fit_transfer']
ml = _load("fit_ml_cento", "fit-ml-cento.py")
ft = ml.ft


def mlp_numpy(x: np.ndarray, layers: list[dict]) -> float:
    cur = x.astype(np.float64)
    for layer in layers:
        W = np.asarray(layer["W"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        cur = W @ cur + b
        if layer.get("activation") == "relu":
            cur = np.maximum(cur, 0.0)
    return float(cur.reshape(-1)[0])


def standardize_with_export(
    row: pd.Series,
    feature_names: list[str],
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    out = np.zeros(len(feature_names), dtype=np.float64)
    for i, name in enumerate(feature_names):
        raw = row[name] if name in row.index else np.nan
        if name.startswith("miss_"):
            out[i] = 1.0 if not np.isfinite(raw) else float(raw)
            continue
        mu = float(mean[i])
        sd = float(std[i]) if float(std[i]) >= 1e-6 else 1.0
        if not np.isfinite(raw):
            out[i] = 0.0
        else:
            out[i] = (float(raw) - mu) / sd
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump Cento MLP Python/JS parity fixture")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "cento-dnn-novalle-6h.json")
    parser.add_argument("--data", type=Path, default=ROOT / "storico")
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "fixtures" / "cento-mlp-parity.json")
    parser.add_argument("--from", dest="from_ts", default="2026-01-01")
    parser.add_argument("--to", dest="to_ts", default=None)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=120, help="Horizon minutes (must exist in model)")
    parser.add_argument("--spike-abs", type=float, default=ft.SPIKE_ABS)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    model = json.loads(args.model.read_text(encoding="utf-8"))
    stations = list(model["stations"])
    feature_names = list(model["featureNames"])
    lookback = int(model["lookbackSteps"])
    use_rain = bool(model.get("useRain"))
    recipe = model.get("featureRecipe") or {}
    poly_points = int(recipe.get("polyDerivPoints") or ml.POLY_NAIVE_POINTS)
    h_key = str(int(args.horizon))
    block = (model.get("horizons") or {}).get(h_key)
    if not block:
        raise SystemExit(f"Missing horizon {h_key} in {args.model}")
    mean = block["mean"]
    std = block["std"]
    layers = block["layers"]
    h_steps = int(args.horizon) // ft.STEP_MINUTES

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    wide = ft.load_wide(args.data, list(config["sensors"].keys()))
    wide, _ = ft.filter_spikes(wide, spike_abs=args.spike_abs)
    rain = None
    if use_rain:
        gauges = [
            (g if isinstance(g, str) else g.get("name"))
            for g in (config.get("rainfall") or {}).get("gauges") or []
        ]
        gauges = [g for g in gauges if g]
        rain = ft.load_rain_basin(args.data / "rain", gauges, wide.index)

    feats, cols = ml.build_feature_matrix(
        wide[stations],
        rain,
        lookback=lookback,
        stations=stations,
        poly_points=poly_points,
        use_soglia_headroom=bool(recipe.get("useSogliaHeadroom")),
        use_soglia_ratio=bool(recipe.get("useSogliaRatio")),
    )
    if cols != feature_names:
        # Keep model order; require same set
        missing = [c for c in feature_names if c not in feats.columns]
        if missing:
            raise SystemExit(f"Feature mismatch, missing e.g. {missing[:5]}")
        feats = feats[feature_names]
        cols = feature_names

    targets = ml.build_targets(wide, h_steps)
    X, y_delta, y0, index = ml.align_xy(feats, cols, targets)

    from_ts = pd.Timestamp(args.from_ts, tz="UTC")
    to_ts = pd.Timestamp(args.to_ts, tz="UTC") if args.to_ts else index.max()
    # index may be tz-naive UTC
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        from_cmp = from_ts.tz_localize(None)
        to_cmp = to_ts.tz_localize(None) if to_ts.tz else to_ts
    else:
        from_cmp = from_ts
        to_cmp = to_ts
    mask = (idx >= from_cmp) & (idx <= to_cmp)
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        raise SystemExit("No samples in requested window")

    rng = np.random.default_rng(args.seed)
    pick = positions if len(positions) <= args.n else np.sort(
        rng.choice(positions, size=args.n, replace=False)
    )

    samples = []
    picked_times: list[pd.Timestamp] = []
    for pos in pick:
        t = pd.Timestamp(index[pos])
        row = X.iloc[pos]
        x_std = standardize_with_export(row, feature_names, mean, std)
        delta_hat = mlp_numpy(x_std, layers)
        feat_vec = []
        for name in feature_names:
            v = row[name]
            feat_vec.append(None if not np.isfinite(v) else round(float(v), 8))
        t_utc = t.tz_convert("UTC") if t.tzinfo is not None else t.tz_localize("UTC")
        samples.append(
            {
                "t": t_utc.isoformat().replace("+00:00", "Z"),
                "y0": float(y0[pos]),
                "deltaTrue": float(y_delta[pos]),
                "deltaHat": delta_hat,
                "yHat": float(y0[pos]) + delta_hat,
                "x": feat_vec,
            }
        )
        picked_times.append(t)

    # Layer B snippets: series lattice for first min(8, n) samples only.
    need_back = lookback + poly_points + 2
    series_by_station = {s: wide[s] for s in stations if s in wide.columns}
    step = pd.Timedelta(minutes=ft.STEP_MINUTES)
    for i, sample in enumerate(samples):
        if i >= 8:
            break
        t = picked_times[i]
        series = {}
        for s, col in series_by_station.items():
            pts = []
            for k in range(need_back):
                ts = t - k * step
                v = col.loc[ts] if ts in col.index else np.nan
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                ts_utc = ts.tz_convert("UTC") if ts.tzinfo is not None else ts.tz_localize("UTC")
                pts.append([
                    ts_utc.isoformat().replace("+00:00", "Z"),
                    None if not np.isfinite(v) else round(float(v), 6),
                ])
            series[s] = pts
        sample["series"] = series
        if rain is not None and use_rain:
            r6 = rain.loc[t, "rain6h"] if t in rain.index else np.nan
            r24 = rain.loc[t, "rain24h"] if t in rain.index else np.nan
            sample["rain"] = {
                "rain6h": None if not np.isfinite(r6) else float(r6),
                "rain24h": None if not np.isfinite(r24) else float(r24),
            }

    payload = {
        "version": 1,
        "kind": "cento_mlp_parity",
        "modelPath": str(args.model.relative_to(ROOT)),
        "horizonMinutes": int(args.horizon),
        "lookbackSteps": lookback,
        "stations": stations,
        "featureNames": feature_names,
        "featureRecipe": recipe,
        "useRain": use_rain,
        "n": len(samples),
        "from": args.from_ts,
        "seed": args.seed,
        "samples": samples,
        "note": (
            "Python build_feature_matrix + exported mean/std/layers from modelPath. "
            "Layer A: mlpForward on dumped x; Layer B: buildWindowFeatureVector vs x."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} ({len(samples)} samples, horizon={args.horizon}′)")


if __name__ == "__main__":
    main()
