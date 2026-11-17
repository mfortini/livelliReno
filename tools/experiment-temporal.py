#!/usr/bin/env python3
"""Experimental Cento MLP with causal interpolated past (linear/PCHIP).

Builds the same feature contract as predict-transfer.js buildWindowFeatureVector
for a denser input grid, trains the existing MLP head per horizon and exports an
experimental node. Does not touch production coeffs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from temporal_sampling import TemporalSampler, MINUTE_MS  # noqa: E402


def _load(name: str, filename: str):
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ml = _load("fit_ml_cento", "fit-ml-cento.py")
ft = ml.ft

DERIV_STEP_MIN = 30
POLY_N = 6
MARGIN_MIN = 1800


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_native(data_dir: Path, stations: list[str]) -> dict[str, TemporalSampler]:
    out = {}
    for name in stations:
        path = ft._station_series_path(data_dir, name)
        if path is None:
            raise SystemExit(f"Missing series for {name} in {data_dir}")
        if path.suffix == ".parquet":
            import duckdb
            con = duckdb.connect()
            df = con.execute(
                "SELECT CAST(time AS TIMESTAMPTZ) AS time, CAST(value AS DOUBLE) AS value "
                "FROM read_parquet(?) WHERE value IS NOT NULL AND isfinite(value)",
                [str(path)],
            ).df()
        else:
            df = pd.read_csv(path, parse_dates=["time"])
        df = df.dropna(subset=["value"]).sort_values("time")
        df = df[~df["time"].duplicated(keep="last")]
        out[name] = TemporalSampler.from_frame(df)
    return out


def poly_derivs_windows(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Y (m,6) oldest→newest at 30′ spacing → d1,d2 per 30′ step (OLS quadratic)."""
    m = Y.shape[0]
    d1 = np.full(m, np.nan)
    d2 = np.full(m, np.nan)
    x = np.arange(-(POLY_N - 1), 1, dtype=float)
    V = np.column_stack([np.ones(POLY_N), x, x * x])
    pinv = np.linalg.pinv(V)
    ok = np.isfinite(Y).all(axis=1)
    if ok.any():
        coef = Y[ok] @ pinv.T
        d1[ok] = coef[:, 1]
        d2[ok] = 2.0 * coef[:, 2]
    need = ~ok
    if need.any():
        tail = Y[need][:, -3:]
        ok3 = np.isfinite(tail).all(axis=1)
        if ok3.any():
            x3 = np.arange(-2, 1, dtype=float)
            V3 = np.column_stack([np.ones(3), x3, x3 * x3])
            pinv3 = np.linalg.pinv(V3)
            coef3 = tail[ok3] @ pinv3.T
            idx = np.flatnonzero(need)[ok3]
            d1[idx] = coef3[:, 1]
            d2[idx] = 2.0 * coef3[:, 2]
    return d1, d2


def build_station_features(
    sampler: TemporalSampler,
    origins_ms: np.ndarray,
    lookback_steps: int,
    input_step_min: int,
    method: str,
) -> dict[str, np.ndarray]:
    """Return per-lag arrays for one station: level/delta/miss/d1/d2."""
    step_ms = input_step_min * MINUTE_MS
    n_orig = len(origins_ms)
    lb = lookback_steps
    times = sampler.times_ms
    values = sampler.values
    level = np.full((n_orig, lb), np.nan, dtype=np.float32)
    delta = np.full((n_orig, lb), np.nan, dtype=np.float32)
    miss = np.ones((n_orig, lb), dtype=np.float32)
    d1 = np.full((n_orig, lb), np.nan, dtype=np.float32)
    d2 = np.full((n_orig, lb), np.nan, dtype=np.float32)

    # target layout per origin: L[0..lb] then deriv windows k-major/i-minor
    n_level = lb + 1
    n_deriv = lb * POLY_N
    for oi, now in enumerate(origins_ms):
        cutoff = int(now)
        right = int(np.searchsorted(times, cutoff, side="right"))
        left = int(np.searchsorted(times, cutoff - MARGIN_MIN * MINUTE_MS, side="left"))
        if right <= left:
            continue
        st = times[left:right]
        sv = values[left:right]
        level_times = now - np.arange(n_level, dtype=np.int64) * step_ms
        k_idx = np.arange(lb, dtype=np.int64)
        # deriv window for lag k ends at now-k*step; samples end-(5-i)*30min
        base = now - k_idx[:, None] * step_ms - (POLY_N - 1 - np.arange(POLY_N)[None, :]) * DERIV_STEP_MIN * MINUTE_MS
        deriv_times = base.reshape(-1)
        targets = np.concatenate([level_times, deriv_times])
        vals = sampler.sample_many(targets, cutoff)
        L = vals[:n_level]
        D = vals[n_level:].reshape(lb, POLY_N)
        level[oi] = L[:lb]
        finite_level = np.isfinite(L[:lb])
        miss[oi] = (~finite_level).astype(np.float32)
        prev = L[1:lb + 1]
        both = finite_level & np.isfinite(prev)
        delta[oi, both] = (L[:lb][both] - prev[both]).astype(np.float32)
        dd1, dd2 = poly_derivs_windows(D.astype(float))
        d1[oi] = dd1.astype(np.float32)
        d2[oi] = dd2.astype(np.float32)
        if (oi + 1) % 2000 == 0:
            log(f"    {oi + 1}/{n_orig} origins")
    return {"level": level, "delta": delta, "miss": miss, "d1": d1, "d2": d2}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "storico")
    p.add_argument("--coeffs", type=Path, default=ROOT / "models" / "transfer-coeffs.json")
    p.add_argument("--out", type=Path, default=ROOT / "models" / "experiments" / "cento-temporal.json")
    p.add_argument("--method", choices=["linear", "pchip"], default="linear")
    p.add_argument("--input-step-minutes", type=int, default=5)
    p.add_argument("--lookback-minutes", type=float, default=1440)
    p.add_argument("--stride", type=int, default=12, help="origins every N×30min (12 = 6 h)")
    p.add_argument("--from", dest="from_ts", default="2020-01-01")
    p.add_argument("--to", dest="to_ts", default=None)
    p.add_argument("--holdout-from", default="2026-01-01")
    p.add_argument("--horizons", type=int, nargs="*", default=None, help="minutes; default all node horizons")
    p.add_argument("--max-origins", type=int, default=None)
    p.add_argument("--feat-cache", type=Path, default=None,
                   help="npz cache of the feature matrix; reused when config matches")
    p.add_argument("--tag", default=None)
    p.add_argument("--spectral", choices=["none", "fft", "wavelet", "both"], default="none",
                   help="append causal spectral features (FFT + Haar à trous) per station")
    p.add_argument("--spectral-window", type=int, default=0,
                   help="spectral window in input steps; 0 = full lookback")
    args = p.parse_args()

    coeffs = json.loads(args.coeffs.read_text(encoding="utf-8"))
    node = coeffs["nodes"]["Cento"]
    stations = list(node["stations"])
    downstream = node.get("downstream", "Cento")
    upstreams = list(node.get("upstreams") or [s for s in stations if s != downstream])
    recipe = dict(node.get("featureRecipe") or {})
    recipe.update({
        "inputStepMinutes": args.input_step_minutes,
        "inputInterpolation": args.method,
        "lookbackMinutes": args.lookback_minutes,
        "maxGapMinutes": 60,
        "derivWindowMinutes": DERIV_STEP_MIN * (POLY_N - 1),
        "stepMinutes": args.input_step_minutes,
    })
    lookback_steps = int(round(args.lookback_minutes / args.input_step_minutes))
    log(f"stations={stations} lookback={lookback_steps}×{args.input_step_minutes}′ method={args.method}")

    log("load wide grid (targets/splits/rain)…")
    wide = ft.load_wide(args.data, stations)
    if "rain" in str(args.data):
        pass
    gauges = [g.get("name") if isinstance(g, dict) else g for g in (json.loads((ROOT / "config.json").read_text())["rainfall"]["gauges"])]
    rain = ft.load_rain_basin(args.data / "rain", gauges, wide.index)

    from_ts = pd.Timestamp(args.from_ts, tz="UTC")
    to_ts = pd.Timestamp(args.to_ts, tz="UTC") if args.to_ts else wide.index.max()
    cand = wide.index[(wide.index >= from_ts) & (wide.index <= to_ts)]
    origins = cand[:: max(1, args.stride)]
    if args.max_origins:
        origins = origins[: args.max_origins]
    log(f"origins={len(origins)} {origins[0]} → {origins[-1]}")

    log("load native series…")
    samplers = load_native(args.data, stations)
    for s in stations:
        samplers[s].method = args.method
        samplers[s].max_gap_minutes = 60.0

    origins_ms = (origins.values.astype("datetime64[ns]").astype("int64") // 1_000_000).astype(np.int64)
    level_series = wide[downstream]
    y0_all = level_series.reindex(origins).to_numpy(dtype=float)

    # feature names in the same order as the production recipe
    def tag(name):
        return name.replace(" ", "_")

    feature_names = []
    for st in stations:
        tg = tag(st)
        for k in range(lookback_steps):
            feature_names += [
                f"level_{tg}_lag{k}", f"delta_{tg}_lag{k}", f"miss_{tg}_lag{k}",
                f"d1_{tg}_lag{k}", f"d2_{tg}_lag{k}",
            ]
    rain_cols = ["rain6h", "rain24h"] if rain is not None else []
    feature_names += rain_cols
    col_index = {name: i for i, name in enumerate(feature_names)}
    n_feat = len(feature_names)
    log(f"features={n_feat}")

    feat_meta = json.dumps({
        "stations": stations, "lookback": lookback_steps,
        "step": args.input_step_minutes, "method": args.method,
        "stride": args.stride, "from": args.from_ts, "to": args.to_ts,
        "maxOrigins": args.max_origins, "nOrigins": int(len(origins)),
        "dataDir": str(args.data),
    }, sort_keys=True)
    X = None
    if args.feat_cache and args.feat_cache.exists():
        try:
            z = np.load(args.feat_cache, allow_pickle=False)
            if str(z["meta"]) == feat_meta and z["X"].shape == (len(origins), n_feat):
                X = z["X"]
                cached_names = [str(x) for x in z["feature_names"]]
                if cached_names == feature_names:
                    log(f"feature cache hit: {args.feat_cache}")
                else:
                    X = None
            else:
                log("feature cache meta mismatch, rebuilding")
        except Exception as exc:  # noqa: BLE001
            log(f"feature cache unreadable ({exc}), rebuilding")
    if X is None:
        X = np.zeros((len(origins), n_feat), dtype=np.float32)
        t0 = time.perf_counter()
        for si, st in enumerate(stations):
            log(f"features station {si + 1}/{len(stations)}: {st}")
            feats = build_station_features(samplers[st], origins_ms, lookback_steps, args.input_step_minutes, args.method)
            tg = tag(st)
            for k in range(lookback_steps):
                base = col_index[f"level_{tg}_lag{k}"]
                # columns are contiguous per lag in feature_names order
                X[:, base + 0] = feats["level"][:, k]
                X[:, base + 1] = feats["delta"][:, k]
                X[:, base + 2] = feats["miss"][:, k]
                X[:, base + 3] = feats["d1"][:, k]
                X[:, base + 4] = feats["d2"][:, k]
            del feats
        if rain is not None:
            rain_sub = rain.reindex(origins)
            X[:, col_index["rain6h"]] = rain_sub["rain6h"].to_numpy(dtype=np.float32)
            X[:, col_index["rain24h"]] = rain_sub["rain24h"].to_numpy(dtype=np.float32)
        log(f"feature matrix {X.shape} in {time.perf_counter() - t0:.1f}s")
        if args.feat_cache:
            args.feat_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.feat_cache, X=X, feature_names=np.array(feature_names), meta=np.array(feat_meta))
            log(f"feature cache written: {args.feat_cache}")

    if args.spectral != "none":
        import spectral_features as sf
        fft_bins = 12 if args.spectral in ("fft", "both") else 0
        wav_scales = 5 if args.spectral in ("wavelet", "both") else 0
        W = max(4, min(lookback_steps, args.spectral_window or lookback_steps))
        spec_blocks = []
        spec_names = []
        t0 = time.perf_counter()
        for st in stations:
            tg = tag(st)
            lvl = np.stack(
                [X[:, col_index[f"level_{tg}_lag{k}"]] for k in range(W)], axis=1
            )
            spec_blocks.append(
                sf.spectral_matrix(lvl, fft_bins=fft_bins, wavelet_scales=wav_scales).astype(np.float32)
            )
            spec_names += sf.spectral_feature_names(tg, fft_bins=fft_bins, wavelet_scales=wav_scales)
        X = np.hstack([X] + spec_blocks)
        feature_names += spec_names
        n_feat = len(feature_names)
        recipe["spectral"] = {
            "enabled": True, "windowSteps": W,
            "fftBins": fft_bins, "waveletScales": wav_scales,
        }
        log(f"spectral +{sum(b.shape[1] for b in spec_blocks)} cols ({args.spectral}, W={W}) "
            f"in {time.perf_counter() - t0:.1f}s")

    Xdf = pd.DataFrame(X, columns=feature_names, index=origins)
    del X

    train_m, val_m, hold_m = ml.split_masks(origins, mode="block", holdout_from=args.holdout_from)
    # require finite target for at least one horizon later; masks per horizon below

    horizon_minutes = [int(h) for h in (args.horizons or list(node["horizons"].keys()))]
    horizon_minutes = sorted({int(h) for h in horizon_minutes})
    log(f"horizons={horizon_minutes}")

    X_std, stats = ml.standardize_features(Xdf, feature_names, train_m)
    del Xdf

    horizons_out = {}
    export_horizons = {}
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tag_exp = args.tag or f"cento-temporal-{args.method}-{args.input_step_minutes}min"

    for minutes in horizon_minutes:
        h_steps = minutes // 30
        targets = ml.build_targets(wide, h_steps)
        y_future = targets["y_future"].reindex(origins).to_numpy(dtype=float)
        delta = targets["delta"].reindex(origins).to_numpy(dtype=float)
        ok = np.isfinite(y0_all) & np.isfinite(delta)
        tr = train_m & ok
        va = val_m & ok
        ho = hold_m & ok
        if tr.sum() < ml.WEAK_POINTS or va.sum() < ml.WEAK_POINTS:
            log(f"skip +{minutes}′: train={tr.sum()} val={va.sum()}")
            continue
        log(f"train +{minutes}′: train={tr.sum()} val={va.sum()} hold={ho.sum()}")
        poly_base = ml.build_poly2_baseline(level_series, origins.to_numpy(), h_steps, n_points=6)
        model, pred, hist = ml.train_dnn(X_std, delta, tr, va)
        metrics = ml.eval_splits(y0_all, delta, pred, tr, va, ho, poly_baseline=poly_base)
        spreads = ml.pack_spreads(y0_all, delta, pred, va)
        mean = [round(stats[c][0], 8) for c in feature_names]
        std = [round(stats[c][1], 8) for c in feature_names]
        export_horizons[str(minutes)] = {
            "mean": mean,
            "std": std,
            "layers": ml.export_mlp_layers(model),
            **spreads,
            "train": metrics["train"],
            "validate": metrics["validate"],
            "validateEvents": metrics["validateEvents"],
            "holdout": metrics["holdout"],
            "holdoutEvents": metrics["holdoutEvents"],
            "earlyStopping": hist,
        }
        horizons_out[str(minutes)] = {
            "horizonMinutes": minutes,
            "nFeatures": n_feat,
            "dnn": {**metrics, "architecture": list(ml.DNN_HIDDEN) + [1], "earlyStopping": hist},
        }
        log(f"+{minutes}′ valEvt={metrics['validateEvents'].get('skill')} holdEvt={metrics['holdoutEvents'].get('skill')}")
        del model, pred

    skill120 = (export_horizons.get("120") or {}).get("validateEvents", {}).get("skill")
    dnn_node = {
        "kind": "window_delta_mlp",
        "experimental": True,
        "downstream": downstream,
        "upstreams": upstreams,
        "stations": stations,
        "lookbackSteps": lookback_steps,
        "stepMinutes": args.input_step_minutes,
        "featureNames": feature_names,
        "featureRecipe": recipe,
        "useRain": rain is not None,
        "rainGauges": gauges if rain is not None else [],
        "horizons": export_horizons,
        "skill120": skill120,
        "reliable": False,
        "trainedAt": trained_at,
        "note": f"Experimental causal {args.method} input at {args.input_step_minutes}′; not for production.",
    }
    compare = {
        "version": 1,
        "kind": "cento_temporal_experiment",
        "tag": tag_exp,
        "downstream": downstream,
        "upstreams": upstreams,
        "stations": stations,
        "lookbackSteps": lookback_steps,
        "lookbackMinutes": args.lookback_minutes,
        "inputStepMinutes": args.input_step_minutes,
        "inputInterpolation": args.method,
        "stride": args.stride,
        "origins": {"from": str(origins[0]), "to": str(origins[-1]), "n": int(len(origins))},
        "holdoutFrom": args.holdout_from,
        "featureRecipe": recipe,
        "features": feature_names,
        "horizons": horizons_out,
        "trainedAt": trained_at,
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(compare, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dnn_out = args.out.with_name(args.out.stem + "-dnn.json")
    dnn_out.write_text(json.dumps({"version": 1, "downstream": downstream, **dnn_node}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"Wrote {args.out} and {dnn_out}")


if __name__ == "__main__":
    main()
