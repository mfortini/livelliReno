#!/usr/bin/env python3
"""Merge three Cento MLP pinball exports (q10/q50/q90) + conformal band inflate.

Uso:
  uv run python tools/merge-cento-quantiles.py \\
    --q10 models/cento-dnn-q10.json \\
    --q50 models/cento-dnn-q50.json \\
    --q90 models/cento-dnn-q90.json \\
    --data ./storico --out models/cento-dnn-quantiles.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _load(name: str, filename: str):
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mlp_numpy(x: np.ndarray, layers: list[dict]) -> float:
    cur = x.astype(np.float64)
    for layer in layers:
        W = np.asarray(layer["W"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        cur = W @ cur + b
        if layer.get("activation") == "relu":
            cur = np.maximum(cur, 0.0)
    return float(cur.reshape(-1)[0])


def standardize_row(row: np.ndarray, mean: list[float], std: list[float], names: list[str]) -> np.ndarray:
    out = np.zeros(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        raw = row[i]
        if name.startswith("miss_"):
            out[i] = 1.0 if not np.isfinite(raw) else float(raw)
            continue
        mu, sd = float(mean[i]), float(std[i])
        if sd < 1e-6:
            sd = 1.0
        out[i] = 0.0 if not np.isfinite(raw) else (float(raw) - mu) / sd
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q10", type=Path, required=True)
    ap.add_argument("--q50", type=Path, required=True)
    ap.add_argument("--q90", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "storico")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "cento-dnn-quantiles.json")
    ap.add_argument("--alpha", type=float, default=0.20, help="Miscoverage target (0.20 → ~80% coverage)")
    ap.add_argument("--spike-abs", type=float, default=3.0)
    args = ap.parse_args()

    q10 = json.loads(args.q10.read_text(encoding="utf-8"))
    q50 = json.loads(args.q50.read_text(encoding="utf-8"))
    q90 = json.loads(args.q90.read_text(encoding="utf-8"))
    for label, m in ("q10", q10), ("q50", q50), ("q90", q90):
        if m.get("kind") != "window_delta_mlp":
            raise SystemExit(f"{label} not window_delta_mlp")

    ml = _load("fit_ml_cento", "fit-ml-cento.py")
    ft = ml.ft
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    stations = list(q50["stations"])
    feature_names = list(q50["featureNames"])
    lookback = int(q50["lookbackSteps"])
    recipe = q50.get("featureRecipe") or {}
    poly_points = int(recipe.get("polyDerivPoints") or 6)

    wide = ft.load_wide(args.data, list(config["sensors"].keys()))
    wide, _ = ft.filter_spikes(wide, spike_abs=args.spike_abs)
    rain = None
    if q50.get("useRain"):
        gauges = [
            (g if isinstance(g, str) else g.get("name"))
            for g in (config.get("rainfall") or {}).get("gauges") or []
        ]
        gauges = [g for g in gauges if g]
        rain = ft.load_rain_basin(args.data / "rain", gauges, wide.index)

    feats, cols = ml.build_feature_matrix(
        wide[stations], rain, lookback=lookback, stations=stations, poly_points=poly_points
    )
    feats = feats[feature_names]

    merged_horizons = {}
    for key in sorted(q50["horizons"], key=int):
        b50 = q50["horizons"][key]
        b10 = q10["horizons"].get(key)
        b90 = q90["horizons"].get(key)
        if not b10 or not b90:
            continue
        h_steps = int(key) // ft.STEP_MINUTES
        targets = ml.build_targets(wide, h_steps)
        X, y_delta, y0, index = ml.align_xy(feats, feature_names, targets)
        train_m, val_m, _hold = ml.split_masks(index, mode="alt")
        # Conformal on outer validate (subsample for speed)
        Xv = X.to_numpy(dtype=float)[val_m]
        yv = y_delta[val_m]
        n_val = len(yv)
        rng = np.random.default_rng(1)
        cap = min(n_val, 1500)
        pick = np.arange(n_val) if n_val <= cap else np.sort(rng.choice(n_val, size=cap, replace=False))
        scores = []
        preds = []  # (d10, d90, y)
        mean50 = np.asarray(b50["mean"], dtype=np.float64)
        std50 = np.asarray(b50["std"], dtype=np.float64)
        for i in pick:
            x = standardize_row(Xv[i], mean50, std50, feature_names)
            d10 = mlp_numpy(x, b10["layers"])
            d90 = mlp_numpy(x, b90["layers"])
            scores.append(max(d10 - yv[i], yv[i] - d90))
            preds.append((d10, d90, yv[i]))
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if len(scores):
            n = len(scores)
            q_level = min(1.0, np.ceil((n + 1) * (1 - args.alpha)) / n)
            qhat = float(np.quantile(scores, q_level))
        else:
            qhat = 0.0
        cov_raw = cov_conf = 0
        n_ok = 0
        for d10, d90, yi in preds:
            if not np.isfinite(yi):
                continue
            n_ok += 1
            lo, hi = min(d10, d90), max(d10, d90)
            if lo <= yi <= hi:
                cov_raw += 1
            if lo - qhat <= yi <= hi + qhat:
                cov_conf += 1
        merged_horizons[key] = {
            "mean": b50["mean"],
            "std": b50["std"],
            "layers": b50["layers"],  # median path = prod-compatible
            "layersQ10": b10["layers"],
            "layersQ50": b50["layers"],
            "layersQ90": b90["layers"],
            "conformalQhat": round(qhat, 6),
            "conformalAlpha": args.alpha,
            "coverageValidateRaw": None if not n_ok else round(cov_raw / n_ok, 4),
            "coverageValidateConformal": None if not n_ok else round(cov_conf / n_ok, 4),
            "nCalibrate": int(len(scores)),
            "spreadLow": b50.get("spreadLow"),
            "spreadHigh": b50.get("spreadHigh"),
            "spreadLowHigh": b50.get("spreadLowHigh"),
            "spreadHighHigh": b50.get("spreadHighHigh"),
            "train": b50.get("train"),
            "validate": b50.get("validate"),
            "validateEvents": b50.get("validateEvents"),
            "holdout": b50.get("holdout"),
            "holdoutEvents": b50.get("holdoutEvents"),
            "earlyStopping": b50.get("earlyStopping"),
        }
        print(
            f"+{key}′ qhat={qhat:.4f} cov_raw={cov_raw / max(1, n_ok):.3f} "
            f"cov_conf={cov_conf / max(1, n_ok):.3f} n={n_ok}"
        )

    skill120 = (merged_horizons.get("120") or {}).get("validateEvents", {}).get("skill")
    out = {
        **{k: q50[k] for k in (
            "kind", "downstream", "upstreams", "acceptors", "stations",
            "lookbackSteps", "stepMinutes", "featureNames", "featureRecipe",
            "useRain", "rainGauges", "thresholdOn", "holdoutBaseline"
        ) if k in q50},
        "version": 1,
        "horizons": merged_horizons,
        "skill120": skill120,
        "reliable": bool(skill120 is not None and skill120 >= 0.10),
        "quantiles": [0.1, 0.5, 0.9],
        "conformalAlpha": args.alpha,
        "note": (
            "M1: pinball q10/q50/q90 + split-conformal inflate on validate. "
            "layers=q50 for median path; layersQ* + conformalQhat for band."
        ),
        "trainedAt": q50.get("trainedAt"),
        "sources": {
            "q10": _rel(args.q10),
            "q50": _rel(args.q50),
            "q90": _rel(args.q90),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
