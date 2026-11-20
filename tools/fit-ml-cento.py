#!/usr/bin/env python3
"""Offline Cento forecast: LightGBM vs small MLP (experimental).

Features (lookback L=48 × 30′ = 24 h) per station in
  Cento + monte (Forcelli, Sostegno, Bonconvento, Casalecchio tiro a volo, chiusa)
  + valle acceptors (Dosso, Opera Reno Panfilia, Secondo Salto, Gallo):
  level[t-k], delta[t-k]=level[t-k]-level[t-k-1] (NaN if either missing), miss[t-k].
Plus rain6h / rain24h basin scalars. Acceptors are backwater boundary features,
not chase targets; at runtime future acceptor levels stay at last observation.

Target: Δy(t→t+h) for h ∈ {1,2,3,4}. Same month-block split as fit-transfer.
Writes models/ml-cento-compare.json — patch transfer-coeffs only with --export-coeffs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Line-buffered stdout so progress shows under tee/pipes without waiting for flush.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

_VERBOSE = True


def log(msg: str = "", *, level: str = "info") -> None:
    """Progress/debug line to stdout (always flushed)."""
    if level == "debug" and not _VERBOSE:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"info": "·", "debug": "…", "ok": "✓", "warn": "!"}.get(level, "·")
    print(f"[{ts}] {prefix} {msg}", flush=True)


def log_step(title: str) -> None:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === {title} ===", flush=True)

ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = "Cento"
# Direct deps + Bonconvento (covers Sostegno gaps) + Casalecchio chain (tiro a volo + chiusa).
UPSTREAMS = [
    "Forcelli",
    "Sostegno Reno",
    "Bonconvento",
    "Casalecchio tiro a volo",
    "Casalecchio chiusa",
]
UPSTREAMS_NO_CHIUSA = [
    "Forcelli",
    "Sostegno Reno",
    "Bonconvento",
    "Casalecchio tiro a volo",
]
EVENT_PAD = pd.Timedelta(hours=6)  # catalog flood windows ±6 h for flood-only train
# Downstream acceptors (backwater): not chase targets.
ACCEPTORS = ["Dosso", "Opera Reno Panfilia", "Secondo Salto", "Gallo"]
# Successive valle edges for chain-delta features (upstream_node - acceptor_node).
CHAIN_PAIRS = [
    ("Cento", "Dosso"),
    ("Dosso", "Opera Reno Panfilia"),
    ("Opera Reno Panfilia", "Secondo Salto"),
    ("Opera Reno Panfilia", "Gallo"),
]
LOOKBACK = 48  # 24 h at 30′
HORIZONS = [1, 2, 3, 4]
EVENT_ABS = 0.10
WEAK_POINTS = 48
# Holdout skill baseline + d1/d2 features: quadratic OLS on recent Cento levels.
# Sweep on event@120: poly2 peaks near 6 points (2.5 h); 4 pts (1.5 h) underfits
# slow rising limbs; wider windows over-extrapolate curvature.
POLY_NAIVE_POINTS = 6
POLY_NAIVE_DEGREE = 2
POLY_NAIVE_MIN = 3
LIN3_POINTS = 3  # residual-lin3 baseline: linear OLS on last 3 Cento samples
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}
LGBM_ROUNDS = 800
LGBM_EARLY = 50
DNN_HIDDEN = (128, 64)
DNN_LR = 1e-3
DNN_BATCH = 512
DNN_EPOCHS = 80
DNN_PATIENCE = 10
DNN_SEED = 42


def _load_fit_transfer():
    path = Path(__file__).resolve().parent / "fit-transfer.py"
    spec = importlib.util.spec_from_file_location("fit_transfer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fit_transfer"] = mod
    spec.loader.exec_module(mod)
    return mod


ft = _load_fit_transfer()


def safe_name(station: str) -> str:
    return station.replace(" ", "_")


def load_soglia_levels(path: Path, prefer: str = "yellow") -> dict[str, float]:
    """Map station → reference soglia. Default yellow (present on all stations)."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for name, block in (payload.get("stations") or {}).items():
        chosen = block.get(prefer)
        if chosen is not None and math.isfinite(float(chosen)) and float(chosen) > 0:
            out[name] = float(chosen)
            continue
        # Fallback order if preferred missing: yellow → orange → red → max soglie
        for key in ("yellow", "orange", "red"):
            v = block.get(key)
            if v is not None and math.isfinite(float(v)) and float(v) > 0:
                out[name] = float(v)
                break
        else:
            soglie = [
                float(v) for v in (block.get("soglie") or [])
                if v is not None and math.isfinite(float(v)) and float(v) > 0
            ]
            if soglie:
                out[name] = max(soglie)
    return out


def tau_frac_lag_steps(tau_steps: int, n: int, phi_max: float) -> list[int]:
    """Phase grid φ=k/n for k=0…n·φ_max → lag steps (dedup, non-negative)."""
    n = max(1, int(n))
    tau_steps = max(0, int(tau_steps))
    phi_max = float(phi_max)
    if phi_max < 0:
        raise ValueError("phi_max must be ≥ 0")
    k_max = int(round(n * phi_max))
    out: list[int] = []
    seen: set[int] = set()
    for k in range(k_max + 1):
        lag = int(round((k / n) * tau_steps))
        if lag < 0 or lag in seen:
            continue
        seen.add(lag)
        out.append(lag)
    if not out:
        out = [0]
    return out


def build_feature_matrix(
    wide: pd.DataFrame,
    rain: pd.DataFrame | None,
    lookback: int = LOOKBACK,
    stations: list[str] | None = None,
    *,
    soglia_levels: dict[str, float] | None = None,
    use_soglia_headroom: bool = False,
    use_soglia_ratio: bool = False,
    soglia_kind: str = "yellow",
    chain_pairs: list[tuple[str, str]] | None = None,
    poly_points: int = POLY_NAIVE_POINTS,
    upstream_tau: dict[str, int] | None = None,
    lag_align_h: int = 0,
    align_mode: str = "shift",
    align_window: int = 16,
    station_lookback: dict[str, int] | None = None,
    station_lag_steps: dict[str, list[int]] | None = None,
    drop_level: bool = False,
    drop_level_stations: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Flat window of level/delta/miss (+ optional soglia headroom/ratio / chain deltas) + rain.

    ``upstream_tau`` (station → lag in 30′ steps) enables lag-aligned features:
    - ``shift``: upstream lag k samples x(t + h - tau_i - k·step);
    - ``augment``: flat window untouched + extra ``a*`` block sampling
      x(t + h - tau_i - j·step) for j in [0, align_window).
    For h ≤ tau_i the wave reaching valle at t+h is already observed upstream.
    Shifts < 0 (h > tau_i) become NaN features with miss=1 — no future peeking.

    ``station_lookback`` (optional): per-station window length in 30′ steps, capped
    by ``lookback``. Used by ``--lookback-tau`` (L_i = τ_i + buffer); unrelated to
    lag-align shift/augment.

    ``station_lag_steps`` (optional): explicit lag indices per station (τ-fraction
    grid). Feature names use the absolute step (``*_lag{s}``).

    ``drop_level`` / ``drop_level_stations``: omit absolute level/headroom/ratio
    (keep delta/d1/d2/miss) so the model stays in increments — no absolute state.
    """
    parts: dict[str, pd.Series] = {}
    feature_cols: list[str] = []
    station_order = list(stations) if stations is not None else list(wide.columns)
    soglia_levels = soglia_levels or {}
    upstream_tau = upstream_tau or {}
    station_lookback = station_lookback or {}
    station_lag_steps = station_lag_steps or {}
    drop_level_stations = drop_level_stations or set()
    prefix = f"headroom_{soglia_kind}" if use_soglia_headroom else "level"
    ratio_prefix = f"ratio_{soglia_kind}"
    n_poly = max(POLY_NAIVE_MIN, int(poly_points))
    nan_series = pd.Series(np.nan, index=wide.index, dtype=float)
    one_series = pd.Series(1.0, index=wide.index, dtype=float)

    def shifted(series: pd.Series, s: int) -> pd.Series:
        """Causal shift: s<0 would peek into the future → all-NaN."""
        if s < 0:
            return nan_series
        return series.shift(s) if s else series

    for station in station_order:
        if station not in wide.columns:
            raise SystemExit(f"Missing station column: {station}")
        level = wide[station]
        sparse = station in station_lag_steps
        if sparse:
            lag_keys = [int(s) for s in station_lag_steps[station]]
            lag_off = 0
            is_aligned_up = False
        else:
            lb = int(station_lookback.get(station, lookback)) if station_lookback else int(lookback)
            lb = max(1, min(lb, int(lookback)))
            lag_keys = list(range(lb))
            is_aligned_up = station in upstream_tau
            lag_off = (
                int(upstream_tau[station]) - int(lag_align_h)
                if is_aligned_up and align_mode == "shift"
                else 0
            )
        drop_level_st = bool(drop_level) or station in drop_level_stations
        prev = level.shift(1)
        miss = level.isna().astype(float)
        tag = safe_name(station)
        ref = soglia_levels.get(station)
        use_head = use_soglia_headroom and ref is not None and ref > 0 and not drop_level_st
        use_ratio = use_soglia_ratio and ref is not None and ref > 0 and not drop_level_st
        # Headroom below soglia: larger ⇒ farther from threshold exceedance.
        signal = (ref - level) if use_head else level
        signal_prev = (ref - prev) if use_head else prev
        signal_delta = signal - signal_prev
        signal_delta = signal_delta.where(level.notna() & prev.notna())
        # Local quadratic OLS on raw level → 1st/2nd derivative estimates (per 30′ step).
        log(f"  poly d1/d2 for {station} (n={n_poly})…", level="debug")
        d1_s, d2_s = series_poly_derivatives(level, n_points=n_poly)
        feat_prefix = prefix if use_head else "level"
        for k in lag_keys:
            s = k + lag_off
            dv = shifted(signal_delta, s)
            mv = shifted(miss, s).fillna(1.0)
            c_dv = f"delta_{tag}_lag{k}"
            c_mv = f"miss_{tag}_lag{k}"
            c_d1 = f"d1_{tag}_lag{k}"
            c_d2 = f"d2_{tag}_lag{k}"
            parts[c_dv] = dv
            parts[c_mv] = mv
            parts[c_d1] = shifted(d1_s, s)
            parts[c_d2] = shifted(d2_s, s)
            if drop_level_st:
                feature_cols.extend([c_dv, c_mv, c_d1, c_d2])
            else:
                c_lv = f"{feat_prefix}_{tag}_lag{k}"
                parts[c_lv] = shifted(signal, s)
                feature_cols.extend([c_lv, c_dv, c_mv, c_d1, c_d2])
        if use_ratio:
            # Fill fraction of soglia (0 ⇒ empty relative to yellow, 1 ⇒ at yellow).
            ratio = level / ref
            ratio_prev = prev / ref
            ratio_delta = (ratio - ratio_prev).where(level.notna() & prev.notna())
            for k in lag_keys:
                c_r = f"{ratio_prefix}_{tag}_lag{k}"
                c_rd = f"ratio_delta_{tag}_lag{k}"
                parts[c_r] = shifted(ratio, k + lag_off)
                parts[c_rd] = shifted(ratio_delta, k + lag_off)
                feature_cols.extend([c_r, c_rd])
        if is_aligned_up and align_mode == "augment":
            # Arriving-wave block: j=0 is the upstream sample expected to reach
            # valle at t+h (x(t+h-tau_i)); j>0 walks further back along the wave.
            a_off = int(upstream_tau[station]) - int(lag_align_h)
            for j in range(max(1, int(align_window))):
                s = a_off + j
                c_lv = f"alevel_{tag}_lag{j}"
                c_dv = f"adelta_{tag}_lag{j}"
                c_mv = f"amiss_{tag}_lag{j}"
                c_d1 = f"ad1_{tag}_lag{j}"
                c_d2 = f"ad2_{tag}_lag{j}"
                parts[c_lv] = shifted(signal, s)
                parts[c_dv] = shifted(signal_delta, s)
                parts[c_mv] = shifted(miss, s).fillna(1.0)
                parts[c_d1] = shifted(d1_s, s)
                parts[c_d2] = shifted(d2_s, s)
                feature_cols.extend([c_lv, c_dv, c_mv, c_d1, c_d2])

    for a, b in (chain_pairs or []):
        if a not in wide.columns or b not in wide.columns:
            continue
        diff = wide[a] - wide[b]
        diff = diff.where(wide[a].notna() & wide[b].notna())
        miss_ab = (wide[a].isna() | wide[b].isna()).astype(float)
        tag = f"{safe_name(a)}__{safe_name(b)}"
        for k in range(lookback):
            c_d = f"chain_{tag}_lag{k}"
            c_m = f"chain_miss_{tag}_lag{k}"
            parts[c_d] = diff.shift(k)
            parts[c_m] = miss_ab.shift(k).fillna(1.0)
            feature_cols.extend([c_d, c_m])

    if rain is not None and {"rain6h", "rain24h"}.issubset(rain.columns):
        parts["rain6h"] = rain["rain6h"]
        parts["rain24h"] = rain["rain24h"]
        feature_cols.extend(["rain6h", "rain24h"])

    feats = pd.DataFrame(parts, index=wide.index)
    return feats, feature_cols


def build_targets(wide: pd.DataFrame, h: int) -> pd.DataFrame:
    y = wide[DOWNSTREAM]
    return pd.DataFrame(
        {
            "y0": y,
            "y_future": y.shift(-h),
            "delta": y.shift(-h) - y,
        },
        index=wide.index,
    )


def align_xy(
    feats: pd.DataFrame,
    feature_cols: list[str],
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Keep rows with finite target y0 and delta; features may contain NaN."""
    frame = feats[feature_cols].copy()
    frame["y0"] = targets["y0"]
    frame["delta"] = targets["delta"]
    ok = frame["y0"].notna() & frame["delta"].notna()
    # Need a full lookback of miss flags defined (after initial warm-up)
    miss_cols = [c for c in feature_cols if c.startswith("miss_")]
    if miss_cols:
        ok &= frame[miss_cols].notna().all(axis=1)
    frame = frame.loc[ok]
    X = frame[feature_cols]
    y = frame["delta"].to_numpy(dtype=float)
    y0 = frame["y0"].to_numpy(dtype=float)
    return X, y, y0, frame.index.to_numpy()


def _poly_deriv_weights(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear maps y[0..n) → (d1, d2) for OLS y≈a0+a1 x+a2 x² with x=-(n-1)..0."""
    x = np.arange(-(n_points - 1), 1, dtype=np.float64)
    v = np.column_stack([np.ones(n_points), x, x * x])
    pinv = np.linalg.pinv(v)  # 3 × n
    return pinv[1].copy(), (2.0 * pinv[2]).copy()


_POLY_D1_W3, _POLY_D2_W3 = _poly_deriv_weights(POLY_NAIVE_MIN)


def series_poly_derivatives(
    level: pd.Series,
    n_points: int = POLY_NAIVE_POINTS,
) -> tuple[pd.Series, pd.Series]:
    """Local quadratic OLS → first/second derivative estimates (per step) at each t."""
    from numpy.lib.stride_tricks import sliding_window_view

    n_points = max(POLY_NAIVE_MIN, int(n_points))
    w_d1, w_d2 = _poly_deriv_weights(n_points)
    vals = level.to_numpy(dtype=float)
    n = len(vals)
    d1 = np.full(n, np.nan)
    d2 = np.full(n, np.nan)
    if n >= n_points:
        win = sliding_window_view(vals, n_points)
        ok = np.isfinite(win).all(axis=1)
        idx = np.flatnonzero(ok) + (n_points - 1)
        d1[idx] = win[ok] @ w_d1
        d2[idx] = win[ok] @ w_d2
    if n >= POLY_NAIVE_MIN:
        win3 = sliding_window_view(vals, POLY_NAIVE_MIN)
        ok3 = np.isfinite(win3).all(axis=1)
        idx3 = np.flatnonzero(ok3) + (POLY_NAIVE_MIN - 1)
        need = ~np.isfinite(d1[idx3])
        if need.any():
            rows = win3[ok3][need]
            d1[idx3[need]] = rows @ _POLY_D1_W3
            d2[idx3[need]] = rows @ _POLY_D2_W3
    return (
        pd.Series(d1, index=level.index, dtype=float),
        pd.Series(d2, index=level.index, dtype=float),
    )


def poly2_extrapolate(window: np.ndarray, h_steps: int) -> float:
    """OLS degree-2 on window (oldest→newest); predict h_steps after the last sample."""
    n = int(window.shape[0])
    x = np.arange(n, dtype=np.float64)
    coef = np.polyfit(x, window.astype(np.float64), POLY_NAIVE_DEGREE)
    return float(np.polyval(coef, (n - 1) + h_steps))


def _poly_extrapolate_weights(n_points: int, h_steps: int) -> np.ndarray:
    """Weights w such that w·y ≈ poly2(y) extrapolated h_steps past the last sample."""
    x = np.arange(n_points, dtype=np.float64)
    v = np.column_stack([np.ones(n_points), x, x * x])
    pinv = np.linalg.pinv(v)  # 3 × n
    xh = float((n_points - 1) + h_steps)
    return (np.array([1.0, xh, xh * xh], dtype=np.float64) @ pinv).copy()


def build_poly2_baseline(
    level: pd.Series,
    index: np.ndarray,
    h_steps: int,
    n_points: int = POLY_NAIVE_POINTS,
) -> np.ndarray:
    """Per-row quadratic extrapolation of Cento to t+h from the last n_points levels."""
    n_points = max(POLY_NAIVE_MIN, int(n_points))
    idx = pd.DatetimeIndex(index)
    cols = [
        level.shift(k).reindex(idx).to_numpy(dtype=float)
        for k in range(n_points - 1, -1, -1)
    ]
    window = np.column_stack(cols)
    y0 = window[:, -1]
    out = np.array(y0, dtype=float, copy=True)  # persist fallback
    w_n = _poly_extrapolate_weights(n_points, h_steps)
    ok = np.isfinite(window).all(axis=1)
    out[ok] = window[ok] @ w_n
    # Fill remaining with 3-point poly when possible
    need = ~ok & np.isfinite(y0)
    if need.any():
        w3 = _poly_extrapolate_weights(POLY_NAIVE_MIN, h_steps)
        tail = window[need][:, -POLY_NAIVE_MIN :]
        ok3 = np.isfinite(tail).all(axis=1)
        idx_need = np.flatnonzero(need)
        out[idx_need[ok3]] = tail[ok3] @ w3
    return out


def _lin_extrapolate_weights(n_points: int, h_steps: int) -> np.ndarray:
    """Weights w such that w·y ≈ linear OLS(y) extrapolated h_steps past the last sample."""
    x = np.arange(n_points, dtype=np.float64)
    v = np.column_stack([np.ones(n_points), x])
    pinv = np.linalg.pinv(v)  # 2 × n
    xh = float((n_points - 1) + h_steps)
    return (np.array([1.0, xh], dtype=np.float64) @ pinv).copy()


def build_lin3_baseline(
    level: pd.Series,
    index: np.ndarray,
    h_steps: int,
    n_points: int = LIN3_POINTS,
) -> np.ndarray:
    """Per-row linear extrapolation of Cento to t+h from the last n_points levels.

    Fallback: persistence (y0) when the window has any non-finite sample.
    """
    n_points = max(2, int(n_points))
    idx = pd.DatetimeIndex(index)
    cols = [
        level.shift(k).reindex(idx).to_numpy(dtype=float)
        for k in range(n_points - 1, -1, -1)
    ]
    window = np.column_stack(cols)
    y0 = window[:, -1]
    out = np.array(y0, dtype=float, copy=True)  # persist fallback
    w_n = _lin_extrapolate_weights(n_points, h_steps)
    ok = np.isfinite(window).all(axis=1)
    out[ok] = window[ok] @ w_n
    return out


def metrics_block(
    y0: np.ndarray,
    delta_obs: np.ndarray,
    delta_pred: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
    baseline_name: str = "persist",
) -> dict:
    obs = y0 + delta_obs
    pred = y0 + delta_pred
    naive = y0 if baseline is None else baseline
    mask = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(naive)
    n = int(mask.sum())
    if n < 8:
        return {
            "rmse": None,
            "skill": None,
            "mae": None,
            "n": n,
            "baseline": baseline_name,
        }
    err_m = float(math.sqrt(np.mean((pred[mask] - obs[mask]) ** 2)))
    err_b = float(math.sqrt(np.mean((naive[mask] - obs[mask]) ** 2)))
    skill = None if err_b == 0 else 1.0 - err_m / err_b
    mae = float(np.mean(np.abs(pred[mask] - obs[mask])))
    out = {
        "rmse": round(err_m, 6),
        "skill": None if skill is None else round(skill, 6),
        "mae": round(mae, 6),
        "n": n,
        "baseline": baseline_name,
        "rmseBaseline": round(err_b, 6),
    }
    if baseline is not None:
        err_p = float(math.sqrt(np.mean((y0[mask] - obs[mask]) ** 2)))
        skill_p = None if err_p == 0 else 1.0 - err_m / err_p
        out["skillPersist"] = None if skill_p is None else round(skill_p, 6)
        out["rmsePersist"] = round(err_p, 6)
    return out


def split_masks(
    index: pd.DatetimeIndex | np.ndarray,
    mode: str | None = None,
    holdout_from: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = pd.DatetimeIndex(index)
    blocks = ft.month_blocks(idx, mode=mode, holdout_from=holdout_from)
    return blocks == 0, blocks == 1, blocks == 2


def inner_split_masks(
    index: pd.DatetimeIndex | np.ndarray,
    outer_train: np.ndarray,
    *,
    mode: str = "alt",
) -> tuple[np.ndarray, np.ndarray]:
    """Split outer-train months into train_inner / val_inner (E2 selection).

    Holdout and outer-validate rows stay False in both masks.
    """
    idx = pd.DatetimeIndex(index)
    naive = idx.tz_convert("UTC").tz_localize(None) if idx.tz is not None else idx
    months = naive.to_period("M")
    train_months = sorted({m for m, keep in zip(months, outer_train) if keep})
    mapping: dict = {}
    if mode == "block":
        mid = max(1, len(train_months) // 2) if train_months else 0
        for i, m in enumerate(train_months):
            mapping[m] = 0 if i < mid else 1
    else:
        for i, m in enumerate(train_months):
            mapping[m] = i % 2
    train_inner = np.zeros(len(idx), dtype=bool)
    val_inner = np.zeros(len(idx), dtype=bool)
    for i, m in enumerate(months):
        if not outer_train[i] or m not in mapping:
            continue
        if mapping[m] == 0:
            train_inner[i] = True
        else:
            val_inner[i] = True
    return train_inner, val_inner


def event_mask(delta: np.ndarray, base: np.ndarray) -> np.ndarray:
    return base & (np.abs(delta) >= EVENT_ABS)


def load_catalog_events(path: Path, station: str = DOWNSTREAM) -> list[dict]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    events = [
        e for e in catalog.get("events", catalog if isinstance(catalog, list) else [])
        if e.get("station") == station
    ]
    if not events and "stations" in catalog:
        events = list((catalog["stations"].get(station) or {}).get("events") or [])
        for e in events:
            e.setdefault("station", station)
    return events


def flood_window_mask(
    index: pd.DatetimeIndex | np.ndarray,
    events: list[dict],
    *,
    pad: pd.Timedelta = EVENT_PAD,
) -> np.ndarray:
    """True for timestamps inside any catalog flood window (start−pad … end+pad)."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    mask = np.zeros(len(idx), dtype=bool)
    for ev in events:
        start = pd.Timestamp(ev["start"])
        end = pd.Timestamp(ev["end"])
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        lo, hi = start - pad, end + pad
        mask |= (idx >= lo) & (idx <= hi)
    return mask


def eval_splits(
    y0: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
    hold_m: np.ndarray,
    *,
    poly_baseline: np.ndarray | None = None,
) -> dict:
    """Train/validate skill vs persistence; holdout skill vs quadratic local poly (stupid predictor)."""
    hold_base = poly_baseline
    hold_name = "poly2" if poly_baseline is not None else "persist"

    def _hold(mask: np.ndarray) -> dict:
        base = None if hold_base is None else hold_base[mask]
        return metrics_block(
            y0[mask], y[mask], pred[mask],
            baseline=base, baseline_name=hold_name,
        )

    return {
        "train": metrics_block(y0[train_m], y[train_m], pred[train_m]),
        "validate": metrics_block(y0[val_m], y[val_m], pred[val_m]),
        "validateEvents": metrics_block(
            y0[event_mask(y, val_m)],
            y[event_mask(y, val_m)],
            pred[event_mask(y, val_m)],
        ),
        "holdout": _hold(hold_m),
        "holdoutEvents": _hold(event_mask(y, hold_m)),
    }


def train_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
) -> tuple[lgb.Booster, np.ndarray]:
    t0 = time.perf_counter()
    log(
        f"LightGBM start: train={int(train_m.sum())} val={int(val_m.sum())} "
        f"feat={X.shape[1]} rounds≤{LGBM_ROUNDS} early={LGBM_EARLY}"
    )
    dtrain = lgb.Dataset(X.iloc[train_m], label=y[train_m], free_raw_data=False)
    dval = lgb.Dataset(X.iloc[val_m], label=y[val_m], reference=dtrain, free_raw_data=False)
    eval_period = 25 if _VERBOSE else 0

    class _LgbmProgress:
        """LightGBM callback: print val metric every N iterations."""

        def __init__(self, period: int) -> None:
            self.period = period
            self.order = 20

        def __call__(self, env) -> None:  # LightGBM CallbackEnv
            it = env.iteration + 1
            if self.period <= 0 or (it % self.period != 0 and it != 1):
                return
            parts = []
            for item in env.evaluation_result_list or []:
                name, metric, value, _hb = item
                parts.append(f"{name}:{metric}={value:.6g}")
            log(f"  LGBM iter {it}/{LGBM_ROUNDS}  " + (" ".join(parts) or "(no eval)"), level="debug")

    callbacks: list = [
        lgb.early_stopping(LGBM_EARLY, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    if eval_period:
        callbacks.append(_LgbmProgress(eval_period))

    booster = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=LGBM_ROUNDS,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=callbacks,
    )
    pred = booster.predict(X, num_iteration=booster.best_iteration)
    log(
        f"LightGBM done in {time.perf_counter() - t0:.1f}s  "
        f"best_iter={booster.best_iteration}",
        level="ok",
    )
    return booster, np.asarray(pred, dtype=float)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...] = DNN_HIDDEN):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def standardize_features(
    X: pd.DataFrame,
    feature_cols: list[str],
    train_m: np.ndarray,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Z-score continuous cols on train; miss_* left as-is; NaN → 0 after scale."""
    arr = X[feature_cols].to_numpy(dtype=float)
    stats: dict[str, tuple[float, float]] = {}
    out = np.zeros_like(arr, dtype=float)
    for j, col in enumerate(feature_cols):
        col_data = arr[:, j]
        if col.startswith("miss_"):
            out[:, j] = np.nan_to_num(col_data, nan=1.0)
            stats[col] = (0.0, 1.0)
            continue
        base = col_data[train_m]
        base = base[np.isfinite(base)]
        mu = float(base.mean()) if len(base) else 0.0
        sd = float(base.std()) if len(base) > 1 else 1.0
        if sd < 1e-6:
            sd = 1.0
        scaled = (col_data - mu) / sd
        out[:, j] = np.nan_to_num(scaled, nan=0.0)
        stats[col] = (mu, sd)
    return out, stats


def train_dnn(
    X_std: np.ndarray,
    y: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
    quantile: float | None = None,
    event_weight: float = 1.0,
    rising_under_weight: float = 1.0,
) -> tuple[MLP, np.ndarray, dict]:
    torch.manual_seed(DNN_SEED)
    np.random.seed(DNN_SEED)
    device = torch.device("cpu")
    model = MLP(X_std.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=DNN_LR)
    loss_fn = nn.MSELoss()
    if quantile is not None:
        # Pinball loss: under-predictions weighted by q, over-predictions by 1-q.
        # q=0.75 on rising limbs behaves like an earlier crossing-time estimate.
        q = float(quantile)

        def pinball(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            err = target - pred
            return torch.mean(torch.maximum(q * err, (q - 1.0) * err))

        loss_fn = pinball

    n_tr, n_va = int(train_m.sum()), int(val_m.sum())
    n_batch = max(1, (n_tr + DNN_BATCH - 1) // DNN_BATCH)
    ruw = float(rising_under_weight) if rising_under_weight else 1.0
    log(
        f"DNN start: in={X_std.shape[1]} hidden={list(DNN_HIDDEN)} "
        f"train={n_tr} val={n_va} batch={DNN_BATCH} (~{n_batch}/epoch) "
        f"epochs≤{DNN_EPOCHS} patience={DNN_PATIENCE} device={device}"
        + (f" risingUnderWeight={ruw:g}" if ruw != 1.0 else "")
    )
    t0 = time.perf_counter()

    x_tr = torch.tensor(X_std[train_m], dtype=torch.float32)
    y_tr = torch.tensor(y[train_m], dtype=torch.float32)
    x_va = torch.tensor(X_std[val_m], dtype=torch.float32)
    y_va = torch.tensor(y[val_m], dtype=torch.float32)
    w_tr = None
    if event_weight and event_weight != 1.0:
        # Event emphasis: |Δy| ≥ EVENT_ABS gets weight event_weight (mean renormalized).
        w_np = 1.0 + (float(event_weight) - 1.0) * (np.abs(y[train_m]) >= EVENT_ABS)
        w_np = w_np / w_np.mean()
        w_tr = torch.tensor(w_np, dtype=torch.float32)
    if w_tr is not None:
        loader = DataLoader(TensorDataset(x_tr, y_tr, w_tr), batch_size=DNN_BATCH, shuffle=True)
    else:
        loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=DNN_BATCH, shuffle=True)

    def batch_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        if quantile is not None:
            err = target - pred
            elem = torch.maximum(float(quantile) * err, (float(quantile) - 1.0) * err)
        else:
            elem = (pred - target) ** 2
            # Rising-only asymmetric MSE: when Δy>0 (rising), under-predictions
            # (pred < target) cost rising_under_weight×; falling / over-pred unchanged.
            if ruw != 1.0:
                rising = target > 0
                under = pred < target
                scale = torch.where(rising & under, torch.full_like(elem, ruw), torch.ones_like(elem))
                elem = elem * scale
        if weight is None:
            return elem.mean()
        return (elem * weight).sum() / weight.sum()

    best_state = None
    best_val = float("inf")
    patience = 0
    history = {"best_epoch": 0, "best_val_mse": None}
    prog_every = 5 if _VERBOSE else 10

    for epoch in range(DNN_EPOCHS):
        model.train()
        train_loss_sum = 0.0
        n_seen = 0
        for batch in loader:
            xb, yb = batch[0], batch[1]
            wb = batch[2] if len(batch) > 2 else None
            opt.zero_grad()
            pred = model(xb)
            loss = batch_loss(pred, yb, wb)
            loss.backward()
            opt.step()
            bs = int(xb.shape[0])
            train_loss_sum += float(loss.item()) * bs
            n_seen += bs
        train_mse = train_loss_sum / max(1, n_seen)
        model.eval()
        with torch.no_grad():
            val_pred = model(x_va)
            val_mse = float(loss_fn(val_pred, y_va).item())
        improved = val_mse < best_val - 1e-8
        if improved:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            history["best_epoch"] = epoch + 1
            history["best_val_mse"] = round(best_val, 8)
        else:
            patience += 1

        ep = epoch + 1
        if improved or ep == 1 or ep % prog_every == 0 or patience >= DNN_PATIENCE:
            mark = "*" if improved else " "
            log(
                f"  DNN epoch {ep:3d}/{DNN_EPOCHS} {mark}  "
                f"train_mse={train_mse:.6g}  val_mse={val_mse:.6g}  "
                f"best={best_val:.6g}@ep{history['best_epoch']}  "
                f"patience={patience}/{DNN_PATIENCE}",
                level="debug",
            )
        if patience >= DNN_PATIENCE:
            log(f"  DNN early stop at epoch {ep} (patience={DNN_PATIENCE})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        full = torch.tensor(X_std, dtype=torch.float32)
        pred = model(full).cpu().numpy().astype(float)
    log(
        f"DNN done in {time.perf_counter() - t0:.1f}s  "
        f"best_epoch={history['best_epoch']} best_val_mse={history['best_val_mse']}",
        level="ok",
    )
    return model, pred, history


def gauge_names(config: dict) -> list[str]:
    raw = list((config.get("rainfall") or {}).get("gauges") or [])
    names: list[str] = []
    for g in raw:
        if isinstance(g, str):
            names.append(g)
        elif isinstance(g, dict) and g.get("name"):
            names.append(str(g["name"]))
    return names


def export_mlp_layers(model: MLP) -> list[dict]:
    """Serialize Linear layers as row-major W (out x in) + bias + activation."""
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    layers: list[dict] = []
    for i, linear in enumerate(linears):
        w = linear.weight.detach().cpu().numpy()
        b = linear.bias.detach().cpu().numpy()
        layers.append(
            {
                "W": [[round(float(v), 8) for v in row] for row in w],
                "b": [round(float(v), 8) for v in b],
                "activation": "linear" if i == len(linears) - 1 else "relu",
            }
        )
    return layers


def residual_spreads_pair(
    y0: np.ndarray,
    delta_obs: np.ndarray,
    delta_pred: np.ndarray,
    mask: np.ndarray,
    q: float = 0.85,
) -> tuple[float | None, float | None]:
    obs = y0[mask] + delta_obs[mask]
    pred = y0[mask] + delta_pred[mask]
    return ft.residual_spreads(obs, pred, q=q)


def pack_spreads(
    y0: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    val_m: np.ndarray,
) -> dict:
    val_obs_delta = y
    lo_all, hi_all = residual_spreads_pair(y0, val_obs_delta, pred, val_m)
    event = val_m & (np.abs(y) >= EVENT_ABS)
    calm = val_m & ~event
    lo_base, hi_base = residual_spreads_pair(y0, val_obs_delta, pred, calm)
    lo_high, hi_high = residual_spreads_pair(y0, val_obs_delta, pred, event, q=0.95)
    if lo_base is None:
        lo_base, hi_base = lo_all, hi_all
    if lo_high is None:
        lo_high, hi_high = lo_all, hi_all
    if lo_high is not None and lo_base is not None:
        lo_high = max(lo_high, lo_base) * 1.25
        hi_high = max(hi_high, hi_base) * 1.25
    return {
        "spreadLow": None if lo_base is None else round(lo_base, 6),
        "spreadHigh": None if hi_base is None else round(hi_base, 6),
        "spreadLowHigh": None if lo_high is None else round(lo_high, 6),
        "spreadHighHigh": None if hi_high is None else round(hi_high, 6),
    }


def ridge_side_by_side(coeffs_path: Path, h_minutes: int) -> dict | None:
    if not coeffs_path.exists():
        return None
    try:
        payload = json.loads(coeffs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    node = (payload.get("nodes") or {}).get(DOWNSTREAM) or {}
    block = (node.get("horizons") or {}).get(str(h_minutes))
    if not block:
        return None
    keys = ("train", "validate", "validateEvents", "holdout", "holdoutEvents")
    out = {}
    for k in keys:
        m = block.get(k) or {}
        out[k] = {
            "rmse": m.get("rmse"),
            "skill": m.get("skill"),
            "mae": m.get("mae"),
            "n": m.get("n"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline LightGBM vs DNN for Cento")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data", type=Path, default=ROOT / "storico")
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: ml-cento-compare-valle.json (or -novalle with --no-acceptors)")
    parser.add_argument("--coeffs", type=Path, default=ROOT / "models" / "transfer-coeffs.json")
    parser.add_argument("--export-dnn", type=Path, default=None,
                        help="Default: models/cento-dnn-valle.json (or cento-dnn-novalle.json with --no-acceptors)")
    parser.add_argument(
        "--export-coeffs",
        type=Path,
        nargs="?",
        const=ROOT / "models" / "transfer-coeffs.json",
        default=None,
        help="Patch nodes.Cento in transfer-coeffs (opt-in; pass flag alone for default path)",
    )
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--spike-abs", type=float, default=ft.SPIKE_ABS)
    parser.add_argument("--no-rain", action="store_true")
    parser.add_argument("--no-acceptors", action="store_true",
                        help="Baseline without valle acceptors (ablation)")
    parser.add_argument(
        "--acceptors",
        nargs="+",
        default=None,
        help="Override valle acceptors (default: Dosso…Gallo). Ignored with --no-acceptors.",
    )
    parser.add_argument(
        "--extra-upstreams",
        nargs="+",
        default=None,
        help="Append monte stations beyond the default UPSTREAMS set (e.g. Bagnetto Reno).",
    )
    parser.add_argument(
        "--no-chiusa",
        action="store_true",
        help="Ablation: drop Casalecchio chiusa from UPSTREAMS (tiro a volo remains)",
    )
    parser.add_argument(
        "--chiusa-weir",
        action="store_true",
        help="WP2.4: replace Casalecchio chiusa level with weir discharge proxy max(h-crest,0)^1.5",
    )
    parser.add_argument(
        "--weir-crest",
        type=float,
        default=None,
        help="Crest for --chiusa-weir (default: 10th percentile of chiusa on train months)",
    )
    parser.add_argument(
        "--flood-only",
        action="store_true",
        help="Restrict train/fit masks to catalog flood windows (±6 h); calm out of loss",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=ROOT / "models" / "events.json",
        help="Catalog for --flood-only windows",
    )
    parser.add_argument(
        "--feature-set",
        choices=("raw", "yellow-headroom", "chain", "yellow-chain"),
        default="raw",
        help=(
            "raw=levels; yellow-headroom=(giallo−livello)+livello/giallo; "
            "chain=Δ tra nodi valle; yellow-chain=both"
        ),
    )
    parser.add_argument(
        "--soglie",
        type=Path,
        default=ROOT / "models" / "arpae-soglie.json",
        help="Cached ARPAE soglie (yellow/orange/red)",
    )
    parser.add_argument("--skip-lgbm", action="store_true", help="Train/export DNN only")
    parser.add_argument(
        "--lag-align",
        nargs="?",
        const="shift",
        default=None,
        choices=("shift", "augment"),
        help=(
            "shift: le finestre upstream partono da x(t+h-tau_i) (perdono le ore recenti di monte); "
            "augment: finestra piatta invariata + blocco align_ centrato sull'onda in arrivo x(t+h-tau_i-j)"
        ),
    )
    parser.add_argument(
        "--lookback-tau",
        action="store_true",
        help=(
            "Per-station lookback L_i = τ_i + buffer (τ from train-month cross-corr); "
            "downstream keeps --lookback. Distinct from --lag-align."
        ),
    )
    parser.add_argument(
        "--lookback-tau-buffer-hours",
        type=float,
        default=4.0,
        help="Extra hours added to each upstream τ for --lookback-tau (default 4 h)",
    )
    parser.add_argument(
        "--tau-frac",
        action="store_true",
        help=(
            "Upstream features on phase grid φ=k/n up to --tau-frac-max·τ "
            "(snap 30′); Cento/acceptors keep --tau-frac-local-lookback flat. "
            "Distinct from --lag-align / --lookback-tau."
        ),
    )
    parser.add_argument(
        "--tau-frac-n",
        type=int,
        default=8,
        help="Points per τ on the phase grid (default 8 → step φ=1/8)",
    )
    parser.add_argument(
        "--tau-frac-max",
        type=float,
        default=2.0,
        help="Max phase φ in units of τ (default 2 → cover up to 2τ)",
    )
    parser.add_argument(
        "--tau-frac-local-lookback",
        type=int,
        default=24,
        help="Flat lookback steps for Cento/acceptors when --tau-frac (default 24 = 12 h)",
    )
    parser.add_argument(
        "--tau-frac-signal",
        choices=("level", "delta"),
        default="level",
        help=(
            "With --tau-frac only: delta=omit absolute level on τ-grid upstreams. "
            "Prefer --feature-signal for flat and τ-grid alike."
        ),
    )
    parser.add_argument(
        "--feature-signal",
        choices=("level", "delta"),
        default="level",
        help=(
            "level=absolute stage + delta/d1/d2 (default); "
            "delta=omit absolute level on all stations (increments only; "
            "target stays Δy, no absolute-state features)"
        ),
    )
    parser.add_argument(
        "--holdout-from",
        type=str,
        default=None,
        help=(
            "ISO date (e.g. 2026-01-01): months ≥ cut → holdout; "
            "train/validate only on earlier months (leakage-free calendar split)"
        ),
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help=(
            "Se impostato (es. 0.75), allena la DNN con pinball loss sul quantile: "
            "anticipa i fronti di piena (timing) a costo di un bias positivo in calma"
        ),
    )
    parser.add_argument(
        "--event-weight",
        type=float,
        default=1.0,
        help="Peso extra per i campioni di train con |Δy| ≥ 0.10 m (default 1 = nessun peso)",
    )
    parser.add_argument(
        "--rising-under-weight",
        type=float,
        default=1.0,
        help=(
            "Su campioni rising (Δy>0), moltiplica la MSE quando pred<target "
            "(sottostima). Default 1 = MSE simmetrica; tipico A/B: 2, 3, 4"
        ),
    )
    parser.add_argument(
        "--rising-only",
        action="store_true",
        help="M3 MoE: restringe il train a campioni Δy>0 (esperto salita); val/hold restano pieni",
    )
    parser.add_argument(
        "--align-window",
        type=int,
        default=16,
        help="Ampiezza blocco augment (step da 30′) attorno a x(t+h-tau_i)",
    )
    parser.add_argument(
        "--max-lag-steps",
        type=int,
        default=LOOKBACK,
        help="Limite superiore per la stima del lag upstream (step da 30′)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=None,
        help="Horizon steps of 30′ (default 1 2 3 4 = 30..120′; e.g. 1..12 = 6 h)",
    )
    parser.add_argument(
        "--poly-points",
        type=int,
        default=POLY_NAIVE_POINTS,
        help=(
            f"Punti OLS per d1/d2 e baseline holdout poly2 "
            f"(default {POLY_NAIVE_POINTS} = {(POLY_NAIVE_POINTS - 1) * 0.5:.1f} h)"
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=("delta", "residual-lin3", "residual-poly2"),
        default="delta",
        help=(
            "delta=predict y(t+h)-y(t) (default); "
            "residual-lin3=predict y(t+h)-lin3(t+h) + lin3_delta_Cento; "
            "residual-poly2=predict y(t+h)-poly2(t+h) + poly2_delta_Cento "
            "(points from --poly-points, default 6)"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("alt", "block"),
        default="alt",
        help="Outer train/validate month split (alt=alternating, block=contiguous halves)",
    )
    parser.add_argument(
        "--selection-split",
        choices=("outer", "inner"),
        default="outer",
        help="Where early-stop / recipe ranking look: outer validate or nested inner validate (E2)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Progress + epoch/iter debug (default on)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Solo messaggi di fase (niente epoch/iter)",
    )
    args = parser.parse_args()

    global _VERBOSE
    _VERBOSE = bool(args.verbose) and not bool(args.quiet)
    poly_points = max(POLY_NAIVE_MIN, int(args.poly_points))
    ft.SPLIT_MODE = args.split
    log(f"split={args.split}  selection-split={args.selection_split}")

    if args.no_acceptors:
        acceptors = []
    elif args.acceptors is not None:
        acceptors = list(args.acceptors)
    else:
        acceptors = list(ACCEPTORS)
    ups = list(UPSTREAMS_NO_CHIUSA if args.no_chiusa else UPSTREAMS)
    if args.extra_upstreams:
        for name in args.extra_upstreams:
            if name not in ups and name != DOWNSTREAM:
                ups.append(name)
    station_list = [DOWNSTREAM, *ups, *acceptors]
    use_yellow = args.feature_set in ("yellow-headroom", "yellow-chain")
    use_chain = args.feature_set in ("chain", "yellow-chain")
    tag = (
        "novalle" if not acceptors else
        "valle-yellow-chain" if args.feature_set == "yellow-chain" else
        "valle-yellow" if args.feature_set == "yellow-headroom" else
        "valle-chain" if args.feature_set == "chain" else
        "valle"
    )
    if acceptors and acceptors != ACCEPTORS:
        tag = f"{tag}-" + "-".join(safe_name(a) for a in acceptors)
    if args.extra_upstreams:
        tag = f"{tag}-x" + "-".join(safe_name(u) for u in args.extra_upstreams)
    tag = f"{tag}-p{poly_points}"
    if args.no_chiusa:
        tag = f"{tag}-nochiusa"
    if args.chiusa_weir and not args.no_chiusa:
        tag = f"{tag}-weir"
    if args.flood_only:
        tag = f"{tag}-flood"
    if args.lag_align:
        tag = f"{tag}-lag{args.lag_align}"
    if args.lookback_tau:
        tag = f"{tag}-lbtau"
    if args.tau_frac:
        n_tf = max(1, int(args.tau_frac_n))
        phi_tf = float(args.tau_frac_max)
        sig_tf = "d" if (
            args.feature_signal == "delta" or args.tau_frac_signal == "delta"
        ) else ""
        tag = f"{tag}-taufrac{n_tf}-m{phi_tf:g}{sig_tf}"
    elif args.feature_signal == "delta":
        tag = f"{tag}-fdelta"
    if args.target_mode == "residual-lin3":
        tag = f"{tag}-rlin3"
    elif args.target_mode == "residual-poly2":
        tag = f"{tag}-rpoly2"
    if args.holdout_from:
        tag = f"{tag}-hf{args.holdout_from[:7].replace('-', '')}"
    if args.quantile is not None:
        tag = f"{tag}-q{int(round(args.quantile * 100))}"
    if args.event_weight != 1.0:
        tag = f"{tag}-ew{args.event_weight:g}"
    if args.rising_under_weight != 1.0:
        tag = f"{tag}-ruw{args.rising_under_weight:g}"
    if args.rising_only:
        tag = f"{tag}-rising"
    if args.export_dnn is None:
        args.export_dnn = ROOT / "models" / f"cento-dnn-{tag}.json"
    if args.out is None:
        args.out = ROOT / "models" / f"ml-cento-compare-{tag}.json"

    log_step(f"fit-ml-cento  tag={tag}  feature-set={args.feature_set}")
    log(f"out={args.out}")
    log(f"export-dnn={args.export_dnn}")
    log(f"skip-lgbm={args.skip_lgbm}  verbose={_VERBOSE}")
    log(f"target-mode={args.target_mode}")
    log(
        f"poly window={poly_points} pts "
        f"({(poly_points - 1) * 0.5:.1f} h)  degree={POLY_NAIVE_DEGREE}"
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stations = list(config["sensors"].keys())
    needed = [s for s in station_list if s in stations]
    missing = [s for s in station_list if s not in stations]
    if missing:
        raise SystemExit(f"Stazioni assenti da config: {missing}")
    log(f"stations ({len(needed)}): {needed}", level="debug")
    log(f"upstreams={ups}  flood-only={args.flood_only}")

    flood_events: list[dict] = []
    if args.flood_only:
        if not args.events.exists():
            raise SystemExit(f"--flood-only requires events catalog at {args.events}")
        flood_events = load_catalog_events(args.events, DOWNSTREAM)
        log(f"flood catalog events for {DOWNSTREAM}: {len(flood_events)}")

    soglia_levels = load_soglia_levels(args.soglie, prefer="yellow") if use_yellow else {}
    if use_yellow and not soglia_levels:
        raise SystemExit(f"Missing soglie in {args.soglie}; fetch ARPAE anagrafica first")
    if use_yellow:
        miss_soglia = [s for s in needed if s not in soglia_levels]
        if miss_soglia:
            raise SystemExit(f"No yellow/soglia for: {miss_soglia}")
        log(f"soglie yellow: { {k: soglia_levels[k] for k in needed if k in soglia_levels} }", level="debug")

    holdout_from = args.holdout_from
    if holdout_from:
        log(f"holdout-from={holdout_from} (calendar cut; train/val before cut)", level="ok")

    log_step("Caricamento storico")
    t_load = time.perf_counter()
    log(f"Carico stazioni da {args.data}…")
    wide_all = ft.load_wide(args.data, stations)
    wide_all, spike_counts = ft.filter_spikes(wide_all, spike_abs=args.spike_abs)
    spiked = {k: v for k, v in spike_counts.items() if v}
    log(
        f"Griglia {ft.STEP_MINUTES}′: {len(wide_all)} step in "
        f"{time.perf_counter() - t_load:.1f}s; spike: {spiked or 'nessuno'}",
        level="ok",
    )
    wide = wide_all[needed].copy()
    weir_crest_used = None
    if args.chiusa_weir and not args.no_chiusa and "Casalecchio chiusa" in wide.columns:
        # WP2.4 weir law proxy: Q ∝ max(h − h_crest, 0)^1.5 (c absorbed in MLP scale).
        series = wide["Casalecchio chiusa"]
        if args.weir_crest is not None:
            weir_crest_used = float(args.weir_crest)
        else:
            train_m0, _, _ = split_masks(
                wide.index, mode=args.split, holdout_from=holdout_from
            )
            base = series.to_numpy(dtype=float)[train_m0]
            base = base[np.isfinite(base)]
            weir_crest_used = float(np.percentile(base, 10)) if len(base) else 0.0
        q = np.maximum(series.to_numpy(dtype=float) - weir_crest_used, 0.0) ** 1.5
        wide["Casalecchio chiusa"] = q
        log(f"chiusa weir crest={weir_crest_used:.4f} → Q∝(h-crest)^1.5", level="ok")

    rain = None
    gauges = gauge_names(config)
    if not args.no_rain:
        log(f"Carico pioggia bacino ({len(gauges)} gauge)…")
        rain = ft.load_rain_basin(args.data / "rain", gauges, wide_all.index)
        if rain is not None:
            log(f"Pioggia: {int(rain['rain6h'].notna().sum())} timestep ok", level="ok")
        else:
            log("Pioggia: assente, fit solo livelli", level="warn")

    chain_pairs = (
        [(a, b) for a, b in CHAIN_PAIRS if a in needed and b in needed]
        if use_chain and acceptors else []
    )

    upstream_tau: dict[str, int] = {}
    station_lookback: dict[str, int] = {}
    station_lag_steps: dict[str, list[int]] = {}
    drop_level_all = args.feature_signal == "delta"
    drop_level_stations: set[str] = set()
    need_tau = bool(args.lag_align) or bool(args.lookback_tau) or bool(args.tau_frac)
    if args.tau_frac and args.lag_align:
        raise SystemExit("--tau-frac and --lag-align are mutually exclusive")
    if args.tau_frac and args.lookback_tau:
        raise SystemExit("--tau-frac and --lookback-tau are mutually exclusive")
    if need_tau:
        log_step("Stima ritardi monte→Cento (solo mesi di train)")
        train_wide, _, _ = split_masks(wide.index, mode=args.split, holdout_from=holdout_from)
        y_down = wide[DOWNSTREAM].to_numpy(dtype=float)
        # τ-fraction may need 2τ beyond the default max-lag-steps (=lookback).
        max_lag_est = int(args.max_lag_steps)
        if args.tau_frac:
            max_lag_est = max(
                max_lag_est,
                int(math.ceil(float(args.tau_frac_max) * int(args.lookback))),
            )
        for up in ups:
            x_up = wide[up].to_numpy(dtype=float)
            lag, r = ft.estimate_lag(
                x_up, y_down, max_lag_est, train_mask=train_wide
            )
            upstream_tau[up] = lag
            log(
                f"τ({up}→{DOWNSTREAM}) = {lag} step = {lag * ft.STEP_MINUTES}′ "
                f"(r={r if r is None else round(r, 3)})",
                level="ok",
            )

    if args.lookback_tau:
        buf_steps = max(0, int(round(float(args.lookback_tau_buffer_hours) * 60 / ft.STEP_MINUTES)))
        min_lb = max(POLY_NAIVE_MIN, poly_points)
        for st in needed:
            if st == DOWNSTREAM or st in acceptors:
                station_lookback[st] = int(args.lookback)
            else:
                tau = int(upstream_tau.get(st, args.lookback))
                station_lookback[st] = max(min_lb, min(int(args.lookback), tau + buf_steps))
        log(
            f"lookback-tau buffer={buf_steps} step "
            f"({args.lookback_tau_buffer_hours:g} h): "
            + ", ".join(f"{s}={station_lookback[s]}" for s in needed),
            level="ok",
        )

    if args.tau_frac:
        n_tf = max(1, int(args.tau_frac_n))
        phi_max = float(args.tau_frac_max)
        local_lb = max(POLY_NAIVE_MIN, int(args.tau_frac_local_lookback))
        if args.tau_frac_signal == "delta" and not drop_level_all:
            # Legacy: drop level only on τ-grid upstreams.
            drop_level_stations = set(ups)
        for st in needed:
            if st == DOWNSTREAM or st in acceptors:
                station_lookback[st] = min(local_lb, int(args.lookback))
            else:
                tau = int(upstream_tau.get(st, 0))
                lags = tau_frac_lag_steps(tau, n_tf, phi_max)
                station_lag_steps[st] = lags
        max_sparse = max((max(v) for v in station_lag_steps.values()), default=0)
        # Ensure --lookback spans the deepest τ-grid sample for chain / warmth.
        if max_sparse + 1 > int(args.lookback):
            args.lookback = max_sparse + 1
        sig_note = "delta-all" if drop_level_all else (
            "delta-ups" if drop_level_stations else "level"
        )
        log(
            f"tau-frac n={n_tf} φ_max={phi_max:g} signal={sig_note} "
            f"local_lb={local_lb}: "
            + ", ".join(
                f"{s}={station_lag_steps[s]}" if s in station_lag_steps
                else f"{s}=flat{station_lookback.get(s, args.lookback)}"
                for s in needed
            ),
            level="ok",
        )
    elif drop_level_all:
        log("feature-signal=delta: omit absolute level on all stations", level="ok")

    # Lag-align uses upstream_tau for shift/augment; lookback-tau / tau-frac alone must not.
    lag_align_tau = upstream_tau if args.lag_align else {}

    log_step("Feature matrix")
    t_feat = time.perf_counter()
    log(
        f"Building features: lookback={args.lookback} yellow={use_yellow} "
        f"ratio={use_yellow} chain={bool(chain_pairs)} d1/d2=poly{poly_points} "
        f"lag-align={bool(lag_align_tau)} lookback-tau={bool(station_lookback) and not station_lag_steps} "
        f"tau-frac={bool(station_lag_steps)}"
    )

    def make_feats(h_steps: int) -> tuple[pd.DataFrame, list[str]]:
        return build_feature_matrix(
            wide, rain, lookback=args.lookback, stations=needed,
            soglia_levels=soglia_levels,
            use_soglia_headroom=use_yellow,
            use_soglia_ratio=use_yellow,
            soglia_kind="yellow",
            chain_pairs=chain_pairs,
            poly_points=poly_points,
            upstream_tau=lag_align_tau,
            lag_align_h=h_steps,
            align_mode=args.lag_align or "shift",
            align_window=args.align_window,
            station_lookback=station_lookback or None,
            station_lag_steps=station_lag_steps or None,
            drop_level=drop_level_all,
            drop_level_stations=drop_level_stations or None,
        )

    feats, feature_cols = make_feats(0)
    use_rain = rain is not None and {"rain6h", "rain24h"}.issubset(rain.columns)
    n_d1 = sum(1 for c in feature_cols if c.startswith("d1_"))
    n_d2 = sum(1 for c in feature_cols if c.startswith("d2_"))
    log(
        f"Feature: {len(feature_cols)} cols in {time.perf_counter() - t_feat:.1f}s "
        f"(d1={n_d1} d2={n_d2} rain={use_rain})",
        level="ok",
    )
    log(f"stations={needed}", level="debug")

    horizons = [int(h) for h in (args.horizons if args.horizons else HORIZONS)]
    if not horizons or any(h < 1 for h in horizons):
        raise SystemExit("--horizons must be positive 30′-steps (e.g. 1 2 3 4)")
    log(f"horizons steps={horizons} → {[h * ft.STEP_MINUTES for h in horizons]}′")

    horizons_out: dict[str, dict] = {}
    export_horizons: dict[str, dict] = {}
    export_feature_cols = list(feature_cols)
    n_h = len(horizons)

    for hi, h in enumerate(horizons, start=1):
        minutes = h * ft.STEP_MINUTES
        log_step(f"Orizzonte {minutes}′  ({hi}/{n_h})")
        t_h = time.perf_counter()
        if lag_align_tau:
            feats, feature_cols = make_feats(h)
        targets = build_targets(wide, h)
        X, y, y0, index = align_xy(feats, feature_cols, targets)
        feat_cols_h = list(feature_cols)
        y_delta = y
        baseline_delta = None
        if args.target_mode == "residual-lin3":
            base = build_lin3_baseline(
                wide[DOWNSTREAM], index, h, n_points=LIN3_POINTS
            )
            baseline_delta = base - y0
            # Train residual r = y(t+h) − baseline; metrics use absolute Δ.
            y = y_delta - baseline_delta
            X = X.copy()
            X["lin3_delta_Cento"] = baseline_delta
            feat_cols_h.append("lin3_delta_Cento")
        elif args.target_mode == "residual-poly2":
            base = build_poly2_baseline(
                wide[DOWNSTREAM], index, h, n_points=poly_points
            )
            baseline_delta = base - y0
            y = y_delta - baseline_delta
            X = X.copy()
            X["poly2_delta_Cento"] = baseline_delta
            feat_cols_h.append("poly2_delta_Cento")
        train_m, val_m, hold_m = split_masks(
            index, mode=args.split, holdout_from=holdout_from
        )
        if args.rising_only:
            rising_m = np.asarray(y_delta, dtype=float) > 0
            n_before = int(train_m.sum())
            train_m = train_m & rising_m
            log(
                f"rising-only train: {int(train_m.sum())}/{n_before} "
                f"({100.0 * int(train_m.sum()) / max(1, n_before):.1f}% kept)"
            )
        if args.flood_only and flood_events:
            flood_m = flood_window_mask(index, flood_events)
            # Calm out of train loss; keep val/hold full for FAR / calm checks in eval.
            n_before = int(train_m.sum())
            train_m = train_m & flood_m
            log(
                f"flood-only train: {int(train_m.sum())}/{n_before} "
                f"(flood rows={int(flood_m.sum())}/{len(flood_m)})"
            )
        if train_m.sum() < WEAK_POINTS or val_m.sum() < WEAK_POINTS:
            log(f"skip: train={train_m.sum()} val={val_m.sum()}", level="warn")
            continue

        fit_train, fit_val = train_m, val_m
        if args.selection_split == "inner":
            fit_train, fit_val = inner_split_masks(index, train_m, mode="alt")
            if fit_train.sum() < WEAK_POINTS or fit_val.sum() < WEAK_POINTS:
                log(
                    f"skip inner: train={fit_train.sum()} val={fit_val.sum()}",
                    level="warn",
                )
                continue

        log("Calcolo baseline holdout poly2…")
        t_poly = time.perf_counter()
        poly_base = build_poly2_baseline(
            wide[DOWNSTREAM], index, h, n_points=poly_points
        )
        n_poly_ok = int(np.isfinite(poly_base).sum())
        log(
            f"poly2 baseline: {n_poly_ok}/{len(poly_base)} finite "
            f"in {time.perf_counter() - t_poly:.1f}s "
            f"(n={poly_points})",
            level="ok",
        )
        log(
            f"samples n={len(y)} train={int(train_m.sum())} "
            f"val={int(val_m.sum())} hold={int(hold_m.sum())} "
            f"fit_train={int(fit_train.sum())} fit_val={int(fit_val.sum())} "
            f"events_val={int(event_mask(y, val_m).sum())} "
            f"events_hold={int(event_mask(y, hold_m).sum())}"
        )

        lgbm_metrics = None
        best_iteration = 0
        if not args.skip_lgbm:
            booster, lgbm_pred = train_lgbm(X[feat_cols_h], y, fit_train, fit_val)
            lgbm_pred_m = (
                lgbm_pred + baseline_delta if baseline_delta is not None else lgbm_pred
            )
            lgbm_metrics = eval_splits(
                y0, y_delta, lgbm_pred_m, train_m, val_m, hold_m, poly_baseline=poly_base
            )
            best_iteration = int(booster.best_iteration or 0)
            he_l = lgbm_metrics["holdoutEvents"]
            log(
                f"LightGBM summary: val={lgbm_metrics['validate']['skill']} "
                f"valEvt={lgbm_metrics['validateEvents']['skill']} "
                f"holdEvt/poly2={he_l.get('skill')} "
                f"(persist {he_l.get('skillPersist')}) iter={best_iteration}",
                level="ok",
            )
        else:
            log("skip LightGBM (--skip-lgbm)")

        log("Standardize features on train…")
        X_std, stats = standardize_features(X, feat_cols_h, train_m)
        model, dnn_pred, dnn_hist = train_dnn(
            X_std,
            y,
            fit_train,
            fit_val,
            quantile=args.quantile,
            event_weight=args.event_weight,
            rising_under_weight=args.rising_under_weight,
        )
        dnn_pred_m = (
            dnn_pred + baseline_delta if baseline_delta is not None else dnn_pred
        )
        log("Eval splits (train/val/hold + events)…")
        dnn_metrics = eval_splits(
            y0, y_delta, dnn_pred_m, train_m, val_m, hold_m, poly_baseline=poly_base
        )
        spreads = pack_spreads(y0, y_delta, dnn_pred_m, val_m)
        he = dnn_metrics["holdoutEvents"]
        log(
            f"DNN summary +{minutes}′: val={dnn_metrics['validate']['skill']} "
            f"valEvt={dnn_metrics['validateEvents']['skill']} "
            f"holdEvt/poly2={he.get('skill')} "
            f"(persist {he.get('skillPersist')}) "
            f"best_epoch={dnn_hist['best_epoch']}  "
            f"wall={time.perf_counter() - t_h:.1f}s",
            level="ok",
        )

        key = str(h * ft.STEP_MINUTES)
        export_horizons[key] = {
            "mean": [round(stats[c][0], 8) for c in feat_cols_h],
            "std": [round(stats[c][1], 8) for c in feat_cols_h],
            "layers": export_mlp_layers(model),
            **spreads,
            "train": dnn_metrics["train"],
            "validate": dnn_metrics["validate"],
            "validateEvents": dnn_metrics["validateEvents"],
            "holdout": dnn_metrics["holdout"],
            "holdoutEvents": dnn_metrics["holdoutEvents"],
            "earlyStopping": dnn_hist,
        }

        ridge = ridge_side_by_side(args.coeffs, h * ft.STEP_MINUTES)
        block = {
            "horizonMinutes": h * ft.STEP_MINUTES,
            "nFeatures": len(feat_cols_h),
            "nSamples": {
                "train": int(train_m.sum()),
                "validate": int(val_m.sum()),
                "holdout": int(hold_m.sum()),
            },
            "dnn": {
                **dnn_metrics,
                "architecture": list(DNN_HIDDEN) + [1],
                "earlyStopping": dnn_hist,
            },
            "ridge": ridge,
        }
        if lgbm_metrics is not None:
            block["lightgbm"] = {**lgbm_metrics, "bestIteration": best_iteration}
        horizons_out[key] = block
        export_feature_cols = feat_cols_h

    skill120 = (export_horizons.get("120") or {}).get("validateEvents", {}).get("skill")
    reliable = skill120 is not None and skill120 >= ft.SKILL_KEEP
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    feature_recipe = {
        "featureSet": args.feature_set,
        "useSogliaHeadroom": use_yellow,
        "useSogliaRatio": use_yellow,
        "usePolyDerivatives": True,
        "polyDerivPoints": poly_points,
        "polyDerivDegree": POLY_NAIVE_DEGREE,
        "sogliaKind": "yellow" if use_yellow else None,
        "sogliaLevels": {k: soglia_levels[k] for k in needed if k in soglia_levels} if use_yellow else {},
        "chainPairs": [list(p) for p in chain_pairs],
        "upstreamLagAlign": bool(lag_align_tau),
        "upstreamLagMode": args.lag_align if lag_align_tau else None,
        "upstreamLagAlignWindow": args.align_window if lag_align_tau else None,
        "upstreamLagMinutes": {
            up: upstream_tau[up] * ft.STEP_MINUTES for up in upstream_tau
        },
        "lookbackTau": bool(args.lookback_tau),
        "lookbackTauBufferHours": (
            float(args.lookback_tau_buffer_hours) if args.lookback_tau else None
        ),
        "stationLookbackSteps": dict(station_lookback) if station_lookback else {},
        "tauFrac": bool(station_lag_steps),
        "tauFracN": int(args.tau_frac_n) if station_lag_steps else None,
        "tauFracMax": float(args.tau_frac_max) if station_lag_steps else None,
        "tauFracSignal": (
            ("delta" if (drop_level_all or drop_level_stations) else "level")
            if station_lag_steps else None
        ),
        "tauFracLocalLookback": (
            int(args.tau_frac_local_lookback) if station_lag_steps else None
        ),
        "stationLagSteps": (
            {k: list(v) for k, v in station_lag_steps.items()} if station_lag_steps else {}
        ),
        "featureSignal": args.feature_signal,
        "dropLevelStations": sorted(drop_level_stations) if drop_level_stations else [],
        "holdoutFrom": holdout_from,
        "quantile": args.quantile,
        "eventWeight": args.event_weight,
        "risingUnderWeight": args.rising_under_weight,
        "chiusaWeir": bool(args.chiusa_weir) and not bool(args.no_chiusa),
        "weirCrest": weir_crest_used,
        "targetMode": (
            "residual_lin3" if args.target_mode == "residual-lin3"
            else "residual_poly2" if args.target_mode == "residual-poly2"
            else "delta"
        ),
        "lin3Points": LIN3_POINTS if args.target_mode == "residual-lin3" else None,
        "poly2Points": poly_points if args.target_mode == "residual-poly2" else None,
    }
    dnn_node = {
        "kind": "window_delta_mlp",
        "downstream": DOWNSTREAM,
        "upstreams": ups,
        "acceptors": acceptors,
        "stations": needed,
        "lookbackSteps": args.lookback,
        "stepMinutes": ft.STEP_MINUTES,
        "featureNames": export_feature_cols,
        "featureRecipe": feature_recipe,
        "useRain": use_rain,
        "rainGauges": gauges if use_rain else [],
        "horizons": export_horizons,
        "skill120": skill120,
        "reliable": reliable,
        "trainedAt": trained_at,
        "quantile": args.quantile,
        "risingUnderWeight": args.rising_under_weight,
        "risingOnly": bool(args.rising_only),
        "floodOnly": bool(args.flood_only),
        "thresholdOn": "event_or_move",
        "holdoutBaseline": "poly2",
        "note": (
            (
                f"featureSet={args.feature_set}; acceptors valle; "
                "headroom = soglia_gialla − livello; "
                "ratio = livello / soglia_gialla; "
                "chain = livello_monte_catena − livello_valle; "
                "d1/d2 = derivate 1ª/2ª da OLS quadratico locale (3–4 punti); "
                "holdout skill vs poly2."
                if acceptors else
                "Ablation without valle acceptors; d1/d2 da OLS quadratico; holdout vs poly2."
            )
            + (
                f" Rising-under-weight={args.rising_under_weight:g} on Δy>0."
                if args.rising_under_weight != 1.0
                else ""
            )
        ),
    }

    compare = {
        "version": 1,
        "kind": "cento_lgbm_vs_dnn",
        "downstream": DOWNSTREAM,
        "upstreams": ups,
        "acceptors": acceptors,
        "stations": needed,
        "featureSet": args.feature_set,
        "featureRecipe": feature_recipe,
        "lookbackSteps": args.lookback,
        "lookbackMinutes": args.lookback * ft.STEP_MINUTES,
        "stepMinutes": ft.STEP_MINUTES,
        "horizonsMinutes": [h * ft.STEP_MINUTES for h in horizons],
        "features": export_feature_cols,
        "useRain": use_rain,
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
        "holdoutBaseline": {
            "kind": "poly2",
            "points": poly_points,
            "minPoints": POLY_NAIVE_MIN,
            "degree": POLY_NAIVE_DEGREE,
            "note": "Holdout skill = 1 − RMSE_model / RMSE_poly2; train/validate restano vs persistenza.",
        },
        "polyPoints": poly_points,
        "trainedAt": trained_at,
        "horizons": horizons_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(compare, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log_step("Export")
    log(f"Wrote {args.out}", level="ok")

    if args.export_dnn:
        args.export_dnn.parent.mkdir(parents=True, exist_ok=True)
        args.export_dnn.write_text(
            json.dumps({"version": 1, "downstream": DOWNSTREAM, **dnn_node}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log(f"Wrote {args.export_dnn}", level="ok")

    if args.export_coeffs and args.export_coeffs.exists():
        coeffs = json.loads(args.export_coeffs.read_text(encoding="utf-8"))
        coeffs.setdefault("nodes", {})[DOWNSTREAM] = dnn_node
        # Keep decorative edges; mark kind for Cento edges.
        for up in UPSTREAMS[:2]:  # direct deps stay as graph edges
            key = f"{up}->{DOWNSTREAM}"
            if key in (coeffs.get("edges") or {}):
                coeffs["edges"][key]["kind"] = "window_delta_mlp"
                coeffs["edges"][key]["reliable"] = reliable
                coeffs["edges"][key]["skill120"] = skill120
                coeffs["edges"][key]["trainedAt"] = trained_at
        args.export_coeffs.write_text(json.dumps(coeffs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log(
            f"Patched {args.export_coeffs} nodes.{DOWNSTREAM} "
            f"[{('OK' if reliable else 'no-skill')}] skill120={skill120}",
            level="ok",
        )
        # Browser loads the compact runtime pack (no indent / train blobs).
        pack = ROOT / "tools" / "pack-runtime-coeffs.js"
        if pack.is_file():
            subprocess.run(["node", str(pack), "--in", str(args.export_coeffs)], check=False)
    elif args.export_coeffs:
        log(f"Skip coeffs patch: {args.export_coeffs} missing", level="warn")

    log(f"Done. skill120(valEvt)={skill120} reliable={reliable} tag={tag}", level="ok")


if __name__ == "__main__":
    main()

