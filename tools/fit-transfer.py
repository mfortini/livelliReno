#!/usr/bin/env python3
"""Train direct-horizon delta ridge transfers (experimental).

For each downstream and each horizon h ∈ HORIZONS (30-min steps):
  Δy(t→t+h) ≈ b + w_local·Δy(t-1→t)
             + Σ_i w_i^win · (x_i(t-τ_i) − x_i(t-τ_i−h))
             + Σ_i w_i^rate · (x_i(t-τ_i) − x_i(t-τ_i−1))
             + Σ_i w_i^win2 · (x_i(t-τ_i−h) − x_i(t-τ_i−2h))
             + Σ_i w_i^rate2 · (x_i(t-τ_i−h) − x_i(t-τ_i−h−1))
             + w_r6·rain6h + w_r24·rain24h   (optional)

Forecast: ŷ(t+h) = y(t) + Δŷ. Persistence is the zero-weight case.

Validation: alternating monthly blocks on non-holdout months; last 3 months
held out. Lags estimated on train months only. Spikes filtered before fit.
Writes models/transfer-coeffs.json for the static dashboard.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def progress(msg: str) -> None:
    """Timestamped progress line (always flushed for tee/pipes)."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

STEP_MINUTES = 30
HORIZONS = [1, 2, 3, 4]  # steps of 30 min (always)
HORIZONS_EXTENDED = [5, 6, 7, 8, 9, 10, 11, 12]  # 150′–360′ when max lag ≥ 360 min
EXTEND_LAG_MIN = 360
WEAK_POINTS = 48
RIDGE_LAMBDA = 1e-2
HIGH_QUANTILE = 0.85
SPIKE_ABS = 3.0  # m in one 30-min step → drop
SKILL_KEEP = 0.10  # min event skill@120 to mark reliable
HOLDOUT_MONTHS = 3
HOLDOUT_FROM: str | None = None  # overridden by --holdout-from
RAIN6_STEPS = 12  # 6 h
RAIN24_STEPS = 48  # 24 h
SPLIT_MODE = "alt"  # "alt" | "block"; overridden by --split


def _station_series_path(data_dir: Path, name: str) -> Path | None:
    """Prefer Parquet (compact); fall back to CSV for JS-era caches."""
    parquet = data_dir / f"{name}.parquet"
    if parquet.exists():
        return parquet
    csv_path = data_dir / f"{name}.csv"
    if csv_path.exists():
        return csv_path
    return None


def load_wide(data_dir: Path, stations: list[str]) -> pd.DataFrame:
    con = duckdb.connect()
    frames: list[pd.DataFrame] = []
    for name in stations:
        path = _station_series_path(data_dir, name)
        if path is None:
            continue
        reader = "read_parquet(?)" if path.suffix == ".parquet" else "read_csv_auto(?, header=true)"
        df = con.execute(
            f"""
            SELECT
              ?::VARCHAR AS station,
              CAST(time AS TIMESTAMPTZ) AS ts,
              CAST(value AS DOUBLE) AS value
            FROM {reader}
            WHERE value IS NOT NULL AND isfinite(value)
            """,
            [name, str(path)],
        ).df()
        frames.append(df)
    if not frames:
        raise SystemExit(f"Nessun CSV/Parquet trovato in {data_dir}")

    all_long = pd.concat(frames, ignore_index=True)
    con.register("levels_long", all_long)
    wide = con.execute(
        f"""
        WITH snapped AS (
          SELECT
            station,
            time_bucket(INTERVAL '{STEP_MINUTES} minutes', ts) AS bucket,
            avg(value) AS value
          FROM levels_long
          GROUP BY 1, 2
        )
        SELECT *
        FROM snapped
        PIVOT (avg(value) FOR station IN ({", ".join(repr(s) for s in stations)}))
        ORDER BY bucket
        """
    ).df()
    return wide.rename(columns={"bucket": "ts"}).set_index("ts").sort_index().dropna(how="all")


def rain_gauge_frame(rain_dir: Path, name: str) -> pd.DataFrame | None:
    """One gauge `time,value` frame; Parquet preferred, CSV fallback."""
    parquet = rain_dir / f"{name}.parquet"
    csv_path = rain_dir / f"{name}.csv"
    path = parquet if parquet.exists() else (csv_path if csv_path.exists() else None)
    if path is None:
        return None
    if path.suffix == ".parquet":
        df = duckdb.connect().execute(
            "SELECT CAST(time AS TIMESTAMPTZ) AS time, CAST(value AS DOUBLE) AS value "
            "FROM read_parquet(?) WHERE value IS NOT NULL",
            [str(path)],
        ).df()
    else:
        df = pd.read_csv(path, parse_dates=["time"])
    return df if not df.empty else None


def load_rain_basin(rain_dir: Path, gauges: list[str], index: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Basin max of 6h/24h cumulated rainfall on the level grid."""
    if not gauges or not rain_dir.is_dir():
        return None
    series: list[pd.Series] = []
    for name in gauges:
        df = rain_gauge_frame(rain_dir, name)
        if df is None:
            continue
        s = df.set_index("time")["value"].astype(float).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        # snap to 30-min grid via resample mean
        s = s.resample(f"{STEP_MINUTES}min").sum()
        series.append(s.rename(name))
    if not series:
        return None
    rain = pd.concat(series, axis=1).reindex(index)
    # treat missing as 0 for accumulation (dry default); keep all-NaN rows as NaN
    any_obs = rain.notna().any(axis=1)
    rain = rain.fillna(0.0)
    basin = rain.max(axis=1)
    out = pd.DataFrame(index=index)
    out["rain6h"] = basin.rolling(RAIN6_STEPS, min_periods=1).sum()
    out["rain24h"] = basin.rolling(RAIN24_STEPS, min_periods=1).sum()
    out.loc[~any_obs, :] = np.nan
    return out


def filter_spikes(wide: pd.DataFrame, spike_abs: float = SPIKE_ABS) -> tuple[pd.DataFrame, dict[str, int]]:
    """Null out single-step jumps larger than spike_abs (sensor glitches / gate ops)."""
    out = wide.copy()
    counts: dict[str, int] = {}
    for col in out.columns:
        s = out[col]
        d = s.diff().abs()
        bad = d > spike_abs
        n = int(bad.sum())
        counts[col] = n
        if n:
            idx = s.index[bad.fillna(False)]
            out.loc[idx, col] = np.nan
            for t in idx:
                pos = s.index.get_loc(t)
                if isinstance(pos, (slice, np.ndarray)):
                    continue
                if pos + 1 < len(s.index):
                    out.iat[pos + 1, out.columns.get_loc(col)] = np.nan
    return out, counts


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if den == 0:
        return None
    return float(np.dot(x, y) / den)


def estimate_lag(
    x: np.ndarray,
    y: np.ndarray,
    max_lag_steps: int,
    train_mask: np.ndarray | None = None,
) -> tuple[int, float | None]:
    """Cross-correlation lag; optional train_mask restricts pairs to train months."""
    best_lag, best_r = 0, None
    n = len(x)
    limit = max(0, min(max_lag_steps, n // 3))
    for lag in range(limit + 1):
        xs = x[: n - lag]
        ys = y[lag:]
        if train_mask is not None:
            m = train_mask[: n - lag] & train_mask[lag:] & np.isfinite(xs) & np.isfinite(ys)
        else:
            m = np.isfinite(xs) & np.isfinite(ys)
        if m.sum() < WEAK_POINTS:
            continue
        r = pearson(xs[m], ys[m])
        if r is None:
            continue
        if best_r is None or abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_lag, best_r


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> np.ndarray | None:
    if X.ndim != 2 or len(y) < X.shape[1] + 2:
        return None
    ones = np.ones((len(y), 1))
    A = np.hstack([ones, X])
    ata = A.T @ A
    diag = np.arange(ata.shape[0])
    ata[diag, diag] += lam
    ata[0, 0] -= lam
    try:
        return np.linalg.solve(ata, A.T @ y)
    except np.linalg.LinAlgError:
        return None


def apply_beta(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return beta[0] + X @ beta[1:]


def rmse(obs: np.ndarray, pred: np.ndarray) -> float | None:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if not mask.any():
        return None
    err = pred[mask] - obs[mask]
    return float(math.sqrt(np.mean(err * err)))


def residual_spreads(
    obs: np.ndarray,
    pred: np.ndarray,
    q: float = 0.85,
) -> tuple[float | None, float | None]:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 20:
        err = rmse(obs, pred)
        return err, err
    resid = pred[mask] - obs[mask]
    low = float(np.quantile(np.maximum(0.0, -resid), q))
    high = float(np.quantile(np.maximum(0.0, resid), q))
    floor = rmse(obs, pred) or 0.05
    return max(low, floor * 0.5), max(high, floor * 0.5)


def month_blocks(
    index: pd.DatetimeIndex,
    mode: str | None = None,
    holdout_from: str | pd.Timestamp | None = None,
) -> np.ndarray:
    """0 = train, 1 = validate, 2 = final holdout.

    Holdout is the last ``HOLDOUT_MONTHS`` calendar months, or every month whose
    start is ≥ ``holdout_from`` when that cut is set (leakage-free calendar split).

    mode:
      - ``alt``: alternating months on the non-holdout span (legacy)
      - ``block``: first half of non-holdout months → train, second half → validate
    Defaults to module ``SPLIT_MODE`` (set by CLI ``--split``).
    """
    mode = mode or SPLIT_MODE
    holdout_from = holdout_from if holdout_from is not None else HOLDOUT_FROM
    naive = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
    months = naive.to_period("M")
    uniq = months.unique().sort_values()
    if holdout_from is not None:
        cut = pd.Timestamp(holdout_from)
        if cut.tzinfo is not None:
            cut = cut.tz_convert("UTC").tz_localize(None)
        holdout = {m for m in uniq if m.to_timestamp() >= cut}
    else:
        holdout = set(uniq[-HOLDOUT_MONTHS:]) if len(uniq) > HOLDOUT_MONTHS else set()
    remaining = [m for m in uniq if m not in holdout]
    mapping: dict = {}
    if mode == "block":
        mid = max(1, len(remaining) // 2) if remaining else 0
        for i, m in enumerate(remaining):
            mapping[m] = 0 if i < mid else 1
    else:
        for i, m in enumerate(remaining):
            mapping[m] = i % 2
    for m in holdout:
        mapping[m] = 2
    return np.array([mapping[m] for m in months], dtype=int)


def standardize_train(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    base = x[train_mask]
    base = base[np.isfinite(base)]
    mu = float(base.mean()) if len(base) else 0.0
    sd = float(base.std()) if len(base) > 1 else 1.0
    if sd < 1e-6:
        sd = 1.0
    return (x - mu) / sd, mu, sd


def high_regime_mask(
    frame: pd.DataFrame,
    upstreams: list[str],
    thresholds: dict[str, dict],
) -> np.ndarray:
    """True when any lagged upstream z-score exceeds its train z85."""
    high = np.zeros(len(frame), dtype=bool)
    for up in upstreams:
        thr = thresholds.get(up)
        if not thr or f"x_{up}" not in frame.columns:
            continue
        x = frame[f"x_{up}"].to_numpy(dtype=float)
        z = (x - thr["mean"]) / thr["std"]
        high |= np.isfinite(z) & (z > thr["z85"])
    return high


def build_horizon_design(
    wide: pd.DataFrame,
    downstream: str,
    upstreams: list[str],
    lags: list[int],
    h: int,
    rain: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    y = wide[downstream]
    parts = {
        "y0": y,
        "y_future": y.shift(-h),
        "dy_local": y - y.shift(1),
    }
    for name, lag in zip(upstreams, lags):
        x = wide[name]
        xl = x.shift(lag)
        xl2 = x.shift(lag + h)
        parts[f"x_{name}"] = xl
        parts[f"dxw_{name}"] = xl - xl.shift(h)
        parts[f"dxr_{name}"] = xl - xl.shift(1)
        parts[f"dxw2_{name}"] = xl2 - xl2.shift(h)
        parts[f"dxr2_{name}"] = xl2 - xl2.shift(1)
    if rain is not None:
        parts["rain6h"] = rain["rain6h"]
        parts["rain24h"] = rain["rain24h"]
    frame = pd.DataFrame(parts, index=wide.index)
    frame["delta"] = frame["y_future"] - frame["y0"]
    frame = frame.dropna()
    if len(frame) < WEAK_POINTS:
        return None
    return frame


def fit_horizon(
    frame: pd.DataFrame,
    upstreams: list[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    holdout_mask: np.ndarray,
    thresholds: dict[str, dict],
    use_rain: bool,
) -> dict | None:
    feature_cols = (
        ["dy_local"]
        + [f"dxw_{u}" for u in upstreams]
        + [f"dxr_{u}" for u in upstreams]
        + [f"dxw2_{u}" for u in upstreams]
        + [f"dxr2_{u}" for u in upstreams]
    )
    if use_rain:
        feature_cols += ["rain6h", "rain24h"]
    X = frame[feature_cols].to_numpy(dtype=float)
    y = frame["delta"].to_numpy(dtype=float)
    y0 = frame["y0"].to_numpy(dtype=float)

    if train_mask.sum() < WEAK_POINTS:
        return None
    beta = ridge_fit(X[train_mask], y[train_mask])
    if beta is None:
        return None

    high = high_regime_mask(frame, upstreams, thresholds)

    def metrics(mask: np.ndarray, regime_spreads: bool = False) -> dict:
        empty = {
            "rmse": None,
            "skill": None,
            "mae": None,
            "spreadLow": None,
            "spreadHigh": None,
            "n": int(mask.sum()),
            "coverage": None,
        }
        if mask.sum() < 8:
            return empty
        pred_delta = apply_beta(X[mask], beta)
        pred_level = y0[mask] + pred_delta
        obs_level = y0[mask] + y[mask]
        persist = y0[mask]
        err_m = rmse(obs_level, pred_level)
        err_p = rmse(obs_level, persist)
        skill = None if err_m is None or err_p is None or err_p == 0 else 1 - err_m / err_p
        lo, hi = residual_spreads(obs_level, pred_level)
        cover = None
        if lo is not None and hi is not None:
            if regime_spreads:
                sub_high = high[mask]
                lo_b, hi_b = residual_spreads(obs_level[~sub_high], pred_level[~sub_high])
                lo_h, hi_h = residual_spreads(obs_level[sub_high], pred_level[sub_high])
                lo_b = lo if lo_b is None else lo_b
                hi_b = hi if hi_b is None else hi_b
                lo_h = lo if lo_h is None else lo_h
                hi_h = hi if hi_h is None else hi_h
                lo_use = np.where(sub_high, lo_h, lo_b)
                hi_use = np.where(sub_high, hi_h, hi_b)
                inside = (obs_level >= pred_level - lo_use) & (obs_level <= pred_level + hi_use)
            else:
                inside = (obs_level >= pred_level - lo) & (obs_level <= pred_level + hi)
            cover = float(inside.mean())
        mae = float(np.mean(np.abs(obs_level - pred_level)))
        return {
            "rmse": None if err_m is None else round(err_m, 6),
            "skill": None if skill is None else round(skill, 6),
            "mae": round(mae, 6),
            "spreadLow": None if lo is None else round(lo, 6),
            "spreadHigh": None if hi is None else round(hi, 6),
            "n": int(mask.sum()),
            "coverage": None if cover is None else round(cover, 6),
        }

    # Regime-specific spreads on validation.
    # Base: calm (not high). High: high-regime OR events, higher quantile so
    # event coverage can reach ~75% (plan success criterion).
    val_obs = y0[val_mask] + y[val_mask]
    val_pred = y0[val_mask] + apply_beta(X[val_mask], beta)
    val_high = high[val_mask]
    val_event = np.abs(y[val_mask]) >= 0.10
    lo_all, hi_all = residual_spreads(val_obs, val_pred)
    lo_base, hi_base = residual_spreads(val_obs[~val_high], val_pred[~val_high], q=0.85)
    elevated = val_high | val_event
    lo_high, hi_high = residual_spreads(val_obs[elevated], val_pred[elevated], q=0.98)
    lo_evt, hi_evt = residual_spreads(val_obs[val_event], val_pred[val_event], q=0.95)
    err_evt = rmse(val_obs[val_event], val_pred[val_event])
    if lo_base is None:
        lo_base, hi_base = lo_all, hi_all
    if lo_high is None:
        lo_high, hi_high = lo_all, hi_all
    # High fork: at least event residuals, never narrower than base; +25% for runtime gap
    for candidate in (lo_evt, err_evt, lo_base):
        if candidate is not None and lo_high is not None:
            lo_high = max(lo_high, candidate)
    for candidate in (hi_evt, err_evt, hi_base):
        if candidate is not None and hi_high is not None:
            hi_high = max(hi_high, candidate)
    if lo_high is not None:
        lo_high *= 1.25
        hi_high *= 1.25

    event_val = val_mask & (np.abs(y) >= 0.10)
    event_hold = holdout_mask & (np.abs(y) >= 0.10)
    k = len(upstreams)
    packed = {
        "b": round(float(beta[0]), 6),
        "wLocal": round(float(beta[1]), 6),
        "wWindow": [round(float(v), 6) for v in beta[2 : 2 + k]],
        "wRate": [round(float(v), 6) for v in beta[2 + k : 2 + 2 * k]],
        "wWindow2": [round(float(v), 6) for v in beta[2 + 2 * k : 2 + 3 * k]],
        "wRate2": [round(float(v), 6) for v in beta[2 + 3 * k : 2 + 4 * k]],
        "features": feature_cols,
        "spreadLow": None if lo_base is None else round(lo_base, 6),
        "spreadHigh": None if hi_base is None else round(hi_base, 6),
        "spreadLowHigh": None if lo_high is None else round(lo_high, 6),
        "spreadHighHigh": None if hi_high is None else round(hi_high, 6),
        "train": metrics(train_mask),
        "validate": metrics(val_mask, regime_spreads=True),
        "validateEvents": metrics(event_val, regime_spreads=True),
        "holdout": metrics(holdout_mask, regime_spreads=True),
        "holdoutEvents": metrics(event_hold, regime_spreads=True),
    }
    if use_rain:
        packed["wRain6"] = round(float(beta[2 + 4 * k]), 6)
        packed["wRain24"] = round(float(beta[2 + 4 * k + 1]), 6)
    return packed


def fit_node(
    wide: pd.DataFrame,
    downstream: str,
    upstreams: list[str],
    max_lag_hours: float,
    rain: pd.DataFrame | None = None,
) -> dict | None:
    t_node = time.time()
    present = [u for u in upstreams if u in wide.columns and downstream in wide.columns]
    if not present or downstream not in wide.columns:
        return None

    progress(f"  {downstream}: stima lag su {present}…")
    blocks = month_blocks(wide.index)
    train_rows = blocks == 0

    max_lag_steps = int(max_lag_hours * 60 / STEP_MINUTES)
    y_all = wide[downstream].to_numpy(dtype=float)
    lags: list[int] = []
    lag_rs: list[float | None] = []
    for up in present:
        x_all = wide[up].to_numpy(dtype=float)
        mask = np.isfinite(x_all) & np.isfinite(y_all)
        if (mask & train_rows).sum() < WEAK_POINTS:
            lags.append(0)
            lag_rs.append(None)
            progress(f"    lag {up}: skip (campioni deboli) → 0′")
            continue
        x_s = pd.Series(x_all).interpolate(limit=2).to_numpy()
        y_s = pd.Series(y_all).interpolate(limit=2).to_numpy()
        lag, r = estimate_lag(x_s, y_s, max_lag_steps, train_mask=train_rows)
        lags.append(lag)
        lag_rs.append(r)
        progress(
            f"    lag {up}: {lag * STEP_MINUTES}′ "
            f"(r={None if r is None else round(float(r), 3)})"
        )

    thresholds: dict[str, dict] = {}
    for up, lag in zip(present, lags):
        xl = wide[up].shift(lag).to_numpy(dtype=float)
        z, mu, sd = standardize_train(xl, train_rows & np.isfinite(xl))
        thr = float(np.nanquantile(z[train_rows], HIGH_QUANTILE))
        thresholds[up] = {"mean": round(mu, 6), "std": round(sd, 6), "z85": round(thr, 6)}

    use_rain = rain is not None and {"rain6h", "rain24h"}.issubset(rain.columns)
    lag_min = max((lag * STEP_MINUTES for lag in lags), default=0)
    horizon_steps = list(HORIZONS)
    if lag_min >= EXTEND_LAG_MIN:
        horizon_steps = horizon_steps + list(HORIZONS_EXTENDED)
    progress(
        f"  {downstream}: fit {len(horizon_steps)} orizzonti "
        f"(maxLag={lag_min}′, rain={'sì' if use_rain else 'no'})…"
    )

    horizons: dict[str, dict] = {}
    for i_h, h in enumerate(horizon_steps, 1):
        t_h = time.time()
        mins = h * STEP_MINUTES
        frame = build_horizon_design(wide, downstream, present, lags, h, rain=rain if use_rain else None)
        if frame is None:
            progress(f"    [{i_h}/{len(horizon_steps)}] +{mins}′: skip (design vuoto)")
            continue
        fb = month_blocks(frame.index)
        packed = fit_horizon(
            frame,
            present,
            fb == 0,
            fb == 1,
            fb == 2,
            thresholds,
            use_rain=use_rain,
        )
        if packed is None:
            progress(f"    [{i_h}/{len(horizon_steps)}] +{mins}′: skip (fit fallito)")
            continue
        key = str(mins)
        # Extended horizons: keep only if event skill clears the threshold
        if h in HORIZONS_EXTENDED:
            skill = (packed.get("validateEvents") or {}).get("skill")
            if skill is None or skill < SKILL_KEEP:
                progress(
                    f"    [{i_h}/{len(horizon_steps)}] +{mins}′: scartato "
                    f"(eventSkill={skill}, soglia {SKILL_KEEP}) in {time.time() - t_h:.1f}s"
                )
                continue
        horizons[key] = packed
        ev = (packed.get("validateEvents") or {}).get("skill")
        ho = (packed.get("holdoutEvents") or {}).get("skill")
        n_tr = (packed.get("train") or {}).get("n")
        progress(
            f"    [{i_h}/{len(horizon_steps)}] +{mins}′: ok "
            f"nTrain={n_tr} valEvt={ev} holdEvt={ho} ({time.time() - t_h:.1f}s)"
        )

    if not horizons:
        progress(f"  {downstream}: nessun orizzonte valido ({time.time() - t_node:.1f}s)")
        return None
    progress(f"  {downstream}: node done in {time.time() - t_node:.1f}s")

    h120 = horizons.get("120") or {}
    skill120 = (h120.get("validateEvents") or {}).get("skill")
    holdout_block = h120.get("holdoutEvents") or {}
    holdout120 = holdout_block.get("skill")
    holdout_n = int(holdout_block.get("n") or 0)
    # Both validate and holdout must clear the bar. Thin holdout (n<30) is not a free pass.
    reliable = (
        skill120 is not None
        and skill120 >= SKILL_KEEP
        and holdout_n >= 30
        and holdout120 is not None
        and holdout120 >= SKILL_KEEP
    )
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "upstreams": present,
        "lagMinutes": [lag * STEP_MINUTES for lag in lags],
        "lagR": [None if r is None else round(float(r), 6) for r in lag_rs],
        "thresholds": thresholds,
        "thresholdOn": "per_upstream_z85",
        "useRain": use_rain,
        "horizons": horizons,
        "skill120": skill120,
        "holdoutSkill120": holdout120,
        "reliable": reliable,
        "samples": int(max((h.get("train", {}).get("n") or 0) for h in horizons.values())),
        "trainedAt": trained_at,
        "kind": "direct_horizon_delta_ridge",
    }


def edge_from_node(node: dict, upstream: str) -> dict | None:
    try:
        i = node["upstreams"].index(upstream)
    except ValueError:
        return None
    h120 = (node.get("horizons") or {}).get("120") or {}
    return {
        "lagMinutes": node["lagMinutes"][i],
        "skill120": node.get("skill120"),
        "reliable": node.get("reliable"),
        "rmse": h120.get("validate", {}).get("rmse"),
        "spreadLow": h120.get("spreadLow") or h120.get("validate", {}).get("spreadLow"),
        "spreadHigh": h120.get("spreadHigh") or h120.get("validate", {}).get("spreadHigh"),
        "trainedAt": node.get("trainedAt"),
        "kind": node.get("kind"),
    }


def fit_config(
    config: dict,
    wide: pd.DataFrame,
    max_lag_hours: float = 12.0,
    rain: pd.DataFrame | None = None,
) -> dict:
    nodes: dict = {}
    edges: dict = {}
    targets = [
        (downstream, meta)
        for downstream, meta in config.get("sensors", {}).items()
        if any(u in config["sensors"] for u in (meta.get("dependencies") or []))
    ]
    progress(f"Fit ridge: {len(targets)} nodi valle da stimare")
    for i, (downstream, meta) in enumerate(targets, 1):
        ups = [u for u in (meta.get("dependencies") or []) if u in config["sensors"]]
        if not ups:
            continue
        progress(f"[{i}/{len(targets)}] ▶ {downstream} ← {ups}")
        node = fit_node(wide, downstream, ups, max_lag_hours, rain=rain)
        if not node:
            progress(f"  skip {downstream}: campioni insufficienti")
            continue
        nodes[downstream] = node
        flag = "OK" if node["reliable"] else "no-skill"
        progress(
            f"  ✓ {downstream}: ups={node['upstreams']} lags={node['lagMinutes']} "
            f"skill120={node['skill120']} holdout={node.get('holdoutSkill120')} "
            f"[{flag}] n={node['samples']} horizons={sorted(node['horizons'], key=int)}"
        )
        for up in node["upstreams"]:
            edge = edge_from_node(node, up)
            if edge:
                edges[f"{up}->{downstream}"] = edge
    all_horizons = sorted({int(h) for n in nodes.values() for h in n["horizons"]}, key=int)
    return {
        "version": 4,
        "kind": "direct_horizon_delta_ridge",
        "stepMinutes": STEP_MINUTES,
        "horizonsMinutes": all_horizons,
        "skillKeep": SKILL_KEEP,
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
        "nodes": nodes,
        "edges": edges,
    }


def holdout_clears(node: dict, keep: float) -> bool:
    """Fat holdout must clear keep; thin holdout (n<30) does not block backtest promotion."""
    block = ((node.get("horizons") or {}).get("120") or {}).get("holdoutEvents") or {}
    n = int(block.get("n") or 0)
    if n < 30:
        return True
    skill = block.get("skill")
    return skill is not None and float(skill) >= keep


def apply_backtest_reliable(payload: dict, summary_path: Path) -> dict:
    """Set reliable from backtest skill@120, still blocked by failed fat holdout."""
    if not summary_path.exists():
        return payload
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    nodes_sum = (summary.get("summary") or {})
    keep = float(payload.get("skillKeep") or SKILL_KEEP)
    for name, node in payload.get("nodes", {}).items():
        skill = (nodes_sum.get(name) or {}).get("skill120")
        if skill is None:
            continue
        node["backtestSkill120"] = round(float(skill), 6)
        node["reliable"] = float(skill) >= keep and holdout_clears(node, keep)
        for up in node.get("upstreams") or []:
            edge = payload.get("edges", {}).get(f"{up}->{name}")
            if edge:
                edge["reliable"] = node["reliable"]
                edge["skill120"] = node["backtestSkill120"]
        flag = "OK" if node["reliable"] else "no-skill"
        print(f"  backtest reliable {name}: skill120={skill:.4f} [{flag}]")
    return payload


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Fit direct-horizon delta transfer coefficients")
    parser.add_argument("--config", type=Path, default=root / "config.json")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=root / "models" / "transfer-coeffs.json")
    parser.add_argument("--max-lag-hours", type=float, default=12.0)
    parser.add_argument("--spike-abs", type=float, default=SPIKE_ABS)
    parser.add_argument(
        "--apply-backtest",
        type=Path,
        default=None,
        help="Optional: override reliable from an existing backtest-summary.json",
    )
    parser.add_argument("--no-rain", action="store_true", help="Ignore rainfall CSVs even if present")
    parser.add_argument(
        "--split",
        choices=("alt", "block"),
        default="alt",
        help="Train/validate month split: alternating (alt) or contiguous halves (block)",
    )
    parser.add_argument(
        "--holdout-from",
        type=str,
        default=None,
        help="ISO date (e.g. 2026-01-01): months ≥ cut → holdout; usare una data futura per includere i mesi recenti in training",
    )
    args = parser.parse_args()

    global SPLIT_MODE, HOLDOUT_FROM
    SPLIT_MODE = args.split
    HOLDOUT_FROM = args.holdout_from
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*inplace keyword.*")
    t0 = time.time()
    progress(f"Split mesi: {SPLIT_MODE}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stations = list(config["sensors"].keys())
    progress(f"Carico {len(stations)} stazioni da {args.data} (DuckDB)…")
    wide = load_wide(args.data, stations)
    progress(f"Griglia {STEP_MINUTES} min: {len(wide)} timestep, {wide.index.min()} → {wide.index.max()}")
    wide, spike_counts = filter_spikes(wide, spike_abs=args.spike_abs)
    spiked = {k: v for k, v in spike_counts.items() if v}
    progress(f"Spike filtrati (|Δ|>{args.spike_abs} m / 30′): {spiked or 'nessuno'}")

    rain = None
    if not args.no_rain:
        raw_gauges = list((config.get("rainfall") or {}).get("gauges") or [])
        gauges = [
            g if isinstance(g, str) else str(g.get("name"))
            for g in raw_gauges
            if isinstance(g, str) or (isinstance(g, dict) and g.get("name"))
        ]
        rain_dir = args.data / "rain"
        progress(f"Carico pioggia bacino ({len(gauges)} pluviometri)…")
        rain = load_rain_basin(rain_dir, gauges, wide.index)
        if rain is not None:
            n_ok = int(rain["rain6h"].notna().sum())
            progress(f"Pioggia bacino: {n_ok} timestep con dato")
        else:
            progress("Pioggia: nessun file in data/rain (fit solo livelli)")

    payload = fit_config(config, wide, max_lag_hours=args.max_lag_hours, rain=rain)
    if args.apply_backtest:
        progress(f"Applico reliable da {args.apply_backtest}…")
        payload = apply_backtest_reliable(payload, args.apply_backtest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    progress(f"Scrivo {args.out}…")
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reliable = sum(1 for n in payload["nodes"].values() if n.get("reliable"))
    progress(
        f"Wrote {len(payload['nodes'])} nodes ({reliable} reliable) / "
        f"{len(payload['edges'])} edges → {args.out} "
        f"(totale {time.time() - t0:.0f}s)"
    )


if __name__ == "__main__":
    main()
