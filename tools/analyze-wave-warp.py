#!/usr/bin/env python3
"""Diagnostic lag + bounded DTW alignment between neighbouring Reno stations.

Diagnostic only: uses complete concluded flood windows (oracle). It does NOT
produce production forecasts. Outputs JSON + Markdown under models/experiments.
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

STEP_MIN = 30
STEP_MS = STEP_MIN * 60_000
MAX_LAG_STEPS = 48  # 24 h
WARP_BAND = 4       # ±2 h residual band after constant lag
WARP_PENALTY = 0.05
MIN_RISE_M = 0.25
MIN_COVERAGE = 0.75

EDGES = [
    ("Forcelli", "Cento"),
    ("Bagnetto Reno", "Cento"),
    ("Casalecchio tiro a volo", "Bonconvento"),
    ("Bonconvento", "Sostegno Reno"),
    ("Sostegno Reno", "Bagnetto Reno"),
    ("Cento", "Dosso"),
    ("Dosso", "Opera Reno Panfilia"),
    ("Opera Reno Panfilia", "Gallo"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def load_series(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / f"{name}.parquet"
    if not path.exists():
        path = data_dir / f"{name}.csv"
    if path.suffix == ".parquet":
        import duckdb
        con = duckdb.connect()
        df = con.execute(
            "SELECT CAST(time AS TIMESTAMPTZ) AS time, CAST(value AS DOUBLE) AS value "
            "FROM read_parquet(?) WHERE value IS NOT NULL AND isfinite(value)", [str(path)]).df()
    else:
        df = pd.read_csv(path, parse_dates=["time"])
    df = df.dropna(subset=["value"]).sort_values("time")
    df = df[~df["time"].duplicated(keep="last")]
    return df


def resample_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Linear interpolation on a regular 30′ lattice inside [start,end]."""
    times = ((df["time"] - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(dtype=np.int64)
    vals = df["value"].to_numpy(dtype=float)
    grid = np.arange(start.value // 1_000_000, end.value // 1_000_000 + 1, STEP_MS, dtype=np.int64)
    if times.size == 0 or grid.size == 0:
        return pd.Series(np.nan, index=pd.DatetimeIndex(grid * 1_000_000))
    out = np.interp(grid, times, vals, left=np.nan, right=np.nan)
    # mask targets outside observed range and across gaps > 60 min
    idx = np.searchsorted(times, grid)
    valid = np.zeros(grid.shape, dtype=bool)
    for i, g in enumerate(grid):
        j = idx[i]
        if j < times.size and times[j] == g:
            valid[i] = True
            continue
        if 0 < j < times.size:
            l, r = times[j - 1], times[j]
            if l <= g <= r and (r - l) <= 60 * 60_000:
                valid[i] = True
        elif j == times.size and times.size and (g - times[-1]) <= 15 * 60_000:
            valid[i] = True
    out[~valid] = np.nan
    return pd.Series(out, index=pd.DatetimeIndex(grid * 1_000_000))


def zscore(s: pd.Series) -> np.ndarray:
    v = s.to_numpy(dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return np.full(v.shape, np.nan)
    mu = np.nanmean(v)
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-9:
        sd = 1.0
    return (v - mu) / sd


def estimate_lag(a: np.ndarray, b: np.ndarray, max_lag: int = MAX_LAG_STEPS) -> tuple[int, float]:
    """Lag L≥0: compare a[t-L] with b[t]; returns best L and Pearson r."""
    best = (0, -np.inf)
    n = min(len(a), len(b))
    for lag in range(0, min(max_lag, n - 10) + 1):
        # downstream b[t+lag] ≈ upstream a[t]: positive lag = wave travel time
        x = a[0:n - lag]
        y = b[lag:n]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 20:
            continue
        if np.nanstd(x[ok]) < 1e-9 or np.nanstd(y[ok]) < 1e-9:
            continue
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        if np.isfinite(r) and r > best[1]:
            best = (lag, r)
    return best


def dtw_path(a: np.ndarray, b: np.ndarray, band: int = WARP_BAND, penalty: float = WARP_PENALTY):
    """Bounded monotonic DTW with free ends inside the band.

    Returns (cost, map_i_to_j, saturated_frac, mean_abs_tau_steps).
    """
    n, m = len(a), len(b)
    a_fill = pd.Series(a).interpolate(limit_direction="both").to_numpy()
    b_fill = pd.Series(b).interpolate(limit_direction="both").to_numpy()
    INF = 1e18
    dp = np.full((n, m), INF, dtype=np.float64)
    back = np.zeros((n, m), dtype=np.int8)  # 0=start, 1=up, 2=diag, 3=left
    scale = max(1, n)
    for i in range(n):
        j0 = max(0, i - band)
        j1 = min(m - 1, i + band)
        for j in range(j0, j1 + 1):
            c = (a_fill[i] - b_fill[j]) ** 2 + penalty * ((j - i) / scale) ** 2 * n
            cand = []
            if i == 0 and j <= band:
                cand.append((0.0, 0))
            if j == 0 and i <= band:
                cand.append((0.0, 0))
            if i > 0:
                if abs((i - 1) - j) <= band and np.isfinite(dp[i - 1, j]):
                    cand.append((dp[i - 1, j], 1))
                if j > 0 and abs((i - 1) - (j - 1)) <= band and np.isfinite(dp[i - 1, j - 1]):
                    cand.append((dp[i - 1, j - 1], 2))
            if j > 0 and abs(i - (j - 1)) <= band and np.isfinite(dp[i, j - 1]):
                cand.append((dp[i, j - 1], 3))
            if not cand:
                continue
            val, mv = min(cand, key=lambda x: x[0])
            dp[i, j] = val + c
            back[i, j] = mv
    # free end: best cell in last row/col within band of the opposite corner
    cand = []
    for j in range(max(0, m - 1 - band), min(m - 1, n - 1 + band) + 1):
        if np.isfinite(dp[n - 1, j]):
            cand.append((dp[n - 1, j], n - 1, j))
    for i in range(max(0, n - 1 - band), min(n - 1, m - 1 + band) + 1):
        if np.isfinite(dp[i, m - 1]):
            cand.append((dp[i, m - 1], i, m - 1))
    if not cand:
        return float("inf"), np.full(n, -1, dtype=np.int64), 1.0, float("nan")
    total, ei, ej = min(cand, key=lambda x: x[0])
    mapping = np.full(n, -1, dtype=np.int64)
    i, j = ei, ej
    while 0 <= i < n and 0 <= j < m:
        mapping[i] = j
        mv = back[i, j]
        if mv == 1:
            i -= 1
        elif mv == 2:
            i -= 1; j -= 1
        elif mv == 3:
            j -= 1
        else:
            break
    valid = mapping >= 0
    dev = np.abs(np.arange(n)[valid] - mapping[valid])
    sat = float(np.mean(dev >= band - 1)) if dev.size else 1.0
    tau = (mapping[valid] - np.arange(n)[valid])
    return float(total), mapping, sat, (float(np.mean(np.abs(tau))) if tau.size else float("nan"))


def affine_cost(a: np.ndarray, b: np.ndarray, scales=(0.8, 0.9, 1.0, 1.1, 1.25)) -> tuple[float, float]:
    """Best uniform time-dilation cost: b stretched/compressed onto a's grid."""
    best = (float("inf"), 1.0)
    n = len(a)
    t = np.arange(n, dtype=float)
    for s in scales:
        src = np.clip(t * s, 0, n - 1)
        bs = np.interp(src, t, b)
        m = np.isfinite(a) & np.isfinite(bs)
        if m.sum() < 20:
            continue
        c = float(np.mean((a[m] - bs[m]) ** 2))
        if c < best[0]:
            best = (c, s)
    return best


def align_pair(up: pd.DataFrame, down: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    a_s = resample_window(up, start, end)
    b_s = resample_window(down, start, end)
    common = a_s.index.intersection(b_s.index)
    a_s = a_s.reindex(common)
    b_s = b_s.reindex(common)
    a = zscore(a_s)
    b = zscore(b_s)
    ok = np.isfinite(a) & np.isfinite(b)
    coverage = float(ok.mean()) if ok.size else 0.0
    if coverage < MIN_COVERAGE or ok.sum() < 50:
        return None
    # trim to finite span
    idx = np.flatnonzero(ok)
    a2 = a[idx[0]: idx[-1] + 1]
    b2 = b[idx[0]: idx[-1] + 1]
    lag, r = estimate_lag(np.nan_to_num(a2, nan=np.nanmean(a2)), np.nan_to_num(b2, nan=np.nanmean(b2)))
    # shift upstream earlier by lag: a_aligned[t] = a[t-lag]
    a_al = np.full_like(a2, np.nan)
    if lag < len(a2):
        a_al[lag:] = a2[: len(a2) - lag]
    a_al = np.nan_to_num(a_al, nan=np.nanmean(a2))
    b_f = np.nan_to_num(b2, nan=np.nanmean(b2))
    m0 = np.isfinite(a2) & np.isfinite(b2)
    noalign_cost = float(np.mean((a2[m0] - b2[m0]) ** 2)) if m0.sum() >= 20 else None
    affine_c, affine_s = affine_cost(a_al, b_f)
    cost, mapping, sat, tau_mean = dtw_path(a_al, b_f)
    diag_cost = float(np.mean((a_al - b_f) ** 2))
    tau = (mapping - np.arange(len(mapping))) * STEP_MIN
    finite_tau = tau[mapping >= 0]
    return {
        "coverage": round(coverage, 3),
        "lagSteps": int(lag),
        "lagMin": int(lag * STEP_MIN),
        "lagCorr": round(r, 4),
        "noAlignCost": round(noalign_cost, 3) if noalign_cost is not None else None,
        "affineCost": round(affine_c, 3),
        "affineScale": affine_s,
        "dtwCost": round(cost, 3),
        "diagCost": round(diag_cost, 3),
        "saturatedFrac": round(sat, 3),
        "tauMeanAbsSteps": round(tau_mean, 2) if np.isfinite(tau_mean) else None,
        "tauMinMean": round(float(np.mean(finite_tau)), 1) if finite_tau.size else None,
        "tauMinAbsP90": round(float(np.quantile(np.abs(finite_tau), 0.9)), 1) if finite_tau.size else None,
        "n": int(len(mapping)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "storico")
    p.add_argument("--events", type=Path, default=ROOT / "models" / "events.json")
    p.add_argument("--out", type=Path, default=ROOT / "models" / "experiments" / "wave-warp.json")
    p.add_argument("--md", type=Path, default=ROOT / "models" / "experiments" / "wave-warp.md")
    p.add_argument("--pad-hours", type=float, default=8.0)
    args = p.parse_args()

    events = json.loads(args.events.read_text(encoding="utf-8"))
    catalog = events.get("events") or events.get("catalog") or []
    # events entries have station/start/peakTime/end ISO strings
    needed = sorted({s for e in EDGES for s in e} | {e.get("station") for e in catalog if e.get("station")})
    series = {name: load_series(args.data, name) for name in needed if (args.data / f"{name}.csv").exists() or (args.data / f"{name}.parquet").exists()}
    log(f"loaded {len(series)} stations, {len(catalog)} catalog events")

    pad = pd.Timedelta(hours=args.pad_hours)
    results = []
    for up, down in EDGES:
        if up not in series or down not in series:
            continue
        down_events = [e for e in catalog if e.get("station") == down]
        edge_rows = []
        for ev in down_events:
            start = pd.Timestamp(ev["start"]) + pd.Timedelta(hours=-1)
            end = pd.Timestamp(ev["end"]) + pd.Timedelta(hours=1)
            start -= pad
            end += pad
            row = align_pair(series[up], series[down], start, end)
            if row is None:
                continue
            row.update({"edge": f"{up}->{down}", "eventPeak": ev.get("peakTime"), "peakLevel": ev.get("peakLevel")})
            edge_rows.append(row)
        if not edge_rows:
            continue
        df = pd.DataFrame(edge_rows)
        summary = {
            "edge": f"{up}->{down}",
            "nEvents": int(len(df)),
            "lagMinMedian": float(df["lagMin"].median()),
            "lagCorrMedian": float(df["lagCorr"].median()),
            "tauAbsP90Median": float(df["tauMinAbsP90"].median()),
            "saturatedFracMean": float(df["saturatedFrac"].mean()),
            "coverageMean": float(df["coverage"].mean()),
            "costNoAlignMedian": float(df["noAlignCost"].median()),
            "costAffineMedian": float(df["affineCost"].median()),
            "costLagOnlyMedian": float(df["diagCost"].median()),
            "costDtwMedian": float(df["dtwCost"].median()),
            "affineScaleMedian": float(df["affineScale"].median()),
        }
        results.append({"summary": summary, "events": edge_rows})
        log(f"{up}->{down}: n={summary['nEvents']} lag={summary['lagMinMedian']:.0f}′ tauP90={summary['tauAbsP90Median']:.1f}′ sat={summary['saturatedFracMean']:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"kind": "wave-warp-diagnostic", "bandSteps": WARP_BAND, "maxLagSteps": MAX_LAG_STEPS, "results": results}, indent=2) + "\n", encoding="utf-8")

    md = ["# Diagnosi lag + DTW (eventi conclusi, diagnostico)", ""]
    md.append("Costi su segnali standardizzati: minore = migliore allineamento. `noAlign` = senza shift, "
              "`affine` = lag + dilatazione uniforme, `lag` = solo lag costante, `dtw` = lag + DTW vincolato.")
    md.append("")
    md.append("| edge | n | lag med (′) | lagCorr | scala affine | costo noAlign | affine | lag | dtw | tauP90 med (′) | sat | cov |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["summary"]
        md.append(f"| {s['edge']} | {s['nEvents']} | {s['lagMinMedian']:.0f} | {s['lagCorrMedian']:.2f} "
                  f"| {s['affineScaleMedian']:.2f} | {s['costNoAlignMedian']:.3f} | {s['costAffineMedian']:.3f} "
                  f"| {s['costLagOnlyMedian']:.3f} | {s['costDtwMedian']:.3f} | {s['tauAbsP90Median']:.1f} "
                  f"| {s['saturatedFracMean']:.2f} | {s['coverageMean']:.2f} |")
    args.md.write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"Wrote {args.out} and {args.md}")


if __name__ == "__main__":
    main()
