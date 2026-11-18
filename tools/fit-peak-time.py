#!/usr/bin/env python3
"""M4: quantile regression of Cento peak-arrival lag from upstream levels.

For each catalog event at Cento, measure lag_min = peak_Cento − peak_upstream
(primary upstream = Forcelli) and regress lag on upstream level / local slope
at issue time (peak − H). Reports median |error| on frozen test events.

Uso:
  uv run python tools/fit-peak-time.py --data ./storico --events models/events.json \\
    --out models/ab-m4-peak-time.md --json models/fase2-m4-peak.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP_MIN = 30
TEST_FROM = pd.Timestamp("2026-01-01T00:00:00Z")


def load_series(path: Path) -> pd.Series:
    if path.suffix == ".csv":
        alt = path.with_suffix(".parquet")
        if alt.exists():
            path = alt
    if path.suffix == ".parquet":
        import duckdb

        df = duckdb.connect().execute(
            "SELECT CAST(time AS TIMESTAMPTZ) AS time, CAST(value AS DOUBLE) AS value "
            "FROM read_parquet(?)",
            [str(path)],
        ).df()
    else:
        df = pd.read_csv(path, parse_dates=[0])
    df.columns = ["time", "value"]
    s = df.set_index("time")["value"].astype(float)
    s.index = pd.to_datetime(s.index, utc=True)
    return s.sort_index()


def peak_near(
    series: pd.Series,
    center: pd.Timestamp,
    window_h: float = 18.0,
    *,
    before_only: bool = False,
) -> tuple[pd.Timestamp, float] | None:
    lo = center - pd.Timedelta(hours=window_h)
    hi = center if before_only else center + pd.Timedelta(hours=window_h)
    w = series.loc[lo:hi]
    if w.empty:
        return None
    i = w.idxmax()
    return i, float(w.loc[i])


def features_at(series: pd.Series, t: pd.Timestamp) -> dict[str, float] | None:
    if t not in series.index:
        # nearest previous on 30′ lattice
        prev = series.loc[:t]
        if prev.empty:
            return None
        t = prev.index[-1]
    v = float(series.loc[t])
    if not np.isfinite(v):
        return None
    prev = series.loc[: t - pd.Timedelta(minutes=STEP_MIN)]
    v1 = float(prev.iloc[-1]) if len(prev) else np.nan
    prev6 = series.loc[: t - pd.Timedelta(hours=3)]
    v6 = float(prev6.iloc[-1]) if len(prev6) else np.nan
    return {
        "level": v,
        "d30": v - v1 if np.isfinite(v1) else 0.0,
        "d3h": v - v6 if np.isfinite(v6) else 0.0,
    }


def pinball_fit(X: np.ndarray, y: np.ndarray, q: float, iters: int = 2000, lr: float = None) -> np.ndarray:
    """Linear pinball via subgradient; returns [bias, ...coefs]. y scaled for stable steps."""
    n, d = X.shape
    y_scale = float(np.std(y)) or 1.0
    ys = y / y_scale
    w = np.zeros(d + 1, dtype=float)
    w[0] = float(np.quantile(ys, q))
    step = lr if lr is not None else 0.05
    for _ in range(iters):
        pred = w[0] + X @ w[1:]
        resid = ys - pred
        # vectorized subgradient of pinball
        sign = np.where(resid >= 0, -q, -(q - 1.0))
        grad = np.empty_like(w)
        grad[0] = sign.mean()
        grad[1:] = (X * sign[:, None]).mean(axis=0) + 1e-2 * w[1:]
        w -= step * grad
    w[0] *= y_scale
    w[1:] *= y_scale
    return w


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return w[0] + X @ w[1:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "storico")
    ap.add_argument("--events", type=Path, default=ROOT / "models" / "events.json")
    ap.add_argument("--downstream", default="Cento")
    ap.add_argument("--upstream", default="Forcelli")
    ap.add_argument("--horizon-min", type=int, default=120)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "ab-m4-peak-time.md")
    ap.add_argument("--json", type=Path, default=ROOT / "models" / "fase2-m4-peak.json")
    args = ap.parse_args()

    catalog = json.loads(args.events.read_text(encoding="utf-8"))
    events = [
        e for e in catalog.get("events", catalog if isinstance(catalog, list) else [])
        if e.get("station") == args.downstream
    ]
    if not events and "stations" in catalog:
        events = catalog["stations"].get(args.downstream, {}).get("events", [])
        for e in events:
            e.setdefault("station", args.downstream)

    y_s = load_series(args.data / f"{args.downstream}.csv")
    x_s = load_series(args.data / f"{args.upstream}.csv")

    rows = []
    for e in events:
        peak_c = pd.Timestamp(e["peakTime"])
        if peak_c.tzinfo is None:
            peak_c = peak_c.tz_localize("UTC")
        pc = peak_near(y_s, peak_c, 6.0)
        # Upstream peak must precede Cento (travel time); search only backward.
        pu = peak_near(x_s, peak_c, 18.0, before_only=True)
        if not pc or not pu:
            continue
        lag_min = (pc[0] - pu[0]).total_seconds() / 60.0
        if lag_min < 0 or lag_min > 18 * 60:
            continue
        issue = pc[0] - pd.Timedelta(minutes=args.horizon_min)
        fx = features_at(x_s, issue)
        fy = features_at(y_s, issue)
        if not fx or not fy:
            continue
        rows.append({
            "peakTime": pc[0].isoformat(),
            "lagMin": lag_min,
            "upLevel": fx["level"],
            "upD30": fx["d30"],
            "upD3h": fx["d3h"],
            "downLevel": fy["level"],
            "downD30": fy["d30"],
            "isTest": pc[0] >= TEST_FROM,
            "peakLevel": pc[1],
        })

    if len(rows) < 20:
        raise SystemExit(f"Too few events with measurable lag: {len(rows)}")

    df = pd.DataFrame(rows)
    train = df[~df["isTest"]]
    test = df[df["isTest"]]
    feat_cols = ["upLevel", "upD30", "upD3h", "downLevel", "downD30"]
    Xtr = train[feat_cols].to_numpy(dtype=float)
    ytr = train["lagMin"].to_numpy(dtype=float)
    # Standardize on train
    mu = Xtr.mean(axis=0)
    sd = np.where(Xtr.std(axis=0) < 1e-6, 1.0, Xtr.std(axis=0))
    Xtrz = (Xtr - mu) / sd
    models = {q: pinball_fit(Xtrz, ytr, q) for q in (0.1, 0.5, 0.9)}

    def eval_split(split: pd.DataFrame) -> dict:
        if split.empty:
            return {"n": 0}
        X = (split[feat_cols].to_numpy(dtype=float) - mu) / sd
        pred50 = predict(models[0.5], X)
        err = pred50 - split["lagMin"].to_numpy(dtype=float)
        # Baseline: constant median train lag
        base = np.full(len(split), float(np.median(ytr)))
        base_err = base - split["lagMin"].to_numpy(dtype=float)
        return {
            "n": int(len(split)),
            "medAbsErr": float(np.median(np.abs(err))),
            "mae": float(np.mean(np.abs(err))),
            "medAbsErrBaseline": float(np.median(np.abs(base_err))),
            "predLagMed": float(np.median(pred50)),
            "trueLagMed": float(np.median(split["lagMin"])),
            "phaseFloorMin": 30.0,
        }

    train_m = eval_split(train)
    test_m = eval_split(test)
    success = (
        test_m.get("n", 0) >= 5
        and test_m["medAbsErr"] < test_m["phaseFloorMin"]
        and test_m["medAbsErr"] < test_m["medAbsErrBaseline"]
    )

    payload = {
        "kind": "m4_peak_time",
        "downstream": args.downstream,
        "upstream": args.upstream,
        "horizonMin": args.horizon_min,
        "featureNames": feat_cols,
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "quantiles": {str(q): {"bias": float(w[0]), "coef": w[1:].tolist()} for q, w in models.items()},
        "train": train_m,
        "test": test_m,
        "success": success,
        "nEvents": len(df),
        "nTrain": int((~df["isTest"]).sum()),
        "nTest": int(df["isTest"].sum()),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# A/B M4 — Tempo di arrivo del picco (Cento)",
        "",
        "**Ipotesi:** il ritardo picco_monte→picco_Cento è prevedibile da livello/pendenza a monte "
        f"a issue = peak−{args.horizon_min}′, meglio della mediana train e sotto i 30′ di ritardo di fase.",
        "",
        "**Criterio:** su test congelato (2026-01-01→) mediana |errore| < 30′ e < baseline mediana.",
        "",
        f"Upstream primario: **{args.upstream}**. Eventi usabili: {len(df)} (train {payload['nTrain']}, test {payload['nTest']}).",
        "",
        "## Metriche",
        "",
        "| Split | n | med\\|err\\| pred (′) | med\\|err\\| baseline (′) | lag vero med (′) |",
        "|---|---:|---:|---:|---:|",
        f"| train | {train_m['n']} | {train_m['medAbsErr']:.1f} | {train_m['medAbsErrBaseline']:.1f} | {train_m['trueLagMed']:.1f} |",
        f"| **test** | {test_m['n']} | **{test_m['medAbsErr']:.1f}** | {test_m['medAbsErrBaseline']:.1f} | {test_m['trueLagMed']:.1f} |",
        "",
        f"**Esito:** {'PASS' if success else 'FAIL — non promuovere'}.",
        "",
        "Nota: stima offline del solo ritardo di picco; non sostituisce la curva di livello.",
        "",
    ]
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"test med|err|={test_m['medAbsErr']:.1f}' baseline={test_m['medAbsErrBaseline']:.1f}' success={success}")


if __name__ == "__main__":
    main()
