#!/usr/bin/env python3
"""M5: dual travel-time τ above/below per-upstream z85 (train-only).

Estimates lag separately on calm vs high-regime train rows, checks non-overlap
of bootstrap CI, and scores a simple persist-delta proxy on frozen-test events
when picking τ by contemporaneous z vs z85.

Uso:
  uv run python tools/ab-m5-tau-regime.py --data ./storico --out models/ab-m5-tau-regime.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP_MIN = 30
HIGH_Q = 0.85
TEST_FROM = pd.Timestamp("2026-01-01T00:00:00Z")
MAX_LAG_H = 12


def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[0])
    df.columns = ["time", "value"]
    s = df.set_index("time")["value"].astype(float)
    s.index = pd.to_datetime(s.index, utc=True)
    return s.sort_index().asfreq(f"{STEP_MIN}min")


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return None
    a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den <= 0:
        return None
    return float((a * b).sum() / den)


def estimate_lag(x: np.ndarray, y: np.ndarray, mask: np.ndarray, max_steps: int) -> tuple[int, float | None]:
    best_lag, best_r = 0, None
    n = len(x)
    for lag in range(0, max_steps + 1):
        xs = x[: n - lag]
        ys = y[lag:]
        m = mask[: n - lag] & mask[lag:]
        r = pearson(xs[m], ys[m]) if m.any() else None
        if r is None:
            continue
        if best_r is None or abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    return best_lag, best_r


def bootstrap_lag_ci(x, y, mask, max_steps, draws=200, seed=1) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.where(mask)[0]
    if len(idx) < 50:
        return (np.nan, np.nan)
    lags = []
    for _ in range(draws):
        pick = rng.choice(idx, size=len(idx), replace=True)
        m = np.zeros_like(mask)
        m[pick] = True
        lag, _ = estimate_lag(x, y, m, max_steps)
        lags.append(lag * STEP_MIN)
    return float(np.quantile(lags, 0.05)), float(np.quantile(lags, 0.95))


def event_skill(y: np.ndarray, pred: np.ndarray, event_mask: np.ndarray) -> float | None:
    m = event_mask & np.isfinite(y) & np.isfinite(pred)
    if m.sum() < 20:
        return None
    err_m = pred[m] - y[m]
    # persist baseline at horizon ≈ use y shifted back? here pred already is forecast of y[t+h]
    # We pass residual vs persist separately.
    return float(err_m.mean())  # placeholder unused


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "storico")
    ap.add_argument("--config", type=Path, default=ROOT / "config.json")
    ap.add_argument("--events", type=Path, default=ROOT / "models" / "events.json")
    ap.add_argument("--stations", default="Gallo,Casalecchio chiusa,Bonconvento")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "ab-m5-tau-regime.md")
    ap.add_argument("--json", type=Path, default=ROOT / "models" / "fase2-m5-tau.json")
    args = ap.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    catalog = json.loads(args.events.read_text(encoding="utf-8"))
    want = [s.strip() for s in args.stations.split(",") if s.strip()]
    max_steps = int(MAX_LAG_H * 60 / STEP_MIN)
    results = []

    for downstream in want:
        ups = [u for u in (config["sensors"].get(downstream, {}).get("dependencies") or [])]
        if not ups:
            continue
        y = load_series(args.data / f"{downstream}.csv")
        train_idx = y.index < TEST_FROM
        # Event mask on test from catalog
        ev = [e for e in catalog["events"] if e["station"] == downstream]
        event_test = pd.Series(False, index=y.index)
        for e in ev:
            t0 = pd.Timestamp(e["start"])
            t1 = pd.Timestamp(e["end"])
            if t1 < TEST_FROM:
                continue
            event_test.loc[max(t0, TEST_FROM):t1] = True

        station_rows = []
        any_nonoverlap = False
        for up in ups:
            if not (args.data / f"{up}.csv").exists():
                continue
            x = load_series(args.data / f"{up}.csv").reindex(y.index)
            x_vals = x.to_numpy(dtype=float)
            y_vals = y.to_numpy(dtype=float)
            train_m = np.asarray(train_idx) & np.isfinite(x_vals) & np.isfinite(y_vals)
            mu = np.nanmean(x_vals[train_m])
            sd = np.nanstd(x_vals[train_m]) or 1.0
            z = (x_vals - mu) / sd
            z85 = float(np.nanquantile(z[train_m], HIGH_Q))
            calm = train_m & (z <= z85)
            high = train_m & (z > z85)
            lag_all, r_all = estimate_lag(x_vals, y_vals, train_m, max_steps)
            lag_c, r_c = estimate_lag(x_vals, y_vals, calm, max_steps)
            lag_h, r_h = estimate_lag(x_vals, y_vals, high, max_steps)
            ci_c = bootstrap_lag_ci(x_vals, y_vals, calm, max_steps)
            ci_h = bootstrap_lag_ci(x_vals, y_vals, high, max_steps)
            nonoverlap = (
                np.isfinite(ci_c[0]) and np.isfinite(ci_h[0])
                and (ci_c[1] < ci_h[0] or ci_h[1] < ci_c[0])
            )
            if nonoverlap:
                any_nonoverlap = True

            # Proxy skill @120′ on test events: ŷ = x(t−τ) − x(t−τ−120) + y(t)
            def proxy(lag_steps: int) -> dict:
                h = 4  # 120′
                pred = np.full_like(y_vals, np.nan)
                for i in range(len(y_vals)):
                    j = i - lag_steps
                    k = j - h
                    if j < 0 or k < 0:
                        continue
                    if not (np.isfinite(x_vals[j]) and np.isfinite(x_vals[k]) and np.isfinite(y_vals[i])):
                        continue
                    # issue at i, forecast y[i+h] ≈ y[i] + (x[j]-x[k]) but we score at i for simplicity
                    # Score: predict y[i] from y[i-h] + Δx over lag window ending at i-h
                    ii = i
                    i0 = i - h
                    j0 = i0 - lag_steps
                    j1 = i0 - lag_steps - h
                    if i0 < 0 or j0 < 0 or j1 < 0:
                        continue
                    if not all(np.isfinite(v) for v in (y_vals[i0], x_vals[j0], x_vals[j1], y_vals[ii])):
                        continue
                    pred[ii] = y_vals[i0] + (x_vals[j0] - x_vals[j1])
                m = event_test.to_numpy() & np.isfinite(pred) & np.isfinite(y_vals)
                if m.sum() < 20:
                    return {"n": int(m.sum()), "rmse": None, "skill": None}
                rmse = float(np.sqrt(np.mean((pred[m] - y_vals[m]) ** 2)))
                base = float(np.sqrt(np.mean((y_vals[np.where(m)[0] - h] - y_vals[m]) ** 2))) if (np.where(m)[0] - h).min() >= 0 else None
                skill = None if not base else 1.0 - rmse / base
                return {"n": int(m.sum()), "rmse": rmse, "skill": skill}

            # Regime-aware: pick τ by z at issue (i-h)
            def proxy_dual() -> dict:
                h = 4
                pred = np.full_like(y_vals, np.nan)
                for i in range(len(y_vals)):
                    i0 = i - h
                    if i0 < 0:
                        continue
                    z0 = z[i0]
                    lag_steps = lag_h if np.isfinite(z0) and z0 > z85 else lag_c
                    j0 = i0 - lag_steps
                    j1 = i0 - lag_steps - h
                    if j0 < 0 or j1 < 0:
                        continue
                    if not all(np.isfinite(v) for v in (y_vals[i0], x_vals[j0], x_vals[j1], y_vals[i])):
                        continue
                    pred[i] = y_vals[i0] + (x_vals[j0] - x_vals[j1])
                m = event_test.to_numpy() & np.isfinite(pred) & np.isfinite(y_vals)
                if m.sum() < 20:
                    return {"n": int(m.sum()), "rmse": None, "skill": None}
                idx = np.where(m)[0]
                if (idx - h).min() < 0:
                    return {"n": int(m.sum()), "rmse": None, "skill": None}
                rmse = float(np.sqrt(np.mean((pred[m] - y_vals[m]) ** 2)))
                base = float(np.sqrt(np.mean((y_vals[idx - h] - y_vals[m]) ** 2)))
                skill = 1.0 - rmse / base if base > 0 else None
                return {"n": int(m.sum()), "rmse": rmse, "skill": skill}

            sc_all = proxy(lag_all)
            sc_dual = proxy_dual()
            station_rows.append({
                "upstream": up,
                "lagAllMin": lag_all * STEP_MIN,
                "lagCalmMin": lag_c * STEP_MIN,
                "lagHighMin": lag_h * STEP_MIN,
                "rAll": r_all,
                "rCalm": r_c,
                "rHigh": r_h,
                "z85": z85,
                "ciCalm": ci_c,
                "ciHigh": ci_h,
                "nonOverlap": nonoverlap,
                "skillAll": sc_all.get("skill"),
                "skillDual": sc_dual.get("skill"),
                "nTest": sc_dual.get("n"),
            })

        improved = any(
            (r["skillDual"] is not None and r["skillAll"] is not None and r["skillDual"] > r["skillAll"] + 0.01)
            for r in station_rows
        )
        success = any_nonoverlap and improved
        results.append({
            "downstream": downstream,
            "upstreams": station_rows,
            "anyNonOverlap": any_nonoverlap,
            "improved": improved,
            "success": success,
        })

    payload = {"kind": "m5_tau_regime", "stations": results}
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# A/B M5 — τ regime-dipendente (ridge stations)",
        "",
        "**Ipotesi:** il ritardo di correlazione cross-corr è diverso sopra/sotto z85 train; "
        "intervalli bootstrap non si sovrappongono e lo skill evento @120′ migliora con τ duale.",
        "",
        "**Criterio:** almeno una stazione con CI non sovrapposti **e** skill duale > skill τ unico (+0.01).",
        "",
    ]
    any_ok = False
    for st in results:
        lines.append(f"## {st['downstream']}")
        lines.append("")
        lines.append("| Upstream | τ all | τ calm [CI90] | τ high [CI90] | non-overlap | skill@120 all | skill dual |")
        lines.append("|---|---:|---|---|:---:|---:|---:|")
        for r in st["upstreams"]:
            cc = f"{r['lagCalmMin']} [{r['ciCalm'][0]:.0f},{r['ciCalm'][1]:.0f}]" if np.isfinite(r["ciCalm"][0]) else f"{r['lagCalmMin']}"
            ch = f"{r['lagHighMin']} [{r['ciHigh'][0]:.0f},{r['ciHigh'][1]:.0f}]" if np.isfinite(r["ciHigh"][0]) else f"{r['lagHighMin']}"
            sa = "n/d" if r["skillAll"] is None else f"{r['skillAll']:.3f}"
            sd = "n/d" if r["skillDual"] is None else f"{r['skillDual']:.3f}"
            lines.append(
                f"| {r['upstream']} | {r['lagAllMin']} | {cc} | {ch} | "
                f"{'yes' if r['nonOverlap'] else 'no'} | {sa} | {sd} |"
            )
        lines.append("")
        lines.append(f"Esito stazione: **{'PASS' if st['success'] else 'FAIL'}** "
                     f"(non-overlap={st['anyNonOverlap']}, improved={st['improved']}).")
        lines.append("")
        any_ok = any_ok or st["success"]
    lines.append("## Decisione")
    lines.append("")
    lines.append(
        "**Promuovere τ duale**." if any_ok else
        "**Non promuovere** — τ duale non batte τ unico con CI disgiunti sul proxy evento."
    )
    lines.append("")
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} any_ok={any_ok}")


if __name__ == "__main__":
    main()
