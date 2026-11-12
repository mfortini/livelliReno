#!/usr/bin/env python3
"""Offline Cento EnKF spike: mass-transfer reservoirs + ensemble filter.

ARPAE gauges do not share a common vertical datum, so absolute head differences
are not physical. The process works on local anomalies z = h − μ_train:

  pulse  = C_f · (Δz_f)_+ + C_s · (Δz_s)_+ + α · rain_step   # mass pulses
  couple = K_f · z_f + K_s · z_s                               # standing transfer
  z_c   ← z_c + (pulse + couple − C_out · z_c) / A
  h_c    = z_c + μ_c

A is effective storage; C_out / A is transfer resistance (lag/drawdown).
Upstreams are lagged boundary drivers. Within the 2 h horizon, lagged upstream
values are already observed (τ ≫ 2 h), so open-loop forecasts reuse them.
Writes models/enkf-cento-summary.json — dashboard wiring only if gate passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = "Cento"
UP_F = "Forcelli"
UP_S = "Sostegno Reno"
STATIONS = [DOWNSTREAM, UP_F, UP_S]
HORIZONS = [1, 2, 3, 4]
EVENT_ABS = 0.10
ENSEMBLE_N = 40
EVAL_EVERY = 4
PROCESS_NOISE = 0.01
OBS_NOISE = 0.02
PARAM_TRIALS = 200
RNG_SEED = 42
DNN_SKILL120_FALLBACK = 0.83
MAX_GAIN = 0.35


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


def gauge_names(config: dict) -> list[str]:
    raw = list((config.get("rainfall") or {}).get("gauges") or [])
    names: list[str] = []
    for g in raw:
        if isinstance(g, str):
            names.append(g)
        elif isinstance(g, dict) and g.get("name"):
            names.append(str(g["name"]))
    return names


def load_rain_step(rain_dir: Path, gauges: list[str], index: pd.DatetimeIndex) -> pd.Series:
    series: list[pd.Series] = []
    for name in gauges:
        path = rain_dir / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["time"])
        if df.empty:
            continue
        s = df.set_index("time")["value"].astype(float).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = s.resample(f"{ft.STEP_MINUTES}min").sum()
        series.append(s.rename(name))
    if not series:
        return pd.Series(0.0, index=index)
    rain = pd.concat(series, axis=1).reindex(index)
    any_obs = rain.notna().any(axis=1)
    rain = rain.fillna(0.0)
    basin = rain.max(axis=1)
    basin.loc[~any_obs] = 0.0
    return basin.astype(float)


def metrics_block(y0: np.ndarray, y_obs: np.ndarray, y_pred: np.ndarray) -> dict:
    persist = y0
    mask = np.isfinite(y_obs) & np.isfinite(y_pred) & np.isfinite(y0)
    n = int(mask.sum())
    if n < 8:
        return {"rmse": None, "skill": None, "mae": None, "n": n}
    err_m = float(math.sqrt(np.mean((y_pred[mask] - y_obs[mask]) ** 2)))
    err_p = float(math.sqrt(np.mean((persist[mask] - y_obs[mask]) ** 2)))
    skill = None if err_p == 0 else 1.0 - err_m / err_p
    mae = float(np.mean(np.abs(y_pred[mask] - y_obs[mask])))
    return {
        "rmse": round(err_m, 6),
        "skill": None if skill is None else round(skill, 6),
        "mae": round(mae, 6),
        "n": n,
    }


def finite_rain(rain: float) -> float:
    return float(rain) if math.isfinite(rain) and rain > 0 else 0.0


def process_step(
    z_c: np.ndarray,
    z_f: float,
    z_f_prev: float,
    z_s: float,
    z_s_prev: float,
    rain: float,
    params: dict,
) -> np.ndarray:
    """Euler step on Cento anomaly ensemble."""
    a = params["A"]
    dz_f = z_f - z_f_prev if math.isfinite(z_f_prev) else 0.0
    dz_s = z_s - z_s_prev if math.isfinite(z_s_prev) else 0.0
    pulse = (
        params["C_f"] * max(dz_f, 0.0)
        + params["C_s"] * max(dz_s, 0.0)
        + params["C_f_down"] * min(dz_f, 0.0)
        + params["C_s_down"] * min(dz_s, 0.0)
        + params["alpha"] * finite_rain(rain)
    )
    couple = params["K_f"] * z_f + params["K_s"] * z_s
    return z_c + (pulse + couple - params["C_out"] * z_c) / a


def sample_params(rng: np.random.Generator) -> dict:
    a = float(rng.uniform(1.0, 8.0))

    def pos_gain(lo: float = 0.01, hi: float = MAX_GAIN) -> float:
        return float(rng.uniform(lo, hi)) * a

    return {
        "A": a,
        "C_f": pos_gain(0.05, MAX_GAIN),
        "C_s": pos_gain(0.05, MAX_GAIN),
        "C_f_down": pos_gain(0.02, MAX_GAIN * 0.8),
        "C_s_down": pos_gain(0.02, MAX_GAIN * 0.8),
        "K_f": pos_gain(0.0, 0.15),
        "K_s": pos_gain(0.0, 0.15),
        "C_out": pos_gain(0.02, 0.25),
        "alpha": pos_gain(0.0, 0.08),
    }


def fit_params(
    z_c: np.ndarray,
    z_f: np.ndarray,
    z_s: np.ndarray,
    rain: np.ndarray,
    train_mask: np.ndarray,
    rng: np.random.Generator,
    trials: int = PARAM_TRIALS,
) -> tuple[dict, float]:
    """Minimize blended 1-step + 4-step open-loop MSE on train (event-weighted)."""
    n = len(z_c)
    idx = np.where(train_mask & np.isfinite(z_c) & np.isfinite(z_f) & np.isfinite(z_s))[0]
    usable = [
        i for i in idx
        if i > 1
        and i + 4 < n
        and np.isfinite(z_c[i - 1])
        and np.isfinite(z_f[i - 1])
        and np.isfinite(z_s[i - 1])
        and np.isfinite(z_f[i : i + 5]).all()
        and np.isfinite(z_s[i : i + 5]).all()
        and np.isfinite(z_c[i + 4])
    ]
    if len(usable) < ft.WEAK_POINTS:
        raise SystemExit("Troppi pochi punti train per il fit EnKF")

    sample = np.array(usable, dtype=int)
    if len(sample) > 2500:
        # Prefer rising/falling windows for multi-step skill.
        deltas = np.abs(z_c[sample + 4] - z_c[sample])
        order = np.argsort(-deltas)
        sample = sample[order[:2500]]

    best_params = None
    best_loss = float("inf")
    for _ in range(trials):
        params = sample_params(rng)
        err1 = []
        err4 = []
        ok = True
        for i in sample:
            p1 = float(
                process_step(
                    np.array([z_c[i]]),
                    float(z_f[i]),
                    float(z_f[i - 1]),
                    float(z_s[i]),
                    float(z_s[i - 1]),
                    float(rain[i]) if np.isfinite(rain[i]) else 0.0,
                    params,
                )[0]
            )
            if not math.isfinite(p1):
                ok = False
                break
            err1.append(p1 - z_c[i + 1])
            z = z_c[i]
            pf, ps = float(z_f[i]), float(z_s[i])
            for k in range(4):
                src = i + 1 + k
                zf, zs = float(z_f[src]), float(z_s[src])
                z = float(
                    process_step(
                        np.array([z]), zf, pf, zs, ps,
                        float(rain[i]) if np.isfinite(rain[i]) else 0.0,
                        params,
                    )[0]
                )
                pf, ps = zf, zs
            if not math.isfinite(z):
                ok = False
                break
            w = 3.0 if abs(z_c[i + 4] - z_c[i]) >= EVENT_ABS else 1.0
            err4.append(w * (z - z_c[i + 4]) ** 2)
        if not ok:
            continue
        loss = float(np.mean(np.square(err1)) + np.mean(err4))
        if math.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_params = params
    if best_params is None:
        raise SystemExit("Fit EnKF fallito: nessuna perdita finita")
    return best_params, math.sqrt(best_loss)


def enkf_update(ensemble: np.ndarray, obs: float, r_obs: float, rng: np.random.Generator) -> np.ndarray:
    n = len(ensemble)
    # Mild inflation so repeated updates cannot collapse the gain to zero.
    spread = float(ensemble.std())
    if spread < 0.5 * r_obs:
        ensemble = ensemble + rng.normal(0.0, r_obs, size=n)
    y = ensemble.copy()
    d = obs + rng.normal(0.0, r_obs, size=n)
    y_mean = float(y.mean())
    x_mean = float(ensemble.mean())
    cov_xy = float(np.dot(ensemble - x_mean, y - y_mean) / max(n - 1, 1))
    var_y = float(np.dot(y - y_mean, y - y_mean) / max(n - 1, 1)) + r_obs**2
    if var_y < 1e-12:
        return ensemble
    return ensemble + (cov_xy / var_y) * (d - y)


def driver_at(series: np.ndarray, t: int, fallback: float) -> float:
    if 0 <= t < len(series) and np.isfinite(series[t]):
        return float(series[t])
    return fallback


def run_enkf_backtest(
    h_c: np.ndarray,
    z_c: np.ndarray,
    z_f: np.ndarray,
    z_s: np.ndarray,
    rain: np.ndarray,
    mu_c: float,
    eval_mask: np.ndarray,
    params: dict,
    rng: np.random.Generator,
    horizons: list[int] = HORIZONS,
    n_ens: int = ENSEMBLE_N,
) -> dict[int, dict]:
    n = len(h_c)
    ens = np.zeros(n_ens, dtype=float)
    for i in range(n):
        if np.isfinite(z_c[i]):
            ens[:] = z_c[i]
            break

    bags: dict[int, list[tuple[float, float, float]]] = {h: [] for h in horizons}
    z_f_prev = 0.0
    z_s_prev = 0.0

    for t in range(n - max(horizons) - 1):
        zf = driver_at(z_f, t, z_f_prev)
        zs = driver_at(z_s, t, z_s_prev)
        rr = finite_rain(rain[t]) if t < n else 0.0
        # Always step (missing upstream → persist last anomaly) and always add Q.
        ens = process_step(ens, zf, z_f_prev, zs, z_s_prev, rr, params)
        ens = ens + rng.normal(0.0, PROCESS_NOISE, size=n_ens)
        if np.isfinite(z_f[t]):
            z_f_prev = zf
        if np.isfinite(z_s[t]):
            z_s_prev = zs
        if np.isfinite(z_c[t]):
            ens = enkf_update(ens, float(z_c[t]), OBS_NOISE, rng)

        if not eval_mask[t] or not np.isfinite(h_c[t]):
            continue
        # Need lagged upstream coverage over the longest horizon for a fair routing test.
        max_h = max(horizons)
        if not (
            np.isfinite(z_f[t : t + max_h + 1]).all()
            and np.isfinite(z_s[t : t + max_h + 1]).all()
        ):
            continue

        y0 = float(h_c[t])
        for h in horizons:
            target = t + h
            if target >= n or not np.isfinite(h_c[target]):
                continue
            members = ens.copy()
            # Filter already advanced with drivers[t]; forecast uses drivers[t+1 … t+h].
            prev_f = float(z_f[t])
            prev_s = float(z_s[t])
            for k in range(h):
                src = t + 1 + k
                zf = float(z_f[src])
                zs = float(z_s[src])
                # Future rain unknown: hold last observed intensity at t.
                rr = finite_rain(rain[t])
                members = process_step(members, zf, prev_f, zs, prev_s, rr, params)
                prev_f, prev_s = zf, zs
            pred = float(np.mean(members) + mu_c)
            bags[h].append((y0, float(h_c[target]), pred))

    out: dict[int, dict] = {}
    for h, rows in bags.items():
        if not rows:
            out[h] = {
                "all": {"rmse": None, "skill": None, "mae": None, "n": 0},
                "events": {"rmse": None, "skill": None, "mae": None, "n": 0},
            }
            continue
        arr = np.array(rows, dtype=float)
        y0, obs, pred = arr[:, 0], arr[:, 1], arr[:, 2]
        delta = obs - y0
        out[h] = {
            "all": metrics_block(y0, obs, pred),
            "events": metrics_block(y0[np.abs(delta) >= EVENT_ABS], obs[np.abs(delta) >= EVENT_ABS], pred[np.abs(delta) >= EVENT_ABS]),
        }
    return out


def load_dnn_skill120(coeffs_path: Path) -> float | None:
    if not coeffs_path.exists():
        return None
    data = json.loads(coeffs_path.read_text())
    node = (data.get("nodes") or {}).get(DOWNSTREAM) or {}
    skill = node.get("skill120")
    return float(skill) if skill is not None and math.isfinite(float(skill)) else None


def train_mean(series: np.ndarray, mask: np.ndarray) -> float:
    base = series[mask & np.isfinite(series)]
    return float(base.mean()) if len(base) else 0.0


def self_test() -> None:
    """Sanity checks for process stability and EnKF update (no I/O)."""
    rng = np.random.default_rng(0)
    params = {
        "A": 3.0,
        "C_f": 0.6,
        "C_s": 0.6,
        "C_f_down": 0.3,
        "C_s_down": 0.3,
        "K_f": 0.1,
        "K_s": 0.1,
        "C_out": 0.3,
        "alpha": 0.05,
    }
    z = np.array([0.5, 0.5, 0.5], dtype=float)
    for _ in range(40):
        z = process_step(z, 1.0, 0.8, 0.5, 0.4, 0.2, params)
    assert np.isfinite(z).all() and float(np.max(np.abs(z))) < 20.0, z
    ens = np.linspace(-0.2, 0.2, 20)
    upd = enkf_update(ens, 1.0, 0.05, rng)
    assert abs(float(upd.mean()) - 1.0) < 0.15, upd.mean()
    print("self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline EnKF spike for Cento")
    parser.add_argument("--data", type=Path, default=ROOT / "storico")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--coeffs", type=Path, default=ROOT / "models" / "transfer-coeffs.json")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "enkf-cento-summary.json")
    parser.add_argument("--trials", type=int, default=PARAM_TRIALS)
    parser.add_argument("--ensemble", type=int, default=ENSEMBLE_N)
    parser.add_argument("--every", type=int, default=EVAL_EVERY)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    config = json.loads(args.config.read_text())
    gauges = gauge_names(config)
    wide = ft.load_wide(args.data, STATIONS)
    wide, spike_counts = ft.filter_spikes(wide)
    missing = [s for s in STATIONS if s not in wide.columns]
    if missing:
        raise SystemExit(f"Stazioni mancanti: {missing}")

    rain_step = load_rain_step(args.data / "rain", gauges, wide.index)
    blocks = ft.month_blocks(wide.index)
    train_m = blocks == 0
    val_m = blocks == 1
    hold_m = blocks == 2

    h_c_raw = wide[DOWNSTREAM].to_numpy(dtype=float)
    h_f_raw = wide[UP_F].to_numpy(dtype=float)
    h_s_raw = wide[UP_S].to_numpy(dtype=float)

    lag_f, r_f = ft.estimate_lag(h_f_raw, h_c_raw, max_lag_steps=24, train_mask=train_m)
    lag_s, r_s = ft.estimate_lag(h_s_raw, h_c_raw, max_lag_steps=24, train_mask=train_m)

    h_f = np.roll(h_f_raw, lag_f)
    h_s = np.roll(h_s_raw, lag_s)
    h_f[:lag_f] = np.nan
    h_s[:lag_s] = np.nan
    rain = rain_step.to_numpy(dtype=float)
    h_c = h_c_raw

    mu_c = train_mean(h_c, train_m)
    mu_f = train_mean(h_f, train_m)
    mu_s = train_mean(h_s, train_m)
    z_c = h_c - mu_c
    z_f = h_f - mu_f
    z_s = h_s - mu_s

    rng = np.random.default_rng(RNG_SEED)
    params, train_rmse = fit_params(z_c, z_f, z_s, rain, train_m, rng, trials=args.trials)

    n = len(h_c)
    base_eval = np.zeros(n, dtype=bool)
    base_eval[:: max(args.every, 1)] = True
    eventish = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if np.isfinite(h_c[i]) and np.isfinite(h_c[i - 1]) and abs(h_c[i] - h_c[i - 1]) >= EVENT_ABS:
            eventish[max(0, i - 2) : min(n, i + 3)] = True
    val_eval = (val_m | hold_m) & (base_eval | eventish)

    metrics = run_enkf_backtest(
        h_c, z_c, z_f, z_s, rain, mu_c, val_eval, params, rng,
        horizons=HORIZONS, n_ens=args.ensemble,
    )

    dnn_skill = load_dnn_skill120(args.coeffs)
    if dnn_skill is None:
        dnn_skill = DNN_SKILL120_FALLBACK
    enkf_skill120 = (metrics.get(4) or {}).get("events", {}).get("skill")
    adopt = bool(
        enkf_skill120 is not None
        and dnn_skill is not None
        and enkf_skill120 >= dnn_skill
    )

    summary = {
        "version": 1,
        "kind": "cento_enkf_mass_transfer",
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
        "note": (
            "Anomaly-space reservoirs with upstream pulses + coupling. "
            "Gauges lack a common datum. A = effective storage; C_out/A = resistance."
        ),
        "downstream": DOWNSTREAM,
        "upstreams": [UP_F, UP_S],
        "stepMinutes": ft.STEP_MINUTES,
        "horizonsMinutes": [h * ft.STEP_MINUTES for h in HORIZONS],
        "ensembleN": args.ensemble,
        "processNoiseM": PROCESS_NOISE,
        "obsNoiseM": OBS_NOISE,
        "means": {"Cento": round(mu_c, 6), UP_F: round(mu_f, 6), UP_S: round(mu_s, 6)},
        "lags": {
            UP_F: {"steps": lag_f, "minutes": lag_f * ft.STEP_MINUTES, "r": None if r_f is None else round(r_f, 6)},
            UP_S: {"steps": lag_s, "minutes": lag_s * ft.STEP_MINUTES, "r": None if r_s is None else round(r_s, 6)},
        },
        "params": {k: round(float(v), 6) for k, v in params.items()},
        "trainOneStepRmse": round(train_rmse, 6),
        "spikeCounts": spike_counts,
        "metrics": {str(h * ft.STEP_MINUTES): metrics[h] for h in HORIZONS},
        "gate": {
            "dnnSkill120Events": round(dnn_skill, 6),
            "enkfSkill120Events": enkf_skill120,
            "adoptRuntime": adopt,
            "reason": (
                "enkf_beats_dnn"
                if adopt
                else "enkf_below_dnn_keep_mlp_apply_smooth"
            ),
        },
        "fittedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["gate"], indent=2))
    print(f"Wrote {args.out}")
    print(f"trainOneStepRmse={train_rmse:.4f} params={summary['params']}")
    for h in HORIZONS:
        ev = metrics[h]["events"]
        al = metrics[h]["all"]
        print(
            f"+{h * ft.STEP_MINUTES:3d}′  all skill={al['skill']} n={al['n']}  "
            f"events skill={ev['skill']} rmse={ev['rmse']} n={ev['n']}"
        )


if __name__ == "__main__":
    main()
