#!/usr/bin/env python3
"""Experimental KF / UKF / EnKF level forecasters for Cento.

Phase A ("da soli"): univariate state-space model.
  state s = [level, velocity]  (velocity = level increment per 30-min step)
  transition f(s):
      y+ = y + v
      v+ = v * (rho + gamma * (y - ymin))        # celerity grows with level
  (rho, gamma) fit by OLS on one-step pairs of the TRAIN months only.
  Q from one-step residuals; R = measurement noise (small grid, KF-selected
  on validation, shared by UKF/EnKF for a fair comparison).

Phase B ("con ingressi"): same transition plus exogenous inputs
  u = [filtered upstream levels..., rain6h, rain24h]
  v+ += B @ u ; input noise = upstream filtered/forecast variance mapped
  through B (rain treated as ~exact). Variance budget reported per horizon.

Filters: linear KF, unscented UKF (sigma points through f), stochastic EnKF.
Evaluation: same origins/splits as experiment-temporal; skill vs poly2 per
horizon via ml.eval_splits. Nothing touches production.
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

STEP_MIN = 30
H_STEPS = 12  # 360 min max horizon


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- model fit

def learn_transition(y: np.ndarray, train_mask: np.ndarray):
    """Fit rho, gamma on one-step velocity pairs within train months.

    v_t = y_t - y_{t-1}; model v_{t+1} = v_t * (rho + gamma*(y_t - ymin)).
    """
    finite = np.isfinite(y)
    d = np.full_like(y, np.nan)
    d[1:] = y[1:] - y[:-1]
    ok_v = finite & np.isfinite(d)
    ok_v[0] = False
    # pair at t: state (y_t, v_t=d_t) -> target v_{t+1}=d_{t+1}
    pair = ok_v & np.roll(ok_v, -1)
    pair[-1] = False
    pair &= train_mask
    t = np.flatnonzero(pair)
    t = t[(t >= 1) & (t + 1 < len(y))]
    v_t = y[t] - y[t - 1]
    v_tp1 = y[t + 1] - y[t]
    y_t = y[t]
    ymin = float(np.nanmin(y[train_mask & finite]))
    X = np.column_stack([v_t, v_t * (y_t - ymin)])
    A = np.linalg.lstsq(X, v_tp1, rcond=None)[0]
    rho, gamma = float(A[0]), float(A[1])
    resid = v_tp1 - X @ A
    q_v = float(np.var(resid)) if resid.size else 1e-4
    return rho, gamma, ymin, q_v, int(t.size)


def learn_B(y, U, train_mask, rho, gamma, ymin):
    """Fit input coefficients B (2 x n_in) on one-step residuals.

    v_{t+1} = v_t*(rho+gamma*(y_t-ymin)) + B_u @ u_t ; B_y row = 0.
    U[t] = inputs at departure time t (causal).
    """
    finite = np.isfinite(y)
    d = np.full_like(y, np.nan)
    d[1:] = y[1:] - y[:-1]
    ok_v = finite & np.isfinite(d)
    ok_v[0] = False
    pair = ok_v & np.roll(ok_v, -1)
    pair[-1] = False
    pair &= train_mask
    t = np.flatnonzero(pair)
    t = t[(t >= 1) & (t + 1 < len(y))]
    v_t = y[t] - y[t - 1]
    v_tp1 = y[t + 1] - y[t]
    base = v_t * (rho + gamma * (y[t] - ymin))
    resid = v_tp1 - base
    X_in = U[t]
    coef, *_ = np.linalg.lstsq(X_in, resid, rcond=None)
    B = np.zeros((2, U.shape[1]))
    B[1, :] = coef
    return B


def f_trans(s, rho, gamma, ymin):
    y = s[..., 0]
    v = s[..., 1]
    return np.stack([y + v, v * (rho + gamma * (y - ymin))], axis=-1)


def jac_trans(s, rho, gamma, ymin):
    y, v = float(s[0]), float(s[1])
    return np.array([
        [1.0, 1.0],
        [gamma * v, rho + gamma * (y - ymin)],
    ])


def f_with_inputs(s, rho, gamma, ymin, B, u):
    out = f_trans(s, rho, gamma, ymin)
    if u is None or B is None:
        return out
    u = np.asarray(u, dtype=float)
    add = B @ u if u.ndim == 1 else u @ B.T
    return out + add


# ---------------------------------------------------------------- filters

def kf_predict(m, P, rho, gamma, ymin, Q, B=None, u=None, Gamma=None):
    F = jac_trans(m, rho, gamma, ymin)
    m_p = f_with_inputs(m, rho, gamma, ymin, B, u)
    P_p = F @ P @ F.T + Q
    if Gamma is not None and B is not None:
        P_p = P_p + B @ Gamma @ B.T
    return m_p, P_p


def kf_update(m, P, y_obs, R):
    H = np.array([1.0, 0.0])
    S = float(H @ P @ H + R)
    if not np.isfinite(S) or S <= 0:
        return m, P
    K = (P @ H) / S
    m = m + K * (y_obs - float(H @ m))
    P = P - np.outer(K, H @ P)
    return m, 0.5 * (P + P.T)


def _sigma_pts(m, P, alpha=0.5, beta=2.0, kappa=0.0):
    n = 2
    lam = alpha * alpha * (n + kappa) - n
    c = n + lam
    try:
        L = np.linalg.cholesky(c * P)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(c * P + np.eye(n) * 1e-10)
    pts = np.empty((2 * n + 1, n))
    pts[0] = m
    pts[1] = m + L[:, 0]
    pts[2] = m - L[:, 0]
    pts[3] = m + L[:, 1]
    pts[4] = m - L[:, 1]
    wm = np.full(2 * n + 1, 1.0 / (2 * c))
    wc = wm.copy()
    wm[0] = lam / c
    wc[0] = lam / c + (1 - alpha * alpha + beta)
    return pts, wm, wc


def ukf_predict(m, P, rho, gamma, ymin, Q, B=None, u=None, Gamma=None):
    pts, wm, wc = _sigma_pts(m, P)
    sig = f_with_inputs(pts, rho, gamma, ymin, B, u)
    m_p = wm @ sig
    d = sig - m_p
    P_p = (d * wc[:, None]).T @ d + Q
    if Gamma is not None and B is not None:
        P_p = P_p + B @ Gamma @ B.T
    return m_p, 0.5 * (P_p + P_p.T)


def ukf_update(m, P, y_obs, R):
    pts, wm, wc = _sigma_pts(m, P)
    z = pts[:, 0]
    z_mean = wm @ z
    dz = z - z_mean
    S = float(wc @ (dz * dz)) + R
    if not np.isfinite(S) or S <= 0:
        return m, P
    dx = pts - m
    Pxz = (dx * wc[:, None]).T @ dz
    K = Pxz / S
    m = m + K * (y_obs - z_mean)
    P = P - np.outer(K, Pxz)
    return m, 0.5 * (P + P.T)


def enkf_predict(X, rho, gamma, ymin, Q, B=None, u=None, Gamma=None, rng=None):
    out = f_with_inputs(X, rho, gamma, ymin, B, u)
    if Gamma is not None and B is not None:
        G = Gamma + np.eye(Gamma.shape[0]) * 1e-12
        Lg = np.linalg.cholesky(G)
        eps = rng.standard_normal((X.shape[0], Gamma.shape[0])) @ Lg.T
        out = out + eps @ B.T
    Lq = np.linalg.cholesky(Q + np.eye(2) * 1e-12)
    out = out + rng.standard_normal((X.shape[0], 2)) @ Lq.T
    return out


def enkf_update(X, y_obs, R, rng):
    N = X.shape[0]
    z = X[:, 0]
    z_mean = z.mean()
    obs_pert = y_obs + np.sqrt(R) * rng.standard_normal(N)
    Xc = X - X.mean(axis=0)
    zc = z - z_mean
    Pxz = (Xc * zc[:, None]).sum(axis=0) / (N - 1)
    Pzz = float((zc * zc).sum() / (N - 1)) + R
    if not np.isfinite(Pzz) or Pzz <= 0:
        return X
    return X + np.outer(obs_pert - z, Pxz / Pzz)


# ---------------------------------------------------------------- driver

def forward_filter(y, rho, gamma, ymin, Q, R, kind, rng, B=None, U=None, Gamma_seq=None):
    """Causal pass; returns filtered mean (T,2) and cov diag (T,2)."""
    T = len(y)
    means = np.full((T, 2), np.nan)
    variances = np.full((T, 2), np.nan)
    finite_idx = np.flatnonzero(np.isfinite(y))
    if finite_idx.size == 0:
        return means, variances
    first = int(finite_idx[0])
    if kind == "enkf":
        N = 200
        X = np.tile(np.array([y[first], 0.0]), (N, 1))
        X[:, 0] += rng.standard_normal(N) * np.sqrt(R)
        X[:, 1] += rng.standard_normal(N) * np.sqrt(max(Q[1, 1], 1e-8))
        m = X.mean(axis=0)
        P = np.cov(X.T)
    else:
        m = np.array([y[first], 0.0])
        P = np.diag([R, max(Q[1, 1], 1e-8)])
    for t in range(first, T):
        if t > first:
            u = None if U is None else U[t - 1]
            G = None if Gamma_seq is None else Gamma_seq[t - 1]
            if kind == "kf":
                m, P = kf_predict(m, P, rho, gamma, ymin, Q, B=B, u=u, Gamma=G)
            elif kind == "ukf":
                m, P = ukf_predict(m, P, rho, gamma, ymin, Q, B=B, u=u, Gamma=G)
            else:
                X = enkf_predict(X, rho, gamma, ymin, Q, B=B, u=u, Gamma=G, rng=rng)
                m = X.mean(axis=0)
                P = np.cov(X.T)
        obs = y[t]
        if np.isfinite(obs):
            if kind == "kf":
                m, P = kf_update(m, P, obs, R)
            elif kind == "ukf":
                m, P = ukf_update(m, P, obs, R)
            else:
                X = enkf_update(X, obs, R, rng)
                m = X.mean(axis=0)
                P = np.cov(X.T)
        means[t] = m
        variances[t] = np.maximum(np.diag(P), 0.0)
    return means, variances


def rollout(mean0, cov0, rho, gamma, ymin, Q, h, kind, rng,
            B=None, input_means=None, input_covs=None):
    """Forecast h steps, no updates. Returns level path (h+1,) and var path."""
    m = np.asarray(mean0, dtype=float).copy()
    P = np.asarray(cov0, dtype=float).copy()
    if kind == "enkf":
        N = 200
        try:
            L = np.linalg.cholesky(cov0 + np.eye(2) * 1e-12)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov0 + np.eye(2) * 1e-9)
        X = m[None, :] + rng.standard_normal((N, 2)) @ L.T
        lvl = [float(m[0])]
        var = [float(P[0, 0])]
        for k in range(h):
            u = None if input_means is None else input_means[k]
            G = None if input_covs is None else input_covs[k]
            X = enkf_predict(X, rho, gamma, ymin, Q, B=B, u=u, Gamma=G, rng=rng)
            lvl.append(float(X[:, 0].mean()))
            var.append(float(X[:, 0].var(ddof=1)))
        return np.array(lvl), np.array(var)
    lvl = [float(m[0])]
    var = [float(P[0, 0])]
    for k in range(h):
        u = None if input_means is None else input_means[k]
        G = None if input_covs is None else input_covs[k]
        if kind == "kf":
            m, P = kf_predict(m, P, rho, gamma, ymin, Q, B=B, u=u, Gamma=G)
        else:
            m, P = ukf_predict(m, P, rho, gamma, ymin, Q, B=B, u=u, Gamma=G)
        lvl.append(float(m[0]))
        var.append(float(P[0, 0]))
    return np.array(lvl), np.array(var)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "storico")
    p.add_argument("--coeffs", type=Path, default=ROOT / "models" / "transfer-coeffs.json")
    p.add_argument("--out", type=Path, default=ROOT / "models" / "experiments" / "kalman.json")
    p.add_argument("--from", dest="from_ts", default="2020-01-01")
    p.add_argument("--to", dest="to_ts", default=None)
    p.add_argument("--holdout-from", default="2026-01-01")
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--max-origins", type=int, default=None)
    p.add_argument("--r-grid", type=float, nargs="*", default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--q-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    coeffs = json.loads(args.coeffs.read_text(encoding="utf-8"))
    node = coeffs["nodes"]["Cento"]
    stations = list(node["stations"])
    downstream = node.get("downstream", "Cento")
    upstreams = list(node.get("upstreams") or [s for s in stations if s != downstream])
    log(f"stations={stations} upstreams={upstreams} downstream={downstream}")

    wide = ft.load_wide(args.data, stations)
    from_ts = pd.Timestamp(args.from_ts, tz="UTC")
    to_ts = pd.Timestamp(args.to_ts, tz="UTC") if args.to_ts else wide.index.max()
    cand = wide.index[(wide.index >= from_ts) & (wide.index <= to_ts)]
    origins = cand[:: max(1, args.stride)]
    if args.max_origins:
        origins = origins[: args.max_origins]
    log(f"origins={len(origins)} {origins[0]} → {origins[-1]}")

    tr_full, _, _ = ml.split_masks(wide.index, mode="block", holdout_from=args.holdout_from)
    series = {s: wide[s].to_numpy(dtype=float) for s in stations}
    trans = {}
    for s in stations:
        rho, gamma, ymin, q_v, npair = learn_transition(series[s], tr_full)
        trans[s] = dict(rho=rho, gamma=gamma, ymin=ymin, q_v=q_v)
        log(f"{s}: rho={rho:.4f} gamma={gamma:.3e} ymin={ymin:.3f} q_v={q_v:.3e} pairs={npair}")

    def est_R(y):
        d = np.diff(y[np.isfinite(y)])
        d = d[np.isfinite(d)]
        if d.size < 100:
            return 1e-3
        mad = np.median(np.abs(d - np.median(d)))
        return float(max((1.4826 * mad) ** 2 / 2.0, 1e-6))

    R_est = {s: est_R(series[s]) for s in stations}
    for s in stations:
        log(f"{s}: R_est={R_est[s]:.3e}")

    def make_Q(s, q_scale):
        return np.diag([1e-8, max(trans[s]["q_v"] * q_scale, 1e-10)])

    origins_pos = np.array([int(wide.index.get_loc(t)) for t in origins])
    tr_m, va_m, ho_m = ml.split_masks(origins, mode="block", holdout_from=args.holdout_from)
    level_series = wide[downstream]
    y0_all = level_series.reindex(origins).to_numpy(dtype=float)
    targets_by_h = {h: ml.build_targets(wide, h) for h in range(1, H_STEPS + 1)}
    deltas = {h: targets_by_h[h]["delta"].reindex(origins).to_numpy(dtype=float)
              for h in range(1, H_STEPS + 1)}
    poly_base = {h: ml.build_poly2_baseline(level_series, origins.to_numpy(), h, n_points=6)
                 for h in range(1, H_STEPS + 1)}

    kinds = ["kf", "ukf", "enkf"]
    results = {}

    # ---------- R selection once, with KF phase A on validation @120 ----------
    log("=== R selection (KF, phase A, val @120) ===")
    Q_c = make_Q(downstream, args.q_scale)
    tr_c = trans[downstream]
    sel_origins = np.flatnonzero(va_m)[::4]  # subset for speed
    best = None
    for R_try in args.r_grid:
        means, variances = forward_filter(
            series[downstream], tr_c["rho"], tr_c["gamma"], tr_c["ymin"],
            Q_c, R_try, "kf", rng)
        se, n = 0.0, 0
        for i in sel_origins:
            pos = origins_pos[i]
            if not np.isfinite(means[pos, 0]):
                continue
            P0 = np.diag(np.nan_to_num(variances[pos], nan=1e-6))
            lvl, _ = rollout(means[pos], P0, tr_c["rho"], tr_c["gamma"],
                             tr_c["ymin"], Q_c, 4, "kf", rng)
            d = deltas[4][i]
            if np.isfinite(d) and np.isfinite(y0_all[i]):
                se += (lvl[4] - y0_all[i] - d) ** 2
                n += 1
        rmse = float(np.sqrt(se / max(n, 1)))
        log(f"  R={R_try:.0e} valRMSE@120={rmse:.4f} (n={n})")
        if best is None or rmse < best[0]:
            best = (rmse, R_try)
    R_sel = best[1] if best else args.r_grid[0]
    log(f"selected R={R_sel:.0e}")

    # ---------- helper: evaluate one forecast table ----------
    def evaluate(level_paths):
        """level_paths: (n_origins, H_STEPS+1) forecast levels (col 0 = y0 model).
        Returns per-horizon eval dicts."""
        out = {}
        for h in range(1, H_STEPS + 1):
            minutes = h * STEP_MIN
            pred = level_paths[:, h] - y0_all
            ev = ml.eval_splits(y0_all, deltas[h], pred, tr_m, va_m, ho_m,
                                poly_baseline=poly_base[h])
            out[str(minutes)] = {"horizonMinutes": minutes, "dnn": ev}
        return out

    # ================= PHASE A =================
    log("=== PHASE A: univariate filters ===")
    for kind in kinds:
        t0 = time.perf_counter()
        means, variances = forward_filter(
            series[downstream], tr_c["rho"], tr_c["gamma"], tr_c["ymin"],
            Q_c, R_sel, kind, rng)
        log(f"{kind} filter pass {time.perf_counter()-t0:.0f}s")
        paths = np.full((len(origins), H_STEPS + 1), np.nan)
        t0 = time.perf_counter()
        for i, pos in enumerate(origins_pos):
            if not np.isfinite(means[pos, 0]):
                continue
            P0 = np.diag(np.nan_to_num(variances[pos], nan=1e-6))
            lvl, _ = rollout(means[pos], P0, tr_c["rho"], tr_c["gamma"],
                             tr_c["ymin"], Q_c, H_STEPS, kind, rng)
            paths[i] = lvl
        log(f"{kind} rollouts {time.perf_counter()-t0:.0f}s")
        per_h = evaluate(paths)
        for h in (4, 8, 12):
            ev = per_h[str(h * STEP_MIN)]["dnn"]
            log(f"  +{h*STEP_MIN}′ valEvt={ev['validateEvents'].get('skill')} "
                f"holdEvt={ev['holdoutEvents'].get('skill')}")
        results[f"{kind}_A"] = {"R": R_sel, "horizons": per_h}
        del means, variances, paths

    # ================= PHASE B =================
    log("=== PHASE B: filters with upstream+rain inputs ===")
    gauges = [g.get("name") if isinstance(g, dict) else g for g in
              json.loads((ROOT / "config.json").read_text())["rainfall"]["gauges"]]
    rain = ft.load_rain_basin(args.data / "rain", gauges, wide.index) \
        if (args.data / "rain").exists() else None
    rain_cols = ["rain6h", "rain24h"] if rain is not None else []
    input_names = [f"lvl_{s}" for s in upstreams] + rain_cols
    n_in = len(input_names)
    log(f"inputs={input_names}")

    # upstream univariate KF filtered levels + variances (input signals)
    up_state = {}
    for s in upstreams:
        Qs = make_Q(s, args.q_scale)
        t0 = time.perf_counter()
        m_arr, v_arr = forward_filter(
            series[s], trans[s]["rho"], trans[s]["gamma"], trans[s]["ymin"],
            Qs, R_est[s], "kf", rng)
        up_state[s] = (m_arr, v_arr)
        log(f"upstream filter {s} ({time.perf_counter()-t0:.0f}s)")

    # U[t] = inputs at time t (levels at departure) -> transition t->t+1 uses U[t]
    # forward_filter uses U[t-1] when predicting t: consistent (departure inputs).
    U = np.zeros((len(wide.index), n_in))
    for j, s in enumerate(upstreams):
        U[:, j] = up_state[s][0][:, 0]
    if rain is not None:
        for j, col in enumerate(rain_cols):
            U[:, len(upstreams) + j] = rain[col].reindex(wide.index).to_numpy(dtype=float)
    U = np.nan_to_num(U, nan=0.0)

    B = learn_B(series[downstream], U, tr_full, tr_c["rho"], tr_c["gamma"], tr_c["ymin"])
    log(f"B (v-row) = {np.round(B[1], 5).tolist()} inputs={input_names}")

    # input noise during filtering: var of upstream filtered level at departure
    Gamma_seq = np.zeros((len(wide.index), n_in, n_in))
    for j, s in enumerate(upstreams):
        Gamma_seq[:, j, j] = np.clip(np.nan_to_num(up_state[s][1][:, 0], nan=R_est[s]), 1e-8, None)
    for j in range(len(upstreams), n_in):
        Gamma_seq[:, j, j] = 1e-6  # rain ~ exact

    for kind in kinds:
        t0 = time.perf_counter()
        means, variances = forward_filter(
            series[downstream], tr_c["rho"], tr_c["gamma"], tr_c["ymin"],
            Q_c, R_sel, kind, rng, B=B, U=U, Gamma_seq=Gamma_seq)
        log(f"{kind}B filter pass {time.perf_counter()-t0:.0f}s")

        # per-origin upstream forecast paths (univariate KF, h=H_STEPS)
        paths = np.full((len(origins), H_STEPS + 1), np.nan)
        var_paths = np.full((len(origins), H_STEPS + 1), np.nan)
        t0 = time.perf_counter()
        for i, pos in enumerate(origins_pos):
            if not np.isfinite(means[pos, 0]):
                continue
            # upstream forecast mean level & var per step k=0..H_STEPS
            in_means = np.zeros((H_STEPS, n_in))
            in_covs = np.zeros((H_STEPS, n_in, n_in))
            for j, s in enumerate(upstreams):
                m_arr, v_arr = up_state[s]
                trs = trans[s]
                Qs = make_Q(s, args.q_scale)
                mu, vu = m_arr[pos], v_arr[pos]
                if not np.isfinite(mu[0]):
                    continue
                P0u = np.diag(np.nan_to_num(vu, nan=1e-6))
                lvl_u, var_u = rollout(mu, P0u, trs["rho"], trs["gamma"],
                                       trs["ymin"], Qs, H_STEPS, "kf", rng)
                in_means[:, j] = lvl_u[:H_STEPS]
                in_covs[:, j, j] = np.clip(var_u[:H_STEPS], 1e-8, None)
            for j in range(len(upstreams), n_in):
                # rain inputs: persist last observed value, var = column variance
                col = U[:, j]
                last = col[:pos + 1]
                last = last[np.isfinite(last)]
                in_means[:, j] = last[-1] if last.size else 0.0
                in_covs[:, j, j] = max(float(np.nanvar(col)), 1e-6)
            P0 = np.diag(np.nan_to_num(variances[pos], nan=1e-6))
            lvl, var = rollout(means[pos], P0, tr_c["rho"], tr_c["gamma"],
                               tr_c["ymin"], Q_c, H_STEPS, kind, rng,
                               B=B, input_means=in_means, input_covs=in_covs)
            paths[i] = lvl
            var_paths[i] = var
        log(f"{kind}B rollouts {time.perf_counter()-t0:.0f}s")
        per_h = evaluate(paths)
        for h in (4, 8, 12):
            ev = per_h[str(h * STEP_MIN)]["dnn"]
            log(f"  +{h*STEP_MIN}′ valEvt={ev['validateEvents'].get('skill')} "
                f"holdEvt={ev['holdoutEvents'].get('skill')}")
        budget = {"120": float(np.nanmedian(var_paths[:, 4])),
                  "360": float(np.nanmedian(var_paths[:, 12]))}
        results[f"{kind}_B"] = {"R": R_sel, "inputs": input_names,
                                "B": B.tolist(), "horizons": per_h,
                                "medianForecastVar": budget}
        del means, variances, paths

    out = {
        "version": 1,
        "kind": "cento_kalman_experiment",
        "trainedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "downstream": downstream,
        "upstreams": upstreams,
        "inputs": input_names,
        "origins": {"from": str(origins[0]), "to": str(origins[-1]), "n": int(len(origins))},
        "holdoutFrom": args.holdout_from,
        "transitions": trans,
        "R_est": R_est,
        "R_selected": R_sel,
        "qScale": args.q_scale,
        "results": results,
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
