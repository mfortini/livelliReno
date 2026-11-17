#!/usr/bin/env python3
"""Global (joint-state) KF / UKF / EnKF for the Reno chain — Cento target.

All stations share one state vector s = [y1,v1, ..., y5,v5] (dim 2*n_stations).
Dynamics are block-diagonal: each station keeps its own learned transition
  y+ = y + v ;  v+ = v*(rho + gamma*(y - ymin))
but the PROCESS NOISE covariance Q is FULL: cross-station residual covariances
are estimated on train months, so a measurement at one station updates the
whole state (information propagates through correlated innovations).
No exogenous inputs — pure model, as requested.

Observation at time t: each station's level (scalar update per station,
missing -> skipped). Forecast: joint rollout h steps, Cento level extracted.

Evaluation identical to experiment-kalman.py (same origins/splits/poly2).
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
ek = _load("experiment_kalman", "experiment-kalman.py")

STEP_MIN = 30
H_STEPS = 12


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def station_residuals(y, train_mask):
    """One-step velocity residuals e_t = v_{t+1} - f_v(state_t) on train pairs."""
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
    return t


def learn_Q_cross(series_list, trans_list, train_mask):
    """Full process-noise covariance on velocity components.

    Returns E (T, n_st) residual matrix (NaN where unavailable) and indices.
    """
    n_st = len(series_list)
    T = len(series_list[0])
    E = np.full((T, n_st), np.nan)
    for i, (y, tr) in enumerate(zip(series_list, trans_list)):
        t = station_residuals(y, train_mask)
        v_t = y[t] - y[t - 1]
        v_tp1 = y[t + 1] - y[t]
        base = v_t * (tr["rho"] + tr["gamma"] * (y[t] - tr["ymin"]))
        E[t, i] = v_tp1 - base
    return E


def cov_pairwise(E):
    """Covariance with pairwise-complete observations."""
    n = E.shape[1]
    C = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            m = np.isfinite(E[:, i]) & np.isfinite(E[:, j])
            if m.sum() < 30:
                C[i, j] = C[j, i] = 0.0
            else:
                c = float(np.cov(E[m, i], E[m, j])[0, 1])
                C[i, j] = C[j, i] = c
    return C


def build_Q_full(E, q_scale=1.0):
    """10x10 (or 2n) process covariance: zeros on level rows/cols,
    cross-cov of velocity residuals on velocity rows/cols."""
    n_st = E.shape[1]
    n = 2 * n_st
    Cv = cov_pairwise(E) * q_scale
    # regularize: ensure PSD-ish
    Cv = 0.5 * (Cv + Cv.T)
    Q = np.zeros((n, n))
    for i in range(n_st):
        for j in range(n_st):
            Q[2 * i + 1, 2 * j + 1] = Cv[i, j]
    # tiny level noise for numerical stability
    for i in range(n_st):
        Q[2 * i, 2 * i] = 1e-8
    return Q, Cv


def f_joint(s, trans_list):
    """s (..., 2*n_st) -> next state, block-diagonal per station."""
    shp = s.shape[:-1]
    r = s.reshape(shp + (len(trans_list), 2))
    out = np.empty_like(r)
    for i, tr in enumerate(trans_list):
        out[..., i, :] = ek.f_trans(r[..., i, :], tr["rho"], tr["gamma"], tr["ymin"])
    return out.reshape(s.shape)


def F_joint(m, trans_list):
    """Block-diagonal Jacobian at state m (n,)."""
    n_st = len(trans_list)
    n = 2 * n_st
    F = np.zeros((n, n))
    for i, tr in enumerate(trans_list):
        Ji = ek.jac_trans(m[2 * i:2 * i + 2], tr["rho"], tr["gamma"], tr["ymin"])
        F[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Ji
    return F


def kf_predict_g(m, P, trans_list, Q):
    F = F_joint(m, trans_list)
    m_p = f_joint(m, trans_list)
    return m_p, F @ P @ F.T + Q


def kf_update_g(m, P, idx, y_obs, R):
    """Scalar update observing station idx level (state index 2*idx)."""
    H = np.zeros(P.shape[0])
    H[2 * idx] = 1.0
    S = float(H @ P @ H + R)
    if not np.isfinite(S) or S <= 0:
        return m, P
    K = (P @ H) / S
    m = m + K * (y_obs - m[2 * idx])
    P = P - np.outer(K, H @ P)
    return m, 0.5 * (P + P.T)


def ukf_predict_g(m, P, trans_list, Q, alpha=0.5, beta=2.0, kappa=0.0):
    n = m.size
    lam = alpha * alpha * (n + kappa) - n
    c = n + lam
    try:
        L = np.linalg.cholesky(c * P)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(c * P + np.eye(n) * 1e-9)
    pts = np.empty((2 * n + 1, n))
    pts[0] = m
    for i in range(n):
        pts[1 + 2 * i] = m + L[:, i]
        pts[2 + 2 * i] = m - L[:, i]
    wm = np.full(2 * n + 1, 1.0 / (2 * c))
    wc = wm.copy()
    wm[0] = lam / c
    wc[0] = lam / c + (1 - alpha * alpha + beta)
    sig = f_joint(pts, trans_list)
    m_p = wm @ sig
    d = sig - m_p
    P_p = (d * wc[:, None]).T @ d + Q
    return m_p, 0.5 * (P_p + P_p.T)


def ukf_update_g(m, P, idx, y_obs, R):
    n = m.size
    alpha, beta, kappa = 0.5, 2.0, 0.0
    lam = alpha * alpha * (n + kappa) - n
    c = n + lam
    try:
        L = np.linalg.cholesky(c * P)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(c * P + np.eye(n) * 1e-9)
    pts = np.empty((2 * n + 1, n))
    pts[0] = m
    for i in range(n):
        pts[1 + 2 * i] = m + L[:, i]
        pts[2 + 2 * i] = m - L[:, i]
    wm = np.full(2 * n + 1, 1.0 / (2 * c))
    wc = wm.copy()
    wm[0] = lam / c
    wc[0] = lam / c + (1 - alpha * alpha + beta)
    z = pts[:, 2 * idx]
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


def enkf_predict_g(X, trans_list, Q, rng):
    out = f_joint(X, trans_list)
    Lq = np.linalg.cholesky(Q + np.eye(Q.shape[0]) * 1e-10)
    return out + rng.standard_normal(X.shape) @ Lq.T


def enkf_update_g(X, idx, y_obs, R, rng):
    N = X.shape[0]
    z = X[:, 2 * idx]
    z_mean = z.mean()
    obs_pert = y_obs + np.sqrt(R) * rng.standard_normal(N)
    Xc = X - X.mean(axis=0)
    zc = z - z_mean
    Pxz = (Xc * zc[:, None]).sum(axis=0) / (N - 1)
    Pzz = float((zc * zc).sum() / (N - 1)) + R
    if not np.isfinite(Pzz) or Pzz <= 0:
        return X
    return X + np.outer(obs_pert - z, Pxz / Pzz)


def forward_filter_g(Y, trans_list, Q, R_vec, kind, rng, station_means):
    """Y: (T, n_st) observed levels (NaN missing). Returns mean (T,n), var (T,n).

    station_means: per-station climatological mean for unobserved init.
    """
    T, n_st = Y.shape
    n = 2 * n_st
    means = np.full((T, n), np.nan)
    variances = np.full((T, n), np.nan)
    # init at first row where downstream (last col? use col 0 = Cento order)
    # stations order = list order; downstream index passed via trans order —
    # caller ensures Cento is index 0? We'll init when ANY station observed and
    # lazily set unobserved stations to climatology with large variance.
    first = int(np.flatnonzero(np.isfinite(Y).any(axis=1))[0])
    m0 = np.zeros(n)
    P0 = np.eye(n) * 1e-6
    for i in range(n_st):
        m0[2 * i] = station_means[i]
        m0[2 * i + 1] = 0.0
        P0[2 * i, 2 * i] = 1.0
        P0[2 * i + 1, 2 * i + 1] = max(Q[2 * i + 1, 2 * i + 1], 1e-6)
    if kind == "enkf":
        N = 200
        X = m0[None, :] + rng.standard_normal((N, n)) @ np.linalg.cholesky(P0 + np.eye(n) * 1e-10).T
    else:
        m, P = m0.copy(), P0.copy()
    for t in range(first, T):
        if t > first:
            if kind == "kf":
                m, P = kf_predict_g(m, P, trans_list, Q)
            elif kind == "ukf":
                m, P = ukf_predict_g(m, P, trans_list, Q)
            else:
                X = enkf_predict_g(X, trans_list, Q, rng)
                m = X.mean(axis=0)
                P = np.cov(X.T)
        row = Y[t]
        for i in range(n_st):
            obs = row[i]
            if not np.isfinite(obs):
                continue
            R = R_vec[i]
            if kind == "kf":
                m, P = kf_update_g(m, P, i, obs, R)
            elif kind == "ukf":
                m, P = ukf_update_g(m, P, i, obs, R)
            else:
                X = enkf_update_g(X, i, obs, R, rng)
                m = X.mean(axis=0)
                P = np.cov(X.T)
        means[t] = m
        variances[t] = np.maximum(np.diag(P), 0.0)
    return means, variances


# ------------------------------------------------- lagged cross coupling
# edges: (target_station, src_station, tau_steps) — tau from wave-warp.json
# median lags (Sostegno->Cento is composite Sostegno->Bagnetto->Cento ~510').
COUPLE_EDGES = [
    ("Bonconvento", "Casalecchio tiro a volo", 9),    # 270'
    ("Sostegno Reno", "Bonconvento", 7),             # 210'
    ("Cento", "Sostegno Reno", 17),                  # ~510' composite
    ("Cento", "Forcelli", 16),                       # 480'
]


def learn_coupling(series_list, stations, trans_list, train_mask, edges, uni_levels):
    """Learn per-target coupling coefficients on train one-step residuals.

    resid_i(t) = v_i(t+1) - base_i(t) ~ sum_e c_e * dz_e(t)
    dz_e(t) = uni_lvl_src[t-tau] - uni_lvl_src[t-tau-1]   (departure-time input)
    Returns C: dict target_idx -> list of (src_idx, tau, coef).
    """
    idx_of = {s: i for i, s in enumerate(stations)}
    per_target = {}
    for tgt, src, tau in edges:
        per_target.setdefault(idx_of[tgt], []).append((idx_of[src], tau))
    C = {}
    for i, y in enumerate(series_list):
        edges_i = per_target.get(i)
        if not edges_i:
            continue
        tr = trans_list[i]
        t_idx = station_residuals(y, train_mask)
        t_idx = t_idx[(t_idx >= 1) & (t_idx + 1 < len(y))]
        v_t = y[t_idx] - y[t_idx - 1]
        v_tp1 = y[t_idx + 1] - y[t_idx]
        base = v_t * (tr["rho"] + tr["gamma"] * (y[t_idx] - tr["ymin"]))
        resid = v_tp1 - base
        cols = []
        for src_idx, tau in edges_i:
            lvl = uni_levels[src_idx]
            ti = t_idx - tau
            dz = np.full(len(t_idx), np.nan)
            ok = (ti >= 1) & (ti - 1 >= 0) & (ti < len(lvl))
            dz[ok] = lvl[ti[ok]] - lvl[ti[ok] - 1]
            cols.append(dz)
        X = np.column_stack(cols)
        ok = np.isfinite(resid) & np.isfinite(X).all(axis=1)
        if ok.sum() < 100:
            continue
        coef, *_ = np.linalg.lstsq(X[ok], resid[ok], rcond=None)
        C[i] = [(src_idx, tau, float(c)) for (src_idx, tau), c in zip(edges_i, coef)]
        log(f"coupling into {stations[i]}: " +
            ", ".join(f"{stations[s]}(tau={t})={c:.4f}" for s, t, c in C[i]))
    return C


def coupling_add(m_next, C, lvl_at, depart_ti):
    """Add coupling increments to velocity rows of next-state array.

    m_next: (..., n) states; lvl_at(src_idx, abs_t) -> level or NaN;
    depart_ti: absolute timeline index of departure (predicting -> +1).
    For each target i: v_i+ += sum c_e * (lvl(src, ti-tau) - lvl(src, ti-tau-1)).
    """
    n_st = m_next.shape[-1] // 2
    u = np.zeros(n_st)
    for i, edges in C.items():
        acc = 0.0
        for src_idx, tau, c in edges:
            a = lvl_at(src_idx, depart_ti - tau)
            b = lvl_at(src_idx, depart_ti - tau - 1)
            if np.isfinite(a) and np.isfinite(b):
                acc += c * (a - b)
        u[i] = acc
    # u indexed by station; state layout interleaved [y0,v0,y1,v1,...]
    out = m_next.copy()
    for i in range(n_st):
        out[..., 2 * i + 1] += u[i]
    return out


def rollout_g(mean0, cov0, trans_list, Q, h, kind, rng):
    """Joint rollout; returns Cento-indexed caller extracts. Returns (h+1, n), (h+1, n)."""
    m = np.asarray(mean0, dtype=float).copy()
    P = np.asarray(cov0, dtype=float).copy()
    n = m.size
    if kind == "enkf":
        N = 200
        try:
            L = np.linalg.cholesky(cov0 + np.eye(n) * 1e-10)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov0 + np.eye(n) * 1e-8)
        X = m[None, :] + rng.standard_normal((N, n)) @ L.T
        path_m = [m.copy()]
        path_v = [np.diag(P).copy()]
        for _ in range(h):
            X = enkf_predict_g(X, trans_list, Q, rng)
            path_m.append(X.mean(axis=0))
            path_v.append(X.var(axis=0, ddof=1))
        return np.array(path_m), np.array(path_v)
    path_m = [m.copy()]
    path_v = [np.diag(P).copy()]
    for _ in range(h):
        if kind == "kf":
            m, P = kf_predict_g(m, P, trans_list, Q)
        else:
            m, P = ukf_predict_g(m, P, trans_list, Q)
        path_m.append(m.copy())
        path_v.append(np.diag(P).copy())
    return np.array(path_m), np.array(path_v)


# ------------------------------------------- coupled (lagged cross-links)

def _u_scalar(C, lvl_at, depart_ti, n_st):
    u = np.zeros(n_st)
    for i, edges in C.items():
        acc = 0.0
        for src_idx, tau, c in edges:
            a = lvl_at(src_idx, depart_ti - tau)
            b = lvl_at(src_idx, depart_ti - tau - 1)
            if np.isfinite(a) and np.isfinite(b):
                acc += c * (a - b)
        u[i] = acc
    return u


def _add_u(state, u):
    """state (..., n) or (N, n): add u_i to each velocity component."""
    out = state.copy()
    n_st = u.size
    for i in range(n_st):
        out[..., 2 * i + 1] += u[i]
    return out


def forward_filter_gc(Y, trans_list, Q, R_vec, kind, rng, station_means, C):
    """Joint filter with lagged cross-coupling.

    Coupling inputs come from this filter's own filtered level history
    (means[ti, 2j]) — causal, self-consistent.
    """
    T, n_st = Y.shape
    n = 2 * n_st
    means = np.full((T, n), np.nan)
    variances = np.full((T, n), np.nan)
    first = int(np.flatnonzero(np.isfinite(Y).any(axis=1))[0])
    m0 = np.zeros(n)
    P0 = np.eye(n) * 1e-6
    for i in range(n_st):
        m0[2 * i] = station_means[i]
        P0[2 * i, 2 * i] = 1.0
        P0[2 * i + 1, 2 * i + 1] = max(Q[2 * i + 1, 2 * i + 1], 1e-6)
    if kind == "enkf":
        N = 200
        X = m0[None, :] + rng.standard_normal((N, n)) @ np.linalg.cholesky(P0 + np.eye(n) * 1e-10).T
    else:
        m, P = m0.copy(), P0.copy()

    def lvl_at(j, ti):
        if ti < first or ti >= T:
            return np.nan
        return means[ti, 2 * j]

    for t in range(first, T):
        if t > first:
            depart = t - 1
            u = _u_scalar(C, lvl_at, depart, n_st)
            if kind == "kf":
                m, P = kf_predict_g(m, P, trans_list, Q)
                m = _add_u(m, u)
            elif kind == "ukf":
                m, P = ukf_predict_g(m, P, trans_list, Q)
                m = _add_u(m, u)
            else:
                X = enkf_predict_g(X, trans_list, Q, rng)
                X = _add_u(X, u)
                m = X.mean(axis=0)
                P = np.cov(X.T)
        row = Y[t]
        for i in range(n_st):
            obs = row[i]
            if not np.isfinite(obs):
                continue
            R = R_vec[i]
            if kind == "kf":
                m, P = kf_update_g(m, P, i, obs, R)
            elif kind == "ukf":
                m, P = ukf_update_g(m, P, i, obs, R)
            else:
                X = enkf_update_g(X, i, obs, R, rng)
                m = X.mean(axis=0)
                P = np.cov(X.T)
        means[t] = m
        variances[t] = np.maximum(np.diag(P), 0.0)
    return means, variances


def rollout_gc(mean0, cov0, trans_list, Q, h, kind, rng, C, hist_means, pos):
    """Coupled joint rollout from origin index pos.

    lvl lookup: ti < pos -> hist_means[ti] (filtered, causal);
                ti >= pos -> this rollout's own path (per member for EnKF).
    Returns Cento level path handled by caller via full state path:
    actually returns (path_m (h+1,n), path_v (h+1,n)).
    """
    n = mean0.size
    n_st = n // 2
    m = np.asarray(mean0, dtype=float).copy()
    P = np.asarray(cov0, dtype=float).copy()
    T_hist = hist_means.shape[0]

    if kind == "enkf":
        N = 200
        try:
            L = np.linalg.cholesky(cov0 + np.eye(n) * 1e-10)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(cov0 + np.eye(n) * 1e-8)
        X = m[None, :] + rng.standard_normal((N, n)) @ L.T
        X_hist = [X.copy()]
        path_m = [m.copy()]
        path_v = [np.diag(P).copy()]

        def lvl_vec(j, ti):
            """Level of station j at absolute index ti: scalar (past) or (N,) vector."""
            if ti < pos:
                if 0 <= ti < T_hist:
                    v = hist_means[ti, 2 * j]
                    return v if np.isfinite(v) else np.nan
                return np.nan
            s = ti - pos
            if 0 <= s < len(X_hist):
                return X_hist[s][:, 2 * j]
            return np.nan

        for k in range(h):
            depart = pos + k
            # per-member coupling
            U = np.zeros((N, n_st))
            for i, edges in C.items():
                acc = np.zeros(N)
                for src_idx, tau, c in edges:
                    a = lvl_vec(src_idx, depart - tau)
                    b = lvl_vec(src_idx, depart - tau - 1)
                    a_arr = np.broadcast_to(np.asarray(a, dtype=float), (N,))
                    b_arr = np.broadcast_to(np.asarray(b, dtype=float), (N,))
                    ok = np.isfinite(a_arr) & np.isfinite(b_arr)
                    acc[ok] += c * (a_arr[ok] - b_arr[ok])
                U[:, i] = acc
            X = enkf_predict_g(X, trans_list, Q, rng)
            for i in range(n_st):
                X[:, 2 * i + 1] += U[:, i]
            X_hist.append(X.copy())
            path_m.append(X.mean(axis=0))
            path_v.append(X.var(axis=0, ddof=1))
        return np.array(path_m), np.array(path_v)

    path_m = [m.copy()]
    path_v = [np.diag(P).copy()]

    def lvl_at(j, ti):
        if ti < pos:
            return hist_means[ti, 2 * j] if 0 <= ti < T_hist else np.nan
        s = ti - pos
        if s < 0 or s >= len(path_m):
            return np.nan
        return path_m[s][2 * j]

    for k in range(h):
        depart = pos + k
        u = _u_scalar(C, lvl_at, depart, n_st)
        if kind == "kf":
            m, P = kf_predict_g(m, P, trans_list, Q)
        else:
            m, P = ukf_predict_g(m, P, trans_list, Q)
        m = _add_u(m, u)
        path_m.append(m.copy())
        path_v.append(np.diag(P).copy())
    return np.array(path_m), np.array(path_v)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT / "storico")
    p.add_argument("--coeffs", type=Path, default=ROOT / "models" / "transfer-coeffs.json")
    p.add_argument("--out", type=Path, default=ROOT / "models" / "experiments" / "kalman-global.json")
    p.add_argument("--from", dest="from_ts", default="2020-01-01")
    p.add_argument("--to", dest="to_ts", default=None)
    p.add_argument("--holdout-from", default="2026-01-01")
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--max-origins", type=int, default=None)
    p.add_argument("--q-scale", type=float, default=1.0)
    p.add_argument("--r", type=float, default=1e-4)
    p.add_argument("--couple", action="store_true",
                   help="add lagged cross-station coupling (wave propagation)")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    coeffs = json.loads(args.coeffs.read_text(encoding="utf-8"))
    node = coeffs["nodes"]["Cento"]
    stations = list(node["stations"])
    downstream = node.get("downstream", "Cento")
    # order: downstream first so state index 0 = Cento
    stations = [downstream] + [s for s in stations if s != downstream]
    log(f"stations(order)={stations}")

    wide = ft.load_wide(args.data, stations)
    from_ts = pd.Timestamp(args.from_ts, tz="UTC")
    to_ts = pd.Timestamp(args.to_ts, tz="UTC") if args.to_ts else wide.index.max()
    cand = wide.index[(wide.index >= from_ts) & (wide.index <= to_ts)]
    origins = cand[:: max(1, args.stride)]
    if args.max_origins:
        origins = origins[: args.max_origins]
    log(f"origins={len(origins)}")

    tr_full, _, _ = ml.split_masks(wide.index, mode="block", holdout_from=args.holdout_from)
    series_list = [wide[s].to_numpy(dtype=float) for s in stations]
    trans_list = []
    for s, y in zip(stations, series_list):
        rho, gamma, ymin, q_v, npair = ek.learn_transition(y, tr_full)
        trans_list.append(dict(rho=rho, gamma=gamma, ymin=ymin, q_v=q_v))
        log(f"{s}: rho={rho:.4f} gamma={gamma:.3e} ymin={ymin:.3f} q_v={q_v:.3e}")

    # full Q with cross-station velocity covariances
    E = learn_Q_cross(series_list, trans_list, tr_full)
    Q, Cv = build_Q_full(E, args.q_scale)
    log("Q velocity cross-cov (x1e4):")
    log(np.array2string(Cv * 1e4, precision=2, suppress_small=True))

    station_means = [float(np.nanmean(y[tr_full & np.isfinite(y)])) for y in series_list]
    R_vec = np.full(len(stations), args.r)

    Y = np.column_stack(series_list)
    origins_pos = np.array([int(wide.index.get_loc(t)) for t in origins])
    tr_m, va_m, ho_m = ml.split_masks(origins, mode="block", holdout_from=args.holdout_from)
    level_series = wide[downstream]
    y0_all = level_series.reindex(origins).to_numpy(dtype=float)
    targets_by_h = {h: ml.build_targets(wide, h) for h in range(1, H_STEPS + 1)}
    deltas = {h: targets_by_h[h]["delta"].reindex(origins).to_numpy(dtype=float)
              for h in range(1, H_STEPS + 1)}
    poly_base = {h: ml.build_poly2_baseline(level_series, origins.to_numpy(), h, n_points=6)
                 for h in range(1, H_STEPS + 1)}

    # ---------- optional lagged cross-coupling ----------
    C = None
    uni_levels = None
    if args.couple:
        def est_R_local(y):
            d = np.diff(y[np.isfinite(y)])
            d = d[np.isfinite(d)]
            if d.size < 100:
                return 1e-3
            mad = np.median(np.abs(d - np.median(d)))
            return float(max((1.4826 * mad) ** 2 / 2.0, 1e-6))
        uni_levels = []
        for i, y in enumerate(series_list):
            Qi = np.diag([1e-8, max(trans_list[i]["q_v"], 1e-10)])
            m_arr, _ = ek.forward_filter(y, trans_list[i]["rho"], trans_list[i]["gamma"],
                                         trans_list[i]["ymin"], Qi, est_R_local(y),
                                         "kf", rng)
            uni_levels.append(m_arr[:, 0])
        C = learn_coupling(series_list, stations, trans_list, tr_full,
                           COUPLE_EDGES, uni_levels)
        log(f"coupling edges learned for targets: {sorted(C.keys())}")

    results = {}
    for kind in ["kf", "ukf", "enkf"]:
        suffix = "_GC" if args.couple else "_G"
        if args.couple:
            t0 = time.perf_counter()
            means, variances = forward_filter_gc(Y, trans_list, Q, R_vec, kind, rng,
                                                 station_means, C)
        else:
            t0 = time.perf_counter()
            means, variances = forward_filter_g(Y, trans_list, Q, R_vec, kind, rng, station_means)
        log(f"{kind}{suffix} filter pass {time.perf_counter()-t0:.0f}s")
        paths = np.full((len(origins), H_STEPS + 1), np.nan)
        var_paths = np.full((len(origins), H_STEPS + 1), np.nan)
        t0 = time.perf_counter()
        for i, pos in enumerate(origins_pos):
            m0 = means[pos]
            if not np.isfinite(m0[0]):
                continue
            P0 = np.diag(np.nan_to_num(variances[pos], nan=1e-6))
            if args.couple:
                pm, pv = rollout_gc(m0, P0, trans_list, Q, H_STEPS, kind, rng,
                                    C, means, pos)
            else:
                pm, pv = rollout_g(m0, P0, trans_list, Q, H_STEPS, kind, rng)
            paths[i] = pm[:, 0]
            var_paths[i] = pv[:, 0]
        log(f"{kind}{suffix} rollouts {time.perf_counter()-t0:.0f}s")
        per_h = {}
        for h in range(1, H_STEPS + 1):
            minutes = h * STEP_MIN
            pred = paths[:, h] - y0_all
            ev = ml.eval_splits(y0_all, deltas[h], pred, tr_m, va_m, ho_m,
                                poly_baseline=poly_base[h])
            per_h[str(minutes)] = {"horizonMinutes": minutes, "dnn": ev}
        for h in (4, 8, 12):
            ev = per_h[str(h * STEP_MIN)]["dnn"]
            log(f"  +{h*STEP_MIN}′ valEvt={ev['validateEvents'].get('skill')} "
                f"holdEvt={ev['holdoutEvents'].get('skill')}")
        results[f"{kind}{suffix}"] = {
            "R": args.r, "qScale": args.q_scale,
            "horizons": per_h,
            "medianForecastVar": {"120": float(np.nanmedian(var_paths[:, 4])),
                                  "360": float(np.nanmedian(var_paths[:, 12]))},
        }
        del means, variances, paths

    out = {
        "version": 1,
        "kind": "cento_kalman_global_experiment",
        "trainedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stations": stations,
        "stateDim": 2 * len(stations),
        "note": "Joint state, block-diagonal dynamics, FULL process-noise covariance "
                "(cross-station residual covariances). " + ("Lagged cross-station "
                "coupling v_i+ += c_e*dz_src(t-tau) learned on train residuals." if args.couple
                else "No exogenous inputs."),
        "transitions": trans_list,
        "coupling": {stations[i]: [{"src": stations[s], "tauSteps": t, "coef": c}
                                  for s, t, c in edges] for i, edges in (C or {}).items()},
        "Q_velocity_crosscov": Cv.tolist(),
        "origins": {"from": str(origins[0]), "to": str(origins[-1]), "n": int(len(origins))},
        "holdoutFrom": args.holdout_from,
        "results": results,
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
