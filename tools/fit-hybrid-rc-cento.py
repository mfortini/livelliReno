#!/usr/bin/env python3
"""Hybrid Cento: GRU → passive RC circuit params → dense rollout.

Sources (monte): Forcelli, Sostegno (+ Bonconvento fill).
Acceptors (valle backwater): Dosso, Opera Reno Panfilia, Secondo Salto, Gallo.
Drain uses softplus(Vc - Va): high acceptor ⇒ little/no drainage.

Surrogate on z-scored levels (not volume-conserving). Soft-adopt only if
holdout event skill@120 beats the current MLP.
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = "Cento"
UP_F = "Forcelli"
UP_S = "Sostegno Reno"
UP_ALT = "Bonconvento"
ACCEPTORS = ["Dosso", "Opera Reno Panfilia", "Secondo Salto", "Gallo"]
STATIONS = [DOWNSTREAM, UP_F, UP_S, UP_ALT, *ACCEPTORS]
LOOKBACK = 48
HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]
# per step: [z,miss,dz] × (1 + 2 sources + 4 acceptors) + rain = 22
N_SERIES = 1 + 2 + len(ACCEPTORS)  # Cento + Forcelli + Sostegno + acceptors
FEAT_PER_STEP = N_SERIES * 3 + 1
EVENT_ABS = 0.10
SIGMA_FLOOR = 1e-3
DNN_SKILL120_FALLBACK = 0.83
GRU_HIDDEN = 32
# G_f, G_s, G_d, G_p, G_2, G_g, C_c, alpha, spread
N_PARAMS = 9
LR = 1e-3
BATCH = 128
EPOCHS = 35
PATIENCE = 8
SEED = 42
EVAL_EVERY = 4
DT = 1.0
DEFAULT_TRAIN_CAP = 12000
DEFAULT_TRAIN_EVENT_KEEP = 8000
G_MIN, G_MAX = 0.02, 2.5
C_MIN, C_MAX = 0.5, 12.0
SURROGATE_NOTE = (
    "Surrogato RC su livelli z-scored: non conserva volumi. "
    "Accettori valle = backwater (softplus drain); forecast congela Va all'ultimo obs."
)


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
    names = []
    for g in list((config.get("rainfall") or {}).get("gauges") or []):
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


def metrics_block(y0, y_obs, y_pred) -> dict:
    mask = np.isfinite(y_obs) & np.isfinite(y_pred) & np.isfinite(y0)
    n = int(mask.sum())
    if n < 8:
        return {"rmse": None, "skill": None, "mae": None, "n": n}
    err_m = float(math.sqrt(np.mean((y_pred[mask] - y_obs[mask]) ** 2)))
    err_p = float(math.sqrt(np.mean((y0[mask] - y_obs[mask]) ** 2)))
    skill = None if err_p == 0 else 1.0 - err_m / err_p
    return {
        "rmse": round(err_m, 6),
        "skill": None if skill is None else round(skill, 6),
        "mae": round(float(np.mean(np.abs(y_pred[mask] - y_obs[mask]))), 6),
        "n": n,
    }


def fit_norm_stats(series: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    base = series[mask & np.isfinite(series)]
    if len(base) == 0:
        return 0.0, 1.0
    mu = float(base.mean())
    sd = float(base.std())
    if not math.isfinite(sd) or sd < SIGMA_FLOOR:
        sd = SIGMA_FLOOR
    return mu, sd


def fill_sostegno(h_s, h_alt, mu_s, sd_s, mu_a, sd_a):
    out = h_s.copy()
    miss = ~np.isfinite(out) & np.isfinite(h_alt)
    if miss.any():
        out[miss] = ((h_alt[miss] - mu_a) / sd_a) * sd_s + mu_s
    return out


def positive_range(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)


def decode_theta(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    # raw: (..., 9)
    return {
        "G_f": positive_range(raw[..., 0], G_MIN, G_MAX),
        "G_s": positive_range(raw[..., 1], G_MIN, G_MAX),
        "G_d": positive_range(raw[..., 2], G_MIN, G_MAX),
        "G_p": positive_range(raw[..., 3], G_MIN, G_MAX),
        "G_2": positive_range(raw[..., 4], G_MIN, G_MAX),
        "G_g": positive_range(raw[..., 5], G_MIN, G_MAX),
        "C_c": positive_range(raw[..., 6], C_MIN, C_MAX),
        "alpha": 0.5 * torch.sigmoid(raw[..., 7]),
        "spread": 0.01 + 0.25 * torch.sigmoid(raw[..., 8]),
    }


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x, beta=5.0)


def rc_step(
    z: torch.Tensor,
    z_f: torch.Tensor,
    z_s: torch.Tensor,
    z_acc: torch.Tensor,  # batch × 4
    rain: torch.Tensor,
    theta: dict[str, torch.Tensor],
    dt: float = DT,
) -> torch.Tensor:
    """Implicit-ish Euler: solve for z_new from linear G terms + softplus drain at z_new ≈ z."""
    g_acc = torch.stack([theta["G_d"], theta["G_p"], theta["G_2"], theta["G_g"]], dim=-1)
    # Use current z for softplus (semi-implicit); sources at new time.
    drain = (g_acc * softplus(z.unsqueeze(-1) - z_acc)).sum(dim=-1)
    inflow = theta["G_f"] * (z_f - z) + theta["G_s"] * (z_s - z)
    # Explicit Euler with C clamp for stability
    dz = (inflow - drain + theta["alpha"] * rain) * (dt / torch.clamp(theta["C_c"], min=C_MIN))
    return z + dz


class RcGRU(nn.Module):
    def __init__(self, n_in: int = FEAT_PER_STEP, hidden: int = GRU_HIDDEN, n_out: int = N_PARAMS):
        super().__init__()
        self.gru = nn.GRU(input_size=n_in, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _out, h_n = self.gru(seq)
        return self.head(h_n.squeeze(0))


def series_features(arr, miss, i):
    v = arr[i]
    m = float(miss[i])
    if not np.isfinite(v):
        return [0.0, 1.0, 0.0]
    prev = arr[i - 1] if i > 0 and np.isfinite(arr[i - 1]) else v
    return [float(v), m, float(v - prev)]


def build_seq_features(z_c, z_f, z_s, z_accs, rain_n, miss_c, miss_f, miss_s, miss_accs, t, lookback=LOOKBACK):
    if t < lookback:
        return None
    rows = []
    for k in range(lookback - 1, -1, -1):
        i = t - k
        row = []
        row.extend(series_features(z_c, miss_c, i))
        row.extend(series_features(z_f, miss_f, i))
        row.extend(series_features(z_s, miss_s, i))
        for za, ma in zip(z_accs, miss_accs):
            row.extend(series_features(za, ma, i))
        r = rain_n[i] if np.isfinite(rain_n[i]) else 0.0
        row.append(float(r))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def collect_indices(z_c, z_f, z_s, z_accs, mask, max_h, lookback=LOOKBACK, use_acceptors=True):
    n = len(z_c)
    out = []
    for t in range(lookback, n - max_h):
        if not mask[t] or not np.isfinite(z_c[t]):
            continue
        if not (
            np.isfinite(z_f[t - lookback + 1 : t + max_h + 1]).all()
            and np.isfinite(z_s[t - lookback + 1 : t + max_h + 1]).all()
            and np.isfinite(z_c[t + 1 : t + max_h + 1]).all()
        ):
            continue
        if use_acceptors:
            ok_acc = True
            for za in z_accs:
                # Acceptors needed on lookback through t (frozen after t); allow NaN in future.
                if not np.isfinite(za[t - lookback + 1 : t + 1]).all():
                    ok_acc = False
                    break
            if not ok_acc:
                continue
        out.append(t)
    return np.asarray(out, dtype=int)


def subsample_train_indices(train_idx, h_c, *, cap, event_keep, seed=SEED):
    meta = {
        "usableTrain": int(len(train_idx)),
        "cap": int(cap),
        "eventKeep": int(event_keep),
        "kept": int(len(train_idx)),
        "lostToCap": 0,
        "mode": "all",
    }
    if cap is None or cap <= 0 or len(train_idx) <= cap:
        return train_idx, meta
    rng = np.random.default_rng(seed)
    deltas = np.abs(h_c[train_idx + 4] - h_c[train_idx])
    order = np.argsort(-deltas)
    n_event = min(max(0, event_keep), cap, len(order))
    keep = list(order[:n_event])
    rest = [i for i in range(len(train_idx)) if i not in set(keep)]
    n_extra = min(cap - len(keep), len(rest))
    if n_extra > 0:
        keep.extend(rng.choice(rest, size=n_extra, replace=False).tolist())
    kept = train_idx[np.asarray(keep, dtype=int)]
    meta.update({
        "kept": int(len(kept)),
        "lostToCap": int(len(train_idx) - len(kept)),
        "mode": "event_then_random",
        "eventSelected": int(n_event),
        "randomSelected": int(n_extra),
    })
    return kept, meta


def rollout(theta, z0, z_f_seq, z_s_seq, z_acc0, rain0, horizons, warm=None):
    """z_*_seq: batch × (max_h+1) with [:,0]=t. z_acc0: batch × 4 frozen. warm: dict of lookback seqs."""
    max_h = max(horizons)
    z = z0
    if warm is not None:
        wf, ws, wacc, wr = warm["f"], warm["s"], warm["acc"], warm["rain"]
        for k in range(1, wf.shape[1]):
            z = rc_step(z, wf[:, k], ws[:, k], wacc[:, k, :], wr[:, k], theta)
    store = {}
    for k in range(1, max_h + 1):
        z = rc_step(z, z_f_seq[:, k], z_s_seq[:, k], z_acc0, rain0, theta)
        if k in horizons:
            store[k] = z
    pred = torch.stack([store[h] for h in horizons], dim=1)
    return pred, theta["spread"]


def export_gru(model: RcGRU) -> dict:
    return {
        "gru": {
            "inputSize": FEAT_PER_STEP,
            "hiddenSize": GRU_HIDDEN,
            "weightIh": model.gru.weight_ih_l0.detach().cpu().numpy().tolist(),
            "weightHh": model.gru.weight_hh_l0.detach().cpu().numpy().tolist(),
            "biasIh": model.gru.bias_ih_l0.detach().cpu().numpy().tolist(),
            "biasHh": model.gru.bias_hh_l0.detach().cpu().numpy().tolist(),
        },
        "head": {
            "W": model.head.weight.detach().cpu().numpy().tolist(),
            "b": model.head.bias.detach().cpu().numpy().tolist(),
        },
    }


def load_dnn_skill120(coeffs_path: Path) -> float | None:
    if not coeffs_path.exists():
        return None
    node = (json.loads(coeffs_path.read_text()).get("nodes") or {}).get(DOWNSTREAM) or {}
    skill = node.get("skill120")
    if skill is None:
        skill = (node.get("horizons") or {}).get("120", {}).get("holdoutEvents", {}).get("skill")
        if skill is None:
            skill = (node.get("horizons") or {}).get("120", {}).get("validateEvents", {}).get("skill")
    return float(skill) if skill is not None and math.isfinite(float(skill)) else None


def self_test() -> None:
    torch.manual_seed(0)
    model = RcGRU()
    seq = torch.randn(2, LOOKBACK, FEAT_PER_STEP)
    theta = decode_theta(model(seq))
    assert float(theta["G_f"].detach().min()) >= G_MIN - 1e-6
    assert float(theta["C_c"].detach().max()) <= C_MAX + 1e-6
    z0 = torch.zeros(2)
    rain = torch.zeros(2)
    # High acceptor blocks drain vs low acceptor
    z_acc_high = torch.ones(2, 4) * 2.0
    z_acc_low = torch.ones(2, 4) * (-2.0)
    z_h = z0.clone()
    z_l = z0.clone()
    for _ in range(6):
        z_h = rc_step(z_h, torch.ones(2), torch.ones(2), z_acc_high, rain, theta)
        z_l = rc_step(z_l, torch.ones(2), torch.ones(2), z_acc_low, rain, theta)
    assert torch.isfinite(z_h).all() and torch.isfinite(z_l).all()
    assert float((z_h - z_l).detach().mean()) >= -1e-4, "high acceptor should not drain more than low"
    # Rising source with acceptors at same level as z → no softplus drain; first step ≥ 0
    z = torch.zeros(1)
    th = decode_theta(torch.zeros(1, 9))
    z1 = rc_step(z, torch.ones(1), torch.zeros(1), torch.zeros(1, 4), torch.zeros(1), th)
    assert float(z1.detach()) >= -1e-5
    print("self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="RC+GRU hybrid Cento with valle acceptors")
    parser.add_argument("--data", type=Path, default=ROOT / "storico")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--coeffs", type=Path, default=ROOT / "models" / "transfer-coeffs.json")
    parser.add_argument("--mlp-valle", type=Path, default=ROOT / "models" / "cento-dnn-valle.json",
                        help="Optional MLP+valle skill reference for gate")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "hybrid-rc-cento-summary.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "hybrid-rc-cento.json")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--train-cap", type=int, default=DEFAULT_TRAIN_CAP)
    parser.add_argument("--train-event-keep", type=int, default=DEFAULT_TRAIN_EVENT_KEEP)
    parser.add_argument("--train-horizons", type=str, default="1,2,3,4")
    parser.add_argument("--no-acceptors", action="store_true", help="Ablation: sources only")
    parser.add_argument("--acceptors-dosso-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    train_h = [int(x) for x in args.train_horizons.split(",") if x.strip()]
    if not train_h or max(train_h) > max(HORIZONS) or min(train_h) < 1:
        raise SystemExit(f"Invalid --train-horizons {args.train_horizons}")

    if args.no_acceptors:
        acceptors = []
    elif args.acceptors_dosso_only:
        acceptors = ["Dosso"]
    else:
        acceptors = list(ACCEPTORS)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config = json.loads(args.config.read_text())
    gauges = gauge_names(config)
    needed = [DOWNSTREAM, UP_F, UP_S, UP_ALT, *acceptors]
    wide = ft.load_wide(args.data, needed)
    wide, spike_counts = ft.filter_spikes(wide)
    for s in [DOWNSTREAM, UP_F, UP_S]:
        if s not in wide.columns:
            raise SystemExit(f"Missing {s}")
    if UP_ALT not in wide.columns:
        wide[UP_ALT] = np.nan
    for a in acceptors:
        if a not in wide.columns:
            raise SystemExit(f"Missing acceptor {a}")

    rain_step = load_rain_step(args.data / "rain", gauges, wide.index)
    blocks = ft.month_blocks(wide.index)
    train_m = blocks == 0
    val_m = blocks == 1
    hold_m = blocks == 2

    h_c = wide[DOWNSTREAM].to_numpy(float)
    h_f = wide[UP_F].to_numpy(float)
    h_s_raw = wide[UP_S].to_numpy(float)
    h_alt = wide[UP_ALT].to_numpy(float)
    h_acc = [wide[a].to_numpy(float) for a in acceptors]

    mu_c, sd_c = fit_norm_stats(h_c, train_m)
    mu_f, sd_f = fit_norm_stats(h_f, train_m)
    mu_s, sd_s = fit_norm_stats(h_s_raw, train_m)
    mu_a, sd_a = fit_norm_stats(h_alt, train_m)
    mu_r, sd_r = fit_norm_stats(rain_step.to_numpy(float), train_m)
    mu_acc, sd_acc = [], []
    for ha in h_acc:
        mu, sd = fit_norm_stats(ha, train_m)
        mu_acc.append(mu)
        sd_acc.append(sd)

    h_s = fill_sostegno(h_s_raw, h_alt, mu_s, sd_s, mu_a, sd_a)
    lag_f, r_f = ft.estimate_lag(h_f, h_c, 24, train_m)
    lag_s, r_s = ft.estimate_lag(h_s, h_c, 24, train_m)
    h_f_lag = np.roll(h_f, lag_f)
    h_s_lag = np.roll(h_s, lag_s)
    h_f_lag[:lag_f] = np.nan
    h_s_lag[:lag_s] = np.nan

    z_c = (h_c - mu_c) / sd_c
    z_f = (h_f_lag - mu_f) / sd_f
    z_s = (h_s_lag - mu_s) / sd_s
    z_accs = [(ha - mu) / sd for ha, mu, sd in zip(h_acc, mu_acc, sd_acc)]
    # Pad to 4 acceptors with zeros if ablation uses fewer (model always has 4 G_a)
    while len(z_accs) < 4:
        z_accs.append(np.zeros_like(z_c))
    rain_n = np.nan_to_num((rain_step.to_numpy(float) - mu_r) / sd_r, nan=0.0)
    miss_c = (~np.isfinite(h_c)).astype(float)
    miss_f = (~np.isfinite(h_f_lag)).astype(float)
    miss_s = (~np.isfinite(z_s)).astype(float)
    miss_accs = [(~np.isfinite(ha)).astype(float) for ha in h_acc]
    while len(miss_accs) < 4:
        miss_accs.append(np.ones_like(z_c))

    max_h = max(HORIZONS)
    use_acc = len(acceptors) > 0
    train_idx_all = collect_indices(z_c, z_f, z_s, z_accs[: len(acceptors)] if use_acc else [], train_m, max_h, use_acceptors=use_acc)
    val_idx = collect_indices(z_c, z_f, z_s, z_accs[: len(acceptors)] if use_acc else [], val_m, max_h, use_acceptors=use_acc)
    hold_idx = collect_indices(z_c, z_f, z_s, z_accs[: len(acceptors)] if use_acc else [], hold_m, max_h, use_acceptors=use_acc)
    if len(train_idx_all) < ft.WEAK_POINTS:
        raise SystemExit(f"Too few train samples: {len(train_idx_all)}")

    train_idx, subsample_meta = subsample_train_indices(
        train_idx_all, h_c, cap=args.train_cap, event_keep=args.train_event_keep, seed=SEED
    )

    def pack(indices: np.ndarray):
        seqs, z0s, zfs, zss, zacc0s = [], [], [], [], []
        warms_f, warms_s, warms_acc, warms_r = [], [], [], []
        rains, targets, y0s, wts = [], [], [], []
        for t in indices:
            feat = build_seq_features(
                z_c, z_f, z_s, z_accs, rain_n, miss_c, miss_f, miss_s, miss_accs, t
            )
            if feat is None:
                continue
            seqs.append(feat)
            z0s.append(z_c[t])
            zfs.append(z_f[t : t + max_h + 1])
            zss.append(z_s[t : t + max_h + 1])
            # Frozen acceptor levels at t (backwater persistence)
            acc0 = np.array([za[t] if np.isfinite(za[t]) else 0.0 for za in z_accs], dtype=np.float32)
            zacc0s.append(acc0)
            warms_f.append(z_f[t - LOOKBACK + 1 : t + 1])
            warms_s.append(z_s[t - LOOKBACK + 1 : t + 1])
            warms_acc.append(np.stack([za[t - LOOKBACK + 1 : t + 1] for za in z_accs], axis=1))
            warms_r.append(rain_n[t - LOOKBACK + 1 : t + 1])
            rains.append(rain_n[t])
            targets.append([h_c[t + h] for h in HORIZONS])
            y0s.append(h_c[t])
            wts.append(3.0 if abs(h_c[t + 4] - h_c[t]) >= EVENT_ABS else 1.0)
        return (
            np.asarray(seqs, np.float32),
            np.asarray(z0s, np.float32),
            np.asarray(zfs, np.float32),
            np.asarray(zss, np.float32),
            np.asarray(zacc0s, np.float32),
            np.asarray(warms_f, np.float32),
            np.asarray(warms_s, np.float32),
            np.asarray(warms_acc, np.float32),
            np.asarray(warms_r, np.float32),
            np.asarray(rains, np.float32),
            np.asarray(targets, np.float32),
            np.asarray(y0s, np.float32),
            np.asarray(wts, np.float32),
        )

    tr = pack(train_idx)
    va = pack(val_idx)
    print(
        f"train={len(tr[0])} (usable={subsample_meta['usableTrain']} cap={args.train_cap}) "
        f"val={len(va[0])} hold={len(hold_idx)} acceptors={acceptors} "
        f"feat/step={FEAT_PER_STEP} loss_h={train_h}"
    )

    model = RcGRU()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(tr[0]), torch.tensor(tr[1]), torch.tensor(tr[2]), torch.tensor(tr[3]),
            torch.tensor(tr[4]), torch.tensor(tr[5]), torch.tensor(tr[6]), torch.tensor(tr[7]),
            torch.tensor(tr[8]), torch.tensor(tr[9]), torch.tensor(tr[10]), torch.tensor(tr[12]),
        ),
        batch_size=BATCH,
        shuffle=True,
    )
    x_va = torch.tensor(va[0])
    z0_va, zf_va, zs_va, za_va = torch.tensor(va[1]), torch.tensor(va[2]), torch.tensor(va[3]), torch.tensor(va[4])
    wf_va, ws_va, wa_va, wr_va = torch.tensor(va[5]), torch.tensor(va[6]), torch.tensor(va[7]), torch.tensor(va[8])
    rn_va, tg_va = torch.tensor(va[9]), torch.tensor(va[10])
    train_h_idx = [HORIZONS.index(h) for h in train_h]

    best_state = None
    best_val = float("inf")
    patience = 0
    history = {"best_epoch": 0, "best_val_mse": None}

    for epoch in range(args.epochs):
        model.train()
        for seq, z0, zf, zs, za, wf, ws, wa, wr, rn, tg, wt in loader:
            opt.zero_grad()
            theta = decode_theta(model(seq))
            # Warm lookback from oldest Cento z, then re-anchor to observed z0 and forecast.
            z_w = seq[:, 0, 0]
            for k in range(1, LOOKBACK):
                z_w = rc_step(z_w, wf[:, k], ws[:, k], wa[:, k, :], wr[:, k], theta)
            pred_z, _ = rollout(theta, z0, zf, zs, za, rn, train_h, warm=None)
            pred_h = pred_z * sd_c + mu_c
            err = (pred_h - tg[:, train_h_idx]) ** 2
            loss = (err.mean(dim=1) * wt).mean() + 0.05 * ((z_w - z0) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            theta = decode_theta(model(x_va))
            z_w = x_va[:, 0, 0]
            for k in range(1, LOOKBACK):
                z_w = rc_step(z_w, wf_va[:, k], ws_va[:, k], wa_va[:, k, :], wr_va[:, k], theta)
            pred_z, _ = rollout(theta, z0_va, zf_va, zs_va, za_va, rn_va, train_h, warm=None)
            pred_h = pred_z * sd_c + mu_c
            val_mse = float(((pred_h - tg_va[:, train_h_idx]) ** 2).mean().item())
        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            history["best_epoch"] = epoch + 1
            history["best_val_mse"] = round(best_val, 8)
            print(f"epoch {epoch+1} val_mse={val_mse:.6f} *")
        else:
            patience += 1
            print(f"epoch {epoch+1} val_mse={val_mse:.6f}")
            if patience >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    def densify_mask(block_mask):
        base = np.zeros(len(h_c), dtype=bool)
        base[::EVAL_EVERY] = True
        eventish = np.zeros(len(h_c), dtype=bool)
        for i in range(1, len(h_c)):
            if np.isfinite(h_c[i]) and np.isfinite(h_c[i - 1]) and abs(h_c[i] - h_c[i - 1]) >= EVENT_ABS:
                eventish[max(0, i - 2) : min(len(h_c), i + 3)] = True
        return collect_indices(
            z_c, z_f, z_s, z_accs[: len(acceptors)] if use_acc else [],
            block_mask & (base | eventish), max_h, use_acceptors=use_acc,
        )

    def eval_indices(indices: np.ndarray):
        if len(indices) == 0:
            return {}, {}
        data = pack(indices)
        with torch.no_grad():
            theta = decode_theta(model(torch.tensor(data[0])))
            pred_z, _ = rollout(
                theta,
                torch.tensor(data[1]),
                torch.tensor(data[2]),
                torch.tensor(data[3]),
                torch.tensor(data[4]),
                torch.tensor(data[9]),
                HORIZONS,
                warm=None,
            )
            pred_h = (pred_z * sd_c + mu_c).cpu().numpy()
        y0, tg = data[11], data[10]
        # High-acceptor bucket: Dosso (or first acceptor) z >= train z85
        high = np.zeros(len(y0), dtype=bool)
        if acceptors:
            za0 = data[4][:, 0]
            thr = float(np.nanquantile(z_accs[0][train_m & np.isfinite(z_accs[0])], 0.85)) if np.isfinite(z_accs[0][train_m]).any() else 0.0
            high = za0 >= thr
        out = {}
        resid_spread = {}
        for j, h in enumerate(HORIZONS):
            obs, pred = tg[:, j], pred_h[:, j]
            all_m = metrics_block(y0, obs, pred)
            ev = np.abs(obs - y0) >= EVENT_ABS
            resid = np.abs(pred - obs)
            mask = np.isfinite(resid)
            mad = float(np.median(resid[mask])) if mask.any() else 0.05
            key = str(h * ft.STEP_MINUTES)
            resid_spread[key] = round(max(0.01, 1.4826 * mad), 6)
            out[key] = {
                "all": all_m,
                "events": metrics_block(y0[ev], obs[ev], pred[ev]),
                "highAcceptor": metrics_block(y0[high], obs[high], pred[high]),
                "lowAcceptor": metrics_block(y0[~high], obs[~high], pred[~high]),
                "meanSpread": resid_spread[key],
            }
        return out, resid_spread

    metrics_validate, spread_val = eval_indices(densify_mask(val_m) if len(densify_mask(val_m)) else val_idx)
    metrics_holdout, spread_hold = eval_indices(densify_mask(hold_m) if len(densify_mask(hold_m)) else hold_idx)
    metrics = metrics_holdout or metrics_validate
    resid_spreads = spread_hold or spread_val

    dnn_skill = None
    if args.mlp_valle.exists():
        node = json.loads(args.mlp_valle.read_text())
        dnn_skill = node.get("skill120")
        if dnn_skill is None:
            dnn_skill = (node.get("horizons") or {}).get("120", {}).get("holdoutEvents", {}).get("skill")
        if dnn_skill is not None:
            dnn_skill = float(dnn_skill)
    if dnn_skill is None:
        dnn_skill = load_dnn_skill120(args.coeffs) or DNN_SKILL120_FALLBACK

    hybrid_skill120 = (metrics.get("120") or {}).get("events", {}).get("skill")
    adopt = bool(hybrid_skill120 is not None and hybrid_skill120 >= dnn_skill)
    mean_spread = resid_spreads.get("120") or 0.05

    means = {DOWNSTREAM: mu_c, UP_F: mu_f, UP_S: mu_s, UP_ALT: mu_a, "rain": mu_r}
    stds = {DOWNSTREAM: sd_c, UP_F: sd_f, UP_S: sd_s, UP_ALT: sd_a, "rain": sd_r}
    for name, mu, sd in zip(ACCEPTORS, mu_acc + [0] * 4, sd_acc + [1] * 4):
        means[name] = mu
        stds[name] = sd

    weights = {
        "version": 1,
        "kind": "rc_gru",
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
        "note": SURROGATE_NOTE,
        "downstream": DOWNSTREAM,
        "upstreams": [UP_F, UP_S],
        "upstreamAlt": UP_ALT,
        "acceptors": ACCEPTORS,
        "acceptorsUsed": acceptors,
        "lookbackSteps": LOOKBACK,
        "stepMinutes": ft.STEP_MINUTES,
        "featPerStep": FEAT_PER_STEP,
        "horizonsMinutes": [h * ft.STEP_MINUTES for h in HORIZONS],
        "trainHorizonsMinutes": [h * ft.STEP_MINUTES for h in train_h],
        "means": means,
        "stds": stds,
        "lags": {
            UP_F: {"steps": lag_f, "minutes": lag_f * ft.STEP_MINUTES, "r": None if r_f is None else round(r_f, 6)},
            UP_S: {"steps": lag_s, "minutes": lag_s * ft.STEP_MINUTES, "r": None if r_s is None else round(r_s, 6)},
        },
        "paramDecode": {"gMin": G_MIN, "gMax": G_MAX, "cMin": C_MIN, "cMax": C_MAX, "dt": DT},
        "spreadLow": mean_spread,
        "spreadHigh": mean_spread,
        "spreadByHorizon": {k: resid_spreads.get(k, mean_spread) for k in (str(h * ft.STEP_MINUTES) for h in HORIZONS)},
        "reliable": adopt,
        "skill120": hybrid_skill120,
        **export_gru(model),
    }
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    args.weights.write_text(json.dumps(weights) + "\n")

    summary = {
        "version": 1,
        "kind": "cento_rc_gru",
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
        "note": SURROGATE_NOTE,
        "downstream": DOWNSTREAM,
        "upstreams": [UP_F, UP_S],
        "acceptors": acceptors,
        "lookbackSteps": LOOKBACK,
        "gruHidden": GRU_HIDDEN,
        "featPerStep": FEAT_PER_STEP,
        "weightsBytes": args.weights.stat().st_size,
        "means": means,
        "stds": stds,
        "lags": weights["lags"],
        "spikeCounts": spike_counts,
        "trainHistory": history,
        "subsample": subsample_meta,
        "nTrain": int(len(tr[0])),
        "nTrainUsable": int(subsample_meta["usableTrain"]),
        "nValidate": int(len(va[0])),
        "nHoldout": int(len(hold_idx)),
        "metricsValidate": metrics_validate,
        "metricsHoldout": metrics_holdout,
        "metrics": metrics,
        "weightsFile": str(args.weights.relative_to(ROOT)) if args.weights.is_relative_to(ROOT) else str(args.weights),
        "gate": {
            "mlpSkill120Events": round(float(dnn_skill), 6),
            "rcSkill120Events": hybrid_skill120,
            "adoptRuntime": adopt,
            "reason": "rc_beats_mlp" if adopt else "rc_below_mlp_keep_mlp",
            "gateSplit": "holdout" if metrics_holdout else "validate",
        },
        "fittedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["gate"], indent=2))
    for m in ("30", "60", "90", "120"):
        if m in metrics:
            ev = metrics[m]["events"]
            print(f"+{m}′ events skill={ev['skill']} rmse={ev['rmse']} n={ev['n']}")


if __name__ == "__main__":
    main()
