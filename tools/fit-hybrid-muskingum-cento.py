#!/usr/bin/env python3
"""Hybrid Cento: tiny GRU → Muskingum multi-reach params → dense rollout.

Browser-ready: exports small JSON weights (not a 12 MB MLP). Levels are
z-scored train-only per station. Two Muskingum reaches (Forcelli, Sostegno)
route into Cento; Bonconvento fills Sostegno gaps. Skill after denorm to m.

Writes models/hybrid-muskingum-cento.json + summary with adoptRuntime gate.
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
STATIONS = [DOWNSTREAM, UP_F, UP_S, UP_ALT]
LOOKBACK = 48  # 24 h
HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]
FEAT_PER_STEP = 10
EVENT_ABS = 0.10
SIGMA_FLOOR = 1e-3
DNN_SKILL120_FALLBACK = 0.83
GRU_HIDDEN = 32
N_PARAMS = 9  # K_f,X_f,K_s,X_s,g_f,g_s,lam,alpha,spread
LR = 1e-3
BATCH = 128
EPOCHS = 35
PATIENCE = 8
SEED = 42
EVAL_EVERY = 4
DT = 1.0  # one step = 30 min in Muskingum units
# Default train subsample: prefer high-Δ windows then random fill.
DEFAULT_TRAIN_CAP = 12000
DEFAULT_TRAIN_EVENT_KEEP = 8000
X_MIN = 0.05
X_MAX = 0.45
# Routing gains stay non-negative; levels remain a surrogate (not volume-conserving).
SURROGATE_NOTE = (
    "Surrogato su livelli z-scored: non conserva volumi. "
    "Parametri Muskingum vincolati a 2·K·X ≤ dt ≤ 2·K·(1−X)."
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
        df = ft.rain_gauge_frame(rain_dir, name)
        if df is None:
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


def muskingum_coeffs(K: torch.Tensor, X: torch.Tensor, dt: float = DT):
    """Return C0,C1,C2. Caller must keep 2·K·X ≤ dt ≤ 2·K·(1−X) so coeffs ≥ 0."""
    denom = K - K * X + 0.5 * dt
    denom = torch.clamp(denom, min=1e-3)
    c0 = (-K * X + 0.5 * dt) / denom
    c1 = (K * X + 0.5 * dt) / denom
    c2 = (K - K * X - 0.5 * dt) / denom
    return c0, c1, c2


def muskingum_step(I2, I1, O1, K, X, dt: float = DT):
    c0, c1, c2 = muskingum_coeffs(K, X, dt)
    return c0 * I2 + c1 * I1 + c2 * O1


def decode_KX(raw_k: torch.Tensor, raw_x: torch.Tensor, dt: float = DT) -> tuple[torch.Tensor, torch.Tensor]:
    """Map unbounded raw → (K, X) inside the Muskingum stability box for this dt."""
    x = X_MIN + (X_MAX - X_MIN) * torch.sigmoid(raw_x)
    k_min = (0.5 * dt) / torch.clamp(1.0 - x, min=1e-3)
    k_max = (0.5 * dt) / torch.clamp(x, min=1e-3)
    k = k_min + (k_max - k_min) * torch.sigmoid(raw_k)
    return k, x


def decode_theta(raw: torch.Tensor, dt: float = DT) -> dict[str, torch.Tensor]:
    # raw: (..., 9) — K/X pairs respect 2 K X <= dt <= 2 K (1-X); gains >= 0.
    k_f, x_f = decode_KX(raw[..., 0], raw[..., 1], dt)
    k_s, x_s = decode_KX(raw[..., 2], raw[..., 3], dt)
    return {
        "K_f": k_f,
        "X_f": x_f,
        "K_s": k_s,
        "X_s": x_s,
        "g_f": 0.05 + 0.95 * torch.sigmoid(raw[..., 4]),
        "g_s": 0.05 + 0.95 * torch.sigmoid(raw[..., 5]),
        "lam": 0.05 + 0.9 * torch.sigmoid(raw[..., 6]),  # blend toward routed inflow
        "alpha": 0.5 * torch.sigmoid(raw[..., 7]),
        "spread": 0.01 + 0.25 * torch.sigmoid(raw[..., 8]),
    }


def subsample_train_indices(
    train_idx: np.ndarray,
    h_c: np.ndarray,
    *,
    cap: int,
    event_keep: int,
    seed: int = SEED,
) -> tuple[np.ndarray, dict]:
    """Optionally cap train windows: top-|Δ| first, then random fill. cap<=0 keeps all."""
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


class MuskingumGRU(nn.Module):
    def __init__(self, n_in: int = FEAT_PER_STEP, hidden: int = GRU_HIDDEN, n_out: int = N_PARAMS):
        super().__init__()
        self.gru = nn.GRU(input_size=n_in, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: batch × T × F
        out, h_n = self.gru(seq)
        return self.head(h_n.squeeze(0))


def build_seq_features(
    z_c, z_f, z_s, rain_n, miss_c, miss_f, miss_s, t: int, lookback: int = LOOKBACK
) -> np.ndarray | None:
    if t < lookback:
        return None
    rows = []
    for k in range(lookback - 1, -1, -1):
        i = t - k
        row = []
        for arr, miss in ((z_c, miss_c), (z_f, miss_f), (z_s, miss_s)):
            v = arr[i]
            m = float(miss[i])
            if not np.isfinite(v):
                row.extend([0.0, 1.0, 0.0])
            else:
                prev = arr[i - 1] if i > 0 and np.isfinite(arr[i - 1]) else v
                row.extend([float(v), m, float(v - prev)])
        r = rain_n[i] if np.isfinite(rain_n[i]) else 0.0
        row.append(float(r))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def collect_indices(z_c, z_f, z_s, mask, max_h, lookback=LOOKBACK):
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
        out.append(t)
    return np.asarray(out, dtype=int)


def warmup_reach(I_seq: torch.Tensor, K: torch.Tensor, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Muskingum over lookback inflows. I_seq: batch × L ending at forecast origin t."""
    # Seed once at the oldest inflow, then warm through history (still a level surrogate).
    I_prev = I_seq[:, 0]
    O = I_seq[:, 0].clone()
    for k in range(1, I_seq.shape[1]):
        I = I_seq[:, k]
        O = muskingum_step(I, I_prev, O, K, X)
        I_prev = I
    return I_prev, O


def rollout(
    theta: dict[str, torch.Tensor],
    z0: torch.Tensor,
    z_f_seq: torch.Tensor,
    z_s_seq: torch.Tensor,
    rain0: torch.Tensor,
    horizons: list[int],
    warm_f: torch.Tensor | None = None,
    warm_s: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """z_f_seq: batch × (max_h+1) with [:,0]=t. Optional warm_* : batch × L ending at t."""
    max_h = max(horizons)
    z = z0
    if warm_f is not None and warm_s is not None and warm_f.shape[1] >= 2:
        I_f_prev, O_f = warmup_reach(warm_f, theta["K_f"], theta["X_f"])
        I_s_prev, O_s = warmup_reach(warm_s, theta["K_s"], theta["X_s"])
    else:
        I_f_prev = z_f_seq[:, 0]
        I_s_prev = z_s_seq[:, 0]
        O_f = z_f_seq[:, 0].clone()
        O_s = z_s_seq[:, 0].clone()
    store = {}
    for k in range(1, max_h + 1):
        I_f = z_f_seq[:, k]
        I_s = z_s_seq[:, k]
        O_f = muskingum_step(I_f, I_f_prev, O_f, theta["K_f"], theta["X_f"])
        O_s = muskingum_step(I_s, I_s_prev, O_s, theta["K_s"], theta["X_s"])
        routed = theta["g_f"] * O_f + theta["g_s"] * O_s
        z = (1.0 - theta["lam"]) * z + theta["lam"] * routed + theta["alpha"] * rain0
        I_f_prev, I_s_prev = I_f, I_s
        if k in horizons:
            store[k] = z
    pred = torch.stack([store[h] for h in horizons], dim=1)
    return pred, theta["spread"]


def export_gru(model: MuskingumGRU) -> dict:
    """Export GRU + head as JSON-friendly arrays (JS can implement GRU)."""
    w_ih = model.gru.weight_ih_l0.detach().cpu().numpy()
    w_hh = model.gru.weight_hh_l0.detach().cpu().numpy()
    b_ih = model.gru.bias_ih_l0.detach().cpu().numpy()
    b_hh = model.gru.bias_hh_l0.detach().cpu().numpy()
    hw = model.head.weight.detach().cpu().numpy()
    hb = model.head.bias.detach().cpu().numpy()
    return {
        "gru": {
            "inputSize": FEAT_PER_STEP,
            "hiddenSize": GRU_HIDDEN,
            "weightIh": w_ih.tolist(),  # (3*H, F)
            "weightHh": w_hh.tolist(),  # (3*H, H)
            "biasIh": b_ih.tolist(),
            "biasHh": b_hh.tolist(),
        },
        "head": {"W": hw.tolist(), "b": hb.tolist()},
    }


def load_dnn_skill120(coeffs_path: Path) -> float | None:
    if not coeffs_path.exists():
        return None
    node = (json.loads(coeffs_path.read_text()).get("nodes") or {}).get(DOWNSTREAM) or {}
    skill = node.get("skill120")
    return float(skill) if skill is not None and math.isfinite(float(skill)) else None


def self_test() -> None:
    torch.manual_seed(0)
    model = MuskingumGRU()
    seq = torch.randn(2, LOOKBACK, FEAT_PER_STEP)
    raw = model(seq)
    theta = decode_theta(raw)
    z0 = torch.zeros(2)
    zf = torch.randn(2, 9)
    zs = torch.randn(2, 9)
    rain = torch.zeros(2)
    warm = torch.randn(2, LOOKBACK)
    pred, sp = rollout(theta, z0, zf, zs, rain, [1, 2, 3, 4], warm, warm)
    assert pred.shape == (2, 4) and torch.isfinite(pred).all()
    assert float(theta["X_f"].detach().max()) <= X_MAX + 1e-6
    # Stability / non-negative C0 for a rising step with O1=0
    for _ in range(32):
        raw = torch.randn(9)
        th = decode_theta(raw)
        for key_k, key_x in (("K_f", "X_f"), ("K_s", "X_s")):
            k = float(th[key_k])
            x = float(th[key_x])
            assert 2 * k * x <= DT + 1e-5
            assert DT <= 2 * k * (1 - x) + 1e-5
            c0, c1, c2 = muskingum_coeffs(th[key_k], th[key_x])
            assert float(c0) >= -1e-6 and float(c1) >= -1e-6 and float(c2) >= -1e-6
        o = float(muskingum_step(torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0), th["K_f"], th["X_f"]))
        assert o >= -1e-6, f"inverse step response {o}"
    # subsample metadata
    idx = np.arange(15000)
    h = np.linspace(0, 1, 15004)
    kept, meta = subsample_train_indices(idx, h, cap=12000, event_keep=8000)
    assert len(kept) == 12000 and meta["lostToCap"] == 3000
    kept_all, meta_all = subsample_train_indices(idx, h, cap=0, event_keep=8000)
    assert len(kept_all) == 15000 and meta_all["mode"] == "all"
    print("self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Muskingum+GRU hybrid Cento")
    parser.add_argument("--data", type=Path, default=ROOT / "storico")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--coeffs", type=Path, default=ROOT / "models" / "transfer-coeffs.json")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "hybrid-muskingum-cento-summary.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "hybrid-muskingum-cento.json")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--train-cap",
        type=int,
        default=DEFAULT_TRAIN_CAP,
        help="Max train windows (0 = use all usable train samples)",
    )
    parser.add_argument(
        "--train-event-keep",
        type=int,
        default=DEFAULT_TRAIN_EVENT_KEEP,
        help="Within --train-cap, keep this many highest-|Δ@+2h| windows first",
    )
    parser.add_argument(
        "--train-horizons",
        type=str,
        default="1,2,3,4",
        help="Comma horizons (steps) used in the training loss; export still covers HORIZONS",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    train_h = [int(x) for x in args.train_horizons.split(",") if x.strip()]
    if not train_h or max(train_h) > max(HORIZONS) or min(train_h) < 1:
        raise SystemExit(f"Invalid --train-horizons {args.train_horizons}; must be subset of {HORIZONS}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config = json.loads(args.config.read_text())
    gauges = gauge_names(config)
    wide = ft.load_wide(args.data, STATIONS)
    wide, spike_counts = ft.filter_spikes(wide)
    for s in [DOWNSTREAM, UP_F, UP_S]:
        if s not in wide.columns:
            raise SystemExit(f"Missing {s}")
    if UP_ALT not in wide.columns:
        wide[UP_ALT] = np.nan

    rain_step = load_rain_step(args.data / "rain", gauges, wide.index)
    blocks = ft.month_blocks(wide.index)
    train_m = blocks == 0
    val_m = blocks == 1
    hold_m = blocks == 2

    h_c = wide[DOWNSTREAM].to_numpy(float)
    h_f = wide[UP_F].to_numpy(float)
    h_s_raw = wide[UP_S].to_numpy(float)
    h_alt = wide[UP_ALT].to_numpy(float)

    mu_c, sd_c = fit_norm_stats(h_c, train_m)
    mu_f, sd_f = fit_norm_stats(h_f, train_m)
    mu_s, sd_s = fit_norm_stats(h_s_raw, train_m)
    mu_a, sd_a = fit_norm_stats(h_alt, train_m)
    mu_r, sd_r = fit_norm_stats(rain_step.to_numpy(float), train_m)

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
    rain_n = np.nan_to_num((rain_step.to_numpy(float) - mu_r) / sd_r, nan=0.0)
    miss_c = (~np.isfinite(h_c)).astype(float)
    miss_f = (~np.isfinite(h_f_lag)).astype(float)
    miss_s = (~np.isfinite(z_s)).astype(float)

    max_h = max(HORIZONS)
    train_idx_all = collect_indices(z_c, z_f, z_s, train_m, max_h)
    val_idx = collect_indices(z_c, z_f, z_s, val_m, max_h)
    hold_idx = collect_indices(z_c, z_f, z_s, hold_m, max_h)
    if len(train_idx_all) < ft.WEAK_POINTS:
        raise SystemExit(f"Too few train samples: {len(train_idx_all)}")

    train_idx, subsample_meta = subsample_train_indices(
        train_idx_all,
        h_c,
        cap=args.train_cap,
        event_keep=args.train_event_keep,
        seed=SEED,
    )

    def pack(indices: np.ndarray):
        seqs, z0s, zfs, zss, warms_f, warms_s, rains, targets, y0s, wts = (
            [], [], [], [], [], [], [], [], [], []
        )
        for t in indices:
            feat = build_seq_features(z_c, z_f, z_s, rain_n, miss_c, miss_f, miss_s, t)
            if feat is None:
                continue
            seqs.append(feat)
            z0s.append(z_c[t])
            zfs.append(z_f[t : t + max_h + 1])
            zss.append(z_s[t : t + max_h + 1])
            warms_f.append(z_f[t - LOOKBACK + 1 : t + 1])
            warms_s.append(z_s[t - LOOKBACK + 1 : t + 1])
            rains.append(rain_n[t])
            targets.append([h_c[t + h] for h in HORIZONS])
            y0s.append(h_c[t])
            wts.append(3.0 if abs(h_c[t + 4] - h_c[t]) >= EVENT_ABS else 1.0)
        return (
            np.asarray(seqs, np.float32),
            np.asarray(z0s, np.float32),
            np.asarray(zfs, np.float32),
            np.asarray(zss, np.float32),
            np.asarray(warms_f, np.float32),
            np.asarray(warms_s, np.float32),
            np.asarray(rains, np.float32),
            np.asarray(targets, np.float32),
            np.asarray(y0s, np.float32),
            np.asarray(wts, np.float32),
        )

    tr = pack(train_idx)
    va = pack(val_idx)
    print(
        f"train={len(tr[0])} (usable={subsample_meta['usableTrain']} cap={args.train_cap}) "
        f"val={len(va[0])} hold={len(hold_idx)} lookback={LOOKBACK} "
        f"loss_horizons={train_h} export_horizons={HORIZONS}"
    )

    model = MuskingumGRU()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(tr[0]),
            torch.tensor(tr[1]),
            torch.tensor(tr[2]),
            torch.tensor(tr[3]),
            torch.tensor(tr[4]),
            torch.tensor(tr[5]),
            torch.tensor(tr[6]),
            torch.tensor(tr[7]),
            torch.tensor(tr[9]),
        ),
        batch_size=BATCH,
        shuffle=True,
    )
    x_va, z0_va, zf_va, zs_va, wf_va, ws_va, rn_va, tg_va = (
        torch.tensor(va[0]),
        torch.tensor(va[1]),
        torch.tensor(va[2]),
        torch.tensor(va[3]),
        torch.tensor(va[4]),
        torch.tensor(va[5]),
        torch.tensor(va[6]),
        torch.tensor(va[7]),
    )
    train_h_idx = [HORIZONS.index(h) for h in train_h]

    best_state = None
    best_val = float("inf")
    patience = 0
    history = {"best_epoch": 0, "best_val_mse": None}

    for epoch in range(args.epochs):
        model.train()
        for seq, z0, zf, zs, wf, ws, rn, tg, wt in loader:
            opt.zero_grad()
            theta = decode_theta(model(seq))
            pred_z, _ = rollout(theta, z0, zf, zs, rn, train_h, wf, ws)
            pred_h = pred_z * sd_c + mu_c
            err = (pred_h - tg[:, train_h_idx]) ** 2
            loss = (err.mean(dim=1) * wt).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            theta = decode_theta(model(x_va))
            pred_z, _ = rollout(theta, z0_va, zf_va, zs_va, rn_va, train_h, wf_va, ws_va)
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

    def eval_indices(indices: np.ndarray) -> tuple[dict, dict[str, float]]:
        """Return metrics per horizon and residual MAD spreads (metres) for fork calibration."""
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
                torch.tensor(data[6]),
                HORIZONS,
                torch.tensor(data[4]),
                torch.tensor(data[5]),
            )
            pred_h = (pred_z * sd_c + mu_c).cpu().numpy()
        y0, tg = data[8], data[7]
        out = {}
        resid_spread: dict[str, float] = {}
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
                "meanSpread": resid_spread[key],
            }
        return out, resid_spread

    def densify_mask(block_mask: np.ndarray) -> np.ndarray:
        base = np.zeros(len(h_c), dtype=bool)
        base[::EVAL_EVERY] = True
        eventish = np.zeros(len(h_c), dtype=bool)
        for i in range(1, len(h_c)):
            if np.isfinite(h_c[i]) and np.isfinite(h_c[i - 1]) and abs(h_c[i] - h_c[i - 1]) >= EVENT_ABS:
                eventish[max(0, i - 2) : min(len(h_c), i + 3)] = True
        return collect_indices(z_c, z_f, z_s, block_mask & (base | eventish), max_h)

    val_eval_idx = densify_mask(val_m)
    hold_eval_idx = densify_mask(hold_m)
    metrics_validate, spread_val = eval_indices(val_eval_idx if len(val_eval_idx) else val_idx)
    metrics_holdout, spread_hold = eval_indices(hold_eval_idx if len(hold_eval_idx) else hold_idx)
    # Gate and exported fork use holdout when available; keep validate separate.
    metrics = metrics_holdout or metrics_validate
    resid_spreads = spread_hold or spread_val

    dnn_skill = load_dnn_skill120(args.coeffs) or DNN_SKILL120_FALLBACK
    hybrid_skill120 = (metrics.get("120") or {}).get("events", {}).get("skill")
    adopt = bool(hybrid_skill120 is not None and hybrid_skill120 >= dnn_skill)

    mean_spread = resid_spreads.get("120") or 0.05
    spread_by_h = {k: resid_spreads.get(k, mean_spread) for k in (str(h * ft.STEP_MINUTES) for h in HORIZONS)}

    weights = {
        "version": 1,
        "kind": "muskingum_gru",
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
        "note": SURROGATE_NOTE,
        "downstream": DOWNSTREAM,
        "upstreams": [UP_F, UP_S],
        "upstreamAlt": UP_ALT,
        "lookbackSteps": LOOKBACK,
        "stepMinutes": ft.STEP_MINUTES,
        "featPerStep": FEAT_PER_STEP,
        "horizonsMinutes": [h * ft.STEP_MINUTES for h in HORIZONS],
        "trainHorizonsMinutes": [h * ft.STEP_MINUTES for h in train_h],
        "means": {DOWNSTREAM: mu_c, UP_F: mu_f, UP_S: mu_s, UP_ALT: mu_a, "rain": mu_r},
        "stds": {DOWNSTREAM: sd_c, UP_F: sd_f, UP_S: sd_s, UP_ALT: sd_a, "rain": sd_r},
        "lags": {
            UP_F: {"steps": lag_f, "minutes": lag_f * ft.STEP_MINUTES, "r": None if r_f is None else round(r_f, 6)},
            UP_S: {"steps": lag_s, "minutes": lag_s * ft.STEP_MINUTES, "r": None if r_s is None else round(r_s, 6)},
        },
        "spreadLow": mean_spread,
        "spreadHigh": mean_spread,
        "spreadByHorizon": spread_by_h,
        "paramDecode": {"xMin": X_MIN, "xMax": X_MAX, "dt": DT, "stableKX": True},
        "reliable": adopt,
        "skill120": hybrid_skill120,
        **export_gru(model),
    }
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    args.weights.write_text(json.dumps(weights) + "\n")

    summary = {
        "version": 1,
        "kind": "cento_muskingum_gru",
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
        "note": SURROGATE_NOTE + " Warm-up storico sul lookback; spread da MAD residui.",
        "downstream": DOWNSTREAM,
        "upstreams": [UP_F, UP_S],
        "lookbackSteps": LOOKBACK,
        "gruHidden": GRU_HIDDEN,
        "weightsBytes": args.weights.stat().st_size,
        "means": weights["means"],
        "stds": weights["stds"],
        "lags": weights["lags"],
        "spikeCounts": spike_counts,
        "trainHistory": history,
        "subsample": subsample_meta,
        "trainHorizonsMinutes": weights["trainHorizonsMinutes"],
        "nTrain": int(len(tr[0])),
        "nTrainUsable": int(subsample_meta["usableTrain"]),
        "nValidate": int(len(va[0])),
        "nHoldout": int(len(hold_idx)),
        "metricsValidate": metrics_validate,
        "metricsHoldout": metrics_holdout,
        "metrics": metrics,
        "weightsFile": "models/hybrid-muskingum-cento.json",
        "gate": {
            "dnnSkill120Events": round(dnn_skill, 6),
            "hybridSkill120Events": hybrid_skill120,
            "adoptRuntime": adopt,
            "reason": "hybrid_beats_dnn" if adopt else "hybrid_below_dnn_keep_mlp",
            "gateSplit": "holdout" if metrics_holdout else "validate",
        },
        "fittedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["gate"], indent=2))
    print(f"weights {args.weights.stat().st_size} bytes → {args.weights}")
    for m in ("30", "60", "90", "120"):
        if m in metrics:
            ev = metrics[m]["events"]
            print(f"+{m}′ events skill={ev['skill']} rmse={ev['rmse']} n={ev['n']}")


if __name__ == "__main__":
    main()
