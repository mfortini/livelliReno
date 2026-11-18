#!/usr/bin/env python3
"""Hybrid Cento: tiny DNN predicts volume-model params; physics rolls out.

Levels are z-scored with train-only (μ, σ) per station (ARPAE gauges do not
share a datum). The MLP maps a short window → bounded θ; Euler process_step
integrates z_c. Loss / skill are computed after denormalization to metres.

Writes models/hybrid-cento-summary.json (+ tiny weights). Dashboard wiring
only if event skill@120 ≥ current window_delta_mlp.
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
LOOKBACK = 6
HORIZONS = [1, 2, 3, 4]
EVENT_ABS = 0.10
MAX_GAIN = 0.35
SIGMA_FLOOR = 1e-3
DNN_SKILL120_FALLBACK = 0.83
PARAM_NAMES = ["A", "C_f", "C_s", "C_f_down", "C_s_down", "K_f", "K_s", "C_out", "alpha"]
HIDDEN = (64, 32)
LR = 1e-3
BATCH = 256
EPOCHS = 40
PATIENCE = 8
SEED = 42
EVAL_EVERY = 4


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


def fit_norm_stats(series: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    base = series[mask & np.isfinite(series)]
    if len(base) == 0:
        return 0.0, 1.0
    mu = float(base.mean())
    sd = float(base.std())
    if not math.isfinite(sd) or sd < SIGMA_FLOOR:
        sd = SIGMA_FLOOR
    return mu, sd


def normalize(series: np.ndarray, mu: float, sd: float) -> np.ndarray:
    out = (series - mu) / sd
    return out


def denorm(z: np.ndarray | float, mu: float, sd: float) -> np.ndarray | float:
    return z * sd + mu


def fill_sostegno(h_s: np.ndarray, h_alt: np.ndarray, mu_s: float, sd_s: float, mu_a: float, sd_a: float) -> np.ndarray:
    """Replace missing Sostegno with Bonconvento mapped into Sostegno z-space then denorm."""
    out = h_s.copy()
    miss = ~np.isfinite(out) & np.isfinite(h_alt)
    if not miss.any():
        return out
    z_alt = (h_alt[miss] - mu_a) / sd_a
    out[miss] = z_alt * sd_s + mu_s
    return out


def process_step_torch(
    z_c: torch.Tensor,
    z_f: torch.Tensor,
    z_f_prev: torch.Tensor,
    z_s: torch.Tensor,
    z_s_prev: torch.Tensor,
    rain: torch.Tensor,
    theta: dict[str, torch.Tensor],
) -> torch.Tensor:
    dz_f = z_f - z_f_prev
    dz_s = z_s - z_s_prev
    pulse = (
        theta["C_f"] * torch.relu(dz_f)
        + theta["C_s"] * torch.relu(dz_s)
        + theta["C_f_down"] * torch.minimum(dz_f, torch.zeros_like(dz_f))
        + theta["C_s_down"] * torch.minimum(dz_s, torch.zeros_like(dz_s))
        + theta["alpha"] * torch.relu(rain)
    )
    couple = theta["K_f"] * z_f + theta["K_s"] * z_s
    return z_c + (pulse + couple - theta["C_out"] * z_c) / theta["A"]


def decode_theta(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map unconstrained MLP outputs to stable positive gains (fractions of A)."""
    # raw: (..., 9)
    a = 1.0 + 7.0 * torch.sigmoid(raw[..., 0])  # [1, 8]
    def gain(i: int, hi: float) -> torch.Tensor:
        return (MAX_GAIN * hi * torch.sigmoid(raw[..., i])) * a

    return {
        "A": a,
        "C_f": gain(1, 1.0),
        "C_s": gain(2, 1.0),
        "C_f_down": gain(3, 0.8),
        "C_s_down": gain(4, 0.8),
        "K_f": gain(5, 0.15 / MAX_GAIN),
        "K_s": gain(6, 0.15 / MAX_GAIN),
        "C_out": gain(7, 0.25 / MAX_GAIN),
        "alpha": gain(8, 0.08 / MAX_GAIN),
    }


class ParamMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int = len(PARAM_NAMES), hidden: tuple[int, ...] = HIDDEN):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.05))
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_feature_row(
    z_c: np.ndarray,
    z_f: np.ndarray,
    z_s: np.ndarray,
    rain_n: np.ndarray,
    miss_c: np.ndarray,
    miss_f: np.ndarray,
    miss_s: np.ndarray,
    t: int,
    lookback: int = LOOKBACK,
) -> np.ndarray | None:
    if t < lookback:
        return None
    feats: list[float] = []
    for k in range(lookback):
        i = t - k
        for arr, miss in ((z_c, miss_c), (z_f, miss_f), (z_s, miss_s)):
            v = arr[i]
            m = miss[i]
            if not np.isfinite(v):
                feats.extend([0.0, 1.0, 0.0])
            else:
                prev = arr[i - 1] if i > 0 and np.isfinite(arr[i - 1]) else v
                feats.extend([float(v), float(m), float(v - prev)])
        r = rain_n[i] if np.isfinite(rain_n[i]) else 0.0
        feats.append(float(r))
    return np.asarray(feats, dtype=np.float32)


def feature_dim(lookback: int = LOOKBACK) -> int:
    # per lag: 3 stations × (z, miss, dz) + rain = 10
    return lookback * 10


def collect_indices(
    z_c: np.ndarray,
    z_f: np.ndarray,
    z_s: np.ndarray,
    mask: np.ndarray,
    max_h: int,
    lookback: int = LOOKBACK,
) -> np.ndarray:
    n = len(z_c)
    out = []
    for t in range(lookback, n - max_h):
        if not mask[t]:
            continue
        if not np.isfinite(z_c[t]):
            continue
        if not (
            np.isfinite(z_f[t - lookback + 1 : t + max_h + 1]).all()
            and np.isfinite(z_s[t - lookback + 1 : t + max_h + 1]).all()
            and np.isfinite(z_c[t + 1 : t + max_h + 1]).all()
        ):
            continue
        out.append(t)
    return np.asarray(out, dtype=int)


def rollout_batch(
    model: ParamMLP,
    x: torch.Tensor,
    z0: torch.Tensor,
    z_f_seq: torch.Tensor,
    z_s_seq: torch.Tensor,
    rain_seq: torch.Tensor,
    horizons: list[int],
) -> torch.Tensor:
    """Return predicted z_c at each horizon; shapes batch×len(horizons)."""
    theta = decode_theta(model(x))
    # z_f_seq: batch × (max_h+1) with index 0 = time t
    max_h = max(horizons)
    z = z0
    preds = []
    zf_prev = z_f_seq[:, 0]
    zs_prev = z_s_seq[:, 0]
    cursor = {h: None for h in horizons}
    for k in range(1, max_h + 1):
        zf = z_f_seq[:, k]
        zs = z_s_seq[:, k]
        rain = rain_seq[:, 0]  # hold rain at t
        z = process_step_torch(z, zf, zf_prev, zs, zs_prev, rain, theta)
        zf_prev, zs_prev = zf, zs
        if k in cursor:
            cursor[k] = z
    return torch.stack([cursor[h] for h in horizons], dim=1)


def export_mlp_layers(model: ParamMLP) -> list[dict]:
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    layers = []
    for i, linear in enumerate(linears):
        w = linear.weight.detach().cpu().numpy()
        b = linear.bias.detach().cpu().numpy()
        layers.append(
            {
                "W": w.tolist(),
                "b": b.tolist(),
                "activation": "linear" if i == len(linears) - 1 else "relu",
            }
        )
    return layers


def load_dnn_skill120(coeffs_path: Path) -> float | None:
    if not coeffs_path.exists():
        return None
    data = json.loads(coeffs_path.read_text())
    node = (data.get("nodes") or {}).get(DOWNSTREAM) or {}
    skill = node.get("skill120")
    return float(skill) if skill is not None and math.isfinite(float(skill)) else None


def self_test() -> None:
    torch.manual_seed(0)
    model = ParamMLP(feature_dim())
    x = torch.randn(4, feature_dim())
    z0 = torch.zeros(4)
    zf = torch.randn(4, 5)
    zs = torch.randn(4, 5)
    rain = torch.zeros(4, 1)
    pred = rollout_batch(model, x, z0, zf, zs, rain, [1, 2, 3, 4])
    assert pred.shape == (4, 4), pred.shape
    assert torch.isfinite(pred).all()
    theta = decode_theta(model(x))
    assert float(theta["A"].detach().min()) >= 1.0 - 1e-5
    print("self-test ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid volume+DNN Cento")
    parser.add_argument("--data", type=Path, default=ROOT / "storico")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--coeffs", type=Path, default=ROOT / "models" / "transfer-coeffs.json")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "hybrid-cento-summary.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "hybrid-cento.json")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config = json.loads(args.config.read_text())
    gauges = gauge_names(config)
    wide = ft.load_wide(args.data, STATIONS)
    wide, spike_counts = ft.filter_spikes(wide)
    for s in [DOWNSTREAM, UP_F, UP_S]:
        if s not in wide.columns:
            raise SystemExit(f"Missing station {s}")
    if UP_ALT not in wide.columns:
        wide[UP_ALT] = np.nan

    rain_step = load_rain_step(args.data / "rain", gauges, wide.index)
    blocks = ft.month_blocks(wide.index)
    train_m = blocks == 0
    val_m = blocks == 1
    hold_m = blocks == 2

    h_c = wide[DOWNSTREAM].to_numpy(dtype=float)
    h_f = wide[UP_F].to_numpy(dtype=float)
    h_s_raw = wide[UP_S].to_numpy(dtype=float)
    h_alt = wide[UP_ALT].to_numpy(dtype=float)

    mu_c, sd_c = fit_norm_stats(h_c, train_m)
    mu_f, sd_f = fit_norm_stats(h_f, train_m)
    mu_s, sd_s = fit_norm_stats(h_s_raw, train_m)
    mu_a, sd_a = fit_norm_stats(h_alt, train_m)
    mu_r, sd_r = fit_norm_stats(rain_step.to_numpy(dtype=float), train_m)

    h_s = fill_sostegno(h_s_raw, h_alt, mu_s, sd_s, mu_a, sd_a)

    lag_f, r_f = ft.estimate_lag(h_f, h_c, max_lag_steps=24, train_mask=train_m)
    lag_s, r_s = ft.estimate_lag(h_s, h_c, max_lag_steps=24, train_mask=train_m)

    h_f_lag = np.roll(h_f, lag_f)
    h_s_lag = np.roll(h_s, lag_s)
    h_f_lag[:lag_f] = np.nan
    h_s_lag[:lag_s] = np.nan

    z_c = normalize(h_c, mu_c, sd_c)
    z_f = normalize(h_f_lag, mu_f, sd_f)
    z_s = normalize(h_s_lag, mu_s, sd_s)
    rain_n = normalize(rain_step.to_numpy(dtype=float), mu_r, sd_r)
    rain_n = np.nan_to_num(rain_n, nan=0.0)

    miss_c = (~np.isfinite(h_c)).astype(float)
    miss_f = (~np.isfinite(h_f_lag)).astype(float)
    miss_s = (~np.isfinite(h_s_lag)).astype(float)
    # after fill, miss_s only if both missing
    miss_s = (~np.isfinite(z_s)).astype(float)

    max_h = max(HORIZONS)
    train_idx = collect_indices(z_c, z_f, z_s, train_m, max_h)
    val_idx = collect_indices(z_c, z_f, z_s, val_m, max_h)
    hold_idx = collect_indices(z_c, z_f, z_s, hold_m, max_h)
    if len(train_idx) < ft.WEAK_POINTS or len(val_idx) < 20:
        raise SystemExit(f"Troppi pochi indici train/val: {len(train_idx)}/{len(val_idx)}")

    def pack(indices: np.ndarray):
        xs, z0s, zfs, zss, rains, targets, y0s, wts = [], [], [], [], [], [], [], []
        for t in indices:
            feat = build_feature_row(z_c, z_f, z_s, rain_n, miss_c, miss_f, miss_s, t)
            if feat is None:
                continue
            xs.append(feat)
            z0s.append(z_c[t])
            zfs.append(z_f[t : t + max_h + 1])
            zss.append(z_s[t : t + max_h + 1])
            rains.append([rain_n[t]])
            # targets in metres for horizons
            targ = [h_c[t + h] for h in HORIZONS]
            targets.append(targ)
            y0s.append(h_c[t])
            w = 3.0 if abs(h_c[t + 4] - h_c[t]) >= EVENT_ABS else 1.0
            wts.append(w)
        return (
            np.asarray(xs, dtype=np.float32),
            np.asarray(z0s, dtype=np.float32),
            np.asarray(zfs, dtype=np.float32),
            np.asarray(zss, dtype=np.float32),
            np.asarray(rains, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            np.asarray(y0s, dtype=np.float32),
            np.asarray(wts, dtype=np.float32),
        )

    tr = pack(train_idx)
    va = pack(val_idx)

    device = torch.device("cpu")
    model = ParamMLP(feature_dim()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    x_tr = torch.tensor(tr[0])
    z0_tr = torch.tensor(tr[1])
    zf_tr = torch.tensor(tr[2])
    zs_tr = torch.tensor(tr[3])
    rn_tr = torch.tensor(tr[4])
    tg_tr = torch.tensor(tr[5])
    wt_tr = torch.tensor(tr[7])
    loader = DataLoader(
        TensorDataset(x_tr, z0_tr, zf_tr, zs_tr, rn_tr, tg_tr, wt_tr),
        batch_size=BATCH,
        shuffle=True,
    )

    x_va = torch.tensor(va[0])
    z0_va = torch.tensor(va[1])
    zf_va = torch.tensor(va[2])
    zs_va = torch.tensor(va[3])
    rn_va = torch.tensor(va[4])
    tg_va = torch.tensor(va[5])
    wt_va = torch.tensor(va[7])

    best_state = None
    best_val = float("inf")
    patience = 0
    history = {"best_epoch": 0, "best_val_mse": None}

    for epoch in range(args.epochs):
        model.train()
        for xb, z0b, zfb, zsb, rnb, tgb, wtb in loader:
            opt.zero_grad()
            pred_z = rollout_batch(model, xb, z0b, zfb, zsb, rnb, HORIZONS)
            pred_h = pred_z * sd_c + mu_c
            err = (pred_h - tgb) ** 2
            # weight last horizon more for events
            loss = (err.mean(dim=1) * wtb).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_z = rollout_batch(model, x_va, z0_va, zf_va, zs_va, rn_va, HORIZONS)
            pred_h = pred_z * sd_c + mu_c
            val_mse = float(((pred_h - tg_va) ** 2).mean().item())
        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            history["best_epoch"] = epoch + 1
            history["best_val_mse"] = round(best_val, 8)
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def eval_pack(pack_t, indices_label: str) -> dict:
        x, z0, zf, zs, rn, tg, y0, _wt = pack_t
        with torch.no_grad():
            pred_z = rollout_batch(
                model,
                torch.tensor(x),
                torch.tensor(z0),
                torch.tensor(zf),
                torch.tensor(zs),
                torch.tensor(rn),
                HORIZONS,
            )
            pred_h = (pred_z * sd_c + mu_c).cpu().numpy()
        out = {}
        for j, h in enumerate(HORIZONS):
            obs = tg[:, j]
            pred = pred_h[:, j]
            y0a = y0
            all_m = metrics_block(y0a, obs, pred)
            ev = np.abs(obs - y0a) >= EVENT_ABS
            ev_m = metrics_block(y0a[ev], obs[ev], pred[ev])
            out[str(h * ft.STEP_MINUTES)] = {"all": all_m, "events": ev_m}
        return out

    # denser eval on val|hold every EVAL_EVERY among collectable indices
    eval_mask = (val_m | hold_m).copy()
    base = np.zeros(len(h_c), dtype=bool)
    base[::EVAL_EVERY] = True
    eventish = np.zeros(len(h_c), dtype=bool)
    for i in range(1, len(h_c)):
        if np.isfinite(h_c[i]) and np.isfinite(h_c[i - 1]) and abs(h_c[i] - h_c[i - 1]) >= EVENT_ABS:
            eventish[max(0, i - 2) : min(len(h_c), i + 3)] = True
    eval_idx = collect_indices(z_c, z_f, z_s, eval_mask & (base | eventish), max_h)
    eval_pack_data = pack(eval_idx) if len(eval_idx) else pack(val_idx)
    metrics = eval_pack(eval_pack_data, "eval")
    train_metrics = eval_pack(tr, "train")
    val_metrics = eval_pack(va, "validate")
    hold_pack = pack(hold_idx) if len(hold_idx) else None
    hold_metrics = eval_pack(hold_pack, "holdout") if hold_pack and len(hold_pack[0]) else {}

    dnn_skill = load_dnn_skill120(args.coeffs)
    if dnn_skill is None:
        dnn_skill = DNN_SKILL120_FALLBACK
    hybrid_skill120 = (metrics.get("120") or {}).get("events", {}).get("skill")
    adopt = bool(
        hybrid_skill120 is not None
        and dnn_skill is not None
        and hybrid_skill120 >= dnn_skill
    )

    layers = export_mlp_layers(model)
    weights = {
        "version": 1,
        "kind": "hybrid_volume_mlp",
        "downstream": DOWNSTREAM,
        "upstreams": [UP_F, UP_S],
        "upstreamAlt": UP_ALT,
        "lookbackSteps": LOOKBACK,
        "stepMinutes": ft.STEP_MINUTES,
        "horizonsMinutes": [h * ft.STEP_MINUTES for h in HORIZONS],
        "paramNames": PARAM_NAMES,
        "featureDim": feature_dim(),
        "means": {
            DOWNSTREAM: mu_c,
            UP_F: mu_f,
            UP_S: mu_s,
            UP_ALT: mu_a,
            "rain": mu_r,
        },
        "stds": {
            DOWNSTREAM: sd_c,
            UP_F: sd_f,
            UP_S: sd_s,
            UP_ALT: sd_a,
            "rain": sd_r,
        },
        "lags": {
            UP_F: {"steps": lag_f, "minutes": lag_f * ft.STEP_MINUTES, "r": None if r_f is None else round(r_f, 6)},
            UP_S: {"steps": lag_s, "minutes": lag_s * ft.STEP_MINUTES, "r": None if r_s is None else round(r_s, 6)},
        },
        "layers": layers,
        "reliable": adopt,
        "skill120": hybrid_skill120,
    }
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    args.weights.write_text(json.dumps(weights) + "\n")

    summary = {
        "version": 1,
        "kind": "cento_hybrid_volume_mlp",
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
        "note": "z-score train-only per station; DNN predicts θ; physics rollout in z-space; metrics in metres.",
        "downstream": DOWNSTREAM,
        "upstreams": [UP_F, UP_S],
        "upstreamAlt": UP_ALT,
        "lookbackSteps": LOOKBACK,
        "stepMinutes": ft.STEP_MINUTES,
        "horizonsMinutes": [h * ft.STEP_MINUTES for h in HORIZONS],
        "means": weights["means"],
        "stds": weights["stds"],
        "lags": weights["lags"],
        "spikeCounts": spike_counts,
        "trainHistory": history,
        "nTrain": int(len(tr[0])),
        "nValidate": int(len(va[0])),
        "nHoldout": int(len(hold_pack[0])) if hold_pack else 0,
        "metrics": {
            "eval": metrics,
            "train": train_metrics,
            "validate": val_metrics,
            "holdout": hold_metrics,
        },
        "weightsFile": str(args.weights.relative_to(ROOT)) if args.weights.is_relative_to(ROOT) else str(args.weights),
        "gate": {
            "dnnSkill120Events": round(dnn_skill, 6),
            "hybridSkill120Events": hybrid_skill120,
            "adoptRuntime": adopt,
            "reason": "hybrid_beats_dnn" if adopt else "hybrid_below_dnn",
        },
        "fittedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["gate"], indent=2))
    print(f"Wrote {args.out} and {args.weights}")
    for minutes, block in metrics.items():
        ev = block["events"]
        print(f"+{minutes}′ events skill={ev['skill']} rmse={ev['rmse']} n={ev['n']}")


if __name__ == "__main__":
    main()
