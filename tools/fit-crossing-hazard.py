#!/usr/bin/env python3
"""WP2.2: discrete-time hazard model for first threshold crossing at Cento.

For each relative offset (+0.5/+1.0/+1.5 m) and each 30′ bin k=1..K (K=12 → 6 h):
  h_k = P(first cross in bin k | survived to k).
Censoring is correct: survivors through K contribute only survival terms.
P(no crossing in 6 h) = ∏_k (1 − h_k).

Uso:
  uv run python tools/fit-crossing-hazard.py --export models/cento-crossing-hazard.json
  uv run python tools/fit-crossing-hazard.py --no-chiusa --export models/cento-crossing-hazard-nochiusa.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS_M = (0.5, 1.0, 1.5)
CENSOR_MIN = 360
STEP_MIN = 30
N_BINS = CENSOR_MIN // STEP_MIN  # 12
HIDDEN = (128, 64)
LR = 1e-3
BATCH = 512
EPOCHS = 80
PATIENCE = 10
SEED = 42


def _load_ml():
    path = ROOT / "tools" / "fit-ml-cento.py"
    spec = importlib.util.spec_from_file_location("fit_ml_cento", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["fit_ml_cento"] = mod
    spec.loader.exec_module(mod)
    return mod


def crossing_bin_labels(level: pd.Series, offset: float, n_bins: int) -> np.ndarray:
    """Return bin index 1..n_bins of first cross, or 0 if censored / never."""
    vals = level.to_numpy(dtype=float)
    n = len(vals)
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        y0 = vals[i]
        if not np.isfinite(y0):
            out[i] = -1
            continue
        thr = y0 + offset
        hi = min(n, i + n_bins + 1)
        hit = 0
        for j in range(i + 1, hi):
            v = vals[j]
            if not np.isfinite(v):
                continue
            if v >= thr:
                hit = j - i  # 1..n_bins
                break
        out[i] = hit if 1 <= hit <= n_bins else 0
    return out


class HazardMLP(nn.Module):
    def __init__(self, n_in: int, n_bins: int, hidden: tuple[int, ...] = HIDDEN):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, n_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits (B, K)


def discrete_hazard_nll(logits: torch.Tensor, event_bin: torch.Tensor) -> torch.Tensor:
    """event_bin: 0 = censored after K, 1..K = fail at that bin (1-indexed)."""
    hazards = torch.sigmoid(logits)  # (B, K)
    surv = 1.0 - hazards
    log_surv = torch.log(surv.clamp_min(1e-7))
    log_haz = torch.log(hazards.clamp_min(1e-7))
    B, K = hazards.shape
    # cumulative log survival before bin k (0-indexed prefix)
    # For fail at k (1-indexed): sum_{j<k} log(1-h_j) + log(h_{k-1})
    # For censor: sum_all log(1-h_j)
    loss = torch.zeros(B, device=logits.device)
    cens = event_bin == 0
    if cens.any():
        loss[cens] = -log_surv[cens].sum(dim=1)
    for k in range(1, K + 1):
        m = event_bin == k
        if not m.any():
            continue
        # survive bins 0..k-2, fail at k-1
        if k > 1:
            loss[m] = -log_surv[m, : k - 1].sum(dim=1) - log_haz[m, k - 1]
        else:
            loss[m] = -log_haz[m, 0]
    return loss.mean()


def hazards_to_mode_and_p(hazards: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """hazards (N,K) → modeMin, pCross, survival curve S."""
    n, k = hazards.shape
    surv = np.cumprod(1.0 - hazards, axis=1)
    # P(fail in bin j) = h_j * S_{j-1}, S_{-1}=1
    pmf = np.empty_like(hazards)
    pmf[:, 0] = hazards[:, 0]
    if k > 1:
        pmf[:, 1:] = hazards[:, 1:] * surv[:, :-1]
    p_cross = pmf.sum(axis=1)
    mode_bin = pmf.argmax(axis=1) + 1  # 1..K
    mode_min = mode_bin.astype(float) * STEP_MIN
    # If most mass on censor, mode = censor
    p_cens = 1.0 - p_cross
    mode_min = np.where(p_cens >= pmf.max(axis=1), float(CENSOR_MIN), mode_min)
    return mode_min, p_cross, surv[:, -1]


def export_layers(model: HazardMLP) -> list[dict]:
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    layers = []
    for i, linear in enumerate(linears):
        w = linear.weight.detach().cpu().numpy()
        b = linear.bias.detach().cpu().numpy()
        layers.append(
            {
                "W": [[round(float(v), 8) for v in row] for row in w],
                "b": [round(float(v), 8) for v in b],
                "activation": "linear" if i == len(linears) - 1 else "relu",
            }
        )
    return layers


def train_hazard(
    X_std: np.ndarray,
    y_bin: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
    n_bins: int,
) -> tuple[HazardMLP, np.ndarray, dict]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = HazardMLP(X_std.shape[1], n_bins)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x_tr = torch.tensor(X_std[train_m], dtype=torch.float32)
    y_tr = torch.tensor(y_bin[train_m], dtype=torch.long)
    x_va = torch.tensor(X_std[val_m], dtype=torch.float32)
    y_va = torch.tensor(y_bin[val_m], dtype=torch.long)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH, shuffle=True)
    best_state = None
    best_val = float("inf")
    patience = 0
    hist = {"best_epoch": 0, "best_val_nll": None}
    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = discrete_hazard_nll(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_nll = float(discrete_hazard_nll(model(x_va), y_va).item())
        if val_nll < best_val - 1e-8:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            hist["best_epoch"] = epoch + 1
            hist["best_val_nll"] = round(best_val, 8)
        else:
            patience += 1
        if patience >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_std, dtype=torch.float32)).numpy()
    hazards = 1.0 / (1.0 + np.exp(-logits))
    print(
        f"  hazard done in {time.perf_counter() - t0:.1f}s  "
        f"best_epoch={hist['best_epoch']} val_nll={hist['best_val_nll']}"
    )
    return model, hazards, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "storico")
    ap.add_argument("--events", type=Path, default=ROOT / "models" / "events.json")
    ap.add_argument("--export", type=Path, default=ROOT / "models" / "cento-crossing-hazard.json")
    ap.add_argument("--no-chiusa", action="store_true")
    ap.add_argument("--flood-only", action="store_true", default=True)
    ap.add_argument("--split", choices=("alt", "block"), default="block")
    ap.add_argument("--gate-p", type=float, default=0.35, help="Default P(crossing) decode gate")
    args = ap.parse_args()

    ml = _load_ml()
    ft = ml.ft
    ups = list(ml.UPSTREAMS_NO_CHIUSA if args.no_chiusa else ml.UPSTREAMS)
    needed = [ml.DOWNSTREAM, *ups]

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    wide = ft.load_wide(args.data, list(config["sensors"].keys()))
    wide, _ = ft.filter_spikes(wide, spike_abs=ft.SPIKE_ABS)
    gauges = []
    for g in (config.get("rainfall") or {}).get("gauges") or []:
        name = g if isinstance(g, str) else (g or {}).get("name")
        if name:
            gauges.append(name)
    rain = ft.load_rain_basin(args.data / "rain", gauges, wide.index) if gauges else None
    wide = wide[needed].copy()

    feats, feature_cols = ml.build_feature_matrix(
        wide, rain, lookback=ml.LOOKBACK, stations=needed, poly_points=ml.POLY_NAIVE_POINTS
    )
    frame = feats[feature_cols].copy()
    frame["y0"] = wide[ml.DOWNSTREAM]
    for off in THRESHOLDS_M:
        frame[f"bin_{off:g}"] = crossing_bin_labels(wide[ml.DOWNSTREAM], off, N_BINS)

    miss_cols = [c for c in feature_cols if c.startswith("miss_")]
    ok = frame["y0"].notna()
    if miss_cols:
        ok &= frame[miss_cols].notna().all(axis=1)
    for off in THRESHOLDS_M:
        ok &= frame[f"bin_{off:g}"] >= 0
    frame = frame.loc[ok]
    index = frame.index
    X = frame[feature_cols]
    train_m, val_m, hold_m = ml.split_masks(index, mode=args.split)
    events = ml.load_catalog_events(args.events, ml.DOWNSTREAM)
    if args.flood_only:
        flood_m = ml.flood_window_mask(index, events)
        train_m = train_m & flood_m
        val_m = val_m & flood_m
        hold_m = hold_m & flood_m
    print(f"samples train={int(train_m.sum())} val={int(val_m.sum())} hold={int(hold_m.sum())}")
    if int(train_m.sum()) < ml.WEAK_POINTS:
        raise SystemExit("too few train samples")

    X_std, stats = ml.standardize_features(X, feature_cols, train_m)
    mean = [stats[c][0] for c in feature_cols]
    std = [stats[c][1] for c in feature_cols]

    thresholds_out: dict = {}
    for off in THRESHOLDS_M:
        y_bin = frame[f"bin_{off:g}"].to_numpy(dtype=np.int64)
        print(f"=== offset +{off:g} m  hit_rate_train={float((y_bin[train_m] > 0).mean()):.3f}")
        model, hazards, hist = train_hazard(X_std, y_bin, train_m, val_m, N_BINS)
        mode_min, p_cross, s_end = hazards_to_mode_and_p(hazards)
        # Validate: MAE on hits only + Brier on p_cross
        hits = val_m & (y_bin > 0)
        mae = None
        if hits.any():
            true_min = y_bin[hits].astype(float) * STEP_MIN
            mae = float(np.mean(np.abs(mode_min[hits] - true_min)))
        brier = float(np.mean((p_cross[val_m] - (y_bin[val_m] > 0).astype(float)) ** 2))
        print(f"  val mae_mode(hits)={mae}  brier_pCross={brier:.4f}")
        thresholds_out[str(off)] = {
            "offsetM": off,
            "censorMin": CENSOR_MIN,
            "nBins": N_BINS,
            "stepMinutes": STEP_MIN,
            "kind": "discrete_hazard",
            "layers": export_layers(model),
            "validateMaeHits": mae,
            "validateBrier": brier,
            "earlyStopping": hist,
        }

    use_rain = rain is not None and {"rain6h", "rain24h"}.issubset(set(rain.columns))
    payload = {
        "version": 1,
        "kind": "cento_crossing_hazard",
        "downstream": ml.DOWNSTREAM,
        "upstreams": ups,
        "stations": needed,
        "lookbackSteps": ml.LOOKBACK,
        "stepMinutes": STEP_MIN,
        "featureNames": feature_cols,
        "featureRecipe": {
            "usePolyDerivatives": True,
            "polyDerivPoints": ml.POLY_NAIVE_POINTS,
            "polyDerivDegree": ml.POLY_NAIVE_DEGREE,
        },
        "mean": [round(m, 8) for m in mean],
        "std": [round(s, 8) for s in std],
        "useRain": use_rain,
        "floodOnly": bool(args.flood_only),
        "noChiusa": bool(args.no_chiusa),
        "adopt": False,
        "decodeGateP": float(args.gate_p),
        "thresholds": thresholds_out,
        "primaryOffsetM": 1.0,
        "disclaimer": "Sperimentale: hazard discreto di attraversamento, non allertistico.",
        "trainedAt": pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z"),
    }
    args.export.parent.mkdir(parents=True, exist_ok=True)
    args.export.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.export}")


if __name__ == "__main__":
    main()
