#!/usr/bin/env python3
"""Offline Cento forecast: LightGBM vs small MLP (experimental).

Features (lookback L=48 × 30′ = 24 h) per station in
  Cento + monte (Forcelli, Sostegno, Bonconvento, Casalecchio tiro a volo)
  + valle acceptors (Dosso, Opera Reno Panfilia, Secondo Salto, Gallo):
  level[t-k], delta[t-k]=level[t-k]-level[t-k-1] (NaN if either missing), miss[t-k].
Plus rain6h / rain24h basin scalars. Acceptors are backwater boundary features,
not chase targets; at runtime future acceptor levels stay at last observation.

Target: Δy(t→t+h) for h ∈ {1,2,3,4}. Same month-block split as fit-transfer.
Writes models/ml-cento-compare.json — patch transfer-coeffs only with --export-coeffs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DOWNSTREAM = "Cento"
# Direct deps + Bonconvento (covers Sostegno gaps) + Casalecchio tiro a volo (earlier Reno).
UPSTREAMS = ["Forcelli", "Sostegno Reno", "Bonconvento", "Casalecchio tiro a volo"]
# Downstream acceptors (backwater): not chase targets.
ACCEPTORS = ["Dosso", "Opera Reno Panfilia", "Secondo Salto", "Gallo"]
LOOKBACK = 48  # 24 h at 30′
HORIZONS = [1, 2, 3, 4]
EVENT_ABS = 0.10
WEAK_POINTS = 48
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}
LGBM_ROUNDS = 800
LGBM_EARLY = 50
DNN_HIDDEN = (128, 64)
DNN_LR = 1e-3
DNN_BATCH = 512
DNN_EPOCHS = 80
DNN_PATIENCE = 10
DNN_SEED = 42


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


def safe_name(station: str) -> str:
    return station.replace(" ", "_")


def build_feature_matrix(
    wide: pd.DataFrame,
    rain: pd.DataFrame | None,
    lookback: int = LOOKBACK,
    stations: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Flat window of level/delta/miss + rain. Index aligned to wide."""
    parts: dict[str, pd.Series] = {}
    feature_cols: list[str] = []
    station_order = list(stations) if stations is not None else list(wide.columns)

    for station in station_order:
        if station not in wide.columns:
            raise SystemExit(f"Missing station column: {station}")
        level = wide[station]
        prev = level.shift(1)
        delta = level - prev
        delta = delta.where(level.notna() & prev.notna())
        miss = level.isna().astype(float)
        tag = safe_name(station)
        for k in range(lookback):
            lv = level.shift(k)
            dv = delta.shift(k)
            mv = miss.shift(k)
            # First rows of the series: miss after shift is NaN → treat as missing
            mv = mv.fillna(1.0)
            c_lv, c_dv, c_mv = f"level_{tag}_lag{k}", f"delta_{tag}_lag{k}", f"miss_{tag}_lag{k}"
            parts[c_lv] = lv
            parts[c_dv] = dv
            parts[c_mv] = mv
            feature_cols.extend([c_lv, c_dv, c_mv])

    if rain is not None and {"rain6h", "rain24h"}.issubset(rain.columns):
        parts["rain6h"] = rain["rain6h"]
        parts["rain24h"] = rain["rain24h"]
        feature_cols.extend(["rain6h", "rain24h"])

    feats = pd.DataFrame(parts, index=wide.index)
    return feats, feature_cols


def build_targets(wide: pd.DataFrame, h: int) -> pd.DataFrame:
    y = wide[DOWNSTREAM]
    return pd.DataFrame(
        {
            "y0": y,
            "y_future": y.shift(-h),
            "delta": y.shift(-h) - y,
        },
        index=wide.index,
    )


def align_xy(
    feats: pd.DataFrame,
    feature_cols: list[str],
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Keep rows with finite target y0 and delta; features may contain NaN."""
    frame = feats[feature_cols].copy()
    frame["y0"] = targets["y0"]
    frame["delta"] = targets["delta"]
    ok = frame["y0"].notna() & frame["delta"].notna()
    # Need a full lookback of miss flags defined (after initial warm-up)
    miss_cols = [c for c in feature_cols if c.startswith("miss_")]
    if miss_cols:
        ok &= frame[miss_cols].notna().all(axis=1)
    frame = frame.loc[ok]
    X = frame[feature_cols]
    y = frame["delta"].to_numpy(dtype=float)
    y0 = frame["y0"].to_numpy(dtype=float)
    return X, y, y0, frame.index.to_numpy()


def metrics_block(
    y0: np.ndarray,
    delta_obs: np.ndarray,
    delta_pred: np.ndarray,
) -> dict:
    obs = y0 + delta_obs
    pred = y0 + delta_pred
    persist = y0
    mask = np.isfinite(obs) & np.isfinite(pred)
    n = int(mask.sum())
    if n < 8:
        return {"rmse": None, "skill": None, "mae": None, "n": n}
    err_m = float(math.sqrt(np.mean((pred[mask] - obs[mask]) ** 2)))
    err_p = float(math.sqrt(np.mean((persist[mask] - obs[mask]) ** 2)))
    skill = None if err_p == 0 else 1.0 - err_m / err_p
    mae = float(np.mean(np.abs(pred[mask] - obs[mask])))
    return {
        "rmse": round(err_m, 6),
        "skill": None if skill is None else round(skill, 6),
        "mae": round(mae, 6),
        "n": n,
    }


def split_masks(index: pd.DatetimeIndex | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = pd.DatetimeIndex(index)
    blocks = ft.month_blocks(idx)
    return blocks == 0, blocks == 1, blocks == 2


def event_mask(delta: np.ndarray, base: np.ndarray) -> np.ndarray:
    return base & (np.abs(delta) >= EVENT_ABS)


def eval_splits(
    y0: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
    hold_m: np.ndarray,
) -> dict:
    return {
        "train": metrics_block(y0[train_m], y[train_m], pred[train_m]),
        "validate": metrics_block(y0[val_m], y[val_m], pred[val_m]),
        "validateEvents": metrics_block(
            y0[event_mask(y, val_m)],
            y[event_mask(y, val_m)],
            pred[event_mask(y, val_m)],
        ),
        "holdout": metrics_block(y0[hold_m], y[hold_m], pred[hold_m]),
        "holdoutEvents": metrics_block(
            y0[event_mask(y, hold_m)],
            y[event_mask(y, hold_m)],
            pred[event_mask(y, hold_m)],
        ),
    }


def train_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
) -> tuple[lgb.Booster, np.ndarray]:
    dtrain = lgb.Dataset(X.iloc[train_m], label=y[train_m], free_raw_data=False)
    dval = lgb.Dataset(X.iloc[val_m], label=y[val_m], reference=dtrain, free_raw_data=False)
    booster = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=LGBM_ROUNDS,
        valid_sets=[dval],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(LGBM_EARLY, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    pred = booster.predict(X, num_iteration=booster.best_iteration)
    return booster, np.asarray(pred, dtype=float)


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...] = DNN_HIDDEN):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def standardize_features(
    X: pd.DataFrame,
    feature_cols: list[str],
    train_m: np.ndarray,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Z-score continuous cols on train; miss_* left as-is; NaN → 0 after scale."""
    arr = X[feature_cols].to_numpy(dtype=float)
    stats: dict[str, tuple[float, float]] = {}
    out = np.zeros_like(arr, dtype=float)
    for j, col in enumerate(feature_cols):
        col_data = arr[:, j]
        if col.startswith("miss_"):
            out[:, j] = np.nan_to_num(col_data, nan=1.0)
            stats[col] = (0.0, 1.0)
            continue
        base = col_data[train_m]
        base = base[np.isfinite(base)]
        mu = float(base.mean()) if len(base) else 0.0
        sd = float(base.std()) if len(base) > 1 else 1.0
        if sd < 1e-6:
            sd = 1.0
        scaled = (col_data - mu) / sd
        out[:, j] = np.nan_to_num(scaled, nan=0.0)
        stats[col] = (mu, sd)
    return out, stats


def train_dnn(
    X_std: np.ndarray,
    y: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
) -> tuple[MLP, np.ndarray, dict]:
    torch.manual_seed(DNN_SEED)
    np.random.seed(DNN_SEED)
    device = torch.device("cpu")
    model = MLP(X_std.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=DNN_LR)
    loss_fn = nn.MSELoss()

    x_tr = torch.tensor(X_std[train_m], dtype=torch.float32)
    y_tr = torch.tensor(y[train_m], dtype=torch.float32)
    x_va = torch.tensor(X_std[val_m], dtype=torch.float32)
    y_va = torch.tensor(y[val_m], dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=DNN_BATCH, shuffle=True)

    best_state = None
    best_val = float("inf")
    patience = 0
    history = {"best_epoch": 0, "best_val_mse": None}

    for epoch in range(DNN_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(x_va)
            val_mse = float(loss_fn(val_pred, y_va).item())
        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            history["best_epoch"] = epoch + 1
            history["best_val_mse"] = round(best_val, 8)
        else:
            patience += 1
            if patience >= DNN_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        full = torch.tensor(X_std, dtype=torch.float32)
        pred = model(full).cpu().numpy().astype(float)
    return model, pred, history


def gauge_names(config: dict) -> list[str]:
    raw = list((config.get("rainfall") or {}).get("gauges") or [])
    names: list[str] = []
    for g in raw:
        if isinstance(g, str):
            names.append(g)
        elif isinstance(g, dict) and g.get("name"):
            names.append(str(g["name"]))
    return names


def export_mlp_layers(model: MLP) -> list[dict]:
    """Serialize Linear layers as row-major W (out x in) + bias + activation."""
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    layers: list[dict] = []
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


def residual_spreads_pair(
    y0: np.ndarray,
    delta_obs: np.ndarray,
    delta_pred: np.ndarray,
    mask: np.ndarray,
    q: float = 0.85,
) -> tuple[float | None, float | None]:
    obs = y0[mask] + delta_obs[mask]
    pred = y0[mask] + delta_pred[mask]
    return ft.residual_spreads(obs, pred, q=q)


def pack_spreads(
    y0: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    val_m: np.ndarray,
) -> dict:
    val_obs_delta = y
    lo_all, hi_all = residual_spreads_pair(y0, val_obs_delta, pred, val_m)
    event = val_m & (np.abs(y) >= EVENT_ABS)
    calm = val_m & ~event
    lo_base, hi_base = residual_spreads_pair(y0, val_obs_delta, pred, calm)
    lo_high, hi_high = residual_spreads_pair(y0, val_obs_delta, pred, event, q=0.95)
    if lo_base is None:
        lo_base, hi_base = lo_all, hi_all
    if lo_high is None:
        lo_high, hi_high = lo_all, hi_all
    if lo_high is not None and lo_base is not None:
        lo_high = max(lo_high, lo_base) * 1.25
        hi_high = max(hi_high, hi_base) * 1.25
    return {
        "spreadLow": None if lo_base is None else round(lo_base, 6),
        "spreadHigh": None if hi_base is None else round(hi_base, 6),
        "spreadLowHigh": None if lo_high is None else round(lo_high, 6),
        "spreadHighHigh": None if hi_high is None else round(hi_high, 6),
    }


def ridge_side_by_side(coeffs_path: Path, h_minutes: int) -> dict | None:
    if not coeffs_path.exists():
        return None
    try:
        payload = json.loads(coeffs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    node = (payload.get("nodes") or {}).get(DOWNSTREAM) or {}
    block = (node.get("horizons") or {}).get(str(h_minutes))
    if not block:
        return None
    keys = ("train", "validate", "validateEvents", "holdout", "holdoutEvents")
    out = {}
    for k in keys:
        m = block.get(k) or {}
        out[k] = {
            "rmse": m.get("rmse"),
            "skill": m.get("skill"),
            "mae": m.get("mae"),
            "n": m.get("n"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline LightGBM vs DNN for Cento")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data", type=Path, default=ROOT / "storico")
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: ml-cento-compare-valle.json (or -novalle with --no-acceptors)")
    parser.add_argument("--coeffs", type=Path, default=ROOT / "models" / "transfer-coeffs.json")
    parser.add_argument("--export-dnn", type=Path, default=None,
                        help="Default: models/cento-dnn-valle.json (or cento-dnn-novalle.json with --no-acceptors)")
    parser.add_argument(
        "--export-coeffs",
        type=Path,
        nargs="?",
        const=ROOT / "models" / "transfer-coeffs.json",
        default=None,
        help="Patch nodes.Cento in transfer-coeffs (opt-in; pass flag alone for default path)",
    )
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--spike-abs", type=float, default=ft.SPIKE_ABS)
    parser.add_argument("--no-rain", action="store_true")
    parser.add_argument("--no-acceptors", action="store_true",
                        help="Baseline without valle acceptors (ablation)")
    parser.add_argument("--skip-lgbm", action="store_true", help="Train/export DNN only")
    args = parser.parse_args()

    acceptors = [] if args.no_acceptors else list(ACCEPTORS)
    station_list = [DOWNSTREAM, *UPSTREAMS, *acceptors]
    if args.export_dnn is None:
        args.export_dnn = ROOT / "models" / (
            "cento-dnn-novalle.json" if args.no_acceptors else "cento-dnn-valle.json"
        )
    if args.out is None:
        args.out = ROOT / "models" / (
            "ml-cento-compare-novalle.json" if args.no_acceptors else "ml-cento-compare-valle.json"
        )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    stations = list(config["sensors"].keys())
    needed = [s for s in station_list if s in stations]
    missing = [s for s in station_list if s not in stations]
    if missing:
        raise SystemExit(f"Stazioni assenti da config: {missing}")

    print(f"Carico stazioni da {args.data}…")
    wide_all = ft.load_wide(args.data, stations)
    wide_all, spike_counts = ft.filter_spikes(wide_all, spike_abs=args.spike_abs)
    spiked = {k: v for k, v in spike_counts.items() if v}
    print(f"Griglia {ft.STEP_MINUTES}′: {len(wide_all)} step; spike: {spiked or 'nessuno'}")
    wide = wide_all[needed].copy()

    rain = None
    gauges = gauge_names(config)
    if not args.no_rain:
        rain = ft.load_rain_basin(args.data / "rain", gauges, wide_all.index)
        if rain is not None:
            print(f"Pioggia bacino: {int(rain['rain6h'].notna().sum())} timestep ({len(gauges)} gauge)")
        else:
            print("Pioggia: assente, fit solo livelli")

    feats, feature_cols = build_feature_matrix(wide, rain, lookback=args.lookback, stations=needed)
    use_rain = rain is not None and {"rain6h", "rain24h"}.issubset(rain.columns)
    print(f"Feature: {len(feature_cols)} (lookback={args.lookback}, stations={needed}, rain={use_rain})")

    horizons_out: dict[str, dict] = {}
    export_horizons: dict[str, dict] = {}

    for h in HORIZONS:
        print(f"Orizzonte {h * ft.STEP_MINUTES}′…")
        targets = build_targets(wide, h)
        X, y, y0, index = align_xy(feats, feature_cols, targets)
        train_m, val_m, hold_m = split_masks(index)
        if train_m.sum() < WEAK_POINTS or val_m.sum() < WEAK_POINTS:
            print(f"  skip: train={train_m.sum()} val={val_m.sum()}")
            continue

        print(
            f"  n={len(y)} train={int(train_m.sum())} "
            f"val={int(val_m.sum())} hold={int(hold_m.sum())}"
        )

        lgbm_metrics = None
        best_iteration = 0
        if not args.skip_lgbm:
            booster, lgbm_pred = train_lgbm(X, y, train_m, val_m)
            lgbm_metrics = eval_splits(y0, y, lgbm_pred, train_m, val_m, hold_m)
            best_iteration = int(booster.best_iteration or 0)
            print(
                f"  LightGBM val skill={lgbm_metrics['validate']['skill']} "
                f"evt={lgbm_metrics['validateEvents']['skill']} "
                f"iter={best_iteration}"
            )

        X_std, stats = standardize_features(X, feature_cols, train_m)
        model, dnn_pred, dnn_hist = train_dnn(X_std, y, train_m, val_m)
        dnn_metrics = eval_splits(y0, y, dnn_pred, train_m, val_m, hold_m)
        spreads = pack_spreads(y0, y, dnn_pred, val_m)
        print(
            f"  DNN      val skill={dnn_metrics['validate']['skill']} "
            f"evt={dnn_metrics['validateEvents']['skill']} "
            f"epoch={dnn_hist['best_epoch']}"
        )

        key = str(h * ft.STEP_MINUTES)
        export_horizons[key] = {
            "mean": [round(stats[c][0], 8) for c in feature_cols],
            "std": [round(stats[c][1], 8) for c in feature_cols],
            "layers": export_mlp_layers(model),
            **spreads,
            "train": dnn_metrics["train"],
            "validate": dnn_metrics["validate"],
            "validateEvents": dnn_metrics["validateEvents"],
            "holdout": dnn_metrics["holdout"],
            "holdoutEvents": dnn_metrics["holdoutEvents"],
            "earlyStopping": dnn_hist,
        }

        ridge = ridge_side_by_side(args.coeffs, h * ft.STEP_MINUTES)
        block = {
            "horizonMinutes": h * ft.STEP_MINUTES,
            "nFeatures": len(feature_cols),
            "nSamples": {
                "train": int(train_m.sum()),
                "validate": int(val_m.sum()),
                "holdout": int(hold_m.sum()),
            },
            "dnn": {
                **dnn_metrics,
                "architecture": list(DNN_HIDDEN) + [1],
                "earlyStopping": dnn_hist,
            },
            "ridge": ridge,
        }
        if lgbm_metrics is not None:
            block["lightgbm"] = {**lgbm_metrics, "bestIteration": best_iteration}
        horizons_out[key] = block

    skill120 = (export_horizons.get("120") or {}).get("validateEvents", {}).get("skill")
    reliable = skill120 is not None and skill120 >= ft.SKILL_KEEP
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    dnn_node = {
        "kind": "window_delta_mlp",
        "downstream": DOWNSTREAM,
        "upstreams": UPSTREAMS,
        "acceptors": acceptors,
        "stations": needed,
        "lookbackSteps": args.lookback,
        "stepMinutes": ft.STEP_MINUTES,
        "featureNames": feature_cols,
        "useRain": use_rain,
        "rainGauges": gauges if use_rain else [],
        "horizons": export_horizons,
        "skill120": skill120,
        "reliable": reliable,
        "trainedAt": trained_at,
        "thresholdOn": "event_or_move",
        "note": (
            "Acceptors are valle backwater features (frozen at last obs in forecast)."
            if acceptors else "Ablation without valle acceptors."
        ),
    }

    compare = {
        "version": 1,
        "kind": "cento_lgbm_vs_dnn",
        "downstream": DOWNSTREAM,
        "upstreams": UPSTREAMS,
        "acceptors": acceptors,
        "stations": needed,
        "lookbackSteps": args.lookback,
        "lookbackMinutes": args.lookback * ft.STEP_MINUTES,
        "stepMinutes": ft.STEP_MINUTES,
        "horizonsMinutes": [h * ft.STEP_MINUTES for h in HORIZONS],
        "features": feature_cols,
        "useRain": use_rain,
        "disclaimer": "Sperimentale: non operativo, non allertistico.",
        "trainedAt": trained_at,
        "horizons": horizons_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(compare, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")

    if args.export_dnn:
        args.export_dnn.parent.mkdir(parents=True, exist_ok=True)
        args.export_dnn.write_text(
            json.dumps({"version": 1, "downstream": DOWNSTREAM, **dnn_node}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.export_dnn}")

    if args.export_coeffs and args.export_coeffs.exists():
        coeffs = json.loads(args.export_coeffs.read_text(encoding="utf-8"))
        coeffs.setdefault("nodes", {})[DOWNSTREAM] = dnn_node
        # Keep decorative edges; mark kind for Cento edges.
        for up in UPSTREAMS[:2]:  # direct deps stay as graph edges
            key = f"{up}->{DOWNSTREAM}"
            if key in (coeffs.get("edges") or {}):
                coeffs["edges"][key]["kind"] = "window_delta_mlp"
                coeffs["edges"][key]["reliable"] = reliable
                coeffs["edges"][key]["skill120"] = skill120
                coeffs["edges"][key]["trainedAt"] = trained_at
        args.export_coeffs.write_text(json.dumps(coeffs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Patched {args.export_coeffs} nodes.{DOWNSTREAM} [{('OK' if reliable else 'no-skill')}] skill120={skill120}")
    elif args.export_coeffs:
        print(f"Skip coeffs patch: {args.export_coeffs} missing")


if __name__ == "__main__":
    main()

