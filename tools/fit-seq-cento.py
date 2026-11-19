#!/usr/bin/env python3
"""Offline end-to-end sequence models (GRU / 1D-CNN / Transformer) for a station.

Same data / month-block split / event skill@120 gate as fit-ml-cento (novalle).
Input tensor [N, L=48, C]: per station (level, delta, miss) + rain6h/rain24h
along the lookback (oldest→newest). Target: Δy(t→t+h), h∈{1,2,3,4}.

Default downstream=Cento. Use --downstream NAME for other nodes (upstreams from
config.dependencies). Does not patch transfer-coeffs. Promote (reliable) only if
validate+holdout event skill@120 beats the production bar for that station.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

_VERBOSE = True

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNSTREAM = "Cento"
DEFAULT_UPSTREAMS = ["Forcelli", "Sostegno Reno", "Bonconvento", "Casalecchio tiro a volo"]
LOOKBACK = 48
HORIZONS = [1, 2, 3, 4]
EVENT_ABS = 0.10
WEAK_POINTS = 48
GRU_HIDDEN = 64
CNN_CHANNELS = (32, 64)
CNN_KERNEL = 5
TF_D_MODEL = 64
TF_NHEAD = 4
TF_LAYERS = 2
TF_DIM_FF = 128
TF_DROPOUT = 0.1
TF_MAX_LEN = 64
LR = 1e-3
BATCH = 256
EPOCHS = 60
PATIENCE = 10
SEED = 42


def log(msg: str = "", *, level: str = "info") -> None:
    if level == "debug" and not _VERBOSE:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"info": "·", "debug": "…", "ok": "✓", "warn": "!"}.get(level, "·")
    print(f"[{ts}] {prefix} {msg}", flush=True)


def log_step(title: str) -> None:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === {title} ===", flush=True)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ft = _load_module("fit_transfer", Path(__file__).resolve().parent / "fit-transfer.py")
ml = _load_module("fit_ml_cento", Path(__file__).resolve().parent / "fit-ml-cento.py")


def safe_name(station: str) -> str:
    return station.replace(" ", "_")


def station_slug(station: str) -> str:
    return station.lower().replace(" ", "-")


def resolve_upstreams(config: dict, downstream: str) -> list[str]:
    meta = (config.get("sensors") or {}).get(downstream) or {}
    ups = [u for u in (meta.get("dependencies") or []) if u in (config.get("sensors") or {})]
    if ups:
        return ups
    if downstream == DEFAULT_DOWNSTREAM:
        return list(DEFAULT_UPSTREAMS)
    return []


def build_channel_matrix(
    wide: pd.DataFrame,
    rain: pd.DataFrame | None,
    stations: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Raw (n × C) channels: level/delta/miss per station + optional rain."""
    parts: list[np.ndarray] = []
    names: list[str] = []
    cont_names: list[str] = []
    n = len(wide)
    for station in stations:
        level = wide[station].to_numpy(dtype=float)
        prev = np.empty_like(level)
        prev[0] = np.nan
        prev[1:] = level[:-1]
        delta = level - prev
        miss = (~np.isfinite(level)).astype(np.float64)
        level_out = np.where(np.isfinite(level), level, np.nan)
        delta_out = np.where(np.isfinite(level) & np.isfinite(prev), delta, np.nan)
        tag = safe_name(station)
        for col, arr, continuous in (
            (f"level_{tag}", level_out, True),
            (f"delta_{tag}", delta_out, True),
            (f"miss_{tag}", miss, False),
        ):
            parts.append(arr)
            names.append(col)
            if continuous:
                cont_names.append(col)
    if rain is not None and {"rain6h", "rain24h"}.issubset(rain.columns):
        for col in ("rain6h", "rain24h"):
            parts.append(rain[col].to_numpy(dtype=float))
            names.append(col)
            cont_names.append(col)
    raw = np.column_stack(parts) if parts else np.zeros((n, 0), dtype=float)
    return raw, names, cont_names


def collect_windows(
    raw: np.ndarray,
    y_level: np.ndarray,
    lookback: int,
    h: int,
    index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows ending at t; targets Δy(t→t+h). Returns X[N,L,C], y, y0, times."""
    n, c = raw.shape
    if n < lookback + h + 8:
        return (
            np.zeros((0, lookback, c)),
            np.zeros(0),
            np.zeros(0),
            np.array([], dtype="datetime64[ns]"),
        )
    # axis=0 window → shape (n-L+1, C, L); transpose to (N, L, C)
    wins = sliding_window_view(raw, lookback, axis=0)
    max_i = n - lookback - h  # inclusive last i with t+h in range
    if max_i < 0:
        return (
            np.zeros((0, lookback, c)),
            np.zeros(0),
            np.zeros(0),
            np.array([], dtype="datetime64[ns]"),
        )
    # wins[i] covers raw[i : i+L], ends at t=i+L-1
    wins = np.asarray(wins[: max_i + 1]).transpose(0, 2, 1)
    t_idx = np.arange(lookback - 1, lookback - 1 + len(wins))
    y0 = y_level[t_idx]
    y_fut = y_level[t_idx + h]
    ok = np.isfinite(y0) & np.isfinite(y_fut)
    X = wins[ok]
    y = (y_fut - y0)[ok]
    y0_ok = y0[ok]
    times = index.to_numpy()[t_idx[ok]]
    return X, y, y0_ok, times


def standardize_seq(
    X: np.ndarray,
    channel_names: list[str],
    cont_names: list[str],
    train_m: np.ndarray,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Z-score continuous channels on train windows; miss_* left as-is; NaN→0/1."""
    out = np.array(X, dtype=np.float64, copy=True)
    stats: dict[str, tuple[float, float]] = {}
    name_to_j = {n: j for j, n in enumerate(channel_names)}
    for name in channel_names:
        j = name_to_j[name]
        col = out[:, :, j]
        if name.startswith("miss_"):
            out[:, :, j] = np.nan_to_num(col, nan=1.0)
            stats[name] = (0.0, 1.0)
            continue
        if name not in cont_names:
            out[:, :, j] = np.nan_to_num(col, nan=0.0)
            stats[name] = (0.0, 1.0)
            continue
        base = col[train_m].ravel()
        base = base[np.isfinite(base)]
        mu = float(base.mean()) if len(base) else 0.0
        sd = float(base.std()) if len(base) > 1 else 1.0
        if sd < 1e-6:
            sd = 1.0
        scaled = (col - mu) / sd
        out[:, :, j] = np.nan_to_num(scaled, nan=0.0)
        stats[name] = (mu, sd)
    return out, stats


class SeqGRU(nn.Module):
    def __init__(self, n_in: int, hidden: int = GRU_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(input_size=n_in, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B × L × C
        _, h_n = self.gru(x)
        return self.head(h_n.squeeze(0)).squeeze(-1)


class SeqCNN(nn.Module):
    def __init__(
        self,
        n_in: int,
        channels: tuple[int, ...] = CNN_CHANNELS,
        kernel: int = CNN_KERNEL,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        pad = kernel // 2
        for ch in channels:
            layers.append(nn.Conv1d(prev, ch, kernel_size=kernel, padding=pad))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(prev, prev),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(prev, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B × L × C → Conv1d wants B × C × L
        z = self.conv(x.transpose(1, 2))
        return self.head(z[:, :, -1]).squeeze(-1)


class _TfEncLayer(nn.Module):
    """Pre-norm encoder block (MHA + ReLU FFN) — weights exported for JS parity."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.n2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        h = self.n2(x)
        ff = self.linear2(self.drop(torch.relu(self.linear1(h))))
        return x + self.drop(ff)


class SeqTransformer(nn.Module):
    def __init__(
        self,
        n_in: int,
        d_model: int = TF_D_MODEL,
        nhead: int = TF_NHEAD,
        nlayers: int = TF_LAYERS,
        dim_ff: int = TF_DIM_FF,
        dropout: float = TF_DROPOUT,
        max_len: int = TF_MAX_LEN,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.in_proj = nn.Linear(n_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(
            [_TfEncLayer(d_model, nhead, dim_ff, dropout) for _ in range(nlayers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B × L × C
        l = x.size(1)
        z = self.in_proj(x) + self.pos[:, :l, :]
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)
        return self.head(z[:, -1, :]).squeeze(-1)


def _round_matrix(arr: np.ndarray, nd: int = 8) -> list:
    return [[round(float(v), nd) for v in row] for row in arr]


def export_head_layers(head: nn.Sequential) -> list[dict]:
    linears = [m for m in head if isinstance(m, nn.Linear)]
    layers: list[dict] = []
    for i, linear in enumerate(linears):
        w = linear.weight.detach().cpu().numpy()
        b = linear.bias.detach().cpu().numpy()
        layers.append(
            {
                "W": _round_matrix(w),
                "b": [round(float(v), 8) for v in b],
                "activation": "linear" if i == len(linears) - 1 else "relu",
            }
        )
    return layers


def export_seq_gru(model: SeqGRU) -> dict:
    w_ih = model.gru.weight_ih_l0.detach().cpu().numpy()
    w_hh = model.gru.weight_hh_l0.detach().cpu().numpy()
    b_ih = model.gru.bias_ih_l0.detach().cpu().numpy()
    b_hh = model.gru.bias_hh_l0.detach().cpu().numpy()
    return {
        "gru": {
            "inputSize": int(w_ih.shape[1]),
            "hiddenSize": int(model.gru.hidden_size),
            "weightIh": _round_matrix(w_ih),
            "weightHh": _round_matrix(w_hh),
            "biasIh": [round(float(v), 8) for v in b_ih],
            "biasHh": [round(float(v), 8) for v in b_hh],
        },
        "headLayers": export_head_layers(model.head),
    }


def export_seq_cnn(model: SeqCNN) -> dict:
    conv_layers: list[dict] = []
    for m in model.conv:
        if isinstance(m, nn.Conv1d):
            w = m.weight.detach().cpu().numpy()  # (out, in, k)
            b = m.bias.detach().cpu().numpy()
            conv_layers.append(
                {
                    "W": [[[round(float(v), 8) for v in ker] for ker in inch] for inch in w],
                    "b": [round(float(v), 8) for v in b],
                    "kernelSize": int(m.kernel_size[0]),
                    "padding": int(m.padding[0]),
                    "activation": "relu",
                }
            )
    return {"convLayers": conv_layers, "headLayers": export_head_layers(model.head)}


def _export_linear(linear: nn.Linear) -> dict:
    w = linear.weight.detach().cpu().numpy()
    b = linear.bias.detach().cpu().numpy()
    return {
        "W": _round_matrix(w),
        "b": [round(float(v), 8) for v in b],
    }


def _export_layernorm(norm: nn.LayerNorm) -> dict:
    return {
        "weight": [round(float(v), 8) for v in norm.weight.detach().cpu().numpy()],
        "bias": [round(float(v), 8) for v in norm.bias.detach().cpu().numpy()],
        "eps": float(norm.eps),
    }


def export_seq_transformer(model: SeqTransformer) -> dict:
    layers_out: list[dict] = []
    for layer in model.layers:
        attn = layer.attn
        layers_out.append(
            {
                "norm1": _export_layernorm(layer.n1),
                "attn": {
                    "nhead": int(model.nhead),
                    "dModel": int(model.d_model),
                    "inProjWeight": _round_matrix(attn.in_proj_weight.detach().cpu().numpy()),
                    "inProjBias": [
                        round(float(v), 8) for v in attn.in_proj_bias.detach().cpu().numpy()
                    ],
                    "outProjWeight": _round_matrix(attn.out_proj.weight.detach().cpu().numpy()),
                    "outProjBias": [
                        round(float(v), 8) for v in attn.out_proj.bias.detach().cpu().numpy()
                    ],
                },
                "norm2": _export_layernorm(layer.n2),
                "linear1": _export_linear(layer.linear1),
                "linear2": _export_linear(layer.linear2),
            }
        )
    return {
        "transformer": {
            "dModel": int(model.d_model),
            "nhead": int(model.nhead),
            "nlayers": len(model.layers),
            "inProj": _export_linear(model.in_proj),
            "pos": _round_matrix(model.pos.detach().cpu().numpy()[0]),
            "layers": layers_out,
            "norm": _export_layernorm(model.norm),
        },
        "headLayers": export_head_layers(model.head),
    }


def train_seq(
    model: nn.Module,
    X_std: np.ndarray,
    y: np.ndarray,
    train_m: np.ndarray,
    val_m: np.ndarray,
    *,
    label: str,
) -> tuple[nn.Module, np.ndarray, dict]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    x_tr = torch.tensor(X_std[train_m], dtype=torch.float32)
    y_tr = torch.tensor(y[train_m], dtype=torch.float32)
    x_va = torch.tensor(X_std[val_m], dtype=torch.float32)
    y_va = torch.tensor(y[val_m], dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH, shuffle=True)

    n_tr, n_va = int(train_m.sum()), int(val_m.sum())
    n_batches = max(1, (n_tr + BATCH - 1) // BATCH)
    log(
        f"{label} start: in={X_std.shape[-1]} L={X_std.shape[1]} "
        f"train={n_tr} val={n_va} batch={BATCH} (~{n_batches}/epoch) "
        f"epochs≤{EPOCHS} patience={PATIENCE} device=cpu"
    )
    t0 = time.perf_counter()
    best_state = None
    best_val = float("inf")
    patience = 0
    history = {"best_epoch": 0, "best_val_mse": None}
    epoch_durs: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.perf_counter()
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred_b = model(xb)
            loss = loss_fn(pred_b, yb)
            loss.backward()
            opt.step()
            bs = int(xb.shape[0])
            running += float(loss.item()) * bs
            n_seen += bs
        train_mse = running / max(1, n_seen)
        model.eval()
        with torch.no_grad():
            val_mse = float(loss_fn(model(x_va), y_va).item())
        star = " "
        if val_mse < best_val - 1e-7:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
            history["best_epoch"] = epoch
            history["best_val_mse"] = round(best_val, 8)
            star = "*"
        else:
            patience += 1
        dt = time.perf_counter() - t_ep
        epoch_durs.append(dt)
        avg = sum(epoch_durs) / len(epoch_durs)
        # If still setting new bests, ETA toward EPOCHS; else toward early-stop.
        eta_epochs = (EPOCHS - epoch) if patience == 0 else max(0, PATIENCE - patience)
        eta_s = avg * eta_epochs
        if _VERBOSE:
            log(
                f"  {label} epoch {epoch}/{EPOCHS}{star}  "
                f"train_mse={train_mse:.6g}  val_mse={val_mse:.6g}  "
                f"best={best_val:.6g}@ep{history['best_epoch']}  "
                f"pat={patience}/{PATIENCE}  "
                f"{dt:.1f}s/ep  eta~{eta_s / 60:.1f}m",
                level="debug",
            )
        if patience >= PATIENCE:
            log(
                f"  {label} early stop at epoch {epoch} "
                f"(no improve for {PATIENCE} epochs)",
                level="info",
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_std, dtype=torch.float32)).numpy()
    log(
        f"{label} done in {time.perf_counter() - t0:.1f}s  "
        f"best_epoch={history['best_epoch']} best_val_mse={history['best_val_mse']}  "
        f"avg={sum(epoch_durs)/max(1,len(epoch_durs)):.1f}s/ep",
        level="ok",
    )
    return model, np.asarray(pred, dtype=float), history


def _finite_skill(value) -> float | None:
    if value is None:
        return None
    try:
        sk = float(value)
    except (TypeError, ValueError):
        return None
    return sk if math.isfinite(sk) else None


def mlp_ref_skills(
    path: Path, minutes: int = 120, downstream: str = DEFAULT_DOWNSTREAM
) -> dict[str, float | None]:
    """Read validateEvents + holdoutEvents.skillPersist @ horizon from compare or coeffs."""
    out = {"validateEvents": None, "holdoutPersist": None, "source": None}
    if not path.exists():
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    block = (payload.get("horizons") or {}).get(str(minutes))
    if block is None and isinstance(payload.get("nodes"), dict):
        node = (payload["nodes"].get(downstream) or {})
        block = (node.get("horizons") or {}).get(str(minutes))
        if block is not None:
            out["source"] = f"nodes.{downstream}"
            out["validateEvents"] = _finite_skill(
                (block.get("validateEvents") or {}).get("skill")
                or node.get("skill120")
            )
            he = block.get("holdoutEvents") or {}
            out["holdoutPersist"] = _finite_skill(
                he.get("skillPersist") if he.get("baseline") != "persist" else he.get("skill")
            )
            if out["holdoutPersist"] is None:
                out["holdoutPersist"] = _finite_skill(he.get("skillPersist") or he.get("skill"))
            return out
    if not isinstance(block, dict):
        return out
    out["source"] = "horizons"
    dnn = block.get("dnn") or block
    out["validateEvents"] = _finite_skill((dnn.get("validateEvents") or {}).get("skill"))
    he = dnn.get("holdoutEvents") or {}
    out["holdoutPersist"] = _finite_skill(he.get("skillPersist"))
    if out["holdoutPersist"] is None and he.get("baseline") == "persist":
        out["holdoutPersist"] = _finite_skill(he.get("skill"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline GRU / 1D-CNN / Transformer end-to-end sequence fit"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data", type=Path, default=ROOT / "storico")
    parser.add_argument(
        "--downstream",
        type=str,
        default=DEFAULT_DOWNSTREAM,
        help="Target station (default Cento); upstreams from config.dependencies",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--mlp-ref",
        type=Path,
        default=None,
        help="Compare JSON for side-by-side skill@120 (Cento: ml-cento-compare-novalle)",
    )
    parser.add_argument(
        "--prod-ref",
        type=Path,
        default=ROOT / "models" / "transfer-coeffs.json",
        help="Production coeffs — hard bar for adoptRuntime (nodes.<downstream>)",
    )
    parser.add_argument("--export-gru", type=Path, default=None)
    parser.add_argument("--export-cnn", type=Path, default=None)
    parser.add_argument("--export-transformer", type=Path, default=None)
    parser.add_argument("--no-export-weights", action="store_true")
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--spike-abs", type=float, default=ft.SPIKE_ABS)
    parser.add_argument("--no-rain", action="store_true")
    parser.add_argument(
        "--arch",
        choices=("both", "gru", "cnn", "transformer", "gru+transformer", "all"),
        default="both",
        help="Which sequence models to train",
    )
    parser.add_argument(
        "--holdout-from",
        type=str,
        default=None,
        help="Calendar cut YYYY-MM-DD: train/val before cut, holdout from cut (leakage-free)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=HORIZONS,
        help="Horizon steps (default 1 2 3 4 = 30..120′)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    global _VERBOSE
    _VERBOSE = bool(args.verbose) and not bool(args.quiet)

    downstream = args.downstream
    slug = station_slug(downstream)
    if args.out is None:
        args.out = ROOT / "models" / f"seq-{slug}-compare.json"
    if args.export_gru is None:
        args.export_gru = ROOT / "models" / f"seq-{slug}-gru.json"
    if args.export_cnn is None:
        args.export_cnn = ROOT / "models" / f"seq-{slug}-cnn.json"
    if args.export_transformer is None:
        args.export_transformer = ROOT / "models" / f"seq-{slug}-transformer.json"
    if args.mlp_ref is None:
        args.mlp_ref = (
            ROOT / "models" / "ml-cento-compare-novalle.json"
            if downstream == DEFAULT_DOWNSTREAM
            else args.prod_ref
        )

    if args.arch == "both":
        arches = ["gru", "cnn"]
    elif args.arch == "all":
        arches = ["gru", "cnn", "transformer"]
    elif args.arch == "gru+transformer":
        arches = ["gru", "transformer"]
    else:
        arches = [args.arch]
    horizons = [int(h) for h in args.horizons]
    holdout_from = args.holdout_from

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if downstream not in (config.get("sensors") or {}):
        raise SystemExit(f"Downstream assente da config: {downstream}")
    upstreams = resolve_upstreams(config, downstream)
    if not upstreams:
        raise SystemExit(f"Nessun upstream in config.dependencies per {downstream}")
    station_list = [downstream, *upstreams]
    stations_cfg = list(config["sensors"].keys())
    needed = [s for s in station_list if s in stations_cfg]
    missing = [s for s in station_list if s not in stations_cfg]
    if missing:
        raise SystemExit(f"Stazioni assenti da config: {missing}")

    log_step(f"fit-seq  {downstream}  arch={'+'.join(arches)}  L={args.lookback}")
    log(f"upstreams={upstreams}")
    log(f"out={args.out}")
    if holdout_from:
        log(f"holdout-from={holdout_from} (calendar cut; train/val before cut)", level="ok")

    log_step("Caricamento storico")
    t_load = time.perf_counter()
    wide_all = ft.load_wide(args.data, stations_cfg)
    wide_all, spike_counts = ft.filter_spikes(wide_all, spike_abs=args.spike_abs)
    spiked = {k: v for k, v in spike_counts.items() if v}
    log(
        f"Griglia {ft.STEP_MINUTES}′: {len(wide_all)} step in "
        f"{time.perf_counter() - t_load:.1f}s; spike: {spiked or 'nessuno'}",
        level="ok",
    )
    wide = wide_all[needed].copy()

    rain = None
    gauges = ml.gauge_names(config)
    if not args.no_rain:
        rain = ft.load_rain_basin(args.data / "rain", gauges, wide_all.index)
        if rain is not None:
            log(f"Pioggia: {int(rain['rain6h'].notna().sum())} timestep ok", level="ok")
        else:
            log("Pioggia: assente, fit solo livelli", level="warn")

    log_step("Canali sequenza")
    raw, channel_names, cont_names = build_channel_matrix(wide, rain, needed)
    use_rain = rain is not None and {"rain6h", "rain24h"}.issubset(
        set(channel_names)
    )
    log(f"C={len(channel_names)} channels: {channel_names}", level="debug")

    y_level = wide[downstream].to_numpy(dtype=float)
    horizons_out: dict[str, dict] = {}
    export_horizons: dict[str, dict[str, dict]] = {a: {} for a in arches}
    mlp_ref = mlp_ref_skills(args.mlp_ref, 120, downstream=downstream)
    prod_ref = mlp_ref_skills(args.prod_ref, 120, downstream=downstream)
    mlp_skill120 = mlp_ref["validateEvents"]
    mlp_hold120 = mlp_ref["holdoutPersist"]
    # Hard adopt bar: production node if present, else compare JSON.
    prod_skill120 = prod_ref["validateEvents"] if prod_ref["validateEvents"] is not None else mlp_skill120
    prod_hold120 = prod_ref["holdoutPersist"] if prod_ref["holdoutPersist"] is not None else mlp_hold120
    log(
        f"Compare ({args.mlp_ref.name}): valEvt@120={mlp_skill120} "
        f"holdPersist@120={mlp_hold120}",
        level="debug",
    )
    log(
        f"Prod bar ({args.prod_ref.name} / {downstream}): valEvt@120={prod_skill120} "
        f"holdPersist@120={prod_hold120}",
        level="debug",
    )

    for hi, h in enumerate(horizons, start=1):
        minutes = h * ft.STEP_MINUTES
        log_step(f"Orizzonte {minutes}′  ({hi}/{len(horizons)})")
        t_h = time.perf_counter()
        X, y, y0, index = collect_windows(
            raw, y_level, args.lookback, h, wide.index
        )
        if len(y) < WEAK_POINTS * 3:
            log(f"skip: n={len(y)}", level="warn")
            continue
        train_m, val_m, hold_m = ml.split_masks(index, holdout_from=holdout_from)
        if train_m.sum() < WEAK_POINTS or val_m.sum() < WEAK_POINTS:
            log(f"skip: train={train_m.sum()} val={val_m.sum()}", level="warn")
            continue

        poly_base = ml.build_poly2_baseline(
            wide[downstream], index, h, n_points=ml.POLY_NAIVE_POINTS
        )
        log(
            f"samples n={len(y)} train={int(train_m.sum())} "
            f"val={int(val_m.sum())} hold={int(hold_m.sum())} "
            f"events_val={int(ml.event_mask(y, val_m).sum())} "
            f"events_hold={int(ml.event_mask(y, hold_m).sum())}"
        )

        X_std, stats = standardize_seq(X, channel_names, cont_names, train_m)
        n_in = X_std.shape[-1]
        block: dict = {
            "horizonMinutes": minutes,
            "nChannels": n_in,
            "channelNames": channel_names,
            "nSamples": {
                "train": int(train_m.sum()),
                "validate": int(val_m.sum()),
                "holdout": int(hold_m.sum()),
            },
            "channelStats": {
                name: {"mean": round(stats[name][0], 8), "std": round(stats[name][1], 8)}
                for name in channel_names
            },
        }

        for ai, arch in enumerate(arches, start=1):
            if arch == "gru":
                model = SeqGRU(n_in, hidden=GRU_HIDDEN)
                arch_label = f"GRU(h={GRU_HIDDEN})"
                export_fn = export_seq_gru
            elif arch == "cnn":
                model = SeqCNN(n_in, channels=CNN_CHANNELS, kernel=CNN_KERNEL)
                arch_label = f"CNN{list(CNN_CHANNELS)}/k{CNN_KERNEL}"
                export_fn = export_seq_cnn
            else:
                model = SeqTransformer(
                    n_in,
                    d_model=TF_D_MODEL,
                    nhead=TF_NHEAD,
                    nlayers=TF_LAYERS,
                    dim_ff=TF_DIM_FF,
                    dropout=TF_DROPOUT,
                    max_len=max(TF_MAX_LEN, args.lookback),
                )
                arch_label = (
                    f"TF(d={TF_D_MODEL},h={TF_NHEAD},L={TF_LAYERS},ff={TF_DIM_FF})"
                )
                export_fn = export_seq_transformer

            job = (
                f"{arch_label} +{minutes}′ "
                f"[h {hi}/{len(horizons)} · arch {ai}/{len(arches)}]"
            )
            log(f"▶ {job}", level="info")
            t_arch = time.perf_counter()
            model, pred, hist = train_seq(
                model, X_std, y, train_m, val_m, label=job
            )
            metrics = ml.eval_splits(
                y0, y, pred, train_m, val_m, hold_m, poly_baseline=poly_base
            )
            spreads = ml.pack_spreads(y0, y, pred, val_m)
            he = metrics["holdoutEvents"]
            log(
                f"✓ {job}: val={metrics['validate']['skill']} "
                f"valEvt={metrics['validateEvents']['skill']} "
                f"holdEvt/poly2={he.get('skill')} "
                f"(persist {he.get('skillPersist')}) "
                f"arch_wall={time.perf_counter() - t_arch:.1f}s "
                f"horizon_wall={time.perf_counter() - t_h:.1f}s",
                level="ok",
            )
            block[arch] = {
                **metrics,
                "architecture": arch_label,
                "earlyStopping": hist,
            }
            weights = export_fn(model)
            export_horizons[arch][str(minutes)] = {
                "mean": [round(stats[c][0], 8) for c in channel_names],
                "std": [round(stats[c][1], 8) for c in channel_names],
                **weights,
                **spreads,
                "train": metrics["train"],
                "validate": metrics["validate"],
                "validateEvents": metrics["validateEvents"],
                "holdout": metrics["holdout"],
                "holdoutEvents": metrics["holdoutEvents"],
                "earlyStopping": hist,
            }

        horizons_out[str(minutes)] = block

    def skill_at(arch: str, minutes: int = 120) -> float | None:
        m = ((horizons_out.get(str(minutes)) or {}).get(arch) or {}).get(
            "validateEvents"
        ) or {}
        sk = m.get("skill")
        return float(sk) if sk is not None and math.isfinite(float(sk)) else None

    def hold_persist_at(arch: str, minutes: int = 120) -> float | None:
        m = ((horizons_out.get(str(minutes)) or {}).get(arch) or {}).get(
            "holdoutEvents"
        ) or {}
        sk = m.get("skillPersist")
        if sk is None:
            sk = m.get("skill") if m.get("baseline") == "persist" else None
        return float(sk) if sk is not None and math.isfinite(float(sk)) else None

    gate = {}
    for arch in arches:
        val_sk = skill_at(arch, 120)
        hold_sk = hold_persist_at(arch, 120)
        beats_compare_val = (
            mlp_skill120 is not None and val_sk is not None and val_sk >= mlp_skill120
        )
        beats_prod_val = (
            prod_skill120 is not None and val_sk is not None and val_sk >= prod_skill120
        )
        # Holdout n_evt is tiny (~30): hard bar is production holdout when known.
        beats_prod_hold = (
            hold_sk is not None
            and hold_sk >= ft.SKILL_KEEP
            and (prod_hold120 is None or hold_sk >= prod_hold120 - 1e-6)
        )
        adopt = bool(beats_prod_val and beats_prod_hold)
        gate[arch] = {
            "validateEventsSkill120": val_sk,
            "holdoutEventsSkillPersist120": hold_sk,
            "mlpValidateEventsSkill120": mlp_skill120,
            "mlpHoldoutEventsSkillPersist120": mlp_hold120,
            "prodValidateEventsSkill120": prod_skill120,
            "prodHoldoutEventsSkillPersist120": prod_hold120,
            "beatsMlpValidate": beats_compare_val,
            "beatsProdValidate": beats_prod_val,
            "beatsProdHoldoutPersist": beats_prod_hold,
            "adoptRuntime": adopt,
            "reason": (
                "beats production MLP on validate+holdout — still needs browser export + backtest"
                if adopt
                else (
                    f"valEvt@120={'above' if beats_prod_val else 'below'} prod; "
                    f"holdPersist@120={'above' if beats_prod_hold else 'below'} "
                    f"prod bar (n_evt hold≈32; no coeffs patch)"
                ),
            ),
        }

    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "version": 1,
        "kind": "seq_e2e",
        "downstream": downstream,
        "upstreams": upstreams,
        "stations": needed,
        "lookbackSteps": args.lookback,
        "lookbackMinutes": args.lookback * ft.STEP_MINUTES,
        "stepMinutes": ft.STEP_MINUTES,
        "channelNames": channel_names,
        "useRain": use_rain,
        "rainGauges": gauges if use_rain else [],
        "arches": arches,
        "holdoutFrom": holdout_from,
        "disclaimer": "Sperimentale: non operativo, non allertistico. Offline only.",
        "mlpRef": str(args.mlp_ref),
        "gate": gate,
        "trainedAt": trained_at,
        "horizons": horizons_out,
        "note": (
            "End-to-end Δy su finestra L×C (non ibrido Muskingum/RC). "
            "Stesso split/eventi della MLP; non patcha transfer-coeffs."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log_step("Export")
    log(f"Wrote {args.out}", level="ok")

    if not args.no_export_weights:
        kind_map = {
            "gru": "window_delta_gru",
            "cnn": "window_delta_cnn",
            "transformer": "window_delta_transformer",
        }
        path_map = {
            "gru": args.export_gru,
            "cnn": args.export_cnn,
            "transformer": args.export_transformer,
        }
        for arch in arches:
            g = gate[arch]
            skill120 = g["validateEventsSkill120"]
            node = {
                "kind": kind_map[arch],
                "downstream": downstream,
                "upstreams": upstreams,
                "stations": needed,
                "channelNames": channel_names,
                "lookbackSteps": args.lookback,
                "stepMinutes": ft.STEP_MINUTES,
                "useRain": use_rain,
                "rainGauges": gauges if use_rain else [],
                "horizons": export_horizons[arch],
                "skill120": skill120,
                # Soft-load in loadCoeffs only if reliable (beats production bar).
                "reliable": bool(g.get("adoptRuntime")),
                "trainedAt": trained_at,
                "thresholdOn": "event_or_move",
                "gate": g,
                "note": (
                    "End-to-end Δy sequence model. Soft-load in loadCoeffs only if reliable; "
                    "use backtest --compare-candidates before promoting."
                ),
            }
            dest = path_map[arch]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps({"version": 1, "downstream": downstream, **node}, indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            log(f"Wrote {dest} (reliable={node['reliable']})", level="ok")

    for arch in arches:
        g = gate[arch]
        log(
            f"gate {arch}: valEvt={g['validateEventsSkill120']} "
            f"vs MLP={g['mlpValidateEventsSkill120']} "
            f"adopt={g['adoptRuntime']}",
            level="ok" if g["adoptRuntime"] else "warn",
        )


if __name__ == "__main__":
    main()
