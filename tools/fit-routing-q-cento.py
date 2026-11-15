#!/usr/bin/env python3
"""WP3: nonlinear reservoir routing in discharge space for Cento.

Rating: Q = softplus(a) * max(h - h0, eps)^b  (a,b,h0 trainable, boxed).
Reach: cascade of n tanks with Qout = k * S^m (m>1 → faster waves at high Q).
Qlat: random-walk state driven by basin rain (optional).

Trains multi-step forecast loss on horizons matching the MLP. Open-loop eval
vs persistence / ARX-style baseline on flood events.

Uso:
  uv run python tools/fit-routing-q-cento.py --data ./storico \\
    --export models/routing-q-cento.json --summary models/routing-q-cento-summary.json
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

ROOT = Path(__file__).resolve().parents[1]
DOWN = "Cento"
UP_F = "Forcelli"
UP_S = "Sostegno Reno"
UP_ALT = "Bonconvento"
STATIONS = [DOWN, UP_F, UP_S, UP_ALT]
HORIZONS = [1, 2, 3, 4]  # 30…120′
N_TANKS = 3
EVENT_ABS = 0.10
DT_HOURS = 0.5  # 30′
SEED = 42
LR = 5e-3
EPOCHS = 60
PATIENCE = 12
BATCH_ISSUES = 256


def _load_ft():
    path = ROOT / "tools" / "fit-transfer.py"
    spec = importlib.util.spec_from_file_location("fit_transfer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["fit_transfer"] = mod
    spec.loader.exec_module(mod)
    return mod


ft = _load_ft()


def gauge_names(config: dict) -> list[str]:
    out = []
    for g in (config.get("rainfall") or {}).get("gauges") or []:
        if isinstance(g, str):
            out.append(g)
        elif isinstance(g, dict) and g.get("name"):
            out.append(str(g["name"]))
    return out


def load_rain_max(rain_dir: Path, gauges: list[str], index: pd.DatetimeIndex) -> np.ndarray:
    series = []
    for name in gauges:
        path = rain_dir / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["time"])
        if df.empty:
            continue
        s = df.set_index("time")["value"].astype(float).sort_index()
        s = s[~s.index.duplicated(keep="last")].resample(f"{ft.STEP_MINUTES}min").sum()
        series.append(s.reindex(index).fillna(0.0))
    if not series:
        return np.zeros(len(index), dtype=float)
    return np.maximum.reduce([s.to_numpy(dtype=float) for s in series])


class RatingCurve(nn.Module):
    """Q = softplus(log_a) * relu(h - h0)^b with soft bounds."""

    def __init__(self, h_med: float, q_scale: float = 1.0):
        super().__init__()
        self.log_a = nn.Parameter(torch.tensor(0.0))
        self.raw_b = nn.Parameter(torch.tensor(0.0))  # → b in (1.1, 2.5)
        self.raw_h0 = nn.Parameter(torch.tensor(0.0))
        self.h_med = float(h_med)
        self.q_scale = float(q_scale)

    def h0(self) -> torch.Tensor:
        # h0 near low water: sigmoid around h_med - 1 m, clipped
        return self.h_med - 2.0 * torch.sigmoid(self.raw_h0) - 0.5

    def b(self) -> torch.Tensor:
        return 1.1 + 1.4 * torch.sigmoid(self.raw_b)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a = torch.nn.functional.softplus(self.log_a) + 1e-3
        head = torch.clamp(h - self.h0(), min=1e-3)
        return self.q_scale * a * torch.pow(head, self.b())


class NonlinearRouting(nn.Module):
    """Two upstream inflows → cascade of tanks → Cento level via inverse rating."""

    def __init__(self, h_stats: dict[str, float], n_tanks: int = N_TANKS):
        super().__init__()
        self.n_tanks = n_tanks
        self.rating = nn.ModuleDict({
            name: RatingCurve(h_stats[name]) for name in STATIONS
        })
        # log k, raw m for each tank
        self.log_k = nn.Parameter(torch.zeros(n_tanks))
        self.raw_m = nn.Parameter(torch.zeros(n_tanks))
        self.w_f = nn.Parameter(torch.tensor(0.5))  # mix Forcelli vs Sostegno
        self.log_qlat_sigma = nn.Parameter(torch.tensor(-2.0))
        self.rain_gain = nn.Parameter(torch.tensor(0.0))
        self.init_s = nn.Parameter(torch.ones(n_tanks))

    def m(self) -> torch.Tensor:
        return 1.2 + 1.3 * torch.sigmoid(self.raw_m)  # (1.2, 2.5)

    def k(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_k) + 1e-3

    def step_tanks(
        self,
        s: torch.Tensor,
        q_in: torch.Tensor,
        q_lat: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """s (B,n), q_in (B,), q_lat (B,) → new s, q_out last tank."""
        k = self.k()
        m = self.m()
        q_prev = q_in + q_lat
        parts = []
        q_out = q_prev
        for i in range(self.n_tanks):
            s_i = torch.clamp(s[:, i], min=1e-4)
            q_out = k[i] * torch.pow(s_i, m[i])
            ds = (q_prev - q_out) * dt
            parts.append(torch.clamp(s_i + ds, min=1e-4))
            q_prev = q_out
        return torch.stack(parts, dim=1), q_out

    def invert_rating_cento(self, q: torch.Tensor) -> torch.Tensor:
        """Closed-form inverse: h = h0 + (Q / (a q_scale))^(1/b)."""
        rc = self.rating[DOWN]
        a = torch.nn.functional.softplus(rc.log_a) + 1e-3
        b = rc.b()
        q_pos = torch.clamp(q, min=1e-6)
        head = torch.pow(q_pos / (rc.q_scale * a), 1.0 / b)
        return rc.h0() + torch.clamp(head, max=50.0)

    def rollout(
        self,
        h_f: torch.Tensor,
        h_s: torch.Tensor,
        h_c0: torch.Tensor,
        rain: torch.Tensor,
        n_steps: int,
        dt: float = DT_HOURS,
    ) -> torch.Tensor:
        """Return predicted Cento levels (B, n_steps) open-loop from t+1..t+n."""
        B = h_c0.shape[0]
        q_f = torch.nan_to_num(self.rating[UP_F](h_f), nan=0.0, posinf=1e3, neginf=0.0)
        q_s = torch.nan_to_num(self.rating[UP_S](h_s), nan=0.0, posinf=1e3, neginf=0.0)
        w = torch.sigmoid(self.w_f)
        q_c0 = torch.nan_to_num(self.rating[DOWN](h_c0), nan=1.0, posinf=1e3, neginf=0.0)
        scale = 0.5 + 0.5 * (q_c0 / (q_c0.detach().mean() + 1e-3)).unsqueeze(1)
        s = torch.clamp(self.init_s.abs().unsqueeze(0).expand(B, -1) * scale, min=1e-3, max=1e3)
        q_lat = torch.zeros(B, device=h_c0.device)
        preds = []
        for t in range(n_steps):
            q_in = w * q_f[:, t] + (1.0 - w) * q_s[:, t]
            innov = torch.nn.functional.softplus(self.log_qlat_sigma) * torch.clamp(rain[:, t], min=0.0, max=50.0)
            q_lat = 0.85 * q_lat + torch.nn.functional.softplus(self.rain_gain) * innov
            q_lat = torch.clamp(q_lat, min=0.0, max=1e3)
            s, q_out = self.step_tanks(s, q_in, q_lat, dt)
            q_out = torch.clamp(q_out, min=1e-6, max=1e4)
            preds.append(self.invert_rating_cento(q_out))
        return torch.stack(preds, dim=1)


def split_masks(index: pd.DatetimeIndex):
    # reuse fit-transfer alt months
    months = index.to_period("M")
    uniq = sorted(months.unique())
    train_m = np.zeros(len(index), dtype=bool)
    val_m = np.zeros(len(index), dtype=bool)
    hold_m = np.zeros(len(index), dtype=bool)
    for i, m in enumerate(uniq):
        mask = np.asarray(months == m, dtype=bool)
        if i >= len(uniq) - 2:
            hold_m |= mask
        elif i % 2 == 0:
            train_m |= mask
        else:
            val_m |= mask
    return train_m, val_m, hold_m


def arx_baseline(y: np.ndarray, h: int, order: int = 4) -> np.ndarray:
    """Simple one-shot ARX: ŷ(t+h) = a0 + sum a_i y(t-i). Fit OLS on finite rows."""
    n = len(y)
    pred = np.full(n, np.nan)
    rows = []
    targets = []
    idx = []
    for t in range(order, n - h):
        if not np.isfinite(y[t + h]):
            continue
        win = y[t - order + 1 : t + 1]
        if not np.all(np.isfinite(win)):
            continue
        rows.append(np.concatenate([[1.0], win[::-1]]))
        targets.append(y[t + h])
        idx.append(t)
    if len(rows) < order + 2:
        return pred
    A = np.asarray(rows)
    b = np.asarray(targets)
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return pred
    for t in range(order, n - h):
        win = y[t - order + 1 : t + 1]
        if not np.all(np.isfinite(win)):
            continue
        pred[t] = float(coef[0] + coef[1:] @ win[::-1])
    return pred


def skill(y0, y_obs, y_pred):
    m = np.isfinite(y0) & np.isfinite(y_obs) & np.isfinite(y_pred)
    if m.sum() < 8:
        return None, None, int(m.sum())
    rmse = float(np.sqrt(np.mean((y_pred[m] - y_obs[m]) ** 2)))
    rmse_p = float(np.sqrt(np.mean((y0[m] - y_obs[m]) ** 2)))
    sk = 1.0 - rmse / rmse_p if rmse_p > 0 else None
    return rmse, sk, int(m.sum())


def phase_lag_min(pred, obs, max_lag=6):
    best, best_r = 0, -1e9
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = pred[-lag:], obs[:lag] if lag != 0 else obs
            b = obs[: len(a)]
        elif lag > 0:
            a, b = pred[:-lag] if lag else pred, obs[lag:]
            a = pred[: len(b)]
        else:
            a, b = pred, obs
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 20:
            continue
        aa, bb = a[m] - a[m].mean(), b[m] - b[m].mean()
        den = np.sqrt((aa * aa).sum() * (bb * bb).sum())
        if den <= 0:
            continue
        r = float((aa * bb).sum() / den)
        if r > best_r:
            best_r, best = r, lag
    return -best * 30  # pred late → positive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "storico")
    ap.add_argument("--export", type=Path, default=ROOT / "models" / "routing-q-cento.json")
    ap.add_argument("--summary", type=Path, default=ROOT / "models" / "routing-q-cento-summary.json")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    wide = ft.load_wide(args.data, list(config["sensors"].keys()))
    wide, _ = ft.filter_spikes(wide, spike_abs=ft.SPIKE_ABS)
    # Fill Sostegno gaps with Bonconvento
    if UP_S in wide.columns and UP_ALT in wide.columns:
        wide[UP_S] = wide[UP_S].fillna(wide[UP_ALT])
    wide = wide[STATIONS].copy()
    rain = load_rain_max(args.data / "rain", gauge_names(config), wide.index)

    h_stats = {s: float(wide[s].median(skipna=True)) for s in STATIONS}
    model = NonlinearRouting(h_stats)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_m, val_m, hold_m = split_masks(wide.index)
    y = {s: wide[s].to_numpy(dtype=float) for s in STATIONS}
    n = len(wide)
    max_h = max(HORIZONS)

    def sample_issues(mask: np.ndarray) -> np.ndarray:
        idx = np.where(mask)[0]
        idx = idx[(idx >= 2) & (idx < n - max_h)]
        ok = []
        for t in idx:
            if all(np.isfinite(y[s][t]) for s in STATIONS) and np.isfinite(y[DOWN][t + max_h]):
                ok.append(t)
        return np.asarray(ok, dtype=int)

    train_idx = sample_issues(train_m)
    val_idx = sample_issues(val_m)
    print(f"issues train={len(train_idx)} val={len(val_idx)}")

    def batch_tensors(issues: np.ndarray, teacher_force_up: bool = False):
        """Boundary: persist upstream@issue (open-loop). teacher_force_up uses obs monte at t+k."""
        B = len(issues)
        h_f = np.zeros((B, max_h), dtype=np.float32)
        h_s = np.zeros((B, max_h), dtype=np.float32)
        rain_b = np.zeros((B, max_h), dtype=np.float32)
        h_c0 = np.zeros(B, dtype=np.float32)
        target = np.zeros((B, max_h), dtype=np.float32)
        for i, t in enumerate(issues):
            h_c0[i] = y[DOWN][t]
            for k in range(max_h):
                src = t + k if teacher_force_up else t
                hf = y[UP_F][src] if np.isfinite(y[UP_F][src]) else y[UP_F][t]
                hs = y[UP_S][src] if np.isfinite(y[UP_S][src]) else y[UP_S][t]
                h_f[i, k] = hf if np.isfinite(hf) else h_stats[UP_F]
                h_s[i, k] = hs if np.isfinite(hs) else h_stats[UP_S]
                rain_b[i, k] = rain[src] if np.isfinite(rain[src]) else rain[t]
                target[i, k] = y[DOWN][t + k + 1]
        return (
            torch.tensor(h_f),
            torch.tensor(h_s),
            torch.tensor(h_c0),
            torch.tensor(rain_b),
            torch.tensor(target),
        )

    best_state = None
    best_val = float("inf")
    patience = 0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(train_idx)
        total = 0.0
        nbat = 0
        for start in range(0, len(train_idx), BATCH_ISSUES):
            issues = train_idx[start : start + BATCH_ISSUES]
            # Teacher-force upstream boundaries while learning tank dynamics.
            h_f, h_s, h_c0, rain_b, target = batch_tensors(issues, teacher_force_up=True)
            opt.zero_grad()
            pred = model.rollout(h_f, h_s, h_c0, rain_b, max_h)
            finite = torch.isfinite(target) & torch.isfinite(pred) & torch.isfinite(h_c0).unsqueeze(1)
            if not finite.any():
                continue
            w = torch.ones_like(target)
            for j, h in enumerate(HORIZONS):
                w[:, h - 1] *= 1.0 + 0.25 * h
            dy = torch.abs(target - h_c0.unsqueeze(1))
            w = w * (1.0 + 3.0 * (dy >= EVENT_ABS).float())
            w = w * finite.float()
            err = (pred - target) ** 2
            loss = (err * w).sum() / w.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            nbat += 1
        model.eval()
        with torch.no_grad():
            h_f, h_s, h_c0, rain_b, target = batch_tensors(
                val_idx[: min(2000, len(val_idx))], teacher_force_up=True
            )
            pred = model.rollout(h_f, h_s, h_c0, rain_b, max_h)
            finite = torch.isfinite(target) & torch.isfinite(pred)
            val_loss = float((((pred - target) ** 2)[finite]).mean().item()) if finite.any() else float("nan")
        print(f"epoch {epoch+1}/{args.epochs} train={total/max(1,nbat):.5f} val={val_loss:.5f}")
        if np.isfinite(val_loss) and val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("early stop")
                break
    if best_state:
        model.load_state_dict(best_state)
    print(f"train wall {time.perf_counter()-t0:.1f}s")

    # Eval open-loop @120 on val event-like samples
    model.eval()
    with torch.no_grad():
        # Gate eval: true open-loop (persist upstream @ issue).
        h_f, h_s, h_c0, rain_b, target = batch_tensors(val_idx, teacher_force_up=False)
        pred = model.rollout(h_f, h_s, h_c0, rain_b, max_h).numpy()
        target_np = target.numpy()
        y0 = h_c0.numpy()

    metrics = {}
    for h in HORIZONS:
        j = h - 1
        rmse, sk, n_s = skill(y0, target_np[:, j], pred[:, j])
        ev = np.abs(target_np[:, j] - y0) >= EVENT_ABS
        rmse_e, sk_e, n_e = skill(y0[ev], target_np[ev, j], pred[ev, j])
        metrics[str(h * 30)] = {
            "all": {"rmse": rmse, "skill": sk, "n": n_s},
            "events": {"rmse": rmse_e, "skill": sk_e, "n": n_e},
        }
        print(f"h={h*30}′ skill_all={sk} skill_evt={sk_e} n_evt={n_e}")

    # ARX baseline on Cento series @120
    y_c = y[DOWN]
    arx = arx_baseline(y_c, h=4, order=4)
    # compare on same val issues
    y0_v = np.array([y_c[t] for t in val_idx], dtype=float)
    yobs_v = np.array([y_c[t + 4] for t in val_idx], dtype=float)
    yarx_v = np.array([arx[t] for t in val_idx], dtype=float)
    yrout_v = pred[:, 3]
    ev = np.abs(yobs_v - y0_v) >= EVENT_ABS
    rmse_arx, sk_arx, n_arx = skill(y0_v[ev], yobs_v[ev], yarx_v[ev])
    rmse_rt, sk_rt, n_rt = skill(y0_v[ev], yobs_v[ev], yrout_v[ev])
    lag_rt = phase_lag_min(yrout_v[ev], yobs_v[ev]) if ev.sum() > 30 else None
    lag_arx = phase_lag_min(yarx_v[ev], yobs_v[ev]) if ev.sum() > 30 else None

    gate_pass = (
        sk_rt is not None
        and sk_arx is not None
        and sk_rt > sk_arx + 0.02
        and (lag_rt is None or abs(lag_rt) <= abs(lag_arx or 999) + 15)
    )
    print(
        f"GATE vs ARX events@120: routing skill={sk_rt} rmse={rmse_rt} lag={lag_rt} | "
        f"ARX skill={sk_arx} rmse={rmse_arx} lag={lag_arx} → pass={gate_pass}"
    )

    # Export compact params
    def rating_export(rc: RatingCurve) -> dict:
        with torch.no_grad():
            return {
                "a": float(torch.nn.functional.softplus(rc.log_a) + 1e-3),
                "b": float(rc.b()),
                "h0": float(rc.h0()),
                "qScale": rc.q_scale,
                "hMed": rc.h_med,
            }

    with torch.no_grad():
        payload = {
            "version": 1,
            "kind": "cento_routing_q",
            "downstream": DOWN,
            "upstreams": [UP_F, UP_S],
            "stations": STATIONS,
            "nTanks": N_TANKS,
            "stepMinutes": ft.STEP_MINUTES,
            "horizonsMinutes": [h * ft.STEP_MINUTES for h in HORIZONS],
            "ratings": {s: rating_export(model.rating[s]) for s in STATIONS},
            "k": [float(x) for x in model.k().tolist()],
            "m": [float(x) for x in model.m().tolist()],
            "wForcelli": float(torch.sigmoid(model.w_f)),
            "rainGain": float(torch.nn.functional.softplus(model.rain_gain)),
            "qlatSigma": float(torch.nn.functional.softplus(model.log_qlat_sigma)),
            "initS": [float(x) for x in model.init_s.abs().tolist()],
            "reliable": False,
            "adopt": False,
            "gatePass": bool(gate_pass),
            "disclaimer": "Sperimentale: routing in portata open-loop, non allertistico.",
            "trainedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    args.export.parent.mkdir(parents=True, exist_ok=True)
    args.export.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    torch.save(model.state_dict(), args.export.with_suffix(".pt"))

    summary = {
        "version": 1,
        "kind": "cento_routing_q_summary",
        "metrics": metrics,
        "gate": {
            "vsArxEvents120": {
                "routingSkill": sk_rt,
                "routingRmse": rmse_rt,
                "routingPhaseLagMin": lag_rt,
                "arxSkill": sk_arx,
                "arxRmse": rmse_arx,
                "arxPhaseLagMin": lag_arx,
                "n": n_rt,
                "pass": bool(gate_pass),
            }
        },
        "note": (
            "Open-loop with persisted upstream at issue time (no future monte). "
            "Gate: event skill@120 > ARX + 0.02."
        ),
        "export": str(args.export.relative_to(ROOT)),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.export}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
