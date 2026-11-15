#!/usr/bin/env python3
"""WP5.1–5.2 after WP3 gate fail: ARX ± routing features / residual correction.

Requires models/routing-q-cento.pt from fit-routing-q-cento.py.
Writes models/wp5-hybrid-summary.json + wp5-hybrid.md. adopt stays false.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("routing", ROOT / "tools" / "fit-routing-q-cento.py")
routing = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["routing_q"] = routing
spec.loader.exec_module(routing)

DOWN = "Cento"
UP_F, UP_S, UP_ALT = "Forcelli", "Sostegno Reno", "Bonconvento"
STATIONS = [DOWN, UP_F, UP_S, UP_ALT]
H = 4
EVENT_ABS = 0.10
ORDER = 4


def skill(y0, y_obs, y_pred):
    m = np.isfinite(y0) & np.isfinite(y_obs) & np.isfinite(y_pred)
    if int(m.sum()) < 20:
        return None, None, int(m.sum())
    rmse = float(np.sqrt(np.mean((y_pred[m] - y_obs[m]) ** 2)))
    rmse_p = float(np.sqrt(np.mean((y0[m] - y_obs[m]) ** 2)))
    return rmse, (1.0 - rmse / rmse_p if rmse_p > 0 else None), int(m.sum())


def fit_arx(X: np.ndarray, y: np.ndarray):
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if m.sum() < X.shape[1] + 5:
        return None
    A = np.column_stack([np.ones(int(m.sum())), X[m]])
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    return coef


def apply_arx(coef, X):
    return coef[0] + X @ coef[1:]


def main() -> None:
    ft = routing.ft
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    export = json.loads((ROOT / "models" / "routing-q-cento.json").read_text(encoding="utf-8"))
    pt_path = ROOT / "models" / "routing-q-cento.pt"
    wide = ft.load_wide(ROOT / "storico", list(config["sensors"].keys()))
    wide, _ = ft.filter_spikes(wide, spike_abs=ft.SPIKE_ABS)
    if UP_S in wide.columns and UP_ALT in wide.columns:
        wide[UP_S] = wide[UP_S].fillna(wide[UP_ALT])
    wide = wide[STATIONS].copy()
    rain = routing.load_rain_max(ROOT / "storico" / "rain", routing.gauge_names(config), wide.index)
    h_stats = {s: float(wide[s].median(skipna=True)) for s in STATIONS}
    model = routing.NonlinearRouting(h_stats)
    if pt_path.exists():
        model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    else:
        print("WARN: missing routing-q-cento.pt — using untrained init")

    y = {s: wide[s].to_numpy(dtype=float) for s in STATIONS}
    n = len(wide)
    train_m, val_m, _ = routing.split_masks(wide.index)
    issues = np.arange(ORDER, n - H)
    ok = np.ones(len(issues), dtype=bool)
    for s in STATIONS:
        ok &= np.isfinite(y[s][issues]) & np.isfinite(y[DOWN][issues + H])
        for k in range(ORDER):
            ok &= np.isfinite(y[DOWN][issues - k])
    issues = issues[ok]

    def predict_batch(idxs: np.ndarray) -> np.ndarray:
        out = np.full(len(idxs), np.nan)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(idxs), 512):
                sl = idxs[start : start + 512]
                B = len(sl)
                h_f = np.zeros((B, H), np.float32)
                h_s = np.zeros((B, H), np.float32)
                rain_b = np.zeros((B, H), np.float32)
                h_c0 = np.zeros(B, np.float32)
                for i, t in enumerate(sl):
                    h_c0[i] = y[DOWN][t]
                    for k in range(H):
                        h_f[i, k] = y[UP_F][t]
                        h_s[i, k] = y[UP_S][t]
                        rain_b[i, k] = rain[t]
                pred = model.rollout(
                    torch.tensor(h_f), torch.tensor(h_s), torch.tensor(h_c0), torch.tensor(rain_b), H
                ).numpy()[:, H - 1]
                out[start : start + B] = pred
        return out

    print(f"routing predict n={len(issues)}…")
    h_hat = predict_batch(issues)
    y0 = y[DOWN][issues]
    y_obs = y[DOWN][issues + H]
    X_y = np.column_stack([y[DOWN][issues - k] for k in range(ORDER)])
    X_phys = np.column_stack([X_y, h_hat, h_hat - y0])
    is_train = train_m[issues]
    is_val = val_m[issues]
    ev = np.abs(y_obs - y0) >= EVENT_ABS

    coef_arx = fit_arx(X_y[is_train], y_obs[is_train])
    coef_phys = fit_arx(X_phys[is_train], y_obs[is_train])
    resid = y_obs - h_hat
    coef_res = fit_arx(X_y[is_train], resid[is_train])

    def eval_pred(pred):
        return skill(y0[is_val & ev], y_obs[is_val & ev], pred[is_val & ev])

    def pack(rmse, sk, n_s):
        return {"rmse": rmse, "skill": sk, "n": n_s}

    rt = pack(*eval_pred(h_hat))
    arx = pack(*eval_pred(apply_arx(coef_arx, X_y))) if coef_arx is not None else pack(None, None, 0)
    phys = pack(*eval_pred(apply_arx(coef_phys, X_phys))) if coef_phys is not None else pack(None, None, 0)
    if coef_res is not None:
        resc = pack(*eval_pred(h_hat + apply_arx(coef_res, X_y)))
    else:
        resc = pack(None, None, 0)

    base, p_sk, r_sk = arx["skill"], phys["skill"], resc["skill"]
    promote = None
    if p_sk is not None and base is not None and p_sk > base + 0.02:
        promote = "arx_plus_routing_feat"
    elif r_sk is not None and base is not None and r_sk > base + 0.02:
        promote = "routing_plus_residual_arx"
    decision = {
        "promote": promote,
        "adopt": False,
        "wp4Skipped": True,
        "reason": (
            f"WP3 gate failed (skip WP4). WP5 vs ARX skill base={base} phys={p_sk} residual={r_sk}. "
            + ("No gain ≥0.02 → stop early." if not promote else f"Research-only promote {promote}.")
        ),
    }
    summary = {
        "version": 1,
        "kind": "wp5_hybrid_summary",
        "horizonMinutes": 120,
        "wp3GatePass": bool(export.get("gatePass")),
        "valEvents": {
            "routing": rt,
            "arx": arx,
            "arx_plus_routing_feat": phys,
            "routing_plus_residual_arx": resc,
        },
        "decision": decision,
    }
    out = ROOT / "models" / "wp5-hybrid-summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (ROOT / "models" / "wp5-hybrid.md").write_text(
        "\n".join(
            [
                "# WP5 — hybrid after WP3 gate fail",
                "",
                "WP4 skipped (WP3 open-loop not better than ARX).",
                "",
                "| model | skill evt@120 | RMSE | n |",
                "|---|---:|---:|---:|",
                f"| routing open-loop | {rt['skill']} | {rt['rmse']} | {rt['n']} |",
                f"| ARX | {arx['skill']} | {arx['rmse']} | {arx['n']} |",
                f"| ARX + routing feat | {phys['skill']} | {phys['rmse']} | {phys['n']} |",
                f"| routing + residual ARX | {resc['skill']} | {resc['rmse']} | {resc['n']} |",
                "",
                f"**promote:** `{promote}`",
                "**adopt:** false",
                f"**motivo:** {decision['reason']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
