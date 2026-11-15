# WP1 diagnosis — Cento flood forecast (research_plan_reno.md rev 4)

**Generated from:** `node tools/wp1-diagnosis.js` → [`wp1-diagnosis.json`](wp1-diagnosis.json)  
**WP0 table:** [`eval-table.md`](eval-table.md) / [`eval-per-event.json`](eval-per-event.json)  
**Disclaimer:** sperimentale; non operativo, non allertistico.

## Corrected assumptions (§2.2)

| Item | Plan assumption | Measured / corrected |
|---|---|---|
| Time step | 30 min | **Confirmed** model grid. Raw ARPAE CSV ≈ 15′. |
| Rain | Not used | **Available** under `storico/rain/`. Left unused for WP0/WP1 new models (MLP/EnKF already have weak rain terms). |
| Downstream | Not used | **Dosso** is nearest valle gauge. Not used as model input in WP1; only residual diagnosis (H11). |
| Travel time > 120′ | Assumed | **Not confirmed as a global floor.** Cross-corr min lag = 0′ on Sostegno in the low (baseflow) band; flood-relevant mid/high lags are mostly **150–750′**. Forcelli/Casalecchio mid–high ≥ 300′. |
| Geometry / DTM | (WP3) | Centerlines: `river_reno.geojson`, `torrente_samoggia.geojson`. **No DTM/LiDAR in repo.** |

## WP0 gate (eval protocol)

Protocol v2 adds ARX, CSI (yellow/orange/red from `arpae-soglie.json`), crossing-time error, pinball on bands, rising/falling RMSE, sub-step phase lag, per-event export, and Cento year folds.

Spotlight **prod @ 120′** (test from 2026-01-01, seed 1):

| Metric | Value |
|---|---|
| skill evt / persist | 0.68 [CI90 0.63–0.76] |
| skill evt / ARX | 0.07 |
| rising / falling RMSE | 0.119 / 0.021 |
| phaseLag (sub-step) | ≈ 26′ |
| CSI yellow / orange | 0.99 / 0.85 |
| pinball (event bands) | 0.015 |

ARX @ 120′ skill evt ≈ 0.53 (better than persist, worse than prod). Year folds (holdout 2023–2026): prod skill evt ≈ 0.65–0.72; ARX ≈ 0.59–0.66.

## Verdict table (H1–H11)

| ID | Status | Summary |
|---|---|---|
| H1 | **confirmed** | Open-loop EnKF@120 skill ≈ 0.03 (event RMSE 0.24); filtered summary skill negative. Filter cannot fix a weak process. |
| H2 | **confirmed** | Hybrids/EnKF/RC stay on levels or anomalies; no mass-conserving Q model in repo. |
| H3 | **not clear** | Lag vs level is mixed (Forcelli/Casalecchio *lengthen* at high level; Sostegno mid→high shortens 330→150′). Wave-speed↑ with Q not established from these bands alone. |
| H4 | **confirmed** | Fixed-θ RC step: 50% rise ≈ 90′ (exponential). Structure is diffusion, not delay. |
| H5 | **not clear** | Rain exists; no explicit `Q_lat` state. Needs WP3 test. |
| H6 | **confirmed** | Innovation variance flood/calm ≈ **11×**. Ensemble spread collapsed (~0.01) in both regimes → constant Q,R too small in floods. |
| H7 | **confirmed** | EnKF 1+4-step fit vs MLP direct multi-horizon — unequal by construction. |
| H8 | **not clear** | q50 (median pinball) lag 21.1′ vs prod MSE 22.6′ — tiny change; discrete lag still 30′. |
| H9 | **confirmed** | Rising RMSE ≈ 6× falling; WP0 now reports CSI / crossing-time / sub-step lag. |
| H10 | **confirmed** | 193 decode ops (166 lift / 27 warp); **78% on samples with no observed crossing**. |
| H11 | **not clear** | Weak residual correlation with Dosso / (Cento−Dosso) on rising limb (~0.09 / 0.06). High-Dosso RMSE *lower* than low-Dosso; second-wave ≈ all-events. Prior valle gate already rejected acceptors on skill@120. |

Hysteresis (top 3 floods): non-zero loop scores vs Sostegno (≈ −0.4 to −0.9) → storage / path dependence present.

## Decision rules (§6) applied

| Rule | Action |
|---|---|
| H10 confirmed | **Next:** WP2 items 2–3 — discrete-time hazard for τ + decode gate on P(crossing) / CSI — before another crossing-time A/B. |
| H1, H2 confirmed; H3 not clear | **Continue WP3** (nonlinear routing in discharge) is still justified by H1/H2/H4; do not treat H3 as proven. |
| H8 not clear | Do **not** prioritize WP2 shape/time losses (6–7) on H8 alone; H10/censoring comes first. |
| H11 not clear | Do **not** force WP2 item 9 / WP3 stage-fall yet; optional light Dosso residual monitor only. |
| H1–H4 not rejected | No “inspect filter bugs first” stop — process structure remains the leading issue. |

## Recommended next package

**WP2 (items 1–3 first):** flood sample weights (not flood-only), hazard model for first crossing, gate decode on CSI. Then re-run protocol metrics (CSI, crossing-time error, rising RMSE, sub-step lag) with bootstrap CIs.

WP3 stays on the roadmap after WP2 gate, with trainable rating curves (no official Q(h) in repo) and geometry only as parameter priors (centerline `L`; DTM still missing).

## How to reproduce

```bash
node --test tests/*.test.js
node tools/backtest.js --protocol --stations Cento \
  --only-candidates persist,poly2,arx,ridge,prod \
  --from 2026-01-01 --seed 1 \
  --json models/eval-table.json --md models/eval-table.md
node tools/wp1-diagnosis.js --json models/wp1-diagnosis.json
```

No production `adopt` / `reliable` flags were changed.
