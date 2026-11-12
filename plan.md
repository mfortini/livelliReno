# Plan: Improve the Prediction Method

## Goal

Increase the quality of the forecast curves and the honesty of the `reliable` flag. Keep the model simple and interpretable.

## Phase 1: Fix method weaknesses

Priority: high. These changes fix correctness problems. They do not add features.

1. Estimate the upstream lags on the train months only.
   - File: `tools/fit-transfer.py`, function `fit_node`.
   - Today `estimate_lag` uses the full series. This leaks validation data into the model.
   - Change: pass a train mask to the lag estimation step.
2. Set the `reliable` flag from the backtest result.
   - File: `tools/fit-transfer.py`, constant `SKILL_KEEP`, and `models/transfer-coeffs.json`.
   - Today the flag uses the skill from the monthly-split fit. The backtest shows lower skill for the same stations.
   - Change: raise `SKILL_KEEP` to `0.10`, or read the skill from `models/backtest-summary.json` after `tools/backtest.js` runs.
   - Expected effect: Dosso (skill ~0.04) loses the flag. Secondo Salto (~0.10) is borderline.
3. Calibrate the forecast fork per regime.
   - Files: `tools/fit-transfer.py`, `predict-transfer.js`.
   - Today one spread covers all conditions. Coverage is ~90% in calm periods and 48-75% in events.
   - Change: compute two spread pairs per horizon. Use the high-regime pair when the upstream driver exceeds the stored `z85` threshold. The thresholds already exist in `transfer-coeffs.json`.
4. Add a stronger baseline to the backtest.
   - File: `tools/backtest.js`.
   - Change: report skill against persistence and against a 3-hour moving average. This stops weak stations from earning skill for free.

## Phase 2: Improve the model

Priority: medium. Each change needs a backtest run before adoption.

5. Add a second lag per upstream station.
   - File: `tools/fit-transfer.py`.
   - Change: for each upstream, fit the lag tau and the lag tau+h. This captures wave dispersion better than one lag.
6. Add rainfall inputs.
   - Files: `tools/fetch-storico.py`, `tools/fit-transfer.py`, `config.json`.
   - Change: download the ARPAE rainfall variable for nearby gauges. Add 6-hour and 24-hour cumulated rainfall as features. Rainfall anticipates the flood wave. Level data upstream only sees the wave after it starts.
   - Risk: more fetch failures and more complexity. Do this only after Phase 1.
7. Add a runtime bias correction.
   - File: `predict-transfer.js`, function `predictNode`.
   - Change: compute an EMA of recent residuals over a 48-hour window. Subtract it from the forecast delta. This compensates for sensor drift and seasonal bias without a new fit.
8. Extend the horizon to 3-4 hours where the lag allows it.
   - File: `tools/fit-transfer.py`, constant `HORIZONS`.
   - Change: add horizons 5-8 for Cento (lag 480 min) and Gallo (lag 360 min). Adopt each horizon only if the backtest skill stays above the threshold.

## Phase 3: Monitoring

9. Add a final holdout split in the fit.
   - File: `tools/fit-transfer.py`, function `month_blocks`.
   - Change: reserve the last three months as an extra holdout. Report its metrics in `transfer-coeffs.json` next to the monthly-split metrics.
10. Densify the backtest.
    - File: `tools/backtest.js` usage.
    - Change: run with `--every 120` for the standard report. Event buckets with n<300 are too small to trust.
11. Track skill drift.
    - Change: store `models/backtest-summary.json` with each fit. Compare the new skill with the stored skill. Print a warning when a station drops by more than 0.05.

## Verification

Run these commands after each change:

```bash
uv run python tools/fit-transfer.py --data ./storico --out models/transfer-coeffs.json
node tools/backtest.js --every 120 --json models/backtest-summary.json
node --test tests/*.test.js
```

Success criteria:

- No station loses skill after the lag-leakage fix.
- Fork coverage on events reaches at least 75% per station.
- Every station with `reliable=true` shows backtest event skill at +120 minutes of 0.10 or more.
- The runtime tests still pass.
