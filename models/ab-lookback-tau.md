# A/B — lookback monte scalato su τ (leakage-free)

**Data:** 2026-11-16  
**Gate promote:** **no** (soft-adopt off; `reliable` invariato)

## Setup

| Voce | Valore |
|---|---|
| Stazioni | Cento + Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo (`--no-acceptors --no-chiusa`) |
| Split | `--holdout-from 2026-01-01` (train/val solo mesi &lt; 2026; test protocollo ≥ 2026-01-01) |
| Horizons | 1…12 (30′…360′) |
| Flat | finestra unica 48 step (24 h) → `cento-dnn-leakfree-flat.json` |
| Lookback-τ | `L_i = min(48, max(6, τ_i + 8))` step, buffer 4 h → `cento-dnn-leakfree-lbtau.json` |
| τ (train months) | Forcelli 510′, Sostegno 180′, Bonconvento 390′, Casalecchio tiro 660′ |
| L_i (step) | Cento 48, Forcelli 25, Sostegno 14, Bonconvento 21, Casalecchio tiro 30 |
| Protocollo | `models/ab-lookback-tau-eval.json` · seed 1 · `--only-candidates prod,ridge,leakfree_flat,leakfree_lbtau` · year-folds off |

Formulazione distinta da `--lag-align` (già chiuso senza promote): qui si **accorcia** la finestra monte a τ+buffer, senza shift di fase.

## Risultati @120′ (eventi test, n=10)

| Modello | skill evt /pers | CI90 skill | rising RMSE | phaseLag′ | NSE evt | KGE evt | peak err med (m) |
|---|---:|---|---:|---:|---:|---:|---:|
| prod (pesi attuali) | 0.680 | [0.626, 0.761] | 0.114 | 25.8 | 0.999 | 0.994 | 0.009 |
| leakfree_flat | 0.682 | [0.630, 0.756] | 0.114 | 26.0 | 0.999 | 0.993 | 0.016 |
| leakfree_lbtau | 0.685 | [0.636, 0.755] | 0.113 | 25.6 | 0.999 | 0.994 | 0.001 |
| ridge | 0.013 | — | 0.372 | 87.2 | 0.987 | 0.980 | −0.040 |
| arx | 0.480 | — | 0.187 | 40.3 | 0.997 | 0.990 | 0.062 |

## @360′ (nota)

| Modello | skill evt | rising RMSE | phaseLag′ | NSE | KGE |
|---|---:|---:|---:|---:|---:|
| prod | 0.547 | 0.452 | 120 | 0.980 | 0.967 |
| leakfree_flat | 0.541 | 0.455 | 120 | 0.980 | 0.952 |
| leakfree_lbtau | 0.546 | 0.452 | 120 | 0.980 | 0.964 |

## Decisione

- Lookback-τ e flat leakage-free sono **indistinguibili** da prod su skill/rising/NSE/KGE: gli intervalli bootstrap a 90% si sovrappongono.
- Nessun miglioramento chiaro del rising limb o del phase lag (~26′ @2 h).
- **No adopt.** Produzione resta `window_delta_mlp` attuale.

## Artefatti

- `models/cento-dnn-leakfree-flat.json`, `models/cento-dnn-leakfree-lbtau.json`
- `models/ml-cento-compare-leakfree-flat.json`, `models/ml-cento-compare-leakfree-lbtau.json`
- `models/ab-lookback-tau-eval.json` / `.md` / `ab-lookback-tau-per-event.json`

## CLI

```bash
uv run python tools/fit-ml-cento.py --no-acceptors --no-chiusa --skip-lgbm --poly-points 6 \
  --holdout-from 2026-01-01 --horizons 1 2 3 4 5 6 7 8 9 10 11 12 \
  --export-dnn models/cento-dnn-leakfree-flat.json --out models/ml-cento-compare-leakfree-flat.json

uv run python tools/fit-ml-cento.py --no-acceptors --no-chiusa --skip-lgbm --poly-points 6 \
  --holdout-from 2026-01-01 --lookback-tau --lookback-tau-buffer-hours 4 \
  --horizons 1 2 3 4 5 6 7 8 9 10 11 12 \
  --export-dnn models/cento-dnn-leakfree-lbtau.json --out models/ml-cento-compare-leakfree-lbtau.json

node tools/backtest.js --protocol --stations Cento --year-folds 0 --seed 1 \
  --only-candidates prod,ridge,leakfree_flat,leakfree_lbtau \
  --extra-mlp leakfree_flat=models/cento-dnn-leakfree-flat.json \
  --extra-mlp leakfree_lbtau=models/cento-dnn-leakfree-lbtau.json \
  --json models/ab-lookback-tau-eval.json --md models/ab-lookback-tau-eval.md \
  --per-event models/ab-lookback-tau-per-event.json
```
