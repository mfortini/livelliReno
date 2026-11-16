# A/B — griglia di fase φ=lag/τ fino a 2τ (densità + delta)

**Data:** 2026-11-16  
**Gate promote:** **no** (soft-adopt off; `reliable` invariato)  
**Piano:** [`ab-tau-fraction-plan.md`](ab-tau-fraction-plan.md)

## Setup

| Voce | Valore |
|---|---|
| Stazioni | Cento + Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo |
| Split | `--holdout-from 2026-01-01` |
| Horizons | 1…12 (30′…360′) |
| Cento | flat locale **24** step (12 h) |
| Monte | `φ=k/n`, `φ_max=2`, `lag=round(φ·τ)` snap 30′ + dedup |
| τ (train) | Forcelli 510′, Sostegno 180′, Bonconvento 390′, Casalecchio tiro 660′ |
| Protocollo | `ab-tau-fraction-eval.json` · seed 1 · year-folds off |

| Candidato | n | signal | nFeat |
|---|---:|---|---:|
| leakfree_flat | — | flat 48 | (riuso) |
| taufrac8 | 8 | level | 442 |
| taufrac16 | 16 | level | 652 |
| taufrac8d | 8 | **delta** (no level monte) | 378 |

## Risultati @120′ (eventi test, n=10)

| Modello | skill evt /pers | CI90 skill | rising RMSE | phaseLag′ | NSE evt | KGE evt |
|---|---:|---|---:|---:|---:|---:|
| prod | 0.680 | [0.626, 0.761] | 0.114 | 25.8 | 0.999 | 0.994 |
| leakfree_flat | 0.682 | [0.630, 0.756] | 0.114 | 26.0 | 0.999 | 0.993 |
| taufrac8 | 0.683 | [0.630, 0.756] | 0.113 | 24.1 | 0.999 | 0.994 |
| taufrac16 | 0.683 | [0.630, 0.758] | 0.113 | 23.8 | 0.999 | 0.994 |
| taufrac8d | 0.678 | [0.630, 0.744] | 0.116 | 27.8 | 0.999 | 0.994 |
| ridge | 0.013 | — | 0.372 | 87.2 | 0.987 | 0.980 |

(Valori arrotondati dalla tabella protocollo; CI da `ab-tau-fraction-eval.md`.)

## @360′ (nota)

Tutti i candidati MLP: skill evt ≈ 0.54–0.55, rising ≈ 0.45–0.46, phaseLag 120′ (saturo). Nessuna separazione.

## Decisione

- **n=8 ≈ n=16 ≈ flat**: densità della griglia irrilevante; CI skill/rising sovrapposti.
- **delta-only** (`taufrac8d`): leggermente peggiore su rising e phaseLag; nessuna riduzione di stato utile in pratica.
- PhaseLag ~24′ vs ~26′ su n=16: rumore, non promote.
- **No adopt.** Produzione resta `window_delta_mlp` flat. Codice CLI/runtime resta disponibile.

## Artefatti

- `models/cento-dnn-leakfree-taufrac8.json` / `…16.json` / `…8d.json`
- `models/ml-cento-compare-leakfree-taufrac*.json`
- `models/ab-tau-fraction-eval.json` / `.md` / `ab-tau-fraction-per-event.json`

## CLI

```bash
uv run python tools/fit-ml-cento.py --no-acceptors --no-chiusa --skip-lgbm --poly-points 6 \
  --holdout-from 2026-01-01 --horizons 1 2 3 4 5 6 7 8 9 10 11 12 \
  --tau-frac --tau-frac-n 8 --tau-frac-max 2 --tau-frac-signal level \
  --export-dnn models/cento-dnn-leakfree-taufrac8.json \
  --out models/ml-cento-compare-leakfree-taufrac8.json

# densità
#   --tau-frac-n 16
# solo incrementi monte (no level):
#   --tau-frac-signal delta

node tools/backtest.js --protocol --stations Cento --year-folds 0 --seed 1 \
  --only-candidates prod,ridge,leakfree_flat,taufrac8,taufrac16,taufrac8d \
  --extra-mlp leakfree_flat=models/cento-dnn-leakfree-flat.json \
  --extra-mlp taufrac8=models/cento-dnn-leakfree-taufrac8.json \
  --extra-mlp taufrac16=models/cento-dnn-leakfree-taufrac16.json \
  --extra-mlp taufrac8d=models/cento-dnn-leakfree-taufrac8d.json \
  --json models/ab-tau-fraction-eval.json --md models/ab-tau-fraction-eval.md \
  --per-event models/ab-tau-fraction-per-event.json
```
