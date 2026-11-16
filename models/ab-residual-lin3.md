# A/B — MLP residuo vs lin3 (Cento)

**Data:** 2026-11-16  
**Gate promote:** **no** (soft-adopt off; `reliable` invariato)

## Contesto

Produzione predice `Δy = y(t+h) − y(t)`. Qui il target diventa il **residuo rispetto a lin3**: OLS lineare sugli ultimi 3 campioni 30′ di Cento → `ŷ_lin3(t+h)`; la MLP stima `r = y(t+h) − ŷ_lin3` e riceve la feature `lin3_delta_Cento = ŷ_lin3 − y0`. Decode: `ŷ = ŷ_lin3 + r̂`. I vicini restano solo lookback passato (niente lin3 sui monte).

## Setup

| Voce | Valore |
|---|---|
| Stazioni | Cento + Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo |
| Split | `--holdout-from 2026-01-01` |
| Horizons | 1…12 |
| Flat level | `cento-dnn-leakfree-flat.json` (riuso) |
| Residual lin3 | `--target-mode residual-lin3` → `cento-dnn-leakfree-rlin3.json` (1203 feat = flat + `lin3_delta_Cento`) |
| Protocollo | `ab-residual-lin3-eval.json` · seed 1 |

## Risultati @120′ (eventi test, n=10)

| Modello | skill evt | CI90 | rising RMSE | phaseLag′ |
|---|---:|---|---:|---:|
| prod | 0.68 | [0.63, 0.76] | 0.11 | 25.8 |
| leakfree_flat | 0.68 | [0.63, 0.76] | 0.11 | 26.0 |
| **rlin3** | 0.69 | [0.63, 0.76] | 0.11 | 24.5 |
| ridge | 0.01 | — | 0.37 | 87.2 |

## Decisione

- Residuo vs lin3 è **pari** a flat/prod: CI skill sovrapposti; rising invariato; phase lag solo marginalmente migliore (−1.3′).
- Nessun guadagno chiaro su rising limb o timing.
- **No adopt.** CLI `--target-mode residual-lin3` resta per esperimenti; runtime decode già supportato.

## CLI

```bash
uv run python tools/fit-ml-cento.py --no-acceptors --no-chiusa --skip-lgbm --poly-points 6 \
  --holdout-from 2026-01-01 --horizons 1 2 3 4 5 6 7 8 9 10 11 12 \
  --target-mode residual-lin3 \
  --export-dnn models/cento-dnn-leakfree-rlin3.json \
  --out models/ml-cento-compare-leakfree-rlin3.json

node tools/backtest.js --protocol --stations Cento --year-folds 0 --seed 1 \
  --only-candidates prod,ridge,leakfree_flat,rlin3 \
  --extra-mlp leakfree_flat=models/cento-dnn-leakfree-flat.json \
  --extra-mlp rlin3=models/cento-dnn-leakfree-rlin3.json \
  --json models/ab-residual-lin3-eval.json --md models/ab-residual-lin3-eval.md \
  --per-event models/ab-residual-lin3-per-event.json
```
