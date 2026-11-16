# A/B — MLP residuo vs poly2 (Cento)

**Data:** 2026-11-16  
**Gate promote:** **no** (soft-adopt off; `reliable` invariato)

## Contesto

Come `ab-residual-lin3`, ma baseline **poly2** (OLS quadratico) sugli ultimi **6** punti 30′ di Cento (2.5 h — stesso sweep già usato per d1/d2 e holdout skill). Target `r = y(t+h) − ŷ_poly2`; feature `poly2_delta_Cento = ŷ_poly2 − y0`; decode `ŷ = ŷ_poly2 + r̂`. Vicini: solo lookback passato.

## Setup

| Voce | Valore |
|---|---|
| Stazioni | Cento + Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo |
| Split | `--holdout-from 2026-01-01` |
| Horizons | 1…12 |
| Flat level | `cento-dnn-leakfree-flat.json` (riuso) |
| Residual poly2 | `--target-mode residual-poly2 --poly-points 6` → `cento-dnn-leakfree-rpoly2.json` |
| Protocollo | `ab-residual-poly2-eval.json` · seed 1 |

## Risultati @120′ (eventi test, n=10)

| Modello | skill evt | CI90 | rising RMSE | phaseLag′ |
|---|---:|---|---:|---:|
| prod | 0.68 | [0.63, 0.76] | 0.11 | 25.8 |
| leakfree_flat | 0.68 | [0.63, 0.76] | 0.11 | 26.0 |
| **rpoly2** | 0.69 | [0.64, 0.76] | 0.11 | 22.4 |
| rlin3 (ref) | 0.69 | [0.63, 0.76] | 0.11 | 24.5 |
| ridge | 0.01 | — | 0.37 | 87.2 |

## Decisione

- Residuo vs poly2(6) è **pari** a flat/prod su skill (CI sovrapposti) e rising.
- Phase lag un po’ migliore (−3.4′ vs prod; −2.1′ vs rlin3) ma non abbastanza da promuovere da solo.
- **No adopt.** CLI `--target-mode residual-poly2` resta per esperimenti.

## CLI

```bash
uv run python tools/fit-ml-cento.py --no-acceptors --no-chiusa --skip-lgbm --poly-points 6 \
  --holdout-from 2026-01-01 --horizons 1 2 3 4 5 6 7 8 9 10 11 12 \
  --target-mode residual-poly2 \
  --export-dnn models/cento-dnn-leakfree-rpoly2.json \
  --out models/ml-cento-compare-leakfree-rpoly2.json

node tools/backtest.js --protocol --stations Cento --year-folds 0 --seed 1 \
  --only-candidates prod,ridge,leakfree_flat,rpoly2 \
  --extra-mlp leakfree_flat=models/cento-dnn-leakfree-flat.json \
  --extra-mlp rpoly2=models/cento-dnn-leakfree-rpoly2.json \
  --json models/ab-residual-poly2-eval.json --md models/ab-residual-poly2-eval.md \
  --per-event models/ab-residual-poly2-per-event.json
```
