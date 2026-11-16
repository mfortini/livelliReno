# A/B — feature solo-delta sulla MLP flat (metodi attuali)

**Data:** 2026-11-16  
**Gate promote:** **no** (soft-adopt off; `reliable` invariato)

## Contesto

Target di previsione è già **Δy** (MLP e ridge). Qui si toglie il **livello assoluto dalle feature** (`level_*`), lasciando `delta` / `miss` / `d1` / `d2` (+ rain): niente ricostruzione di stato assoluto negli input.

Il ridge di produzione usa già solo incrementi (`dy_local`, `dxw`/`dxr`, …) — non è un nuovo candidato feature.

## Setup

| Voce | Valore |
|---|---|
| Stazioni | Cento + Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo |
| Split | `--holdout-from 2026-01-01` |
| Horizons | 1…12 |
| Flat level | `cento-dnn-leakfree-flat.json` (riuso) |
| Flat delta | `--feature-signal delta` → `cento-dnn-leakfree-fdelta.json` (962 feat = 4×48×5 + rain) |
| Protocollo | `ab-feature-delta-eval.json` · seed 1 |

## Risultati @120′ (eventi test, n=10)

| Modello | skill evt | CI90 | rising RMSE | phaseLag′ |
|---|---:|---|---:|---:|
| prod | 0.68 | [0.63, 0.76] | 0.11 | 25.8 |
| leakfree_flat (level+Δ) | 0.68 | [0.63, 0.76] | 0.11 | 26.0 |
| **fdelta** (solo Δ) | 0.67 | [0.62, 0.74] | 0.12 | 27.7 |
| ridge (già Δ-feat) | 0.01 | — | 0.37 | 87.2 |

## Decisione

- Solo-delta sulla flat MLP è **pari o leggermente peggiore** (CI sovrapposti; rising e lag non migliorano).
- Il livello assoluto nella finestra porta informazione utile (datum / offset di regime) oltre ai tassi.
- **No adopt.** CLI `--feature-signal delta` resta per esperimenti.

## CLI

```bash
uv run python tools/fit-ml-cento.py --no-acceptors --no-chiusa --skip-lgbm --poly-points 6 \
  --holdout-from 2026-01-01 --horizons 1 2 3 4 5 6 7 8 9 10 11 12 \
  --feature-signal delta \
  --export-dnn models/cento-dnn-leakfree-fdelta.json \
  --out models/ml-cento-compare-leakfree-fdelta.json

node tools/backtest.js --protocol --stations Cento --year-folds 0 --seed 1 \
  --only-candidates prod,ridge,leakfree_flat,fdelta \
  --extra-mlp leakfree_flat=models/cento-dnn-leakfree-flat.json \
  --extra-mlp fdelta=models/cento-dnn-leakfree-fdelta.json \
  --json models/ab-feature-delta-eval.json --md models/ab-feature-delta-eval.md \
  --per-event models/ab-feature-delta-per-event.json
```
