# A/B M1 — loss asimmetrica rising-under-weight (Cento)

**Data:** 2026-11-20
**Ipotesi:** penalizzare la sottostima solo su Δy>0 riduce risingMin / rising RMSE senza ripetere ew4/q75.
**Ricetta:** novalle, poly-points 6, orizzonti 30–360′, `--rising-under-weight α`.
**Disclaimer:** sperimentale; non operativo, non allertistico.

## Fit

| α | tag | skill120 valEvt | reliable | export |
|---:|---|---:|---|---|
| 3 | `novalle-p6-ruw3` | 0.825 | true | [`cento-dnn-ruw3.json`](cento-dnn-ruw3.json) |
| 2 | in training | — | — | `cento-dnn-ruw2.json` |
| 4 | previsto | — | — | `cento-dnn-ruw4.json` |

## Protocollo Fase 0 (test ≥2026-01-01, seed 1)

Fonte: [`eval-table-cento-ruw.md`](eval-table-cento-ruw.md) / [`eval-per-event-cento-ruw.json`](eval-per-event-cento-ruw.json).

| modello | h | skill evt | rising RMSE | risingMin med | falling RMSE | phaseLag′ | peak err med |
|---|---:|---:|---:|---:|---:|---:|---:|
| prod | 120 | 0.674 | 0.121 | **−0.560** | 0.020 | 27.0 | 0.027 |
| ruw3 | 120 | 0.682 | 0.119 | **−0.560** | 0.019 | 24.2 | 0.011 |
| prod | 360 | 0.544 | 0.473 | **−1.690** | 0.064 | 120 | 0.092 |
| ruw3 | 360 | 0.547 | 0.468 | **−1.690** | 0.072 | 120 | 0.081 |

Paired Δ risingMin (ruw3−prod) su eventi test `peak≥4` (n=6):

| h | mean | median | p10 | p90 |
|---:|---:|---:|---:|---:|
| 120 | +0.008 | 0.000 | 0.000 | +0.045 |
| 360 | −0.002 | 0.000 | −0.013 | 0.000 |

## Gate adozione (piano)

- CI90 Δ risingMin@120 e @360 > 0 su `peak≥4` → **FAIL** (mediana 0; guadagno ≪ 15 cm)
- falling entro +10% → OK @120, borderline peggio @360 (0.072 vs 0.064)
- phaseLag non peggiora >15′ → OK (leggero miglioramento @120)

## Decisione

**Non promuovere** ruw3 in prod / `transfer-coeffs`.
La loss asimmetrica sul Δy non sposta il residuo minimo in salita: conferma che il collo di bottiglia è strutturale (fronte/lag), non solo bias di ampiezza sul MSE.

**Next:** M3 mixture-of-experts salita (`--rising-only` + gate causale).

```bash
node tools/backtest.js --protocol --stations Cento \
  --only-candidates persist,prod,ruw3 \
  --extra-mlp ruw3=models/cento-dnn-ruw3.json \
  --from 2026-01-01 --seed 1 --year-folds 0 \
  --json models/eval-table-cento-ruw.json
```
