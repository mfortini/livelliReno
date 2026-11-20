# A/B M4 — retrain leakage-free (`--holdout-from 2026-01-01`)

**Data:** 2026-11-20
**Export:** [`cento-dnn-longhist.json`](cento-dnn-longhist.json) · skill120 valEvt≈0.806 · `reliable` fit=true (non in soft-load UI)
**Protocollo:** [`eval-table-cento-m4.md`](eval-table-cento-m4.md) / [`eval-per-event-cento-m4-longhist.json`](eval-per-event-cento-m4-longhist.json)
**Disclaimer:** sperimentale; non operativo, non allertistico.

## Setup

Stessa ricetta prod (novalle, poly-points 6, h=30–360′). Train/val solo mesi **&lt; 2026-01-01**; test protocollo ≥2026-01-01.

## Risultati test 2026 (seed 1)

| modello | h | skill evt | rising RMSE | risingMin med | falling RMSE | phaseLag′ | peak err med |
|---|---:|---:|---:|---:|---:|---:|---:|
| prod | 120 | 0.674 | 0.121 | **−0.560** | 0.020 | 27′ | 0.027 |
| longhist | 120 | 0.667 | 0.123 | **−0.560** | 0.027 | 29′ | 0.048 |
| prod | 360 | 0.544 | 0.473 | **−1.690** | 0.064 | 120′ | 0.092 |
| longhist | 360 | 0.539 | 0.478 | **−1.690** | 0.066 | 120′ | 0.080 |

Paired Δ risingMin (longhist−prod) su eventi `peak≥4` (n=6): mediana **0** @120 e @360 (mean −0.9 / −3.9 cm).

## Gate

- Δ risingMin ≥ +0.15 m → **FAIL**
- falling entro +10% → **FAIL** @120 (0.027 vs 0.020, +35%)

## Decisione

**Non promuovere** longhist al posto di prod. Il retrain leakage-free è utile come baseline onesta (skill un filo più basso del prod “optimistic”), ma **non** chiude il gap di salita: conferma che il problema non è solo leakage sul test 2026.

## Contesto piano

- M1 ruw3 / M3 MoE rising-only: stesso esito su `risingMin` (invariato).
- Resta utile in prod: **M2** (UI salita + inflate banda).
- Prossime leve fuori da MSE/retrain flat: loss coda/L^p su salita, testa picco, o routing in Q.
