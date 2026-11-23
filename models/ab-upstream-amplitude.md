# A/B — prior ampiezza monte (peak-within-h) Cento

**Data:** 2026-11-23  
**Artefatto:** [`cento-upstream-amplitude.json`](cento-upstream-amplitude.json) (`adopt: true`)  
**Fit:** `node tools/fit-upstream-amplitude.js` — ridge su Δ picco entro h da livelli/Δ monte (eventi Cento peak≥2 m).

## Runtime

In regime salita (`detectRisingRegime`): solleva la mediana MLP se il prior peak-within-h è più alto (`applyUpstreamAmplitudeLift`). Soft-merge in `loadCoeffs` + Predizioni.

## Test 2026 (8 eventi Cento peak≥2, ramo salita, stride 1 h)

| | MAE@2h | bias@2h | MAE@4h | MAE@6h | falling MAE@2h |
|---|---:|---:|---:|---:|---:|
| prod | 0.045 | −0.032 | 0.105 | 0.196 | 0.018 |
| +amp | 0.049 | −0.009 | **0.097** | **0.172** | 0.018 |

Sottostime grosse (err&lt;−0.15 m su Δy&gt;0.05): 33%→28% @6h.

## Piena 23/11 (00–10Z live)

| | MAE@2h | MAE@4h | MAE@6h | bias@6h |
|---|---:|---:|---:|---:|
| prod | 0.096 | 0.103 | 0.130 | −0.11 |
| +amp | **0.074** | 0.123 | 0.161 | **≈0** |

A +6h MAE un filo peggiore (overshoot a tratti), bias azzerato; @2h migliora.

## Gate

- Rising MAE@4–6h test: **PASS**
- Falling: **PASS** (invariato)
- Soft-adopt in salita: **sì** (`adopt: true`)
