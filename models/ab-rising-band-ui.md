# A/B — UI banda salita (M2 in runtime)

**Data:** 2026-11-23  
**Obiettivo:** in regime salita, bordo alto della banda come riferimento operativo (anche senza soft-merge q10/q90 da 48 MB).

## Modifiche

- `loadCoeffs` chiama subito `softMergeCentoRisingInflate` (JSON leggero).
- Predizioni: `emphasizeQ90` se rising ∧ (quantili **o** `risingBand`); meta/pannello citano inflate attivo.
- Test: `applyRisingUpperInflate` + `predictNode` con `risingBand`.

## Hindcast piena 23/11 (00–10Z, oriz. 2/4/6 h)

| | upper coverage | MAE bordo alto | MAE mediana |
|---|---:|---:|---:|
| senza inflate | 60.7% | 0.11 m | 0.11 m |
| con inflate ×2 | **72.1%** | 0.15 m | 0.11 m |

La mediana non cambia (atteso); il bordo alto copre meglio ma resta sotto il target calibrazione 85% su questo evento.

## Gate

- UI/runtime: **sì** (M2 già calibrato; solo wiring).
- Non promuove una nuova mediana.
