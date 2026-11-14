# A/B M3 — Blend GRU≤2 h + ridge Gallo

**Ipotesi:** GRU fino a ~90′, raccordo lineare 90–150′, ridge oltre → skill ≥ GRU@2 h e ≥ ridge@6 h, senza salto.

**Criterio:** skill evento non inferiore a `seq_gru` @120′ e a `ridge`/`prod` @360′; raccordo monotono (runtime testato).

**Protocollo:** `--every 120`, test `2026-01-01`→, seed 1. Runtime `blend_gru_ridge` in `predict-transfer.js`.

## Skill evento

| Modello | @120′ | CI90 | peak time med (′) | @360′ |
|---|---:|---|---:|---:|
| seq_gru | **0.754** | — | 0 | n/d (≤120′) |
| **blend_gru_ridge** | 0.721 | — | 270 | **0.470** |
| ridge | 0.603 | [0.48, 0.72] | 270 | 0.470 |
| prod (ridge) | 0.594 | [0.48, 0.70] | 270 | 0.469 |

Δ(blend − gru) @120 ≈ −0.033 (blend sotto GRU).  
Δ(blend − ridge) @360 ≈ 0 (pari).  
Δ(blend − prod) @120 ≈ +0.13 (migliore del prod attuale).

## Decisione

| Condizione | Esito |
|---|---|
| ≥ GRU @2 h | **FAIL** (−0.03) |
| ≥ ridge @6 h | **PASS** |
| Raccordo runtime | **PASS** (unit test; regime `gru`/`blend`/`ridge`) |

**Promozione soft-adopt:** no (criterio stretto fallisce sul corto).  
**Utilità:** il blend è il miglior compromesso 2 h+6 h vs prod-only ridge; candidabile in Fase 3 se si accetta −0.03 @120′ in cambio della banda 6 h con corto quasi-GRU.

## File

- Runtime: `forecastBlendGruRidge`, kind `blend_gru_ridge`
- Protocol candidate: `blend_gru_ridge` in `tools/protocol-mode.js`
- Eval: `models/fase2-m3-eval.json`
- GRU backup: `models/seq-gallo-gru.reliable-120.json`
