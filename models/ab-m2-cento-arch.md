# A/B M2 — Rivalutazione architetture Cento (GRU / 1D-CNN / ibridi)

**Ipotesi (prima del run):** sulla tabella unica Fase 0 il GRU o la CNN possono battere o eguagliare la MLP prod a +2 h; in quel caso andrebbe ripensato soft-adopt.

**Criterio di successo:** differenza di skill evento vs `prod` con CI90 che non comprende lo zero; a parità resta la MLP. Peak error non peggiore.

**Protocollo:** `models/eval-table.json`, test `2026-01-01`→, seed 1. Nessun nuovo addestramento.

## Skill evento @120′ (Cento)

| Modello | skillEvt | CI90 | vs poly2 | peak lvl med (m) | peak time med (′) | @360′ |
|---|---:|---|---:|---:|---:|---:|
| **prod (MLP)** | **0.667** | [0.58, 0.76] | 0.449 | +0.02 | +30 | 0.525 |
| seq_cnn | 0.666 | [0.58, 0.76] | 0.447 | 0.00 | 0 | n/d (≤120′) |
| seq_gru | 0.639 | [0.55, 0.73] | 0.402 | +0.02 | 0 | n/d |
| ridge | 0.206 | [0.07, 0.38] | −0.32 | −0.02 | −90 | 0.112 |
| poly2 | 0.396 | [0.19, 0.63] | 0 | −0.02 | −30 | −0.44 |
| muskingum_gru | 0.053 | [−0.03, 0.14] | −0.57 | +0.02 | +240 | 0.126 |
| rc_gru | −0.146 | [−0.22, −0.08] | −0.90 | −0.05 | −90 | −0.09 |

Δ(prod − seq_cnn) ≈ 0.001 (rumore; CI sovrapposti).  
Δ(prod − seq_gru) ≈ 0.028 (CI sovrapposti; GRU non superiore).  
Ibridi sotto soglia di promozione (cond. 1: lower CI ≥ 0.10 fallisce per muskingum/rc).

## Decisione

| Architettura | Esito | Runtime |
|---|---|---|
| **MLP `window_delta_mlp`** | **Resta** modello Cento | `transfer-coeffs` / prod |
| seq CNN | Pari a MLP @120 ma senza banda 6 h | resta offline (`reliable=false`) |
| seq GRU | Inferiore / non superiore | resta offline |
| Muskingum / RC | Scartati | soft-adopt OFF |

**Promozione:** nessuna. Soft-adopt GRU/CNN Cento resta spento.