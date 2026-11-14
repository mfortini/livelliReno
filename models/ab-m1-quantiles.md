# A/B M1 — MLP quantili 10-50-90 + conformal (Cento)

**Ipotesi (prima del run):** pinball q10/q50/q90 + inflate conformal per orizzonte dà copertura banda ≈80% e picco migliore della MLP MSE in prod.

**Criterio di successo:** copertura banda vicina all’80%; errore sul picco minore di prod.

**Protocollo:** `--every 120` sul test congelato `2026-01-01`→, seed 1, candidati `prod` vs `mlp_q` (`models/cento-dnn-quantiles.json`).

## Skill / banda @120′

| Modello | skillEvt | CI90 | peak lvl med (m) | peak time med (′) | band cov all | band cov evt |
|---|---:|---|---:|---:|---:|---:|
| prod (MSE MLP) | 0.713 | [0.68, 0.75] | **+0.015** | +30 | 0.872 | 0.871 |
| **mlp_q** (pinball+conf) | **0.725** | [0.69, 0.77] | +0.020 | +30 | **0.880** | **0.887** |

Δ skill (q − prod) ≈ +0.01 (CI sovrapposti).  
Copertura banda ~88% (target ~80% su validate conformal; leggermente larga ma utilizzabile).  
Errore sul picco **non** migliora (+0.020 vs +0.015).

Validate fit (export): skillEvt q50@120 ≈ 0.83; coverage conformal validate subsample ≈ 0.80.

## Decisione

| Voce | Esito |
|---|---|
| Mediana q50 vs prod | Pari / lievemente meglio — **non** sostituisce prod da sola |
| Banda 10–90 + conformal | **Utile** per Fase 3 (UI banda) |
| Errore picco | **FAIL** criterio |

**Promozione runtime mediana:** no.  
**Promozione banda (Fase 3):** sì come soft-load opzionale di `cento-dnn-quantiles.json` per spread, senza cambiare il path mediano finché il picco non migliora.

## File

- `models/cento-dnn-q10.json` / `q50` / `q90` — fit pinball
- `models/cento-dnn-quantiles.json` — merge + `conformalQhat` per orizzonte
- `models/fase2-m1-eval.json` — protocollo A/B
- Runtime: `parseWindowMlpNode` + `forecastWindowMlp` leggono `layersQ*` / `conformalQhat`
