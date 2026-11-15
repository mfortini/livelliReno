# A/B M8 — correzione online di fase + bias (Cento MLP)

**Data:** 2026-11-15  
**Ipotesi (prima del run):** assimilazione online su residui recenti (time-shift gated ±60′ + bias EMA di livello) riduce `phaseLagMin` e RMSE rising @120′ senza peggiorare skillEvt / falling / FAR rispetto a prod.

**Criterio di successo:** RMSE rising @120′ chiaramente migliore di prod; `phaseLagMin` @120′ ridotto; falling e skill all non peggiori.

## Contesto misura

Piena Cento aggiornata (storico fino al 2026-11-15): start `2026-11-09`, **peak 7.17 m @ 2026-11-14T07:00Z**, end ~16 nov.  
Asimmetria prod confermata in [`ab-rising-falling-nov.md`](ab-rising-falling-nov.md):

| Slice | rising@120 RMSE | falling@120 RMSE |
|---|---:|---:|
| Nov event / prod | **0.175** (n=237) | **0.019** (n=35) |
| Test 2026 eventi / prod | 0.102 | 0.022 |

## Setup

- Runtime: `estimateMlpOnlineCorrection` + `applyMlpOnlineCorrection` in `predict-transfer.js`
- Opt-in: `predictNode({ onlineCorrection: true })` (default **off**)
- Finestra causale 12 h, stride 1 h, horizon replay 2 h; phase gated ±60′ se \|r\|≥0.5
- Candidati: `prod` (MLP prod, smooth=0) vs `mlp_online` (stessi pesi + online)

## Risultati — evento Nov (spotlight)

Fonte: `models/fase2-m8-eval.json`

| Modello | h | n | RMSE | skill vs persist | phaseLagMin | rising RMSE | falling RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| prod | 120′ | 280 | **0.161** | **0.568** | 30′ | **0.175** | **0.018** |
| mlp_online | 120′ | 280 | 0.172 | 0.538 | 30′ | 0.184 | 0.079 |
| prod | 360′ | 280 | 0.528 | 0.466 | 120′ | 0.574 | 0.047 |
| mlp_online | 360′ | 201† | 0.253 | 0.717 | 30′ | 0.274 | 0.057 |

† n più basso @360′: lo shift di fase sposta knot fuori dalla finestra di match — guadagno apparente non confrontabile a parità di campione.

Catalogo test (`ab-rising-falling-nov.json`): mlp_online peggiora rising@120 (0.119 vs 0.102) e falling@120 (0.047 vs 0.022).

## Decisione

| Voce | Esito |
|---|---|
| RMSE rising @120′ | **FAIL** (peggio di prod) |
| phaseLagMin @120′ | **FAIL** (invariato a 30′) |
| Falling / skill all | **FAIL** (peggio) |
| Promozione runtime | **no** — resta `onlineCorrection` opt-in solo per ricerca |

**Chiusura:** come M4/M5. Il residuo recente non ri-fasa in modo stabile la spezzata a orizzonti indipendenti; il bias EMA aiuta poco in salita e danneggia la discesa.  
**Prossima leva** (ciclo separato): target **crossing-time** / tempo di attraversamento soglia, non correzione post-hoc sulla mediana Δy.

## File

- Runtime: `predict-transfer.js` (`estimateMlpOnlineCorrection`, `applyMlpOnlineCorrection`)
- Backtest candidate: `mlp_online` in `tools/backtest.js --compare-candidates`
- Misura limb: `tools/ab-rising-falling.js` → `models/ab-rising-falling-nov.md`
- Eval: `models/fase2-m8-eval.json`
