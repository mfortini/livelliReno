# Fase 2 — Sintesi esperimenti modello

Test congelato `2026-01-01`→. Misura primaria: protocollo Fase 0 (`--every 120` per A/B pesanti).

| ID | Esito | Promuovere? | Nota |
|---|---|---|---|
| M2 | MLP resta | — | CNN pari @120 senza 6 h; GRU/ibridi sotto |
| M1 | Banda OK, picco no | banda sì / mediana no | `cento-dnn-quantiles.json`; skillEvt 0.73 vs 0.71 |
| M3 | Blend utile ma < GRU@2h | no soft-adopt | Runtime `blend_gru_ridge` pronto per Fase 3 |
| M4 | FAIL | no | lag viaggio ~hours; err 139′ |
| M5 | FAIL | no | CI disgiunti su Gallo ma skill peggiora |
| M6 | chiuso | no | manca pioggia prevista |
| M7 | chiuso | no | mancano manovre Napoleonico; valle n=0 in protocollo |

**Gate uscita Fase 2:** nessun pacchetto supera tutte le regole di promozione per sostituire prod. Materiale pronto per Fase 3 selettiva (banda quantili Cento; eventuale blend Gallo come opt-in).
