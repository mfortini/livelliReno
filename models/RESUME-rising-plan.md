# Piano Cento rising — stato finale sessione 2026-11-20

## Chiuso

| Step | Esito | Artefatti |
|------|--------|-----------|
| M0 metriche | OK | protocollo + `risingMin` |
| M1 ruw3 | **FAIL** | `ab-rising-asymmetric.md`, `cento-dnn-ruw3.json` |
| M2 UI/banda | **OK (adottabile)** | inflate `cento-rising-band-inflate.json`, Predizioni |
| M3 MoE rising | **FAIL** | `ab-rising-moe.md`, `cento-dnn-rising.json` (`reliable:false`) |
| M4 longhist | **FAIL** vs gate salita | `ab-m4-longhist.md`, `cento-dnn-longhist.json`, `eval-table-cento-m4.*` |
| M4 QC/folds | OK | `wpC-inventario-qc.md`, `eval-table-cento-m4-folds.*` |

**Prod non patchato** (né ruw né longhist né MoE).

## Opzionale non fatto

- `ruw2` / `ruw4` (ruw3 già conclusivo)
- retrain year-fold completo dei pesi (folds solo eval)

## Se riparti

Nuove leve (non MSE flat): loss L4/focal su salita, testa “picco entro h”, routing portata.
