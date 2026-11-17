# Esperimento: input MLP a 10 minuti, previsioni a passo 10 minuti — Cento

_Data: 2026-11-15. Esito: **negativo, nessuna promozione.**_

## Protocollo

- Training: `tools/experiment-temporal.py --method linear
  --input-step-minutes 10 --lookback-minutes 1440` → 144 lag/stazione,
  3602 feature totali, orizzonti 30–360′, stessi split/origini.
- Previsione: backtest con emissioni **ogni 10 minuti** (`--every 10`),
  stessi timestamp per tutti i candidati; solo MLP a confronto
  (`--only mlp_prod,mlp_temporal_10m,poly2`).
- Il modello 30′ (`mlp_prod`) emette a ogni tick da 10′ usando le proprie
  feature a passo 30′: la cadenza di emissione non dipende dal passo input.

## Risultati — backtest gen–mar 2026, 12.961 emissioni (metriche su n=4312)

| Candidato | skill@120′ | skill@240′ | skill@360′ | rmse@120′ |
|---|---:|---:|---:|---:|
| **mlp_prod (train 30′)** | **0.765** | **0.682** | **0.613** | **0.0334** |
| mlp_temporal_10m (train 10′) | 0.663 | 0.574 | 0.532 | 0.0479 |
| poly2 | 0.613 | 0.336 | 0.018 | 0.0550 |

Holdout evento del trainer 10′: valEvt@120 0.664, holdEvt@120 0.258 —
in linea col pattern degli esperimenti precedenti (debole a 120′,
migliore a orizzonti lunghi).

## Conclusioni

1. **L'input a 10 minuti peggiora** rispetto al 30′, come già osservato a
   5 minuti (0.655) e PCHIP (0.576): more points → più varianza, stessa
   informazione. Il pattern è monotono: 30′ > 10′ > 5′.
2. **Le emissioni a passo 10′ non cambiano lo skill** del modello 30′
   (0.765 vs 0.769 a passo 120′): la densificazione temporale delle
   previsioni è neutra per l'accuratezza — le emissioni ravvicinate sono
   correlate, non indipendenti.
3. Il backtest ora supporta `--every 10` e `--only <keys>` per scan
   mirati; nuovo candidato `mlp_temporal_10m`.

## File

- `models/experiments/cento-temporal-linear-10min{,-dnn}.json`
- `models/experiments/feat-10min-lin.npz` (cache feature)
- `models/cento-10min-backtest.json` (riepilogo; `cento-candidates-backtest.json`
  è stato sovrascritto dall'ultima corsa `--only`)

## Riproduzione

```bash
uv run python tools/experiment-temporal.py --method linear \
  --input-step-minutes 10 --lookback-minutes 1440 --stride 12 \
  --feat-cache models/experiments/feat-10min-lin.npz \
  --out models/experiments/cento-temporal-linear-10min.json \
  --tag cento-temporal-linear-10min
node tools/backtest.js --compare-candidates Cento --from 2026-01-01 \
  --to 2026-04-01 --every 10 --only mlp_prod,mlp_temporal_10m,poly2 \
  --json models/cento-10min-backtest.json
```
