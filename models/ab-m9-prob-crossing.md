# A/B M9 — crossing-time probabilistico + train piena (Cento)

**Data:** 2026-11-15  
**Ipotesi:** train flood-only + Casalecchio chiusa + quantili di τ (crossing-time) + decode (warp/lift) sulla fan di livello riducono rising RMSE e phaseLagMin; curva «più probabile» = moda di τ.

## Dati e recipe

- Storico esteso: **2023-01 → 2026-11** (ARPAE ok da 2023-01); catalogo 913 eventi (Cento 65).
- Upstream: Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo, **Casalecchio chiusa**.
- Train mask: solo finestre catalogo ±6 h (`--flood-only`).
- Crossing-time: soglie relative +0.5/+1.0/+1.5 m, pinball q10/q50/q90 di τ, censura 6 h → `models/cento-crossing-time.json` (+ ablazione `…-nochiusa.json`).
- Quantili livello flood-only q10/q50/q90 → `models/cento-dnn-flood-quantiles.json` / `cento-dnn-flood-bands.json` (non sostituiscono le bande prod).

## Decode runtime

In `predict-transfer.js`: soft-load crossing; `parseWindowMlpNode` conserva `crossingTime`.  
Con `useCrossingTime` / `adopt`:

1. se il path MLP raggiunge già +offset → **time-warp** alla moda di τ;
2. altrimenti **level-lift** costruttivo (ramp fino a thr a modeTau).

UI Predizioni: etichetta «più probabile» + fan; `adopt` resta **false**.

## Risultati (spotlight Nov + test 2026)

Fonte: `models/fase2-m9-eval.json` (run con lift attivo).

| Slice | h | RMSE | skill | phaseLag | rising RMSE | falling RMSE |
|---|---:|---:|---:|---:|---:|---:|
| nov/prod | 120′ | **0.159** | **0.572** | 30′ | **0.175** | **0.019** |
| nov/cross+chiusa | 120′ | 0.276 | 0.255 | **−30′** | 0.304 | 0.019 |
| nov/cross-nochiusa | 120′ | 0.261 | 0.297 | 0′ | 0.286 | 0.058 |
| test/prod | 120′ | **0.080** | **0.724** | 30′ | **0.103** | **0.022** |
| test/cross+chiusa | 120′ | 0.259 | 0.110 | 0′ | 0.298 | 0.189 |

## Decisione

| Voce | Esito |
|---|---|
| Rising@120 vs prod | **FAIL** (lift troppo aggressivo / overshoot) |
| phaseLagMin@120 | migliorato (−30′ / 0′ vs +30′) ma a costo di RMSE |
| Casalecchio chiusa vs nochiusa | nochiusa leggermente meno peggio su rising; **non decisivo** |
| Soft-adopt crossing | **no** (`adopt: false`) |
| Bande flood-only | tenute come artefatto di ricerca; soft-merge prod resta `cento-dnn-bands.json` |

**Chiusura motivata:** il timing crossing-time sposta la fase nella direzione giusta, ma il decode lift/warp sulla spezzata Δy **peggiora** rising e skill. Serve un modello che generi la traiettoria di piena dal target τ (non un post-processing sulla mediana MSE). Codice e fit restano in repo per cicli successivi.

## File

- `tools/fit-crossing-time.py`, `tools/fit-ml-cento.py` (`--flood-only`, `--no-chiusa`, UPSTREAMS+chiusa)
- `tools/ab-m9-prob-crossing.js` → questo report + `fase2-m9-eval.json`
- Runtime: `mergeCrossingTime`, `predictCrossingTau`, `warpForecastByCrossing`, `softMergeCentoCrossing`
- UI: `predizioni.html` testi «più probabile»
