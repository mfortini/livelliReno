# Esperimento: passato a 5 minuti (interpolazione causale) — Cento

Stato: **COMPLETATO — nessuna variante promuovibile** (lineare e PCHIP
entrambi peggiori del baseline 30′).

## Setup

- Stesso contratto feature della produzione (`buildWindowFeatureVector`), ma
  input ricampionati causalmente a 5 minuti: 288 step = 24 h di lookback
  (invariato in tempo fisico), derivate su finestra fisica equivalente.
- Interpolazione lineare e PCHIP (monotona, senza overshoot), gap > 60′
  restano mancanti, nessuna estrapolazione oltre l'ultima misura.
- Target e orizzonti invariati: incrementi di 30 minuti, 30–360′.
- Stesse origini (stride 6 h), stessi split mese-blocco, stesso holdout
  ≥ 2026-01-01 del trainer di produzione.
- Implementazione: `tools/temporal_sampling.py` + `temporal-sampling.js`
  (stessa logica), `tools/experiment-temporal.py`.

## Backtest A/B (gen–mar 2026, 2043 emissioni, skill vs poly2)

| Candidato | skill@120′ | skill@240′ | skill@360′ | rmse@120′ |
|---|---:|---:|---:|---:|
| mlp_prod (30′ baseline) | **0.769** | **0.682** | **0.612** | 0.0472 |
| mlp_temporal_lin (5′ lineare) | 0.655 | 0.614 | 0.543 | 0.0707 |
| mlp_temporal_pchip (5′ PCHIP) | 0.576 | 0.570 | 0.534 | 0.0867 |
| seq_cnn | 0.770 | – | – | 0.0471 |
| seq_gru | 0.736 | – | – | 0.0541 |
| poly2 | 0.643 | 0.368 | 0.055 | 0.0730 |

Holdout del trainer (skill eventi): lineare negativo a 30–90′
(-0.70 / -0.14 / -0.05); PCHIP ancora peggio (-0.76 / -0.72 / -0.19 / -0.02
fino a 120′). Entrambi positivi solo da 150′ in su — coerente col backtest.

**Lettura:** densificare il passato interpolato NON migliora la previsione;
peggiora a tutti gli orizzonti valutati. L'ipotesi "il modello vede salti e
quindi salta" non è confermata: la MLP a step fissi impara già la
regolarità temporale dai target a 30′; i punti interpolati aggiungono
varianza senza informazione e sembrano favorire fit/overfit (train_mse
molto più basso del baseline).

## Diagnosi deformazione (DTW, eventi conclusi — oracolo)

`models/experiments/wave-warp.{json,md}` — 8 archi, 169–309 eventi ciascuno.

- Lag mediano plausibile dove `lagCorr` alto: Forcelli→Cento 480′,
  Casalecchio→Bonconvento 270′, Bonconvento→Sostegno 210′,
  Opera→Gallo 420′.
- **Scala affine sempre 1.00**: nessuna evidenza di dilatazione uniforme
  sistematica dopo il lag costante.
- `tauP90` mediano ~120′ (metà del bordo ±2h) e saturazione 0.25–0.74:
  la deformazione residua è ampia e instabile; sui due archi con
  lagCorr basso (Sostegno→Bagnetto 0.25, Dosso→Opera 0.45) le pairing
  evento sono probabilmente ambigue (IQR del lag: 60–1050′ e 30–540′).
- Correlazione lag–picco: solo Cento→Dosso è marcata (r=0.54); sugli altri
  archi debole/negativa → il lag non è chiaramente funzione del livello.

**Lettura:** esiste una deformazione locale, ma non sembra strutturata in
modo ripetibile e semplice (né affine né funzione monotona del picco).
Un modello causale di warp dovrebbe prima dimostrare di stimare τ(t)
meglio del lag costante *sul passato disponibile* — passo non ancora fatto.

## Limiti / non fatto

- Replay causale del DTW (feature di deformazione stimata solo col passato)
  **non implementato**: la diagnosi mostra deformazione debole/ambigua e
  l'interpolazione da sola è già negativa; il valore atteso è basso.
- Testo Python/JS parity su fixture condivise parziale (verificato a mano
  su sequenza di test; test automatici JS in `tests/temporal-sampling.test.js`).
- PCHIP: completato, conferma il trend (peggiore del lineare).
- Nessun modello `reliable` toccato; i JSON in `models/experiments/` sono
  solo candidati forzati nel backtest.

## Riproduzione

```bash
# trainer (cache feature per rerun veloci)
uv run python tools/experiment-temporal.py --method linear \
  --horizons 30 60 90 120 150 180 210 240 270 300 330 360 \
  --out models/experiments/cento-temporal-linear-5min.json

# diagnosi warp
uv run python tools/analyze-wave-warp.py \
  --out models/experiments/wave-warp.json --md models/experiments/wave-warp.md

# backtest candidati
node tools/backtest.js --compare-candidates Cento --from 2026-01-01 \
  --to 2026-04-01 --every 120
```
