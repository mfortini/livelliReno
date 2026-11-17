# Stato esperimenti temporali (aggiornato live)

_Ultimo aggiornamento: ESP VIDEO — nuovo esperimento "spettrale" (FFT + wavelet Haar à trous)
COMPLETATO: feature spettrali peggiorano il modello (skill@120 0.611 vs
0.769 baseline). Nessuna promozione. Dettagli in `spectral-report.md`._

COMPLETATO — input 10′ peggiora come 5′: skill@120 0.663 vs 0.765 prod.
Emissioni a passo 10′ neutre per lo skill. Dettagli `temporal-10min-report.md`.
Backtest completo `--every 120` ripristinato con `mlp_temporal_10m` incluso.

**Pulizia (15:30)**: eliminati cache feature `feat-*.npz`, `smoke-*` e i
pesi `*-dnn.json` dei candidati (~1.2 GB). Restano report, metriche JSON
e backtest. I modelli si rigenerano con i comandi nei report; in backtest
`mlp_temporal_*`/`mlp_spectral` risultano assenti finché non riaddestrati._

Le job in corso sono in background; questo file viene aggiornato man mano.
Per vederlo: `watch -n 30 models/experiments/STATUS.md` oppure `tail -f`.

## Stato job

| Job | Stato | Note |
|---|---|---|
| Analisi wave-warp (diagnostica DTW) | ✅ completata | `models/experiments/wave-warp.{json,md}` |
| Training 5′ lineare (DNN) | ✅ completata | `cento-temporal-linear-5min{,-dnn}.json` |
| Backtest candidati (gen–mar 2026) | ✅ completata | `models/cento-candidates-backtest.json` |
| Training 5′ PCHIP (DNN) | ✅ completato | orizzonti 30–360′, cache `feat-pchip-5min.npz` |
| Backtest PCHIP | ✅ completato | `models/cento-candidates-backtest.json` |
| Replay DTW causale | ⏳ non ancora implementato | vedi limiti sotto |
| Feature spettrali FFT+wavelet | ✅ completato — negativo | `spectral-report.md`, `cento-spectral-30min{,-dnn}.json`, candidato `mlp_spectral` nel backtest |
| MLP input 10′ + previsioni 10′ | ✅ completato — negativo | `temporal-10min-report.md`, `cento-temporal-linear-10min{,-dnn}.json`, `cento-10min-backtest.json` |

## Risultati parziali

### Backtest gen–mar 2026, skill vs poly2 (n=2041, step 120′)

| Candidato | skill@120′ | skill@240′ | skill@360′ | rmse@120′ |
|---|---:|---:|---:|---:|
| mlp_prod (baseline) | **0.769** | **0.682** | **0.612** | 0.0472 |
| mlp_temporal_lin (5′ lineare) | 0.655 | 0.614 | 0.543 | 0.0707 |
| mlp_temporal_pchip (5′ PCHIP) | 0.576 | 0.570 | 0.534 | 0.0867 |
| seq_cnn | 0.770 | – | – | 0.0471 |
| seq_gru | 0.736 | – | – | 0.0541 |
| poly2 | 0.643 | 0.368 | 0.055 | 0.0730 |

**Entrambi i ricampionamenti a 5 minuti sono PEGGIORI del baseline 30′**
su tutte le soglie; PCHIP è peggiore del lineare. Coerente col holdout del
trainer (skill negativo a 30–120′ per entrambi). Esperimento chiuso:
l'interpolazione del passato non aiuta la previsione.

### Diagnosi DTW (eventi conclusi, oracolo — non predittiva)

| edge | n | lag med | lagCorr | scala affine | costo noAlign→affine→lag→dtw |
|---|---:|---:|---:|---:|---|
| Forcelli→Cento | 169 | 480′ | 0.88 | 1.00 | 0.63 → 0.31 → 0.31 → 54.3* |
| Bagnetto→Cento | 143 | 120′ | 0.90 | 1.00 | 0.21 → 0.23 → 0.23 → 35.9* |
| Casalecchio→Bonconvento | 181 | 270′ | 0.95 | 1.00 | 0.39 → 0.13 → 0.13 → 20.4* |
| Bonconvento→Sostegno | 134 | 210′ | 0.96 | 1.00 | 0.21 → 0.11 → 0.11 → 16.4* |
| Sostegno→Bagnetto | 309 | 390′ | **0.25** | 1.00 | 1.81 → 1.19 → 1.38 → 173.7* |
| Cento→Dosso | 169 | 150′ | 1.00 | 1.00 | 0.04 → 0.03 → 0.03 → 2.2* |
| Dosso→Opera Panfilia | 243 | 60′ | **0.45** | 1.00 | 1.09 → 0.71 → 1.08 → 110.5* |
| Opera Panfilia→Gallo | 168 | 420′ | 0.90 | 1.00 | 0.43 → 0.33 → 0.33 → 60.5* |

\* `dtwCost` è su scala diversa (costo accumulato del percorso, non MSE):
confrontare solo `noAlign`/`affine`/`lag` fra loro; `dtw` vs `lag` va letto
insieme a `tauP90`/`sat`. Scala affine sempre 1.00 → nessuna evidenza di
dilatazione uniforme; `tauP90` al 50% del bordo (±2h) → deformazione residua
ampia e/o boundary saturation alta sugli archi deboli.

## Report completo

`models/experiments/temporal-5min-report.md` — setup, tabella A/B, diagnosi
DTW, limiti e comandi di riproduzione.

## Limiti noti

- Il DTW completo-evento è diagnostico; il **replay causale** (stimare la
  deformazione solo col passato disponibile) non è ancora implementato.
- I lag stimati sono buoni dove lagCorr è alto; Sostegno→Bagnetto e
  Dosso→Opera hanno correlazione bassa → eventi ambigui o copertura insufficiente.
- Nessun modello sperimentale è promosso: `reliable` resta false.


Nuovo esperimento KF/UKF/EnKF (`tools/experiment-kalman.py`): fase A
univariata, fase B con ingressi upstream+pioggia e varianza input
propagata. In corso: run completo.


## KF/UKF/EnKF (in corso)

- `tools/experiment-kalman.py`: filtri per-stazione, stato [livello,velocità],
  transizione `v+ = v·(rho+gamma·(y-ymin))` stimata su train; fase A senza
  ingressi, fase B con livelli upstream filtrati + pioggia (varianza input
  propagata via B·Γ·Bᵀ). **Completato**: `kalman.json` — skill modesto
  (valEvt@120 ~0.14–0.27), molto sotto la MLP.
- `tools/experiment-kalman-global.py`: stato congiunto 10-D di tutte le
  stazioni, Q di processo piena con covarianze incrociate; `--couple` aggiunge
  coppiaggio ritardato monte→valle (lag da wave-warp). **Completato** — vedi `kalman-report.md`.


### Sintesi KF/UKF/EnKF

- Per-stazione A (no input): valEvt@120 ~0.14 — debole.
- Con ingressi B (upstream+pioggia, varianza input propagata): ~0.27 — meglio
  ma sempre sotto MLP.
- Globale (stato congiunto, Q incrociata): ~0.148 — neutro.
- Globale accoppiato (lag DTW monte→valle): ~0.165 — piccolo guadagno reale.
- KF≈UKF≈EnKF: non linearità troppo debole per distinguerli.
- Utili come stimatori di stato con incertezza attribuita (R/Q/Γ);
  non competitivi come previsori. Nessuna promozione.
