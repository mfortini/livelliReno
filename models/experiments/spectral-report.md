# Esperimento: feature spettrali (FFT + wavelet) — Cento

_Data: 2026-11-15. Esito: **negativo, nessuna promozione.**_

## Ipotesi

Le oscillazioni di piena (onde di piena, smorzi, onde secondarie) non sono
descritte bene dai soli livelli/derivate punto-punto. Feature spettrali
causali sullo stesso passato potrebbero dare al modello una vista sulle
frequenze dominanti del segnale.

## Feature aggiunte (per stazione, 28 feature)

Finestra causale delle ultime `W` misure della griglia di input
(`W = lookback = 48` a passo 30′ → 24 h; per il 5′ sarebbe 288):

- `spec_*_cov`: frazione di dati validi nella finestra;
- `spec_*_fft1..12`: `log1p(|X_j|/W)` della DFT diretta, bin j=1..12
  (periodi ~24h→2h a passo 30′), su segnale detrendato (media rimossa) e
  finestra Hann;
- `spec_*_centroid`, `spec_*_dombin`, `spec_*_band{lo,mid,hi}`: centroide,
  bin dominante e frazioni di potenza nelle bande j≤3 / 4–8 / ≥9;
- `spec_*_wav{k}rms`, `spec_*_wav{k}last` (k=1..5): scomposizione Haar
  à trous (media mobile a spirale di 2^k), RMS e ultimo dettaglio per scala.

Riempimento NaN per interpolazione lineare sugli indici (estremi = ultimo
valido); finestre con <2 punti validi → zeri + copertura bassa. Tutto
calcolato solo sul passato disponibile all'emissione (causale).

## Implementazione

- `tools/spectral_features.py` — reference Python (numpy);
- `temporal-sampling.js` — `spectralFeatures()` / `spectralFeatureNames()`,
  parità numerica JS↔Py verificata (<1e-9 sullo stesso input con gap);
- `predict-transfer.js` — `buildWindowFeatureVector` emette le feature se
  `featureRecipe.spectral` è presente (i modelli esistenti non lo dichiarano
  → comportamento invariato);
- `tools/experiment-temporal.py --spectral {fft,wavelet,both}` — riusa le
  colonne `level_*` già costruite (costo ~0) e la cache feature.

Test in `tests/temporal-sampling.test.js`: lunghezza/nomi, bin dominante su
senoide a periodo noto, gap NaN, finestra costante e corta, causalità
(dati futuri non modificano la finestra), parità dei nomi per ogni
combinazione fftBins/waveletScales.

## Risultati

Training: `--method linear --input-step-minutes 30 --lookback-minutes 1440`,
`--spectral both` → 1342 feature (1202 base + 140 spettrali), orizzonti
30–360′, stessi split/origini degli esperimenti precedenti.

Holdout evento del trainer (skill vs poly2):

| h | valEvt | holdEvt |
|---|---:|---:|
| 120′ | 0.674 | 0.242 |
| 240′ | 0.650 | 0.650 |
| 360′ | 0.617 | 0.731 |

Backtest A/B gen–mar 2026 (`models/cento-candidates-backtest.json`,
n=2041, step 120′):

| Candidato | skill@120′ | skill@240′ | skill@360′ | rmse@120′ |
|---|---:|---:|---:|---:|
| **mlp_prod** | **0.769** | **0.682** | **0.612** | **0.0472** |
| mlp_temporal_lin (5′) | 0.655 | 0.614 | 0.543 | 0.0707 |
| mlp_temporal_pchip (5′) | 0.576 | 0.570 | 0.534 | 0.0867 |
| **mlp_spectral** | 0.611 | 0.558 | 0.481 | 0.0795 |

## Conclusione

Le feature spettrali **peggiorano** il modello: skill@120 0.611 vs 0.769 del
baseline produttivo, e anche peggio del ricampionamento 5′ lineare. Holdout
evento debole a 120′ (0.242). Ipotesi probabile: con 48 campioni e DFT a 12
bin il contenuto spettrale è rumoroso e i LEVEL/derivate locali portano già
l'informazione utile; le feature aggiuntive aumentano la varianza senza
segnale nuovo. Coerente con gli esiti negativi di interpolazione e DTW:
la MLP attuale è già vicina al limite informativo del segnale disponibile.

Nessuna promozione. Il candidato resta solo nel backtest
(`--mlp-spectral`, default `models/experiments/cento-spectral-30min-dnn.json`).

## Riproduzione

```bash
uv run python tools/experiment-temporal.py --method linear \
  --input-step-minutes 30 --lookback-minutes 1440 --stride 12 \
  --spectral both --feat-cache models/experiments/feat-30min-lin.npz \
  --out models/experiments/cento-spectral-30min.json \
  --tag cento-spectral-30min
node tools/backtest.js --compare-candidates Cento --from 2026-01-01 \
  --to 2026-04-01 --every 120 --json models/cento-candidates-backtest.json
```
