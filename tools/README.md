# Fitting offline dei trasferimenti monte→valle

**Sperimentale.** I coefficienti e le predizioni derivate **non** sono previsioni di piena, **non** hanno valore operativo/allertistico e **non** vanno usate per decisioni.

## Ambiente (`uv`)

```bash
uv sync
uv run python tools/fetch-storico.py --months 24
uv run python tools/fit-transfer.py --data ./storico --out models/transfer-coeffs.json
uv run python tools/fit-ml-cento.py --data ./storico --skip-lgbm \
  --export-dnn models/cento-dnn.json --export-coeffs models/transfer-coeffs.json
node tools/backtest.js --every 120 --json models/backtest-summary.json
```

### Fase 0 — protocollo di valutazione unico

Un solo comando produce la tabella skill per stazione/orizzonte/modello (runtime JS, campionamento 30′, eventi da catalogo, baseline persist / poly2(6) / ridge, bootstrap CI 90%). **Non** aggiorna `reliable`.

```bash
node tools/build-event-catalog.js --data ./storico --out models/events.json
node tools/backtest.js --protocol --data ./storico \
  --from 2026-01-01 --events models/events.json --seed 1 \
  --json models/eval-table.json --md models/eval-table.md
```

Opzionale: `--stations Cento,Gallo` per un sottoinsieme. Due run consecutive con lo stesso `--seed` devono dare gli stessi numeri. I pesi attuali possono aver visto mesi del test via split a mesi alterni → numeri assoluti ottimistici; ranking omogeneo sì (retrain pulito = Fase 1).

Dipendenze: DuckDB, pandas, NumPy, LightGBM, PyTorch CPU (vedi `pyproject.toml`).
Cento in produzione browser: `kind: window_delta_mlp` (pesi JSON) dopo `fit-ml-cento.py --export-coeffs`.

Accettori valle (Dosso, Opera Reno Panfilia, Secondo Salto, Gallo) come feature backwater:

```bash
uv run python tools/fit-ml-cento.py --skip-lgbm                 # → cento-dnn-valle.json
uv run python tools/fit-ml-cento.py --skip-lgbm --no-acceptors # ablazione
uv run python tools/fit-hybrid-rc-cento.py --data ./storico    # RC+GRU; soft-adopt solo se reliable
node tools/backtest.js --compare-candidates Cento --candidates-json models/cento-candidates-backtest.json
```

Gate corrente: MLP+valle e RC **non** battono la MLP senza valle → non promuovere (`models/cento-valle-gate.json`).

## Sequenza end-to-end (GRU / 1D-CNN, offline)

Stesso split/eventi della MLP novalle; predice Δy su `[N, L=48, C]`. Default Cento; altre stazioni con `--downstream NAME` (upstream da `config.dependencies`). **Non** patcha `transfer-coeffs` — export `models/seq-{slug}-gru.json`. Soft-load in `loadCoeffs` se `reliable: true`.

```bash
uv run python -u tools/fit-seq-cento.py --data ./storico
# → models/seq-cento-compare.json + seq-cento-gru.json + seq-cento-cnn.json
uv run python -u tools/fit-seq-cento.py --downstream Gallo --arch gru --data ./storico
uv run python -u tools/fit-seq-cento.py --arch gru --horizons 4   # smoke @120′
node tools/backtest.js --compare-candidates Cento --every 120 \
  --candidates-json models/cento-candidates-backtest.json
```

Soft-load browser: `models/seq-cento-gru.json` con `reliable: true` (preferito su MLP Cento in `loadCoeffs`). CNN resta offline finché non batte la prod.

### Spezzata sulla curva Cento (DNN)

La spezzata rossa non è dinamica idraulica: `forecastWindowMlp` in `predict-transfer.js` valuta quattro MLP indipendenti a 30/60/90/120′ sulla stessa finestra e Plotly unisce i nodi con segmenti. Non c’è vincolo di tasso tra orizzonti (il path affine legacy aveva uno smooth EMA; il path MLP usa solo `maxStep` + smooth leggero). Densificare i passi o un filtro di stato (EnKF) evitano il kink per costruzione.

## Spike EnKF Cento (offline)

Modello a serbatoi su **anomalie** locali (i livelli ARPAE non condividono lo stesso zero) Forcelli + Sostegno → Cento, con pioggia bacino come forcing. Parametri (`A`, `C_*`, `alpha`) stimati offline; online solo lo stato anomalia Cento. Output `models/enkf-cento-summary.json` con gate `adoptRuntime` vs skill DNN@120. Non è wired in dashboard se il gate fallisce (resta lo smooth EMA sul path MLP).

```bash
uv run python tools/fit-enkf-cento.py --data ./storico --out models/enkf-cento-summary.json
```

## Hybrid volume + DNN Cento (offline)

Livelli z-score (μ/σ train-only per stazione) → DNN piccola predice θ del modello a volumi → rollout Euler denso. Metriche in metri dopo denorm. Gate vs MLP `skill120`; wiring browser solo se vince.

```bash
uv run python tools/fit-hybrid-cento.py --data ./storico --out models/hybrid-cento-summary.json
uv run python tools/fit-hybrid-cento.py --self-test
```

Esito corrente (2026-11-12): event skill@120 ≈ **0.38** vs DNN ≈ **0.83** → `adoptRuntime: false`, resta `window_delta_mlp` + smooth. Pesi tiny in `models/hybrid-cento.json` solo per ricerca.

## Hybrid Muskingum + GRU Cento (browser-ready)

Routing Muskingum multi-reach + GRU (hidden 32, lookback 24 h) che predice K/X/guadagni. Export ~100 KB. Soft-load da `models/hybrid-muskingum-cento.json` in `loadCoeffs` **solo se** `reliable: true`. Surrogato su livelli (non conserva volumi); decode K/X vincolato a `2·K·X ≤ dt ≤ 2·K·(1−X)`, warm-up sul lookback, spread da MAD residui. Metriche validate e holdout separate.

```bash
uv run python tools/inventory-storico-cento.py --out models/storico-cento-inventory.json
uv run python tools/fit-hybrid-muskingum-cento.py --data ./storico \
  --train-cap 12000 --train-event-keep 8000 \
  --out models/hybrid-muskingum-cento-summary.json \
  --weights models/hybrid-muskingum-cento.json
# Confronto full-train (nessun sottocampionamento):
uv run python tools/fit-hybrid-muskingum-cento.py --data ./storico --train-cap 0 \
  --out models/hybrid-muskingum-cento-summary-full.json \
  --weights models/hybrid-muskingum-cento-full.json
```

Esito corrente (pesi legacy): event skill@120 ≈ **0.07** vs DNN ≈ **0.83** → `adoptRuntime: false`; in pagina resta la MLP. Non rigenerare/promuovere i pesi finché holdout e `--compare-candidates` non battono la MLP.

Confronto sperimentale omogeneo (non scrive `reliable`):

```bash
node tools/backtest.js --compare-candidates Cento --every 120 \
  --candidates-json models/cento-candidates-backtest.json
```

## Scarico storico ARPAE (open data)

```bash
uv run python tools/fetch-storico.py --months 24
uv run python tools/fetch-storico.py --from 2024-01 --to 2026-11
```

Variabili: **B13215** (livello) e **B13011** 15′ (pioggia, pluviometri in `config.rainfall.gauges`).
CSV livelli in `./storico`, pioggia in `./storico/rain`, cache in `./storico/raw` (gitignored).

## Modello `direct_horizon_delta_ridge` (v4)

Per ogni valle e ogni orizzonte \(h\) (30–120′; fino a 240′ se il lag massimo ≥ 360′ e lo skill regge):

\[
\begin{aligned}
\Delta y(t\!\to\!t+h) \approx\
& b + w_{\mathrm{local}}\Delta y_{\mathrm{local}} \\
&+ \sum_i w_i^{\mathrm{win}}\bigl(x_i(t-\tau_i)-x_i(t-\tau_i-h)\bigr)
 + \sum_i w_i^{\mathrm{rate}}\bigl(x_i(t-\tau_i)-x_i(t-\tau_i-\Delta)\bigr) \\
&+ \sum_i w_i^{\mathrm{win2}}\bigl(x_i(t-\tau_i-h)-x_i(t-\tau_i-2h)\bigr)
 + \sum_i w_i^{\mathrm{rate2}}\bigl(x_i(t-\tau_i-h)-x_i(t-\tau_i-h-\Delta)\bigr) \\
&+ w_{r6}\,\mathrm{rain6h} + w_{r24}\,\mathrm{rain24h}
\end{aligned}
\]

Previsione: \(\hat y(t+h)=y(t)+\widehat{\Delta y}-\mathrm{biasEMA}_{48h}\). La persistenza è il caso a pesi nulli.

- **Lag** \(\tau_i\) stimato solo sui mesi di train (niente leakage)
- **Secondo lag** \(\tau_i+h\) per dispersione dell’onda
- **Spike filter**: salti \(|\Delta|>3\) m / 30′ annullati prima del fit
- **Split**: mesi alterni train/validate; ultimi 3 mesi holdout
- **Forchetta a due regimi**: spread base vs high quando un monte supera `z85`
- **Affidabilità**: dopo il backtest, `reliable=true` solo se skill@120′ eventi \(\ge 0.10\) (anche vs media mobile 3 h nel report)

Output `models/transfer-coeffs.json`. La dashboard soft-load il file.

### Lag-aligned upstream features (esperimento, non adottato)

`--lag-align {shift,augment}` stima τᵢ sui mesi di train e ri-indicizza le
finestre upstream sull'onda in arrivo x(t+h−τᵢ). Esito negativo/marginale:
vedi `models/ab-lagalign.md`. Il runtime legge `featureRecipe.upstreamLag*`.

### Loss varianti (timing / eventi)

`--quantile 0.75` allena la DNN con pinball loss sul quantile: la curva
anticipa i fronti di piena (proxy di crossing-time) ma è sbilanciata in alto
in calma — candidato separato `mlp_q75`, non sostituisce la mediana.
`--event-weight N` pesa i campioni |Δy| ≥ 0.10 m ×N nella loss di train:
attacca la sottostima sui fronti senza cambiare la semantica del punto.

## Backtest

```bash
node tools/backtest.js --every 120 --json models/backtest-summary.json
```

Campiona ogni 2 h e densifica a 30′ intorno agli eventi. Confronta RMSE modello vs persistenza e vs media mobile 3 h. Aggiorna i flag `reliable` nei coefficienti e segnala drift di skill > 0.05 rispetto al summary precedente.
