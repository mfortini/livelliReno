# Fitting offline dei trasferimenti monte→valle

**Sperimentale.** I coefficienti e le predizioni derivate **non** sono previsioni di piena, **non** hanno valore operativo/allertistico e **non** vanno usate per decisioni.

## Ambiente (`uv`)

```bash
uv sync
uv run python tools/fetch-storico.py --months 24
uv run python tools/fit-transfer.py --data ./storico --out models/transfer-coeffs.json
node tools/backtest.js --every 120 --json models/backtest-summary.json
```

Dipendenze: DuckDB, pandas, NumPy (vedi `pyproject.toml`).

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

## Backtest

```bash
node tools/backtest.js --every 120 --json models/backtest-summary.json
```

Campiona ogni 2 h e densifica a 30′ intorno agli eventi. Confronta RMSE modello vs persistenza e vs media mobile 3 h. Aggiorna i flag `reliable` nei coefficienti e segnala drift di skill > 0.05 rispetto al summary precedente.
