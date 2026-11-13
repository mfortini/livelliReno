# Report modelli di previsione livelli — Reno / Cento

**Data snapshot:** 2026-11-13  
**Ambito:** dashboard sperimentale `livelliReno` (Comune di Cento)  
**Disclaimer:** le previsioni **non** sono operative né allertistiche; non sostituiscono bollettini o soglie territoriali ufficiali.

Questo documento riassume (1) cosa gira oggi in pagina Predizioni, (2) cosa è stato provato e scartato, (3) metriche e limiti noti, per revisione da parte di un esperto idrologo / modellista.

---

## 1. Obiettivo del sistema

Prevedere il **livello idrometrico** (m) a orizzonti da 30′ a al più 6 h, a partire da:

- serie ARPAE 30′ (livelli) e pioggia 15′ (pluviometri in `config.json`);
- grafo di dipendenze monte→valle (`config.dependencies`);
- soglie ARPAE solo come riferimento grafico (non come probabilità di rischio).

**Target modellato:** quasi sempre \(\Delta y(t\!\to\!t+h)\); la curva mostrata è \(y(t)+\widehat{\Delta y}\) (con bias/smooth dove previsto).  
**Baseline di confronto:** persistenza \(y(t+h)=y(t)\); in fit anche media mobile 3 h e, per Cento, polinomio locale (poly2) su 2.5 h.

**Metrica di gate:** *event skill* a +2 h  
\(\mathrm{skill}=1-\mathrm{RMSE}_{mod}/\mathrm{RMSE}_{base}\) su campioni con \(|\Delta y|\ge 0.10\,\mathrm{m}\).  
Soglia di affidabilità runtime: \(\mathrm{skill@120}\ge 0.10\) (validate+holdout dove applicabile, poi backtest).  
Orizzonte mostrato in UI: fino a dove lo skill per-orizzonte resta ≥ 0.10 (`maxReliableHorizonMinutes`, cap 12 h).

---

## 2. Cosa è in produzione oggi (browser)

File principale: `models/transfer-coeffs.json` (v4). Soft-load aggiuntivo: `models/seq-{slug}-gru.json` **solo se** `reliable: true` (sostituisce il nodo).

| Stazione | Kind runtime | Affidabile | Orizzonte UI tipico | Note |
|---|---|:---:|---|---|
| **Cento** | `window_delta_mlp` | sì | **fino a 6 h** (knot 30′…360′) | MLP senza accettori valle; rain on; d1/d2 OLS 6 pt |
| **Gallo** | `direct_horizon_delta_ridge` | sì | **fino a 6 h** | Ridge estesa; GRU migliore a 2 h ma soft-adopt OFF (trade-off banda) |
| **Casalecchio chiusa** | ridge in coeffs + **GRU soft-adopt** | sì (GRU) | **2 h** | `seq-casalecchio-chiusa-gru.json` `reliable: true` |
| Bonconvento, Forcelli, Sostegno Reno, … | ridge | **no** | nessuna curva | holdout/backtest sotto soglia o dati mancanti |
| Dosso, Opera Reno Panfilia, Secondo Salto | ridge | no | — | valle di Cento; skill bassi / serie incomplete |

Smooth EMA di default sul path ridge: **0.15**. Cento MLP: smooth tipicamente 0 (nessun EMA pesante).

### 2.1 Cento — MLP `window_delta_mlp` (modello principale)

- **Input:** finestra piatta ~24 h su monte (Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo): livello, Δ, miss; pioggia bacino; derivate d1/d2 da OLS quadratico su 6 punti (2.5 h). **Senza** accettori valle.
- **Architettura:** MLP a pesi JSON valutata **per orizzonte** (spezzata Plotly = nodi indipendenti 30′…360′, non un’unica ODE).
- **Fit (month-block):** skill eventi validate @120′ ≈ **0.82**; @360′ ≈ **0.80** (`cento-dnn-novalle-6h.json` / nodo in `transfer-coeffs`).
- **Backtest runtime** (`backtest-summary.json`, campionamento ogni 2 h): skill@120 ≈ **0.48** vs persistenza, ≈ **0.67** vs MA 3 h — protocollo più severo dello split di fit; interpretare come lower bound operativo.

### 2.2 Altre stazioni — ridge `direct_horizon_delta_ridge`

Per ogni valle e ogni \(h\):

\[
\begin{aligned}
\Delta y(t\!\to\!t+h) \approx\
& b + w_{\mathrm{local}}\Delta y_{\mathrm{local}} \\
&+ \sum_i w_i^{\mathrm{win}}\bigl(x_i(t-\tau_i)-x_i(t-\tau_i-h)\bigr)
 + \sum_i w_i^{\mathrm{rate}}(\ldots) \\
&+ \text{secondo lag }\tau_i+h\text{ (dispersione)} + w_{r6}\mathrm{rain6h}+w_{r24}\mathrm{rain24h}
\end{aligned}
\]

- \(\tau_i\) stimato **solo** sui mesi di train (cross-correlazione).
- Forchetta a due regimi quando un monte supera `z85`.
- Spike filter: salti \(|\Delta|>3\) m / 30′ esclusi dal fit.
- Split: mesi alterni train/validate; ultimi ~3 mesi holdout.

**Gallo (prod):** skill fit @120 ≈ **0.77**, holdout ≈ **0.62** (n_hold≈25, documentato); @360 ≈ **0.58**.

### 2.3 Soft-adopt sequenziale GRU

| File | Stazione | reliable | skill@120 fit | Uso |
|---|---|:---:|---:|---|
| `seq-cento-gru.json` | Cento | **false** | 0.75 (&lt; MLP 0.82) | non carica |
| `seq-gallo-gru.json` | Gallo | **false** | 0.86 (&gt; ridge) | spento di proposito per tenere banda 6 h ridge |
| `seq-casalecchio-chiusa-gru.json` | Casalecchio chiusa | **true** | 0.25 (&gt; ridge 0.14) | soft-adopt ON |

---

## 3. Cosa è stato provato (e perché non è in prod)

### 3.1 Feature engineering Cento (MLP)

| Esperimento | Esito | Decisione |
|---|---|---|
| Accettori valle (Dosso, ORP, Secondo Salto, Gallo) | validate ≤ novalle | non promuovere |
| Yellow headroom / ratio soglia | holdout a tratti meglio, validate misto | non promuovere |
| d1/d2 OLS su 4 / 6 / 12 punti | 6 pt migliore compromesso | **adottato** (poly-points 6) |
| LightGBM | confrontato in fit | skip in recipe prod (`--skip-lgbm`) |

Dettaglio storico: `models/cento-metodi-report.md`, gate `cento-mlp-feature-ab-gate.json`.

### 3.2 Estensione orizzonte a 6 h

| Confronto | @120 | @360 | Decisione |
|---|---:|---:|---|
| Cento MLP@120 vs MLP@6h | 0.826 → 0.820 | — → **0.80** | **adottare MLP@6h** (`ab-mlp120-vs-mlp6h.md`) |
| Cento ridge@6h | 0.67 | 0.42 | non sostituisce MLP (−0.15 @120) |
| Gallo GRU@120 vs ridge@6h | 0.86 vs 0.77 | — vs 0.58 | **ridge@6h** per banda lunga; GRU backup (`ab-gru120-vs-ridge6h.md`) |
| GRU@6h Cento/Gallo | non migliore del 120-only | — | scartati (`seq-*-gru.6h-discarded.json`) |

### 3.3 Allineamento lag upstream (`--lag-align`)

Ipotesi: la curva “insegue” perché la finestra MLP non è allineata al tempo di propagazione \(\tau_i\).

| Variante | skill@120 backtest | phase lag@120 | Esito |
|---|---:|---:|---|
| MLP prod (flat) | 0.690 | ~30′ | baseline |
| lag **shift** | 0.631 | ~30′ | peggiore |
| lag **augment** | 0.697 | ~30′ | ~pari |
| phase relabel −30′ | 0.698 | — | guadagno minimo, non adottato |

**Conclusione (`ab-lagalign.md`):** il ritardo di fase (~30′ @2 h, ≥120′ @6 h) **esiste**, ma **non** si corregge con feature lag-aligned. Cause residue: shrinkage della media condizionata, celerità variabile in piena, (eventuale) smoothing. La finestra piatta 24 h contiene già l’onda in arrivo.

### 3.4 Modelli fisici / ibridi (tutti gated OFF)

| Metodo | skill@120 eventi (ordine) | vs MLP Cento ~0.83 | Adopt |
|---|---:|---|---|
| Seq GRU end-to-end | fit 0.75 / backtest ~0.72 | sotto @120; a tratti meglio highAcc | no (Cento) |
| Seq 1D-CNN | backtest ~0.68 | no | no |
| Hybrid volume + DNN | ~0.38 | no | no |
| Muskingum + GRU | ~0.07 | no | no |
| RC + GRU | ~−0.05 | no | no |
| EnKF mass-transfer | ~−0.08 | no | no |

I pesi restano in repo per ricerca (`hybrid-*.json`, `enkf-cento-summary.json`); soft-load solo se `reliable: true`.

### 3.5 Baseline stupida poly2 (solo Cento)

Sweep lookback: massimo skill vs persistenza a +2 h con **6 punti / 2.5 h** (≈0.45). Finestre più lunghe peggiorano il predittore → non confrontare `holdPoly2` tra finestre diverse; per A/B finestra usare `holdPersist`.

---

## 4. Protocollo dati e valutazione (da conoscere per leggere i numeri)

**Diagnosi divario fit/backtest (Fase 1):** [`models/fase1-diagnosi.md`](fase1-diagnosi.md) — il gap 0.82→0.67 non è un bug JS; usare la scala protocollo per i gate.

**Fonte unica per confronti tra modelli (Fase 0):** [`models/eval-table.md`](eval-table.md) / [`models/eval-table.json`](eval-table.json), prodotta da:

```bash
node tools/build-event-catalog.js
node tools/backtest.js --protocol --from 2026-01-01 --seed 1
```

Protocollo: runtime JS (`predict-transfer.js`), campionamento denso 30′, orizzonti 30–360′, eventi da [`models/events.json`](events.json) (non da `|Δy|≥0.10`), baseline persist / poly2(6) / ridge, skill all + in-evento, errore picco, bootstrap CI 90% per evento. Test congelato: **2026-01-01 → fine storico**. I numeri di fit Python sotto **non** sono più la scala di confronto.

Pipeline storica (ancora usata per training / gate `reliable` legacy):

1. **Storico:** `tools/fetch-storico.py` → `./storico` (livelli B13215, pioggia B13011).
2. **Fit ridge:** `fit-transfer.py` → `transfer-coeffs.json`.
3. **Fit Cento MLP:** `fit-ml-cento.py --no-acceptors --skip-lgbm --poly-points 6 --horizons 1…12 --export-coeffs`.
4. **Fit seq:** `fit-seq-cento.py` (export separato, non patcha coeffs).
5. **Backtest legacy:** `node tools/backtest.js --every 120` aggiorna `reliable` se skill eventi ≥ 0.10;  
   `--compare-candidates Cento` **non** promuove modelli sperimentali.
6. Caveat Fase 0: i pesi fitted con mesi alterni possono sovrapporsi al test → skill assoluti potenzialmente ottimistici; usare la tabella per ranking, non mescolare con skill di fit.

---

## 5. Limiti noti (per l’esperto)

- **Non idraulico:** nessuna conservazione di volume; Muskingum/RC sono surrogati su livelli (zeri ARPAE non comuni).
- **Orizzonte fisso, non tempo di picco:** non si prevede il crossing time; solo livello a \(t+h\).
- **Shrinkage:** i fronti di piena tendono a essere sottostimati (media condizionata).
- **Fase:** ritardo tipico ~30′ @2 h sulla MLP Cento; non risolto da lag features.
- **Spezzata MLP:** orizzonti indipendenti → kink possibili tra knot (non dinamica continua).
- **Stazioni valle / incomplete:** molte `reliable=false` o fetch falliti in backtest → nessuna curva in UI (corretto, non un bug di plot).
- **Gate binario:** sotto skill 0.10 la curva sparisce; non c’è downgrade graduale di confidenza oltre lo spread/forchetta.

---

## 6. Domande aperte su cui chiedere parere

1. È accettabile pubblicare una banda a 6 h su Cento con skill fit ~0.80 ma backtest @120 ~0.48, purché esplicitamente non allertistica?
2. Per Gallo, preferire **skill @2 h** (GRU 0.86, banda corta) o **banda 6 h** (ridge, skill @2 h 0.77) come ora?
3. Vale la pena un modello di **tempo di attraversamento** (quantili del lag di picco) invece di sole regressioni a orizzonte fisso?
4. \(\tau\) **regime-dipendente** (due celerità su z85 / portata) merita un nuovo A/B, dato che lo shift fisso ha fallito?
5. Soft-adopt GRU su Casalecchio chiusa (skill assoluto basso ~0.25 ma meglio del ridge): tenere o alzare l’asticella assoluta oltre il confronto relativo?

---

## 7. Indice file utili

| File | Contenuto |
|---|---|
| `models/transfer-coeffs.json` | Coefficienti runtime (prod) |
| `models/cento-dnn-novalle-6h.json` | Pesi MLP Cento 6 h |
| `models/ab-mlp120-vs-mlp6h.md` | A/B orizzonte Cento |
| `models/ab-gru120-vs-ridge6h.md` | A/B GRU vs ridge / Gallo |
| `models/ab-lagalign.md` | A/B lag features + phase lag |
| `models/cento-metodi-report.md` | Report metodi precedente (Cento) |
| `models/backtest-summary.json` | Skill backtest runtime (legacy) |
| `models/eval-table.json` / `.md` | **Tabella unica Fase 0** (skill + CI) |
| `models/events.json` | Catalogo piene per valutazione |
| `models/cento-candidates-backtest.json` | Confronti candidati (no promote) |
| `tools/README.md` | Comandi di fit / backtest / protocol |

---

*Generato come snapshot del repository; allineare eventuali decisioni successive aggiornando questo file e gli A/B in `models/`.*
