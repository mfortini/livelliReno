# Report completo — esperimenti di previsione livelli Reno / Cento

**Data:** 2026-11-16  
**Ambito:** dashboard sperimentale `livelliReno` (Comune di Cento)  
**Disclaimer:** previsioni **non** operative né allertistiche; non sostituiscono bollettini o soglie territoriali ufficiali.

Questo documento consolida tutto ciò che è stato provato sul problema della previsione del livello idrometrico a Cento (e, dove rilevante, Gallo e altre stazioni), con spiegazione delle tecniche e dei risultati. Fonti: A/B in `models/ab-*.md`, fasi 0–3, `research_plan_reno.md` / `research_plan_final.md`, `report-esperto-modelli.md`, `cento-metodi-report.md`, `wp1-diagnosis.md`.

---

## 0. Verdetto in una frase

Nessuna variante ha battuto in modo chiaro e promuovibile la **MLP di produzione** (`window_delta_mlp`, monte senza valle, orizzonti 30′–360′) sul rising limb / skill eventi @120′. Il ritardo di fase (~30′ @2 h, ≥120′ @6 h) è un **pavimento strutturale** della formulazione a orizzonte fisso Δy, non un bug di feature o di runtime.

| Voce | Valore |
|---|---|
| Soft-adopt / patch `transfer-coeffs` (cicli recenti) | **no** |
| `reliable` | invariato |
| Modello prod Cento | **window_delta_mlp** — Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo |
| Gallo | ridge @6 h (GRU soft-adopt OFF) |
| Banda Cento q10/q90 | soft-merge da `cento-dnn-bands.json` (mediana prod invariata) |

---

## 1. Problema e formulazione di base

### 1.1 Obiettivo

Prevedere il **livello** (m) a orizzonti da 30′ fino a 6 h (cap UI 12 h) a partire da:

- serie ARPAE 30′ (livelli) e pioggia 15′;
- grafo di dipendenze monte→valle;
- soglie ARPAE solo come riferimento grafico.

**Target modellato:** quasi sempre \(\Delta y(t\to t+h)\); la curva mostrata è \(y(t)+\widehat{\Delta y}\).  
**Baseline:** persistenza \(y(t+h)=y(t)\); in fit anche media mobile 3 h e poly2 locale (6 punti / 2.5 h).

**Gate di affidabilità:** event skill @120′  
\(\mathrm{skill}=1-\mathrm{RMSE}_{mod}/\mathrm{RMSE}_{base}\) su campioni con \(|\Delta y|\ge 0.10\,\mathrm{m}\) (fit) o su eventi di catalogo (protocollo Fase 0).  
Soglia runtime: skill@120 ≥ 0.10.

### 1.2 Sintomo centrale

L’errore è concentrato sul **rising limb**:

| Slice | rising RMSE @120′ | falling RMSE @120′ | Rapporto |
|---|---:|---:|---:|
| Test 2026 eventi / prod | ~0.10–0.11 | ~0.02 | ≈ 5–6× |
| Evento Nov 2026 / prod | 0.175 | 0.019 | ≈ 9× |

Phase lag tipico MLP Cento: **~26–30′ @2 h**, **~120′ @6 h** (metrica satura a ±120′).

### 1.3 Perché la MSE “ritarda”

Interpretazione consolidata (M9 + WP1):

1. Un modello addestrato su errore quadratico stima la **media condizionata**.
2. Se il tempo di arrivo del fronte è incerto, la media di molti fronti possibili è un fronte **lento e ritardato**.
3. L’RMSE **punisce due volte** un fronte acuto con piccolo errore di timing; un fronte liscio e tardi può avere RMSE migliore pur essendo fisicamente peggiore.

Quindi: spostare davvero la fase richiede di modellare il **timing** nella traiettoria (o cambiare target), non solo aggiungere feature di lag.

---

## 2. Cosa c’è in produzione oggi

| Stazione | Kind | Affidabile | Orizzonte UI | Note |
|---|---|:---:|---|---|
| **Cento** | `window_delta_mlp` | sì | **fino a 6 h** | MLP senza accettori valle; rain on; d1/d2 OLS 6 pt |
| **Gallo** | `direct_horizon_delta_ridge` | sì | **fino a 6 h** | GRU migliore a 2 h ma soft-adopt OFF (trade-off banda) |
| **Casalecchio chiusa** | ridge + **GRU soft-adopt** | sì (GRU) | **2 h** | skill assoluto basso ma > ridge |
| Altre monte / valle | ridge | spesso **no** | nessuna curva | sotto soglia o dati incompleti |

Smooth EMA default sul path ridge: **0.15**. Cento MLP: smooth tipicamente 0.

### 2.1 MLP Cento — tecnica

- Input: finestra piatta ~24 h su 4 monte (livello, Δ, miss), pioggia bacino, derivate d1/d2 da OLS quadratico su 6 punti.
- Architettura: MLP a pesi JSON **per orizzonte** (nodi indipendenti 30′…360′, non un’ODE).
- Fit month-block: skill eventi validate @120′ ≈ **0.82**; @360′ ≈ **0.80**.
- Protocollo Fase 0 (test ≥ 2026-01-01): skill evt ≈ **0.67–0.68** — scala più severa e out-of-sample; usare questa per i confronti.

### 2.2 Ridge — tecnica

Per ogni valle e ogni \(h\):

\[
\Delta y \approx b + w_{\mathrm{local}}\Delta y_{\mathrm{local}}
+ \sum_i w_i^{\mathrm{win}}\bigl(x_i(t-\tau_i)-x_i(t-\tau_i-h)\bigr)
+ \text{rate} + \text{secondo lag} + \text{rain}
\]

- \(\tau_i\) stimato solo sui mesi di train (cross-correlazione).
- Forchetta a due regimi sopra `z85`.
- Split: mesi alterni train/validate; ultimi ~3 mesi holdout (poi anche `--holdout-from 2026-01-01` leakage-free).

---

## 3. Protocollo di valutazione (come leggere i numeri)

### Fase 0 — tabella unica

```bash
node tools/build-event-catalog.js
node tools/backtest.js --protocol --from 2026-01-01 --seed 1
```

- Runtime JS reale (`predict-transfer.js`)
- Eventi da catalogo (non solo `|Δy|≥0.10`)
- Baseline persist / poly2(6) / ridge / ARX
- Skill all + in-evento, phase lag (anche sub-step), rising/falling RMSE, CSI soglie, bootstrap CI 90%
- Test congelato: **2026-01-01 → fine storico**

**Non mescolare** skill di fit Python (mesi alterni, spesso ottimistici) con skill di protocollo.

### Fase 1 — divario fit 0.82 vs protocollo 0.67

| Causa | Esito |
|---|---|
| Parità Python ≠ JS | **smentita** (max\|Δ\| &lt; 1e-5) |
| Campionamento every-120 vs 30 vs catalogo | **nullo** sul test congelato |
| Split alt vs block | lieve (~0.02–0.03) |
| Early-stop / selezione su validate | ~0.05 di ottimismo sul numero di fit |
| Overlap test ⊂ train (mesi alt) | **residuo strutturale principale** |

Conclusione: il gap non è un bug di runtime; la scala operativa è il protocollo Fase 0.

Stressor: publish lag 90′ → skill 0.35; drop Forcelli → 0.62.

---

## 4. Linea temporale degli esperimenti

```
Feature MLP / seq / ibridi fisici (pre Nov)
    → Estensione 6 h (MLP Cento, ridge Gallo)          [ADOTTATO]
    → Lag-align + timing-loss (ew4, q75)               [no]
    → Fase 0–1 protocollo e diagnosi gap               [fatto]
    → Fase 2 M1–M9 (quantili, arch, blend, peak, τ, …) [quasi tutto no]
    → Fase 3 banda + gate UX                           [parziale]
    → Research plan WP0–WP6 (hazard, routing Q, …)     [no adopt]
    → Post-WP6: lookback-τ, φ/τ, feature-delta         [no adopt]
```

---

## 5. Feature engineering Cento (MLP)

### Tecnica

Ablazione sistematica di input: accettori valle, yellow headroom / ratio soglia, derivate OLS su finestre 4/6/12 punti, LightGBM.

### Risultati (skill@120 eventi, scala fit)

| Variante | valEvt | hold persist | Decisione |
|---|---:|---:|---|
| MLP novalle (baseline) | 0.826 | 0.772 | base storica |
| + valle raw | 0.802 | 0.766 | no |
| + yellow+ratio | 0.824 | 0.840 | holdout su, validate misto → no |
| novalle + d1/d2 n=4 | 0.831 | 0.736 | |
| novalle + d1/d2 n=6 | 0.820 | 0.726 | **poly-points 6 adottato** (compromesso) |
| novalle + d1/d2 n=12 | 0.820 | 0.778 | baseline poly2 più debole a n=12 |
| LightGBM | confrontato | — | skip in recipe prod |

**Poly2 come predittore stupido:** skill vs persistenza massima a **6 punti / 2.5 h** (~0.45); finestre più lunghe peggiorano.

**Lezione:** gli accettori valle non aiutano (e peggiorano); le derivate locali su 2.5 h sì; non promuovere feature che migliorano solo lo holdout a scapito del validate.

Dettaglio: `cento-metodi-report.md`.

---

## 6. Estensione orizzonte a 6 h

### Tecnica

Addestrare knot aggiuntivi 150′…360′ (stessa architettura MLP / ridge), decidere soft-adopt GRU vs banda lunga.

### Risultati

| Confronto | @120 | @360 | Decisione |
|---|---:|---:|---|
| Cento MLP@120 → MLP@6h | 0.826 → 0.820 | — → **0.80** | **adottare MLP@6h** |
| Cento ridge@6h | 0.67 | 0.42 | non sostituisce MLP (−0.15 @120) |
| Gallo GRU@120 vs ridge@6h | 0.86 vs 0.77 | — vs 0.58 | **ridge@6h** (banda lunga); GRU OFF |
| GRU@6h Cento/Gallo | non migliore del 120-only | — | scartati |

**Lezione:** su Cento la MLP scala bene a 6 h; su Gallo si preferisce la copertura temporale della ridge allo skill corto della GRU.

File: `ab-mlp120-vs-mlp6h.md`, `ab-gru120-vs-ridge6h.md`.

---

## 7. Allineamento lag upstream (`--lag-align`)

### Tecnica / ipotesi

La MLP “insegue” perché la finestra piatta non è allineata al tempo di propagazione \(\tau_i\).

Varianti:

- **shift:** finestra monte centrata su \(t+h-\tau_i\)
- **augment:** finestra piatta + blocco di 16 step attorno a \(t+h-\tau_i\)
- **phase relabel −30′:** target spostato di −30′ (diagnostica)
- **Loss:** `--quantile 0.75` (pinball alto), `--event-weight 4` (eventi ×4)

τ train tipici: Forcelli ~480–510′, Sostegno ~150–180′, Bonconvento ~360–390′, Casalecchio tiro ~660′.

### Risultati backtest

| Candidato | skill@120 | phaseLag@120 |
|---|---:|---:|
| mlp_prod (flat) | 0.690 | ~30′ |
| lag **shift** | 0.631 | ~30′ |
| lag **augment** | 0.697 | ~30′ |
| phase30 | 0.698 | — |
| mlp_ew4 | 0.699 | 30′ |
| mlp_q75 | 0.669 | 30′ |

### Decisione: **no adopt**

**Lezione chiave:** il ritardo **esiste** ed è misurato, ma **non** si corregge allineando le feature a τ. La finestra 24 h contiene già l’onda. Cause residue: shrinkage della media condizionata, celerità variabile, formulazione Δy a orizzonte fisso. ew4 migliora RMSE marginale ma non il lag → pavimento strutturale.

File: `ab-lagalign.md`.

---

## 8. Architetture sequenziali e ibridi fisici

### Tecniche

| Metodo | Idea |
|---|---|
| **Seq GRU / 1D-CNN** | Sequenza [N, L=48, C] → Δy multi-orizzonte end-to-end |
| **Muskingum + GRU** | Routing Muskingum su livelli + correzione GRU |
| **RC + GRU** | Rete RC (equivalente elettrico) + GRU |
| **Hybrid volume + DNN** | Bilancio di volume surrogato + DNN |
| **EnKF mass-transfer** | Filtro di Kalman ensemble su trasferimento di massa |

### Risultati (ordini di grandezza)

| Metodo | skill@120 (scala) | vs MLP ~0.67–0.83 | Adopt |
|---|---:|---|---|
| Seq GRU | fit 0.75 / protocol ~0.64 / backtest ~0.72 | sotto o pari (CI sovrapposti) | no Cento |
| Seq CNN | protocol ~0.67 | pari a MLP @120, senza 6 h | no |
| Hybrid volume | ~0.38 | no | no |
| Muskingum+GRU | ~0.05–0.07 | no | no |
| RC+GRU | negativo | no | no |
| EnKF | negativo / open-loop ~0.03 | no | no |

**M2 (rivalutazione su tabella Fase 0):** MLP resta; CNN pari; GRU non superiore; ibridi sotto soglia.

**Lezione:** i surrogati “fisici” su **livelli** (non portata) falliscono: il livello non è una quantità conservata; l’RC dà diffusione non ritardo di trasporto; il filtro non ripara un process model sbagliato (H1/H2/H4 confermate in WP1). Soft-adopt solo se `reliable: true` (Casalecchio chiusa GRU sì; Cento/Gallo no).

File: `ab-m2-cento-arch.md`, `cento-metodi-report.md`.

---

## 9. Fase 2 — A/B M1…M9 (ciclo Nov 2025–2026)

### M1 — Quantili + conformal

**Tecnica:** pinball q10/q50/q90 + inflate conformal per orizzonte.  
**Risultato:** skillEvt 0.725 vs prod 0.713 (CI sovrapposti); copertura banda ~88% (target ~80%); errore picco **non** migliora.  
**Decisione:** mediana **no**; banda **sì** (soft-load Fase 3).

### M3 — Blend GRU≤2 h + ridge (Gallo)

**Tecnica:** GRU fino ~90′, raccordo lineare 90–150′, ridge oltre.  
**Risultato:** @120 0.721 &lt; GRU 0.754; @360 = ridge. Runtime OK.  
**Decisione:** soft-adopt **no** (criterio stretto); runtime `blend_gru_ridge` pronto ma spento.

### M4 — Tempo di arrivo del picco

**Tecnica:** regressione lag picco Forcelli→Cento da livello/pendenza a issue = peak−120′.  
**Risultato:** med|err| test **139′** vs baseline 165′ (lag vero ~450–600′).  
**Decisione:** **FAIL** — confonde tempo di viaggio della piena con ritardo di fase della curva (~30′).

### M5 — τ regime-dipendente (ridge)

**Tecnica:** τ calm vs high (soglia z85); bootstrap CI.  
**Risultato:** dove i CI non si sovrappongono (Gallo), lo skill **peggiora**; altrove overlap.  
**Decisione:** **no**.

### M6 / M7 — Chiusure motivate

- **M6** Casalecchio: manca pioggia **prevista** (NWP); osservata già usata, insufficiente.
- **M7** Valle / Napoleonico: mancano serie manovre/paratoie; protocollo n=0 su stazioni valle.

### M8 — Correzione online di fase + bias

**Tecnica:** assimilazione causale su residui 12 h: time-shift gated ±60′ + bias EMA; opt-in `onlineCorrection`.  
**Risultato (Nov):** rising@120 peggiora (0.184 vs 0.175); phaseLag invariato 30′; falling peggiora.  
**Decisione:** **no** — post-hoc sulla spezzata a orizzonti indipendenti non ri-fasa in modo stabile.

### M9 — Crossing-time probabilistico + decode

**Tecnica:**

1. Train flood-only (±6 h catalogo)
2. Quantili di τ (tempo di primo crossing di +0.5/+1.0/+1.5 m), pinball, censura 6 h
3. Decode: time-warp alla moda di τ, altrimenti **level-lift** costruttivo
4. Ablazione ± Casalecchio chiusa

**Risultato @120′ test:**

| Modello | RMSE | skill | phaseLag | rising | falling |
|---|---:|---:|---:|---:|---:|
| prod | **0.080** | **0.724** | 30′ | **0.103** | **0.022** |
| cross+chiusa | 0.259 | 0.110 | **0′** | 0.298 | 0.189 |

La fase **migliora** (direzione corretta) ma rising/skill **crollano** (overshoot del lift).  
WP1: 78% dei decode su campioni **senza** crossing osservato (H10).  
**Decisione:** adopt **false**. Serve generare la traiettoria dal timing, non post-processare la mediana MSE.

File: `ab-m1`…`ab-m9`, `fase2-sintesi.md`.

---

## 10. Fase 3 — Integrazione UI (senza cambio mediana)

| Voce | Esito |
|---|---|
| Banda q10/q90 + conformal | soft-merge `cento-dnn-bands.json` |
| Gate isteresi enter 0.10 / exit 0.05 | `skill-gates.json` |
| Coda oltre orizzonte affidabile | sfumata / allargata |
| Monte mancante / stale | nota + banda ×1.5 |
| Blend Gallo / crossing | runtime presente, soft-adopt OFF |

---

## 11. Research plan WP0–WP6 (ciclo fisico / hazard)

Piano: `research_plan_reno.md` rev 4. Domanda guida:

> Un modello con struttura fisica può generare la traiettoria di piena col timing corretto e battere la MLP?

### WP0 — Protocollo

Aggiunti ARX, CSI, crossing-time error, pinball, rising/falling, phase lag sub-step, per-event, year folds. Gate: tabella unica con CI.

### WP1 — Diagnosi ipotesi

| ID | Stato | Sintesi |
|---|---|---|
| H1 | **confirmed** | Errore dei filtri dal process model, non dal filtro |
| H2 | **confirmed** | Livello non conservato; RC/EnKF su livelli senza mass balance |
| H3 | not clear | Celerità ↑ con Q non chiara dai lag per banda |
| H4 | **confirmed** | RC = diffusione, non ritardo di trasporto |
| H5 | not clear | Afflussi laterali non espliciti |
| H6 | **confirmed** | Innovazioni flood/calm ≈ 11×; ensemble collapse |
| H7 | **confirmed** | EnKF 1-step vs MLP multi-horizon: confronto iniquo |
| H8 | not clear | Loss assoluta/mediana non riduce chiaramente il lag |
| H9 | **confirmed** | RMSE nasconde valore del timing corretto |
| H10 | **confirmed** | Overshoot M9 da censoring + decode senza gate |
| H11 | not clear | Backwater Dosso: correlazione residui debole |

### WP2 — Data-driven migliorato

| Esperimenti | Esito |
|---|---|
| Hazard discreto P(crossing) + gate decode | rising 0.109 vs prod **0.103** — **no adopt** (decode 35 vs 193 di M9: gate funziona, ma non batte) |
| Weir law su Casalecchio chiusa \(Q=c(h-h_c)^{1.5}\) | valEvt 0.819 vs baseline 0.834 — **no** |
| Extra upstream (Bagnetto, far monte, tributari) | Bagnetto **pari**; valle/Dosso **peggiora** (rising 0.124–0.136) |
| ew4 / lag-align / rain | già chiusi senza promote |

### WP3 — Routing in portata Q

**Tecnica:** rating \(Q=a(h-h_0)^b\) + cascata di serbatoi non lineari / Muskingum-Cunge, train multi-step.  
**Gate:** open-loop deve battere ARX su piene.  
**Risultato:** skill eventi **−10.2** vs ARX **0.64** — **gate FAIL**.

### WP4 — EnKF sul modello WP3

**Saltato** (regola: se WP3 ≤ ARX, stop filtro).

### WP5 — Ibridi

ARX+routing ≈ ARX (0.641); residual correction peggiore → **stop early**.

### WP6 — Decisione

Soft-adopt **no**; tenere MLP prod. Prossimo ciclo utile solo con **rating Q(h) ufficiale** e routing open-loop che batta ARX.

File: `research_plan_final.md`, `wp1-diagnosis.md`.

---

## 12. Ciclo post-WP6 (2026-11-16) — finestre e segnali

Tutti con split **leakage-free** `--holdout-from 2026-01-01`, protocollo seed 1, n=10 eventi test @120′.

### 12.1 Lookback-τ

**Idea:** accorciare la finestra monte a \(L_i=\tau_i+4\,\mathrm{h}\) (cap 48 step), senza shift di fase (diverso da lag-align).

| Modello | skill evt | rising RMSE | phaseLag′ |
|---|---:|---:|---:|
| prod | 0.680 | 0.114 | 25.8 |
| leakfree_flat | 0.682 | 0.114 | 26.0 |
| leakfree_lbtau | 0.685 | 0.113 | 25.6 |

CI sovrapposti → **no adopt**. Accorciare la finestra non aiuta: il rumore remoto non era il collo di bottiglia.

### 12.2 Griglia di fase φ = lag/τ (fino a 2τ)

**Idea:** campionare i monte su griglia di fase \(\phi=k/n\), \(\phi_{\max}=2\), non ogni 30′ su 24 h. Ablazione densità n=8 vs 16; variante delta-only.

| Modello | skill | rising | phaseLag′ |
|---|---:|---:|---:|
| prod / flat | 0.68 | 0.114 | ~26 |
| taufrac8 / 16 | 0.683 | 0.113 | ~24 |
| taufrac8d (delta) | 0.678 | 0.116 | 27.8 |

**no adopt.** Densità irrilevante; delta-only leggermente peggiore. Δ lag ~2′ = rumore.

### 12.3 Feature solo-delta (flat)

**Idea:** togliere `level_*` dalle feature (target già Δy); lasciare delta/miss/d1/d2/rain.

| Modello | skill | rising | phaseLag′ |
|---|---:|---:|---:|
| flat level+Δ | 0.68 | 0.11 | 26 |
| fdelta solo Δ | 0.67 | 0.12 | 27.7 |

**no adopt.** Il livello assoluto porta informazione di regime/datum oltre ai tassi.

File: `ab-lookback-tau.md`, `ab-tau-fraction.md`, `ab-feature-delta.md`.

---

## 13. Mappa tecnica → esito

| Famiglia di tecnica | Cosa speravamo | Cosa abbiamo imparato | Promote? |
|---|---|---|---|
| Più feature valle / soglie | backwater / contesto | valle peggiora o è neutrale | no |
| Derivate OLS | dinamica locale | 6 pt utile | **sì (prod)** |
| Orizzonti lunghi MLP | banda 6 h | skill@360 ≈ 0.80 | **sì** |
| Lag-aligned features | togliere phase lag | lag strutturale, non da feature | no |
| Loss ew4 / q75 | ridurre shrinkage | RMSE ±, lag invariato | no |
| GRU / CNN | battere MLP | pari o sotto su protocollo | no Cento |
| Muskingum / RC / EnKF su livelli | fisica | process model sbagliato | no |
| τ duale / peak-time | celerità / timing | lag viaggio ≠ lag fase | no |
| Online phase correction | ri-fasare online | peggiora rising/falling | no |
| Crossing-time + lift | timing corretto | fase OK, overshoot RMSE | no |
| Hazard + gate | riparare M9 | meno decode, non batte prod | no |
| Routing in Q | mass balance | fallisce gate vs ARX (senza rating ufficiale) | no |
| Lookback-τ / φ-grid / delta-feat | meglio usare τ | indistinguibili da flat | no |
| Banda quantili | incertezza | copertura utile | **sì (banda only)** |
| Ridge Gallo @6 h | copertura temporale | trade-off vs GRU corto | **sì (Gallo)** |

---

## 14. Produzione attuale — numeri di riferimento

| Scala | Skill evt @120′ Cento | Note |
|---|---:|---|
| Fit validate (month-block) | ≈ 0.82 | ottimistico / overlap |
| Protocollo Fase 0 (≥2026) | ≈ 0.67–0.68 | **scala confronti** |
| Rising / falling RMSE @120 | ≈ 0.11 / 0.02 | asimmetria strutturale |
| Phase lag @120 / @360 | ≈ 26–30′ / 120′ | satura a ±120′ @6 h |
| Skill @360 protocollo | ≈ 0.54–0.55 | ancora utile in UI |

---

## 15. Cosa resta aperto (prossimi cicli utili)

1. **Rating Q(h) ufficiale** ARPAE/Consorzio → riprovare routing open-loop che batta ARX prima di EnKF/ibridi.
2. **Generatore di traiettoria** condizionato a τ (non post-process sulla mediana MSE) — M9 ha mostrato che il timing è nei dati ma il decode lift fallisce.
3. **Pioggia prevista (NWP)** per Casalecchio / orizzonti &gt; tempo di viaggio minimo.
4. **Manovre Cavo Napoleonico** se si vuole modellare le stazioni di valle.
5. Metriche che non doppio-puniscono il timing (CSI, crossing-time error, curve “most probable”) **accanto** alla mediana MSE — già in protocollo WP0; non usarle da sole per promuovere curve overshooting.

---

## 16. Indice file A/B e sintesi

| File | Contenuto |
|---|---|
| `ab-mlp120-vs-mlp6h.md` | Estensione MLP Cento a 6 h — **adopt** |
| `ab-gru120-vs-ridge6h.md` | GRU vs ridge / Gallo — ridge @6 h |
| `ab-lagalign.md` | Lag features + ew4/q75 — no |
| `ab-m1-quantiles.md` | Banda pinball+conformal — banda sì |
| `ab-m2-cento-arch.md` | GRU/CNN/ibridi su Fase 0 — no |
| `ab-m3-gallo-blend.md` | Blend GRU+ridge — soft-adopt no |
| `ab-m4-peak-time.md` | Lag picco — FAIL |
| `ab-m5-tau-regime.md` | τ duale — FAIL |
| `ab-m6-m7-close.md` | Dati mancanti — chiuso |
| `ab-m8-online-phase.md` | Correzione online — FAIL |
| `ab-m9-prob-crossing.md` | Crossing-time+decode — no |
| `ab-rising-falling-nov.md` | Asimmetria rising/falling Nov |
| `ab-lookback-tau.md` | Lookback scalato su τ — no |
| `ab-tau-fraction.md` | Griglia φ/τ — no |
| `ab-feature-delta.md` | Feature solo-delta — no |
| `research_plan_reno.md` | Piano WP0–WP6 |
| `research_plan_final.md` | Riassunto WP + post-ciclo |
| `wp1-diagnosis.md` | H1–H11 |
| `fase1-diagnosi.md` | Gap fit/backtest |
| `fase2-sintesi.md` | Tabella M1–M7 |
| `fase3-integrazione.md` | Banda/gate UI |
| `report-esperto-modelli.md` | Snapshot esperto (13 nov) |
| `cento-metodi-report.md` | Feature/seq/ibridi storici |

---

*Report consolidato 2026-11-16. Allineare decisioni successive aggiornando questo file e i singoli A/B.*
