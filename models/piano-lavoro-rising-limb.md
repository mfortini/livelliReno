# Piano di lavoro — ridurre l'errore in salita a Cento (senza pioggia)

**Data:** 2026-11-17
**Vincoli:** niente dati di pioggia, né osservata nuova né prevista (radar/NWP esclusi). Sono disponibili le serie storiche di livello scaricabili (ARPAE Dext3r e simili).
**Punto di partenza:** MLP prod `window_delta_mlp`, protocollo Fase 0. Rising RMSE @120′ ≈ 0.11 m, errori in salita sugli eventi grandi fino a oltre 0.5 m.
**Idea guida:** il report del 17 nov mostra che con *gli stessi dati* il modello è saturo. Questo piano quindi non cambia architettura: lavora su **più eventi, più stazioni, test più potente e un'uscita adatta alla salita**.

---

## Sequenza e durata indicativa

```
Sett. 1      WP-A Diagnosi errori in salita   ┐ in parallelo
Sett. 1      WP-B Test appaiato sugli A/B già fatti ┘
Sett. 2–4    WP-C Storico lungo (download + QC + catalogo)
Sett. 4–5    WP-D Retrain MLP prod su storico lungo + CV per anno
Sett. 5–7    WP-E Stazioni extra / esperto salita (solo se A lo giustifica)
Sett. 7–8    WP-F Uscita operativa in salita (quantile alto, probabilità soglia)
Opzionale    WP-G Idraulico 1D (solo se si ottengono le sezioni)
```

---

## WP-A — Diagnosi degli errori in salita (1 settimana)

**Domanda:** da dove vengono i ~50 cm? Da pochi eventi grandi, dal timing o da acqua non vista a monte?

Attività:
1. Dai `*-per-event.json` del protocollo: errore in salita per **evento × orizzonte** (60′, 120′, 240′, 360′). Separare RMSE ed errore massimo.
2. Scomporre l'errore di ogni salita in:
   - **ampiezza**: picco previsto vs osservato, a parità di allineamento;
   - **timing**: shift ottimo, con cross-corr o DTW sul singolo evento;
   - **residuo** non spiegato da nessuno dei due.
3. Errore vs **livello di picco dell'evento** a Cento: il modello sottostima sistematicamente gli eventi oltre il range visto in training? (estrapolazione)
4. Errore vs **origine dell'onda**: per ogni salita, quale monte si è mosso prima (Forcelli, Sostegno, Bonconvento, Casalecchio). Ci sono salite a Cento **senza segnale proporzionato a monte**? (afflusso laterale, H5)
5. Quota dell'errore in salita dovuta a orizzonti h > τ della stazione dominante (es. Sostegno τ ≈ 150–180′): è la parte strutturalmente non osservabile senza pioggia.

**Output:** `wpA-diagnosi-salita.md` con una tabella di classificazione degli eventi (estrapolazione / timing / laterale / h>τ).
**Gate:** decide le priorità di WP-C/E.
- Domina l'estrapolazione → priorità a WP-C/D.
- Domina il laterale → WP-E1.
- Domina il timing → WP-F (e accettare il pavimento).

---

## WP-B — Rianalisi degli A/B chiusi con test appaiato (2–3 giorni, in parallelo)

**Problema:** con n=10 eventi le decisioni sono state prese su «CI sovrapposti», un criterio poco potente. Un guadagno reale di 0.03–0.05 non si vede.

Attività:
1. Per ogni candidato chiuso (rpoly2, rlin3, lbtau, taufrac, ew4, lag-augment, CNN): **bootstrap appaiato sulla differenza** di skill/rising RMSE tra candidato e prod, ricampionando *eventi* e non campioni.
2. Riportare P(diff > 0) e CI90 della differenza.
3. Valutare anche un **ensemble semplice** (media prod + rpoly2 + CNN, oppure 5 seed della MLP prod): riduce la varianza senza cambiare informazione e costa quasi zero.

**Output:** tabella `wpB-paired.md`.
**Gate:** un candidato con CI90 della differenza tutto > 0 sulla salita torna in gioco per WP-D.

---

## WP-C — Storico lungo: download, QC, catalogo (2–3 settimane)

È la leva principale disponibile senza pioggia: più piene grandi in training.

Attività:
1. **Inventario** su Dext3r: per Cento, Forcelli, Sostegno, Bonconvento, Casalecchio tiro a volo e chiusa, Bagnetto, Gallo, più tutte le stazioni di livello sul Reno e sugli affluenti tra Casalecchio e Cento, con anno di inizio, passo (orario vs 30′) e % lacune.
2. **Download** del massimo storico disponibile. Conservare i grezzi in `data/raw-historic/` senza toccarli.
3. **QC**, il punto critico:
   - cambi di **zero idrometrico / datum**: salti a gradino, confronto dei livelli di magra anno per anno;
   - cambi di **passo** (orario → 30′): uniformare. Il passato orario non va interpolato a 30′ per il training (vedi esperimenti 5′/10′); meglio un training a passo nativo, oppure una coerenza a 60′;
   - spostamenti di stazione, sensori sostituiti, valori saturati al picco;
   - variazioni d'alveo (erosione o deposito) che spostano il livello di base: valutare una normalizzazione per anno sul livello di magra.
4. **Catalogo eventi esteso** con `build-event-catalog.js`, segnando gli eventi grandi (maggio 2023 e altre piene storiche presenti nei dati).
5. Statistica: quanti eventi sopra le soglie gialla/arancione/rossa prima e dopo l'estensione.

**Output:** dataset storico pulito, `wpC-inventario-qc.md`, catalogo esteso.
**Gate:** proseguire se il numero di eventi grandi aumenta in modo sostanziale (indicativamente ×2 o più).

---

## WP-D — Retrain MLP prod su storico lungo + nuovo protocollo (1–2 settimane)

Attività:
1. **Stessa ricetta prod** (stesse feature, d1/d2 6 pt, orizzonti 30′–360′): cambia solo il periodo di training.
2. **Protocollo a fold per anno** (leave-one-year-out o blocchi di 2 anni) su tutti gli eventi del catalogo esteso, più il test congelato ≥ 2026 come controllo.
3. Metriche come Fase 0, più:
   - rising RMSE ed **errore massimo in salita** per classe di evento (piccolo / medio / grande);
   - errore al picco e crossing-time error sulle soglie.
4. Confronto **appaiato** (metodo WP-B) prod vs prod-storico-lungo.
5. Eventuale rivalutazione dei candidati salvati da WP-B sullo storico lungo.

**Criterio di adozione:** rising RMSE o errore massimo in salita sugli eventi grandi migliori con CI90 della differenza > 0, senza peggiorare falling e skill complessivo oltre il rumore.

---

## WP-E — Stazioni extra ed esperto di salita (2 settimane, condizionato a WP-A)

**E1 — Stazioni aggiuntive** (se WP-A mostra salite non spiegate dal monte):
- aggiungere gli idrometri degli affluenti/sottobacini intermedi trovati in WP-C, uno alla volta, con test appaiato;
- esperienza passata: Bagnetto pari, valle/Dosso peggiora. Quindi solo stazioni **a monte** con τ verso Cento ≥ orizzonte target.

**E2 — Esperto di salita** (se domina l'ampiezza o l'estrapolazione):
- mixture of experts: gate causale (monte in crescita e d1 Cento > 0) → MLP «salita» addestrata solo su campioni di salita, MLP prod per il resto, raccordo morbido;
- loss asimmetrica sulla salita (sottostima penalizzata di più), da distinguere dal reweighting ew4 già provato;
- se il problema è l'estrapolazione: feature di livello relative (rispetto alla magra annuale) e un vincolo di monotonia sulla risposta ai monte.

**Gate:** come WP-D, sempre con test appaiato.

---

## WP-F — Uscita operativa in salita (1–2 settimane)

Accettare che una parte del timing sia irriducibile e renderlo utile in UI.

Attività:
1. In regime di salita (gate WP-E2) mostrare in evidenza il **q75/q90** della banda esistente, con la mediana come riferimento.
2. **Probabilità di superamento soglia** a 2/4/6 h dalla banda quantilica (con calibrazione conformal).
3. Valutazione: CSI, POD/FAR sulle soglie, crossing-time error, copertura della banda **solo in salita**. Oggi la banda è calibrata su tutto: verificare che non sia troppo stretta in salita, ed eventualmente fare una calibrazione separata per regime.

**Output:** `wpF-uscita-salita.md` e una modifica UI dietro flag.

---

## WP-G — Idraulico 1D (opzionale, lungo periodo)

Solo se si ottengono **sezioni trasversali** del tratto Casalecchio → Cento (Regione / AIPo / Consorzio di Bonifica) e una **scala delle portate** affidabile a monte.
- HEC-RAS unsteady con contorno di monte da livello→portata e assimilazione sui livelli osservati.
- È l'unico approccio «fisico» non ancora provato seriamente (i fallimenti WP3 riguardavano surrogati senza geometria).
- Per ora: solo richiesta dati, nessuno sviluppo.

---

## Fuori ambito (per il vincolo sulla pioggia)

- NWP / radar: esclusi. Si accetta che per h > τ delle stazioni vicine (Sostegno ≈ 150–180′) parte della salita resti non prevedibile.
- Nuove architetture sugli stessi dati: chiuso dal report del 17 nov.

## Regole trasversali

- Split leakage-free sempre (`--holdout-from`), più fold per anno da WP-D in poi.
- Decisioni su **differenza appaiata**, non su CI sovrapposti.
- Ogni WP chiude con un `ab-*.md` o `wp*-*.md` e l'aggiornamento del report consolidato.
