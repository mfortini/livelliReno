# Piano di ricerca — miglioramento previsioni livelli Reno / Cento

13 novembre 2026

## Sintesi

La ricerca deve stabilire, su un'unica scala di misura, quale modello prevede meglio il livello a ogni stazione e fino a quale orizzonte è corretto mostrarlo. Oggi non lo sappiamo: lo stesso modello di Cento vale 0.82, 0.69 o 0.48 di skill a +2 h secondo il protocollo usato.

Il lavoro è in quattro fasi, in quest'ordine:

1. **Fase 0 — Misura.** Un solo protocollo di valutazione per tutti i modelli. Nessun altro esperimento parte prima.
2. **Fase 1 — Diagnosi.** Capire perché lo skill cala da 0.82 (fit) a 0.48 (backtest).
3. **Fase 2 — Modello.** Esperimenti mirati su banda di incertezza, Gallo, tempo di picco, Casalecchio chiusa e stazioni di valle.
4. **Fase 3 — Integrazione.** Portare in runtime e in interfaccia solo ciò che ha superato le regole di promozione.

Risultati attesi alla fine:

- una tabella unica di skill per stazione, orizzonte e modello, con intervalli di confidenza;
- una banda di previsione con copertura verificata, al posto della sola curva;
- una decisione motivata sull'orizzonte a 6 h per Cento e Gallo;
- una stima del tempo di arrivo del picco a Cento.

Fuori dal perimetro: uso allertistico delle previsioni, modelli idraulici completi del tratto, aggiunta di nuove stazioni alla dashboard.

## Punto di partenza

In produzione tre stazioni mostrano una curva: Cento (MLP, fino a 6 h), Gallo (ridge, fino a 6 h) e Casalecchio chiusa (GRU, 2 h). Le altre stazioni sono sotto la soglia di affidabilità e non mostrano nulla. Fonte: report modelli del 13 novembre 2026.

La review del report ha trovato otto problemi. I primi quattro riguardano la misura e bloccano ogni confronto tra modelli.

| # | Problema | Conseguenza per il team |
| --- | --- | --- |
| 1 | Tre scale di skill mescolate (fit, backtest §3.3, backtest §2.1) | I confronti tra modelli non sono validi. Il GRU di Cento, scartato, ha un backtest (~0.72) sopra quello della MLP (0.69 o 0.48) |
| 2 | Divario fit/backtest non spiegato (0.82 contro 0.48 a +2 h) | Non sappiamo quale numero descrive l'uso reale |
| 3 | Skill calcolato solo sui campioni con variazione osservata ≥ 0.10 m | La persistenza perde per costruzione; i falsi allarmi nei tratti stabili non contano |
| 4 | Campioni piccoli e autocorrelati (n≈32 e n≈25 a 30′) | Differenze di 0.006 sono rumore; il flag `reliable` può cambiare a ogni run |
| 5 | Banda a 6 h senza backtest | A 360′ esiste solo il valore di fit (0.80) |
| 6 | Baseline deboli | Il poly2 arriva a ~0.45: se la scala è la stessa, la MLP in backtest (0.48) aggiunge poco |
| 7 | Condizioni reali non simulate | Ritardo di pubblicazione ARPAE, dati grezzi, stazione di monte mancante |
| 8 | Limiti strutturali noti | Fronti sottostimati, ritardo di fase (~30′ a +2 h, ≥120′ a +6 h), nessuna stima del tempo di picco |

## Domande di ricerca

Ogni domanda ha un'ipotesi dichiarata prima dell'esperimento e una misura che la conferma o la smentisce.

| ID | Domanda | Ipotesi | Come si misura | Fase |
| --- | --- | --- | --- | --- |
| D1 | Quanto vale davvero ogni modello? | Su scala unica le classifiche cambiano; il GRU di Cento può superare la MLP | Tabella unica di skill con intervalli di confidenza | 0 |
| D2 | Perché lo skill cala da 0.82 a 0.48? | Tre cause sommate: selezione sul validate, mesi alterni, differenze Python/JS | Un esperimento per causa, una alla volta | 1 |
| D3 | La rete batte una baseline seria? | Il margine su poly2 è piccolo a +2 h e cresce con l'orizzonte | Skill contro poly2 e contro ridge, per orizzonte | 0 |
| D4 | Fino a che orizzonte la previsione è utile? | L'anticipo reale a Cento è circa 4 h, non 6 | Skill e ritardo di fase per orizzonte, in backtest | 0–1 |
| D5 | Una banda a quantili è meglio della curva singola? | Riduce la sottostima dei fronti e dà una copertura verificabile | Copertura empirica della banda 10–90, errore sul picco | 2 |
| D6 | Si può prevedere il tempo di arrivo del picco? | Sì, con errore inferiore al ritardo di fase attuale | Errore sul tempo di picco, per evento | 2 |
| D7 | Cosa manca a Casalecchio chiusa e alle stazioni di valle? | Mancano input (pioggia, manovre idrauliche), non architettura | Skill prima e dopo l'aggiunta di ogni input | 2 |

## Fase 0 — Protocollo di valutazione unico

Tutti i modelli, in produzione e scartati, vengono rivalutati con lo stesso strumento, sugli stessi dati e contro le stesse baseline. È il prerequisito di ogni fase successiva.

**Specifica del protocollo**

- **Esecuzione nel runtime reale.** La valutazione gira sul codice JS che gira in pagina, a partire da `tools/backtest.js`. Il fit in Python non produce più numeri di skill da citare.
- **Split per tempo e per evento.** Il test è un periodo finale mai usato per fit, scelta delle feature o iperparametri. Nessuna piena è divisa tra train e test.
- **Catalogo degli eventi.** Un file con le piene dello storico: inizio, fine, picco, stazione. Tutte le statistiche si aggregano per evento, non per campione a 30′.
- **Campionamento denso.** Ogni passo a 30′, non uno ogni 2 h.
- **Tutti gli orizzonti.** Da 30′ a 360′ per ogni modello, anche per quelli addestrati solo a 120′.
- **Baseline fisse.** Persistenza, poly2 a 6 punti, ridge. La media mobile a 3 h esce dai confronti.

**Metriche**

| Metrica | Cosa dice | Note |
| --- | --- | --- |
| Skill su tutti i campioni | Valore medio, tratti stabili compresi | Penalizza i falsi allarmi |
| Skill in evento | Valore durante le piene | Eventi scelti dal catalogo, non dalla variazione osservata |
| Errore sul picco (m) | Sottostima dei fronti | Uno per evento |
| Errore sul tempo di picco (min) | Ritardo di fase | Uno per evento |
| POD e FAR sui superamenti di soglia | Utilità per chi guarda il grafico | Soglie ARPAE disegnate in pagina |
| Intervallo di confidenza al 90% | Quanto fidarsi del numero | Bootstrap per evento |

**Attività**

- [x] Costruire il catalogo eventi e documentare periodo dello storico, numero di piene e livello massimo in train per stazione
- [x] Definire e congelare il periodo di test (`2026-01-01` → fine storico)
- [x] Estendere `tools/backtest.js`: campionamento denso, tutti gli orizzonti, metriche della tabella, bootstrap (`--protocol`)
- [x] Aggiungere poly2 e ridge come baseline per ogni stazione
- [x] Rieseguire tutti i modelli del report, compresi GRU, 1D-CNN e ibridi
- [x] Pubblicare la tabella unica in `models/` e sostituire i numeri del report

**Fatto quando:** esiste un solo comando che produce la tabella per ogni stazione, e due esecuzioni consecutive danno gli stessi numeri.

Comando: `node tools/build-event-catalog.js && node tools/backtest.js --protocol --from 2026-01-01 --seed 1`

## Fase 1 — Diagnosi del divario fit/backtest

Cinque esperimenti brevi, ciascuno isola una causa possibile del calo da 0.82 a 0.48 sulla MLP di Cento. Si cambia una sola cosa per volta e si misura con il protocollo della Fase 0.

| ID | Causa sospetta | Esperimento | Se confermata |
| --- | --- | --- | --- |
| E1 | Differenze tra fit Python e runtime JS | Stessi input alle due implementazioni; confronto delle feature e dell'uscita campione per campione | Correggere il runtime; test di parità fisso in CI |
| E2 | Validate usato anche per scegliere feature e iperparametri | Ripetere la selezione (punti OLS, accettori, architettura) su un validate interno al train; misurare sul test congelato | Rifare le scelte del §3.1 con il nuovo schema |
| E3 | Mesi alterni misurano interpolazione | Confrontare split a mesi alterni e split per blocchi temporali, a parità di modello | Abbandonare lo split a mesi alterni |
| E4 | Campionamento ogni 2 h e definizione dei campioni evento | Ricalcolare lo skill con campionamento denso e con eventi da catalogo | Il numero di riferimento diventa quello denso |
| E5 | Condizioni operative | Backtest con ritardo di pubblicazione ARPAE simulato, dati grezzi al posto dello storico validato, una stazione di monte spenta durante la piena | Aggiungere queste condizioni al protocollo standard |

E1 va per primo: costa poco e, se trova un difetto, cambia tutti gli altri numeri.

**Prodotto della fase:** una nota in `models/` che attribuisce il divario alle cause, con la quota spiegata da ciascuna. Se una quota importante resta non spiegata, la fase non è chiusa.

## Fase 2 — Esperimenti sul modello

Sette pacchetti di lavoro indipendenti, in ordine di priorità. Ognuno si confronta con il modello in produzione sulla tabella unica e passa o no secondo le regole di promozione.

| ID | Pacchetto | Stazione | Cosa si fa | Successo se | Priorità |
| --- | --- | --- | --- | --- | --- |
| M1 | Previsione a quantili | Cento, poi Gallo | Perdita a quantili 10-50-90 al posto dell'MSE; calibrazione della banda per orizzonte con conformal prediction | Copertura della banda vicina all'80%; errore sul picco minore dell'attuale | Alta |
| M2 | Rivalutazione GRU e 1D-CNN | Cento | Nessun nuovo addestramento: lettura della tabella unica della Fase 0 | Decisione documentata su quale architettura resta | Alta |
| M3 | Combinazione GRU + ridge | Gallo | GRU fino a 2 h, ridge oltre, raccordo pesato tra i due | Skill non inferiore al GRU a 2 h e alla ridge a 6 h; nessun salto visibile nel raccordo | Alta |
| M4 | Tempo di arrivo del picco | Cento | Regressione a quantili del ritardo di picco in funzione del livello alle stazioni di monte | Errore mediano sul tempo di picco sotto i 30′ del ritardo di fase attuale | Media |
| M5 | Ritardo τ dipendente dal regime | Stazioni con ridge | Due valori di τ sopra e sotto `z85`, stimati solo sul train | Skill in evento migliore, intervalli non sovrapposti | Media |
| M6 | Nuovi input | Casalecchio chiusa | Intensità di pioggia, stazioni più a monte, pioggia prevista | Limite inferiore dello skill sopra la soglia di promozione | Media |
| M7 | Stazioni di valle | Dosso, Opera Reno Panfilia, Secondo Salto | Verificare se le manovre al Cavo Napoleonico spiegano gli skill bassi; se sì, aggiungerle come input | Almeno una stazione sopra soglia, oppure chiusura motivata | Bassa |

**Regole comuni**

- Ipotesi e criterio di successo scritti prima di lanciare l'esperimento.
- Un file A/B per pacchetto in `models/`, con lo stesso formato degli attuali.
- Il periodo di test congelato si legge una sola volta per pacchetto, a scelte fatte.
- Un esito negativo si documenta come uno positivo.

**Rimandato:** Muskingum ed EnKF. Hanno senso solo con una scala di deflusso che converta i livelli in portate; da riaprire se il dato diventa disponibile.

## Fase 3 — Integrazione in runtime e interfaccia

Entra in pagina solo ciò che ha superato le regole di promozione. Le modifiche toccano il formato dei modelli, il gate e la pagina Predizioni.

- [ ] **Formato dei modelli.** Estendere `transfer-coeffs.json` e i file `seq-*.json` con i tre quantili per orizzonte e con l'intervallo di confidenza dello skill.
- [ ] **Test di parità in CI.** Ogni export di pesi verifica che Python e JS diano la stessa uscita sugli stessi input.
- [ ] **Gate con isteresi.** `reliable` si decide sul limite inferiore dell'intervallo, con due soglie distinte per entrare e uscire, così non cambia a ogni run.
- [ ] **Banda al posto della curva.** La pagina mostra mediana e banda 10–90. La banda si allarga con l'orizzonte e sfuma, invece di troncarsi a `maxReliableHorizonMinutes`.
- [ ] **Raccordo tra modelli.** Supporto runtime alla combinazione pesata GRU + ridge (M3).
- [ ] **Tempo di picco.** Se M4 passa, indicazione in pagina dell'intervallo di arrivo stimato del picco.
- [ ] **Dati mancanti.** Comportamento esplicito quando una stazione di monte non risponde: banda più larga o curva sospesa, secondo l'esito di E5.
- [ ] **Testi in pagina.** Aggiornare il disclaimer e spiegare cosa significa la banda.

## Regole di promozione

Un modello sostituisce quello in produzione solo se soddisfa tutte le condizioni sotto, misurate sul periodo di test congelato. Le soglie numeriche sono una proposta da confermare a fine Fase 0, quando si conosce la larghezza reale degli intervalli.

| # | Condizione | Soglia proposta |
| --- | --- | --- |
| 1 | Skill in evento contro persistenza, limite inferiore dell'intervallo al 90% | ≥ 0.10 a ogni orizzonte mostrato |
| 2 | Skill contro poly2, limite inferiore | > 0 |
| 3 | Confronto con il modello in produzione | Differenza di skill con intervallo che non comprende lo zero; a parità, resta il modello in produzione |
| 4 | Errore sul picco | Non peggiore del modello in produzione |
| 5 | Copertura della banda 10–90 | Tra 70% e 90% a ogni orizzonte mostrato |
| 6 | FAR sui superamenti di soglia | Non peggiore del modello in produzione |
| 7 | Parità Python/JS | Test superato |

Il flag `reliable` usa la condizione 1 con isteresi: si accende a 0.10 e si spegne sotto 0.05. Un confronto solo relativo, come 0.25 contro 0.14 a Casalecchio chiusa, non basta più.

## Sequenza e impegno

Il piano dura tra 9 e 12 settimane per una persona a tempo pieno. È una stima da confermare: il report non indica quante persone lavorano al progetto.

| Fase | Nome | Durata | Gate in uscita |
| --- | --- | --- | --- |
| 0 | Misura | 2 settimane | Tabella unica riproducibile |
| 1 | Diagnosi | 1–2 settimane | Divario spiegato causa per causa |
| 2 | Modello | 4–6 settimane | Regole di promozione superate dal pacchetto |
| 3 | Integrazione | 2 settimane | — |

La fase successiva non parte finché il criterio di uscita della precedente non è soddisfatto.

**Dipendenze dentro la Fase 2**

- M2 non richiede addestramento: è disponibile alla chiusura della Fase 0.
- M3 (Gallo) dipende da M2, che conferma o smentisce il vantaggio del GRU.
- M1 precede la Fase 3: la banda in pagina dipende dai quantili.
- M6 e M7 partono solo dopo la verifica della disponibilità dei dati.
- Con due persone, M1 e M3–M4 possono procedere in parallelo e la Fase 2 si accorcia.

A fine Fase 1 conviene un punto di controllo con l'idrologo: è il momento in cui i numeri del report vengono sostituiti.

## Rischi e decisioni aperte

Il rischio principale è che lo storico contenga troppo poche piene per distinguere i modelli: in quel caso il risultato onesto è una banda larga e un orizzonte più corto.

| Rischio | Effetto | Risposta |
| --- | --- | --- |
| Poche piene indipendenti nello storico | Intervalli larghi, nessun pacchetto supera le regole | Allungare lo storico se ARPAE lo consente; altrimenti accettare orizzonti più corti |
| Lo skill reale è vicino a 0.48, non a 0.82 | L'orizzonte a 6 h di Cento va ridotto | Decidere sulla tabella unica; mostrare 3–4 h finché manca il backtest a 360′ |
| Il periodo di test viene consultato troppe volte | Torna il problema del validate riusato | Una lettura per pacchetto, registrata nel file A/B |
| Pioggia prevista o manovre idrauliche non disponibili come dato | M6 e M7 non partono | Verificare le fonti prima di pianificare i due pacchetti |
| La banda a quantili confonde chi guarda la pagina | Lettura errata come soglia di allerta | Testo esplicativo e prova con alcuni utenti prima del rilascio |

**Decisioni da prendere prima di partire**

- [ ] Chi lavora al piano e con quale disponibilità
- [x] Quale periodo dello storico diventa il test congelato → **2026-01-01 → fine storico** (ultimi 3 mesi senza piene a Cento)
- [x] Se ridurre subito l'orizzonte di Cento a 3–4 h in attesa della Fase 0, o lasciare 6 h → **lasciare 6 h**; decidere dopo la tabella unica
- [ ] Se l'idrologo che rivede il report può validare il catalogo eventi e le soglie di promozione
- [ ] Disponibilità dei dati per M6 (pioggia prevista) e M7 (manovre al Cavo Napoleonico)
