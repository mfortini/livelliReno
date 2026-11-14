# A/B M6 / M7 — Verifica dati e chiusura

## M6 — Nuovi input (Casalecchio chiusa)

**Ipotesi:** intensità di pioggia, stazioni più a monte o pioggia prevista alzano il lower CI skill ≥ 0.10.

**Disponibilità dati (2026-11-14):**

| Sorgente | Stato |
|---|---|
| Pioggia osservata (8 pluviometri in `storico/rain/`) | **Sì** — già usata dove `useRain` |
| Stazioni monte Casalecchio (`Sasso Marconi`, `Vergato DX`) | **Sì** — già in config |
| Pioggia **prevista** (NWP / nowcast) | **No** nello storico del repo |
| Altre stazioni monte oltre il DAG attuale | Non inventariate / non in CSV |

**Tabella unica @120′ (prod):** skillEvt ≈ 0.052, CI90 [−0.00, 0.17] — sotto soglia promozione.

**Esito:** **chiusura motivata** finché manca la pioggia prevista (o un nuovo sensore). La pioggia osservata da sola non ha portato Casalecchio sopra soglia nel fit attuale. Riaprire quando esiste una serie forecast allineata al reticolo 30′.

## M7 — Stazioni di valle (Dosso, Opera Reno Panfilia, Secondo Salto)

**Ipotesi:** le manovre al Cavo Napoleonico spiegano skill bassi; aggiungerle come input alza almeno una stazione sopra soglia.

**Disponibilità:**

| Voce | Stato |
|---|---|
| CSV livelli valle | **Sì** (`storico/{Dosso,Opera Reno Panfilia,Secondo Salto}.csv`) |
| Eventi catalogo | Sì (34 / 35 / 24) |
| Serie manovre / paratoie / portate Cavo Napoleonico | **Assenti** nel repo |
| Protocollo Fase 0 su queste stazioni | `nAll=0` per prod/ridge (nodi non affidabili / allineamento insufficiente → nessun campione scorabile) |

**Esito:** **chiusura motivata** — senza dato manovre non si testa l’ipotesi. I livelli valle restano fuori soft-adopt. Riaprire se ARPAE/Consorzio pubblica una serie operativa delle manovre sul Napoleonico.

## Decisione congiunta

Né M6 né M7 partono come fit: manca l’input decisivo. Nessuna promozione.
