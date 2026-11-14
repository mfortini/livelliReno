# A/B M4 — Tempo di arrivo del picco (Cento)

**Ipotesi:** il ritardo picco_monte→picco_Cento è prevedibile da livello/pendenza a monte a issue = peak−120′, meglio della mediana train e sotto i 30′ di ritardo di fase.

**Criterio:** su test congelato (2026-01-01→) mediana |errore| < 30′ e < baseline mediana.

Upstream primario: **Forcelli**. Eventi usabili: 29 (train 20, test 9).

## Metriche

| Split | n | med\|err\| pred (′) | med\|err\| baseline (′) | lag vero med (′) |
|---|---:|---:|---:|---:|
| train | 20 | 131.8 | 165.0 | 600.0 |
| **test** | 9 | **139.0** | 165.0 | 450.0 |

**Esito:** FAIL — non promuovere.

Il lag picco-picco Forcelli→Cento ha mediana ~450–600′ (tempo di viaggio della piena), non i ~30′ di ritardo di fase della curva MLP. La regressione batte la baseline mediana sul test (139′ vs 165′) ma resta lontana dalla soglia 30′. Senza feature di celerità/regime più ricche non promuovere.

Nota: stima offline del solo ritardo di picco; non sostituisce la curva di livello.
