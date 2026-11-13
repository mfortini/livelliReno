# Fase 1 — Diagnosi del divario fit / backtest (Cento MLP @120′)

**Data:** 2026-11-13  
**Protocollo di misura:** Fase 0 (`node tools/backtest.js --protocol --from 2026-01-01 --seed 1`)  
**Disclaimer:** sperimentale; non operativo / non allertistico.

## 1. Numeri di riferimento

| Scala | Skill evento @120′ | Origine |
|---|---:|---|
| Fit validateEvents (mesi alterni, `\|Δy\|≥0.10`) | **0.820** | `cento-dnn-novalle-6h.json` |
| Protocol Fase 0 (JS `prod`, catalogo, test 2026-01→) | **0.667** CI90 [0.58, 0.76] | `eval-table.json` |
| Backtest legacy full-history (`--every 120`) | **~0.48** | `backtest-summary` (report) |
| Legacy sullo **stesso** test 2026-01→ | **0.666** | `fase1-e4-every120.json` |

Divario da spiegare sul protocollo: **0.820 − 0.667 ≈ 0.15**.  
Il salto 0.48 → 0.67 **non** è campionamento sul test congelato (vedi E4): lo 0.48 era una finestra/aggregazione diversa.

## 2. Decomposizione

| Causa | Esito | Δ skill stimato (verso 0.82→0.67) | Note |
|---|---|---:|---|
| **E1** Python ≠ JS | **Smentita** | ~0 | Layer A `mlpForward` max\|Δ\|&lt;1e-5; Layer B level/delta/d1/d2 allineati sul reticolo 30′ (`tests/fixtures/cento-mlp-parity.json`) |
| **E4** Campionamento / def. evento | **Quasi nullo sul test congelato** | ~0 | every-120 = every-30 = 0.666; protocol catalogo = 0.667 |
| **E3** Mesi alterni vs blocchi | **Parziale / lieve** | ~0.02–0.03 | Fit valEvt: alt 0.820 → block 0.789; protocol: prod 0.667 vs `mlp_block` 0.648 (CI sovrapposti) |
| **E2** Validate riusato per selezione | **Parziale** | ~0.05 sul *numero di fit* | Early-stop su validate interno: outer valEvt scende a 0.773; recipe vincitrice resta poly-6 / no-acceptors |
| **Overlap test⊂train (alt)** | **Strutturale** | residuo principale | Gen–ago 2026 entrano nei mesi di train/val; il test Fase 0 non è out-of-sample puro |
| **E5** Condizioni operative | Stressor, non spiega il divario fit | — | Lag 90′: 0.667→**0.347**; drop Forcelli: →**0.617**; raw: **bloccato** (solo storico validato in repo) |

**Residuo non spiegato come bug di runtime:** dopo E1–E4, il gap 0.82→0.67 è coerente con (a) metrica di fit su validate a mesi alterni ottimistica (~0.05 da E2 + ~0.03 da E3) e (b) overlap temporale del test. Non resta un difetto di parità Python/JS.

## 3. Dettaglio esperimenti

### E1 — Parità Python / JS

- Script: `tools/dump-mlp-parity.py` → `tests/fixtures/cento-mlp-parity.json`
- Test CI: `E1 Layer A/B` in `tests/predict-transfer.test.js`
- **Conclusione:** runtime fedele ai pesi esportati; **quota runtime = 0**.

### E4 — Campionamento e eventi

| Setup | skillEvt@120 Cento |
|---|---:|
| `--every 120 --from 2026-01-01` | 0.666 |
| `--every 30 --from 2026-01-01` | 0.666 |
| `--protocol` catalogo | 0.667 |

Densificazione legacy attorno a `\|Δ\|` rende every-120 già equivalente a every-30 sugli eventi. Il catalogo non sposta il numero. **Chiudere E4:** il riferimento operativo resta il protocollo Fase 0; lo 0.48 storico non va confrontato con lo 0.82 di fit.

### E3 — Split `alt` vs `block`

- CLI: `--split {alt,block}` in `fit-transfer.py` / `fit-ml-cento.py`
- Pesi: `models/cento-dnn-block-split.json`
- Protocol (`fase1-e3-protocol.json`): prod **0.667** vs block **0.648** (CI si sovrappongono)
- **Raccomandazione:** preferire `block` nei fit futuri (meno interpolazione); non urgente per la UI. Non spiega da solo il divario.

### E2 — Selezione su validate interno

| Recipe (early-stop inner, skill riportato = outer valEvt) | valEvt@120 |
|---|---:|
| poly4 novalle | 0.744 |
| **poly6 novalle** | **0.773** |
| poly12 novalle | 0.715 |
| poly6 valle | 0.731 |
| poly6 novalle, selection **outer** (prod-like) | **0.820** |

Vincitore invariato rispetto a §3.1. La differenza 0.820−0.773 ≈ **0.05** è ottimismo da early-stop/ranking sullo stesso validate esterno. **Non rifare §3.1.**

### E5 — Condizioni operative

| Stressor | skillEvt@120 prod | Δ vs baseline 0.667 |
|---|---:|---:|
| `--publish-lag-min 90` | 0.347 | −0.32 |
| `--drop-upstream Forcelli` | 0.617 | −0.05 |
| dati grezzi vs storico | — | **bloccato** (nessun corpus raw distinto) |

Promuovere lag e dropout come stressor standard del protocollo in Fase 2/3; il lag di pubblicazione è il degrado più grave.

## 4. Verdetto e go/no-go Fase 2

- Il divario **0.82 (fit) → 0.67 (protocol)** non è un bug JS; è soprattutto **scala di valutazione + split/selezione + overlap**.
- Il **0.48** del vecchio backtest non è la metrica di riferimento sul test congelato.
- **Go Fase 2** con numeri di gate presi dal protocollo Fase 0 (limite inferiore CI), non dallo skill di fit.
- Prima di citare skill assoluti “onesti” out-of-sample: refit con `--split block` e test che non rientri nei mesi di train (punto di controllo idrologo).

## 5. Artefatti

| File | Ruolo |
|---|---|
| `tests/fixtures/cento-mlp-parity.json` | E1 |
| `models/fase1-e4-every120.json` / `every30` | E4 |
| `models/cento-dnn-block-split.json` | E3 |
| `models/fase1-e3-protocol.json` | E3 protocol |
| `models/fase1-e2/*` | E2 grid |
| `models/fase1-e5-lag90.json` | E5 lag |
| `models/fase1-e5-drop-forcelli.json` | E5 dropout |
