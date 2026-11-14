# Fase 3 — Integrazione runtime / Predizioni

Selettiva rispetto alla Fase 2: entra solo ciò che ha materiale utilizzabile (banda Cento, gate, UX). Nessun cambio del modello mediano in produzione.

| Voce piano | Esito |
|---|---|
| Formato modelli (quantili + CI skill) | `cento-dnn-bands.json` + `skill-gates.json`; soft-merge runtime |
| Parità Python/JS in CI | già presente (E1); resta obbligatoria ad ogni export |
| Gate isteresi enter 0.10 / exit 0.05 | `reliableFromSkill` + soft-load `skill-gates.json` |
| Banda che sfuma | Predizioni: coda oltre `maxReliableHorizonMinutes` tratteggiata/allargata |
| Raccordo GRU+ridge (M3) | runtime `blend_gru_ridge` pronto; **soft-adopt OFF** (criterio M3) |
| Tempo di picco (M4) | **saltato** (FAIL Fase 2) |
| Monte mancante | avviso + banda ×1.5 (`missingUpstreams` / `widenForecastBand`) |
| Testi disclaimer | aggiornati in `predizioni.html` |

## File

- `models/cento-dnn-bands.json` — Q10/Q90 + conformal (~48 MB), soft-load dopo primo paint
- `models/skill-gates.json` — skill@120 + CI90 da eval-table
- `tools/export-cento-bands.py`, `tools/export-skill-gates.py`
- Runtime: `predict-transfer.js`; UI: `predizioni.html`

## Rigenerare

```bash
uv run python tools/export-cento-bands.py
uv run python tools/export-skill-gates.py
node --test tests/*.test.js
```
