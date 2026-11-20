# WP-C / M4 — Inventario e QC storico Cento

**Data:** 2026-11-20
**Scope:** stazioni MLP Cento (Cento + Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo, Casalecchio chiusa).
**Disclaimer:** diagnostica dati; non allertistica.

## Inventario

| Fonte | Periodo | Note |
|---|---|---|
| `storico/Cento.csv` | 2006-01-01 → 2026-11-15 | ~486k righe; passo misto 15′/30′ (resample 30′ in fit) |
| Catalogo eventi | 302 piene Cento | `models/events.json` · testFrom 2026-01-01 (9 eventi) |
| Inventory JSON | 2024-12 → 2026-11 | [`storico-cento-inventory.json`](storico-cento-inventory.json) — finestra corta; lo storico CSV è più lungo |

Eventi per anno (catalogo): 2006–2025 train-heavy; 2026 test=9. Anni con ≥15 piene: 2008–10, 2013–14, 2018–19, 2023–25.

## QC rapido (gate M4)

1. **Datum / zero idrometrico:** nessuna correzione automatica applicata. Controllare a occhio salti di magra anno-su-anno prima di un retrain “storico lungo” come unico segnale.
2. **Passo:** grezzo spesso 15′; training su griglia 30′ (nearest). Non interpolare il passato orario a 30′ per il fit.
3. **Gap:** inventory corto mostra gap >2h limitati su Cento (3) e Casalecchio chiusa (1). Sul lungo periodo usare `tools/report-piene-lag-align.js` / coverage CSV.
4. **Picchi patologici:** nel replay picchi (`analisi-cento-picchi`) due emissioni `peak−h` con |err|>1.5 m (es. 2015-03-17) → trattare come outlier QC, non come segnale modello.

## Retrain previsto (stessa ricetta o M1 se adottata)

```bash
# Stessa ricetta prod (+ eventuale --rising-under-weight α se M1 passa il gate)
uv run python -u tools/fit-ml-cento.py --no-acceptors --skip-lgbm --poly-points 6 \
  --horizons 1 2 3 4 5 6 7 8 9 10 11 12 \
  --holdout-from 2026-01-01 \
  --export-dnn models/cento-dnn-longhist.json

# Protocollo + year folds (non tocca reliable)
node tools/backtest.js --protocol --stations Cento \
  --only-candidates persist,prod \
  --extra-mlp longhist=models/cento-dnn-longhist.json \
  --from 2026-01-01 --seed 1 \
  --json models/eval-table-cento-m4.json --md models/eval-table-cento-m4.md
```

**Criterio adozione M4:** su eventi `peakLevel≥4`, CI90 della Δ risingMin (o rising RMSE) vs prod tutto > 0; falling RMSE entro +10%.

## Stato 2026-11-20

- Storico lungo **già presente** in `storico/*.csv` (dal 2006): il collo di bottiglia non è il download.
- Inventory aggiornato: `uv run python tools/inventory-storico-cento.py` → ~9.9M punti idrometrici; hold≈3613.
- **QC critico:** `Sostegno Reno` ha mesi a zero / anomali nel test 2025-09→2026-02 (e 2009-08/09). Impatta feature MLP in piena recente — da mascherare o escludere nei fold.
- Retrain leakage-free con `--holdout-from 2026-01-01` e year folds è il passo successivo dopo la decisione M1.
