# Cento — report metodi di previsione (sperimentale)

Skill@120 su eventi (`|Δy|≥0.10 m`). Train/validate vs **persistenza**; holdout anche vs **poly2** locale.
**Non allertistico.** Holdout eventi n≈32 → interpretare con cautela.

## Decisioni

- **In prod:** `window_delta_mlp` senza valle (`transfer-coeffs` / `cento-dnn.json`), soft-load se `reliable`.
- **Non promossi:** bracci valle, yellow/ratio, d1/d2, ibridi Muskingum/RC/volume/EnKF.
- **Finestra OLS:** 6 punti (2.5 h) ottima come predittore stupido; a 12 punti (5.5 h) la baseline poly2 peggiora → `holdPoly2` non confrontabile tra finestre; per A/B finestra usare `holdPersist`.

## MLP feature A/B (skill@120 eventi)

| Metodo | Feature | valEvt | hold persist | hold poly2 | Nota |
|---|---|---:|---:|---:|---|
| MLP novalle (prod storico) | level+Δ+miss monte, rain | 0.826 | 0.772 | — | Baseline operativa |
| MLP + valle raw | + accettori valle | 0.802 | 0.766 | — | Non batte novalle |
| MLP valle + chain Δ | Δ catena | 0.795 | 0.793 | — | Holdout su, validate giù |
| MLP valle yellow headroom | giallo−livello | 0.802 | 0.782 | — | Prima del ratio |
| MLP valle yellow+ratio | headroom + livello/giallo | 0.824 | 0.840 | 0.833 | Miglior holdout pre-d1/d2 (poly n=4) |
| MLP valle yellow-chain | yellow+ratio+chain | 0.804 | 0.817 | — | Meno netto |
| MLP novalle + d1/d2 (n=4) | OLS 1.5 h | 0.831 | 0.736 | 0.723 | Validate su |
| MLP yellow+ratio + d1/d2 (n=4) | + d1/d2 | 0.819 | 0.821 | 0.812 | Holdout sopra novalle |
| MLP novalle + d1/d2 (n=6) | OLS 2.5 h | 0.820 | 0.726 | 0.748 | |
| MLP yellow+ratio + d1/d2 (n=6) | OLS 2.5 h | **0.821** | 0.730 | 0.752 | Miglior validate p6/p12 |
| MLP novalle + d1/d2 (n=12) | OLS 5.5 h | 0.820 | **0.778** | 0.836† | Miglior holdPersist |
| MLP yellow+ratio + d1/d2 (n=12) | OLS 5.5 h | 0.806 | 0.725 | 0.796† | Validate giù |

† Baseline poly2 più debole a n=12 (sweep predittore: skill vs persist 0.45@6pt → 0.27@12pt).

Pesi A/B finestra: `cento-dnn-novalle.json` (p6), `cento-dnn-novalle-p12.json`, `cento-dnn-valle-yellow-p6.json`, `cento-dnn-valle-yellow-p12.json`. Gate: `cento-mlp-feature-ab-gate.json`.

## Ibridi (gate holdout eventi)

| Metodo | skill120 evt | vs MLP | Adopt |
|---|---:|---|---|
| Muskingum + GRU | 0.068 | MLP 0.826 | no |
| RC + GRU | −0.054 | MLP+valle 0.802 | no |
| Hybrid volume + DNN | 0.381 | MLP 0.826 | no |
| EnKF mass transfer | −0.077 | MLP 0.826 | no |

## Poly2 come predittore stupido (solo Cento, n_evt=3774)

Skill vs persistenza a +2 h:

| n punti | lookback | skill |
|---:|---:|---:|
| 3 | 1.0 h | 0.038 |
| 4 | 1.5 h | 0.387 |
| **6** | **2.5 h** | **0.453** |
| 8 | 3.5 h | 0.402 |
| 12 | 5.5 h | 0.269 |
| 16 | 7.5 h | 0.133 |
| 24 | 11.5 h | −0.118 |

Lineare a 4 pt: skill 0.52. d1 ACF resta alta a 3–6 h (0.70→0.44); d2 grezza a 30′ è rumorosa.

## Backtest runtime

`models/backtest-summary.json`: Cento skill120≈0.48 vs persistenza, ≈0.67 vs MA 3 h — protocollo diverso dallo split month-block del fit MLP.

## Tooling

```bash
uv run python -u tools/fit-ml-cento.py --data ./storico --feature-set yellow-headroom --skip-lgbm --poly-points 12
uv run python -u tools/fit-ml-cento.py --data ./storico --no-acceptors --skip-lgbm --poly-points 6
```

`--poly-points` regola d1/d2 e la baseline holdout poly2. Progress verbose di default (`-q` per silenziare).
