# A/B: Cento MLP@120 (prod) vs MLP@6h

Date: 2026-11-12. Recipe: novalle, `--poly-points 6`, `--skip-lgbm`, rain on.  
Metric: `validateEvents.skill` vs persistenza.

## Skill eventi per orizzonte

| min | MLP@120 baseline (prod) | MLP@6h (nuovo) | Δ |
|---:|---:|---:|---:|
| 30 | 0.864 | 0.871 | +0.007 |
| 60 | 0.825 | 0.839 | +0.014 |
| 90 | 0.833 | 0.832 | −0.001 |
| **120** | **0.826** | **0.820** | **−0.006** |
| 150 | — | 0.833 | |
| 180 | — | 0.836 | |
| 210 | — | 0.822 | |
| 240 | — | 0.815 | |
| 270 | — | 0.816 | |
| 300 | — | 0.818 | |
| 330 | — | 0.800 | |
| 360 | — | **0.799** | |

Confronto col ridge@6h Cento (A/B precedente): skill@360 ridge ≈ **0.42** vs MLP ≈ **0.80**.

## Decisione

- skill@120 sostanzialmente invariata (−0.006); `reliable=true`.
- Orizzonti lunghi restano forti (≈0.80 a 6 h) → **adottare MLP@6h** in `transfer-coeffs.json` nodes.Cento.
- Baseline salvata in `models/cento-mlp-120-baseline.json`; pesi in `models/cento-dnn-novalle-6h.json`.

## Runtime

Cento: `window_delta_mlp` con knot 30′–360′ → `maxReliableHorizonMinutes` = 360 → banda **6 h** in Predizioni.  
GRU Cento resta `reliable=false` (non soft-adotta).
