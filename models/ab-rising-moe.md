# A/B M3 — MoE salita Cento (rising-only expert)

**Expert:** `models/cento-dnn-rising.json` · skill120=0.581846 · risingOnly=true
**Gate:** Δ risingMin med ≥ +0.15 m vs prod; falling RMSE entro +10%.

| modello | h | rising RMSE | risingMin med | falling RMSE | peak err med |
|---|---:|---:|---:|---:|---:|
| prod | 120 | 0.121 | -0.560 | 0.021 | 0.018 |
| moe | 120 | 0.118 | -0.560 | 0.022 | 0.022 |
| prod | 360 | 0.475 | -1.690 | 0.067 | 0.015 |
| moe | 360 | 0.473 | -1.690 | 0.068 | 0.015 |

| h | Δ risingMin (moe−prod) | falling rel | gate |
|---:|---:|---:|---|
| 120 | 0.000 | 4.0% | FAIL |
| 360 | 0.000 | 1.7% | FAIL |

**Decisione:** non promuovere MoE; tenere prod + M2 UI/banda.

`models/cento-dnn-rising.json` impostato a `reliable: false` (soft-adopt off).
