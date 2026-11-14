# A/B M5 — τ regime-dipendente (ridge stations)

**Ipotesi:** il ritardo di correlazione cross-corr è diverso sopra/sotto z85 train; intervalli bootstrap non si sovrappongono e lo skill evento @120′ migliora con τ duale.

**Criterio:** almeno una stazione con CI non sovrapposti **e** skill duale > skill τ unico (+0.01).

## Gallo

| Upstream | τ all | τ calm [CI90] | τ high [CI90] | non-overlap | skill@120 all | skill dual |
|---|---:|---|---|:---:|---:|---:|
| Opera Reno Panfilia | 390 | 30 [0,240] | 330 [330,540] | yes | -7.834 | -8.894 |

Esito stazione: **FAIL** (non-overlap=True, improved=False).

## Casalecchio chiusa

| Upstream | τ all | τ calm [CI90] | τ high [CI90] | non-overlap | skill@120 all | skill dual |
|---|---:|---|---|:---:|---:|---:|
| Sasso Marconi | 90 | 0 [0,180] | 120 [90,150] | no | -0.087 | -0.148 |
| Vergato DX | 180 | 720 [89,720] | 210 [120,270] | no | -0.233 | -0.404 |

Esito stazione: **FAIL** (non-overlap=False, improved=False).

## Bonconvento

| Upstream | τ all | τ calm [CI90] | τ high [CI90] | non-overlap | skill@120 all | skill dual |
|---|---:|---|---|:---:|---:|---:|
| Casalecchio tiro a volo | 270 | 300 [240,331] | 330 [270,391] | no | 0.021 | -0.018 |

Esito stazione: **FAIL** (non-overlap=False, improved=False).

## Decisione

**Non promuovere** — τ duale non batte τ unico con CI disgiunti sul proxy evento.
