# Diagnosi lag + DTW (eventi conclusi, diagnostico)

Costi su segnali standardizzati: minore = migliore allineamento. `noAlign` = senza shift, `affine` = lag + dilatazione uniforme, `lag` = solo lag costante, `dtw` = lag + DTW vincolato.

| edge | n | lag med (′) | lagCorr | scala affine | costo noAlign | affine | lag | dtw | tauP90 med (′) | sat | cov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Forcelli->Cento | 169 | 480 | 0.88 | 1.00 | 0.631 | 0.313 | 0.313 | 54.308 | 120.0 | 0.74 | 1.00 |
| Bagnetto Reno->Cento | 143 | 120 | 0.90 | 1.00 | 0.212 | 0.225 | 0.225 | 35.921 | 120.0 | 0.54 | 0.98 |
| Casalecchio tiro a volo->Bonconvento | 181 | 270 | 0.95 | 1.00 | 0.393 | 0.133 | 0.133 | 20.356 | 120.0 | 0.56 | 1.00 |
| Bonconvento->Sostegno Reno | 134 | 210 | 0.96 | 1.00 | 0.213 | 0.107 | 0.107 | 16.410 | 120.0 | 0.53 | 0.98 |
| Sostegno Reno->Bagnetto Reno | 309 | 390 | 0.25 | 1.00 | 1.807 | 1.193 | 1.381 | 173.695 | 120.0 | 0.33 | 0.98 |
| Cento->Dosso | 169 | 150 | 1.00 | 1.00 | 0.038 | 0.034 | 0.034 | 2.187 | 90.0 | 0.25 | 1.00 |
| Dosso->Opera Reno Panfilia | 243 | 60 | 0.45 | 1.00 | 1.092 | 0.707 | 1.076 | 110.454 | 120.0 | 0.42 | 1.00 |
| Opera Reno Panfilia->Gallo | 168 | 420 | 0.90 | 1.00 | 0.430 | 0.325 | 0.325 | 60.462 | 120.0 | 0.66 | 0.99 |
