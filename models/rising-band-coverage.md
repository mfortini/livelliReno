# Copertura banda Cento — solo salita (M2)

**Eventi test:** 9 da 2026-01-01T00:00:00.000Z
**Rising coverage:** 75.6% (n=1629)
**Falling coverage:** 96.0% (n=1116)
**Gate ≥85% rising:** FAIL

| h (′) | rising cov | falling cov |
|---:|---:|---:|
| 120 | 82.5% | 97.3% |
| 240 | 75.1% | 95.4% |
| 360 | 69.2% | 95.1% |

Banda troppo stretta in salita: applicare inflate upper in regime salita.

**Inflate calibrato:** ×2 → coverage rising 85.1% (PASS) → `models/cento-rising-band-inflate.json`
