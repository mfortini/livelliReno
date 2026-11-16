# Research plan — riassunto risultati (Cento)

**Data:** 2026-11-16 · Piano: [`research_plan_reno.md`](research_plan_reno.md) rev 4  
**Disclaimer:** sperimentale; non operativo, non allertistico.

## Decisione

| Voce | Valore |
|---|---|
| Soft-adopt / patch `transfer-coeffs` | **no** |
| `reliable` | invariato |
| Modello di produzione | **window_delta_mlp** — Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo (no valle, no chiusa) |

Nessun work package ha battuto la MLP di produzione su rising RMSE / skill eventi @120′.

## Tabella unica WP0–WP6

| WP | Esito | Metrica chiave @120′ |
|---|---|---|
| WP0 protocollo | fatto | `eval-table.md` |
| WP1 diagnosi | fatto | H1/H2/H4/H6/H7/H9/H10 confirmed; H3/H8/H11 not clear (`wp1-diagnosis.md`) |
| Stazioni monte/valle | **no adopt** | baseline rising **0.105**; Bagnetto pari; valle peggiora (Dosso 0.133) |
| WP2 hazard+gate | **no adopt** | rising 0.109 vs prod **0.103** (decode 35 vs M9 193) |
| WP2 weir chiusa | **no adopt** | valEvt skill 0.819 vs baseline 0.834 |
| WP2 ew4 / lag-align / rain | già chiusi | nessun promote |
| WP3 routing in Q | **gate FAIL** | skill eventi **−10.2** vs ARX **0.64** |
| WP4 EnKF | saltato | regola: WP3 ≤ ARX |
| WP5 ibrido | stop early | ARX+routing ≈ ARX (0.641); residual peggiore |
| WP6 | questo file | adopt false |

### Dettaglio stazioni (event eval ≥ 2026-01-01)

| Variante | RMSE | skill | rising | falling |
|---|---:|---:|---:|---:|
| baseline | **0.080** | **0.720** | **0.105** | **0.019** |
| +Bagnetto | 0.080 | 0.719 | 0.105 | 0.021 |
| +far monte | 0.081 | 0.718 | 0.105 | 0.021 |
| +tributari | 0.082 | 0.716 | 0.106 | 0.020 |
| Dosso / +chain | 0.096–0.105 | 0.63–0.66 | 0.124–0.136 | 0.029–0.034 |

## Ipotesi

| ID | Stato |
|---|---|
| H1, H2, H4, H6, H7, H9, H10 | confirmed |
| H3, H5, H8, H11 | not clear |

## Codice utile (nessun peso sperimentale in repo)

- `tools/fit-crossing-hazard.py`, `tools/fit-routing-q-cento.py`, `tools/fit-wp5-hybrid.py`
- `tools/fit-ml-cento.py` (`--extra-upstreams`, `--acceptors`, `--chiusa-weir`)
- `predict-transfer.js` (hazard + `decodeGateP`; soft-load se il JSON esiste)

## Ciclo post-WP6 (2026-11-16)

| Voce | Esito |
|---|---|
| Protocollo NSE/KGE su eventi | fatto (`eval-protocol.js` / `protocol-mode.js`) |
| Retrain leakage-free (`--holdout-from 2026-01-01`) | fatto — flat ≈ prod |
| Lookback-τ (`--lookback-tau`, L_i=τ_i+4 h) | **no adopt** — pari a flat; CI skill/rising sovrapposti (`ab-lookback-tau.md`) |
| Griglia φ≤2τ (`--tau-frac`, n=8/16, signal level/delta) | **no adopt** — ≈ flat; densità e delta-only irrilevanti (`ab-tau-fraction.md`) |
| Feature solo-delta flat (`--feature-signal delta`) | **no adopt** — ≤ flat level+Δ; rising/lag non migliorano (`ab-feature-delta.md`) |
| Soft-adopt / `reliable` | invariato |

## Raccomandazione

Tenere la MLP di produzione. Per un prossimo ciclo: **rating Q(h) ufficiale se disponibile**; routing open-loop che batta ARX prima di EnKF/ibridi. Lookback-τ, griglia φ/τ e retrain calendar cut già chiusi senza promote.
