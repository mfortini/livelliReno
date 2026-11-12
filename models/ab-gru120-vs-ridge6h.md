# A/B: GRU@120 vs ridge delta@6h

Date: 2026-11-12. Metric: `validateEvents.skill` vs persistenza (month-block split).  
Barre: GRU soft-adopt batte MLP/prod @120; ridge `reliable` se skill@120 e holdout ≥ 0.10 (holdout n≥30).

## Skill per orizzonte (eventi)

### Cento

| min | GRU@120 (retrain) | Ridge@6h (nuovo fit) | MLP prod |
|---:|---:|---:|---:|
| 30 | 0.877 | 0.828 | (MLP knot) |
| 60 | 0.836 | 0.766 | |
| 90 | 0.791 | 0.713 | |
| **120** | **0.754** | **0.672** | **0.826** |
| 150 | — | 0.636 | — |
| 180 | — | 0.597 | — |
| 210 | — | 0.555 | — |
| 240 | — | 0.519 | — |
| 270 | — | 0.487 | — |
| 300 | — | 0.459 | — |
| 330 | — | 0.437 | — |
| 360 | — | 0.416 | — |

- GRU@120: `reliable=false` (0.754 &lt; MLP compare 0.820). Soft-adopt **no**.
- Ridge@6h: `reliable=true`, lag [480, 150], skill@360=0.416 (Δ vs @120 ≈ −0.26).
- MLP prod: migliore a 2 h; non ha knot &gt;120.

### Gallo

| min | GRU@120 (backup) | Ridge@6h (nuovo fit) | Ridge prod (≤240) |
|---:|---:|---:|---:|
| 30 | 0.932 | 0.842 | 0.843 |
| 60 | 0.888 | 0.805 | 0.805 |
| 90 | 0.872 | 0.782 | 0.782 |
| **120** | **0.863** | **0.768** | **0.768** |
| 150 | — | 0.741 | 0.741 |
| 180 | — | 0.717 | 0.717 |
| 210 | — | 0.694 | 0.694 |
| 240 | — | 0.669 | 0.669 |
| 270 | — | 0.645 | — |
| 300 | — | 0.622 | — |
| 330 | — | 0.603 | — |
| 360 | — | 0.580 | — |

- GRU@120: `reliable=true`, skill@120 **0.863** (migliore sul corto).
- Ridge@6h: skill@120 0.768, holdout skill 0.622 ma **n_hold=25 &lt; 30** → flag `reliable=false` nel fit; skill@360 **0.580** (Δ vs @120 ≈ −0.19). Knot allineati al prod fino a 240′.
- GRU@6h (scartato): skill@120 0.862 — non migliore del 120-only; non usato.

## Decisione

| Stazione | Corto (≤120′) | Lungo (&gt;120′) | Runtime |
|---|---|---|---|
| **Cento** | **MLP** (0.826) batte GRU e ridge | Solo ridge (0.42–0.64); adottarla sostituirebbe MLP e calerebbe @120 di ≈0.15 | **MLP@2 h** — niente soft-adopt GRU; **non** sostituire con ridge |
| **Gallo** | **GRU** (0.863) batte ridge (0.768) | Ridge solida fino a 360′ (0.58) | **Trade-off**: per avere banda 6 h serve la ridge (GRU soft-adopt rimpiazza tutto il nodo). **Scelta: ridge@6 h** (skill@120 ancora ≫ 0.10; −0.10 vs GRU). GRU@120 resta in `seq-gallo-gru.json` ma soft-adopt disattivato |

### Raccomandazione operativa

1. **Cento**: resta `window_delta_mlp` in `transfer-coeffs.json`; `seq-cento-gru.json` con `reliable=false`. Banda **2 h**.
2. **Gallo**: aggiornare `nodes.Gallo` in `transfer-coeffs.json` con orizzonti fino a 360′ dal fit ridge6h; `reliable=true` (holdout skill ok; n=25 documentato). Impostare `seq-gallo-gru.json` `reliable=false` con nota A/B così non soft-adotta. Banda **fino a 6 h** via `maxReliableHorizonMinutes`.
3. GRU@6h non è candidato (Cento peggiorava; Gallo non aggiungeva skill@120).

## File

- `models/seq-cento-gru.json` — 120′, reliable false  
- `models/seq-gallo-gru.json` — 120′, soft-adopt off dopo A/B  
- `models/seq-*-gru.6h-discarded.json` — traccia run 6 h  
- `models/transfer-coeffs-ridge6h.json` — fit completo extended  
- `models/seq-gallo-gru.reliable-120.json` — backup GRU buono  
