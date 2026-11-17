# Esperimento: filtri di Kalman (KF / UKF / EnKF) — Cento

_Data: 2026-11-17. Esito: **skill predittivo modesto, molto sotto la MLP;
la famiglia resta però utile come stimatore di incertezza di stato.**_

## Modello

Stato per stazione `s = [livello, velocità]` (velocità = incremento per passo
da 30′). Transizione stimata sui soli mesi di train:

```
y⁺ = y + v
v⁺ = v · (rho + gamma·(y − ymin))        # celerità crescente col livello
```

`(rho, gamma)` per OLS sulle coppie one-step; `Q` = covarianza dei residui;
`R` = rumore di misura (selezionato su griglia {1e-4,1e-3,1e-2} con KF,
condiviso da UKF/EnKF). Filtri: KF lineare, UKF (sigma-point, n=2 o 10),
EnKF stocastico (N=200, osservazioni perturbate).

**Fase A** (`experiment-kalman.py`): filtri indipendenti per stazione,
senza ingressi.

**Fase B**: ingressi = livelli delle stazioni upstream filtrate (KF) +
`rain6h`/`rain24h`; `B` stimato per LS sui residui; **varianza degli input**
propagata come `B·Γ·Bᵀ` con `Γ` = varianza filtrata/prevista upstream
(pioggia ~esatta). La varianza di previsione è quindi attribuita a: R
(misura), Q (processo) e Γ (input), e riportata come varianza marginale.

**Fase G** (`experiment-kalman-global.py`): stato congiunto 10-D di tutte le
stazioni; dinamica block-diagonale ma **Q piena** con covarianze di processo
incrociate stimate dai residui → la misura di una stazione aggiorna lo stato
delle altre. `--couple` aggiunge coppiamento ritardato
`v_i⁺ += Σ c_e·Δz_src(t−τ_e)` con τ dai lag mediani DTW
(Casalecchio→Bonconvento 270′, Bonconvento→Sostegno 210′,
Sostegno→Cento ~510′ composito, Forcelli→Cento 480′) e coefficienti `c`
stimati per LS — la piena si propaga deterministicamente monte→valle.

## Risultati — skill evento di validazione @120′ (vs poly2)

| Modello | valEvt@120 | holdEvt@120 | valEvt@360 | holdEvt@360 |
|---|---:|---:|---:|---:|
| MLP prod (riferimento) | ~0.66* | ~0.6* | ~0.62* | ~0.7* |
| KF-A / UKF-A / EnKF-A | 0.144 / 0.145 / 0.142 | −1.11 | 0.051 / 0.050 / 0.049 | 0.35 |
| KF-B / UKF-B / EnKF-B | 0.271 / 0.271 / 0.271 | −0.83 | 0.182 | 0.43 |
| KF-G / UKF-G / EnKF-G | 0.148 | −1.10 | 0.053 | 0.35 |
| KF-GC / UKF-GC / EnKF-GC | 0.165 / 0.165 / 0.168 | −1.07 | 0.077 / 0.076 / 0.079 | 0.36 |

\* stime dal trainer/backtest precedente, stessa splits/origini.

## Lettura

- **KF ≈ UKF ≈ EnKF** ovunque: la non linearità `gamma·(y−ymin)` è troppo
  debole per distinguere i filtri; lo UKF collassa al KF, l'EnKF alla media.
- **Ingressi (B) > nessun input (A)**: +0.13 di skill@120 — le stazioni
  upstream portano informazione reale, ma serve la MLP per sfruttarla.
- **Globale neutro vs per-stazione**: la Q incrociata migliora la stima di
  stato (varianza ridotta) ma non la previsione — atteso, perché la
  propagazione monte→valle è solo accoppiamento di rumore contemporaneo.
- **Coppia ritardata (GC)**: piccolo miglioramento reale
  (+0.02 valEvt@120, +0.025@240/360 vs G) — la fisica di propagazione
  aiuta, ma i coefficienti stimati sono piccoli (es. Forcelli→Cento 0.013)
  perché il segnale arriva diluito e disperso; il lag singolo non basta.
- **Incertezza**: varianza di previsione mediana @120 ≈ 0.0165 (σ≈0.13 m),
  @360 ≈ 0.105 (σ≈0.32 m) — bande molto più larghe dell'RMSE MLP (0.047):
  oneste ma poco informative operative.

## Conclusioni

1. Il filtro di Kalman come **previsore** non è competitivo con la MLP:
   lo stato [livello, velocità] non ha memoria sufficiente e la dinamica
   stimata è troppo semplice per la piena.
2. Come **stimatore di stato con incertezza attribuita** (R/Q/Γ) è corretto
   e funziona: se servisse una nowcast stima con bande, la fase B è il
   punto di partenza.
3. La propagazione con lag fisso aiuta poco: servirebbe la deformazione
   temporale continua (DTW causale) — già scartata diagnostica — o una
   rappresentazione più ricca (es. livello idrometrico normalizzato,
   portata stimata).

## File

- `tools/experiment-kalman.py`, `tools/experiment-kalman-global.py`
- `models/experiments/kalman.json`, `kalman-global.json`,
  `kalman-global-coupled.json`

## Riproduzione

```bash
uv run python tools/experiment-kalman.py --out models/experiments/kalman.json
uv run python tools/experiment-kalman-global.py --out models/experiments/kalman-global.json
uv run python tools/experiment-kalman-global.py --couple \
  --out models/experiments/kalman-global-coupled.json
```
