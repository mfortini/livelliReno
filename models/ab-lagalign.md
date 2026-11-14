# A/B: feature upstream allineate al ritardo τᵢ (Cento MLP)

Data: 2026-11-12. Ipotesi: la previsione "insegue" l'osservato perché le
variazioni a monte non sono sfruttate — la MLP `window_delta_mlp` usa una
finestra piatta identica per ogni orizzonte e deve imparare da sola il tempo
di propagazione.

## Setup

- τᵢ stimati sui soli mesi di train (`ft.estimate_lag`, max 48 step):
  Forcelli 480′ (r=0.86), Sostegno Reno 150′ (0.88), Bonconvento 360′ (0.99),
  Casalecchio tiro a volo 660′ (0.96).
- `shift`: lag k → x(t+h−τᵢ−k·step) (la finestra upstream si sposta indietro).
- `augment`: finestra piatta invariata + blocco `a*` di 16 step attorno a
  x(t+h−τᵢ). Shift <0 (h>τᵢ) → miss=1, nessun peek nel futuro.
- Stessi split/eventi/iperparametri del baseline `ml-cento-compare-novalle-6h`.
- Metrica di fase nuova in backtest.js: `phaseLagMin` = shift dell'osservato
  che minimizza l'RMSE del candidato (range ±120′, satura).

## Risultati fit (validateEvents skill / holdoutEvents vs poly2)

| h   | flat val | shift val | augment val | flat hold | shift hold | augment hold |
|-----|----------|-----------|-------------|-----------|------------|--------------|
| 30  | 0.871    | 0.820     | 0.863       | 0.722     | 0.422      | 0.606        |
| 120 | 0.820    | 0.648     | 0.813       | 0.748     | 0.428      | 0.819        |
| 240 | 0.815    | 0.606     | 0.819       | 0.908     | 0.703      | 0.898        |
| 360 | 0.799    | 0.692     | 0.780       | 0.915     | 0.837      | 0.904        |

## Risultati backtest (--compare-candidates, skill vs persistenza)

| candidato       | skill120 | skill240 | skill360 | lag120 | lag360 | skill120 highAcc |
|-----------------|----------|----------|----------|--------|--------|------------------|
| persist         | 0        | 0        | 0        | 120′   | 120′   | 0                |
| mlp_prod        | 0.690    | 0.629    | 0.582    | 30′    | 120′   | 0.754            |
| mlp_lagshift    | 0.631    | 0.544    | 0.511    | 30′    | 120′   | 0.715            |
| mlp_lagaugment  | 0.697    | 0.631    | 0.576    | 30′    | 120′   | 0.778            |
| mlp_phase30     | 0.698    | 0.654    | –        | 30′    | –      | 0.722            |
| mlp_phase60     | 0.591    | 0.644    | –        | −30′   | –      | 0.568            |
| mlp_valle_ema065| 0.383    | –        | –        | 60′    | –      | 0.488            |
| seq_gru         | 0.676    | –        | –        | 30′    | –      | 0.812            |

## Conclusioni

1. **Il ritardo esiste ed è misurato**: ~30′ @2h e ~120′ @6h per la MLP in
   produzione (satura a ±120′, quindi @6h è ≥).
2. **Non è un problema di feature mancanti**: allineare la finestra a τᵢ non
   riduce il lag (augment ~pari, shift peggiore). La finestra piatta da 24 h
   contiene già l'onda in arrivo; il collo di bottiglia è altrove.
3. Cause residue del ritardo:
   - regressione alla media condizionata → sottostima dei fronti di piena;
   - celerità variabile: τ fisso da cross-correlazione è sbagliato in regime
     di piena (l'onda viaggia più veloce);
   - smoothing EMA: 0.65 aggiunge ~30′ (prod usa smooth=0, ridge EMA 0.15);
   - oltre τᵢ l'informazione di monte è fisicamente esaurita.
4. `phase30` (relabel −30′) migliora lievemente RMSE @120–240′: c'è un bias
   temporale sistematico ~30′. Effetto piccolo; non adottato.
5. **Non promuovere** lagshift/lagaugment. Il codice resta disponibile
   (`--lag-align {shift,augment}`, recipe `upstreamLag*` nel runtime).

## Seconda tornata: loss orientate al timing (2026-11-12 pomeriggio)

Ipotesi: il ritardo viene dalla media condizionata, non dalle feature.
Testate due varianti di loss (stessi split/feature del baseline flat):

- `--quantile 0.75`: pinball loss sul quantile alto — proxy di crossing-time.
- `--event-weight 4`: i campioni |Δy| ≥ 0.10 m pesano 4× nella MSE di train.

Fit (validateEvents skill): ew4 **0.835** @120 / 0.802 @360 (baseline 0.820/0.799);
q75 0.781 @120 / 0.778 @360 (atteso più basso: q75 ≠ media).

Backtest `--compare-candidates` (skill vs persistenza, phaseLagMin):

| candidato  | skill120 | skill240 | skill360 | lag120 | lag360 | rmse120 | highAcc120 |
|------------|----------|----------|----------|--------|--------|---------|------------|
| mlp_prod   | 0.690    | 0.629    | 0.582    | 30′    | 120′   | 0.0576  | 0.754      |
| mlp_ew4    | 0.699    | 0.630    | 0.584    | 30′    | 120′   | 0.0559  | 0.773      |
| mlp_q75    | 0.669    | 0.625    | 0.581    | 30′    | 120′   | 0.0615  | 0.690      |

## Conclusioni aggiornate

6. **Nessuna variante riduce il phaseLagMin**: 30′ @2h / 120′ @6h identici per
   prod, shift, augment, phase, q75, ew4. Con target Δy(t→t+h) il ritardo
   residuo è un pavimento strutturale della formulazione, non una carenza di
   feature, pesi o quantili. La metrica satura a ±120′: @6h è un limite inferiore.
7. ew4 migliora marginalmente RMSE (−3% @120′) e regime-high (+2 pt) ma non il
   timing → **non promosso** (convenzione: vincitore chiaro). Pesi in
   `models/cento-dnn-ew4.json`; q75 in `models/cento-dnn-q75.json`.
8. Per spostare davvero il ritardo serve cambiare formulazione del target:
   classificatore/regressore sul **tempo di attraversamento di una soglia**
   (crossing-time), oppure correzione di stato online (assimilazione) che
   ri-fasi la curva sull'ultimo residuo — non solo in livello.
