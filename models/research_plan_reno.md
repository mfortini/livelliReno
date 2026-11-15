# Research Plan: Flood Level Forecast for the Reno River at Cento

Revision 4, 2026-11-15. This revision includes the result of the A/B test M9, the downstream (backwater) effect and the use of geometry data.

Language: ASD-STE100 rules applied manually. No STE checker was used.

## 1. Purpose

This plan has two functions:

- It finds why the physical filter models (Kalman filter, EnKF, electrical equivalent) give larger errors than the MLP.
- It gives a sequence of tests to decrease the error and the phase lag on the rising limb at Cento.

The primary question is:

> Can a model with physical structure generate the flood trajectory with the correct timing, and be better than the production MLP?

## 2. Known Data and Assumptions

### 2.1 Known data

| Item | Value |
|---|---|
| Target | Level at Cento |
| Upstream gauges | Forcelli, Sostegno Reno, Bonconvento, Casalecchio tiro a volo, Casalecchio chiusa |
| Period | 2023-01 to 2026-11 (ARPAE) |
| Event catalog | 913 events, 65 at Cento |
| Production model | MLP with quantile bands (`cento-dnn-bands.json`) |
| Production result at 120 min (test) | RMSE 0.080, skill 0.724, phase lag +30 min, rising RMSE 0.103, falling RMSE 0.022 |
| M9 result at 120 min (test) | RMSE 0.259, skill 0.110, phase lag 0 min, rising RMSE 0.298, falling RMSE 0.189 |

### 2.2 Assumptions

Correct these assumptions before you start.

| Item | Assumption |
|---|---|
| Time step | 30 minutes (the phase lag values are multiples of 30) |
| Rain data | Not used at this time |
| Downstream gauges | Not used at this time |
| Filter models | Linear state-space model or RC network, with levels as states |
| Filter training | Parameters adjusted on the one-step error or on the likelihood |
| Travel time | The travel time from the nearest upstream gauge to Cento is longer than 120 minutes |

## 3. Interpretation of the M9 Result

1. The production error is on the rising limb. The rising RMSE is five to nine times the falling RMSE.
2. The crossing-time model moves the phase in the correct direction. Thus the upstream data contains the timing information.
3. A model trained on the squared error gives the conditional mean. When the arrival time is not certain, the mean of many possible fronts is a slow, late front.

   This is a probable cause of the +30 minute lag.
4. The RMSE gives a double penalty to a sharp front with a small timing error. A smooth, late front gets a smaller RMSE.

   Thus the RMSE comparison is not neutral for a trajectory at the mode of τ.
5. The falling RMSE of M9 on the test set is 0.189, against 0.022. The decode operates when no flood front is present, or the flood-only model lost the recession dynamics.
6. A pinball loss on τ with censoring at 6 hours gives biased quantiles. The model has no explicit probability of "no crossing".

   The level-lift then makes a ramp for events that do not cross the threshold. This agrees with the overshoot.
7. Casalecchio chiusa is a level upstream of a weir. Large discharge changes give small level changes there.

   As a raw level, it adds little information to Casalecchio tiro a volo.
8. One step of 30 minutes is the resolution of the phase lag. A difference of 30 minutes on few events is not statistically reliable.

Conclusion: the M9 conclusion is correct. The timing must be part of the trajectory model, not a correction after the model.
A routing model with a wave speed that changes with discharge is a trajectory generator of this type.

## 4. Hypotheses

| ID | Hypothesis |
|---|---|
| H1 | The error of the filter models comes from the process model, not from the filter. A filter cannot repair an incorrect model structure. |
| H2 | Level is not a conserved quantity. Discharge and volume are conserved. An RC network on levels does not obey mass balance. |
| H3 | The wave speed increases with discharge. A model with a constant delay is too slow during floods. |
| H4 | An RC stage gives diffusion, not transport delay. A river reach is primarily a delay. |
| H5 | Lateral inflow (rain, tributaries without gauges) is not in the model. The MLP estimates it indirectly from the recent level changes. |
| H6 | Constant noise covariances (Q, R) are adjusted to the steady state. During floods the filter gain is then too small. |
| H7 | The filter is trained for one step. The MLP is trained directly for the forecast horizon. The comparison is not equal. |
| H8 | The phase lag of the MLP comes from the squared-error loss with a timing that is not certain (Section 3, item 3). |
| H9 | The RMSE hides the value of a correct timing (Section 3, item 4). |
| H10 | The M9 overshoot comes from the censoring of τ and from a decode without a gate (Section 3, items 5 and 6). |
| H11 | The downstream level controls the discharge at Cento (backwater). The discharge is a function of the level and of the water-surface slope, not of the level only. |

Note about H11: the Reno at Cento is a lowland river with a small bed slope. A high downstream level decreases the slope.
Then the same inflow gives a higher level at Cento, and the wave moves more slowly.
Hydraulic structures downstream of Cento can also change the level at Cento.
Models with only upstream inputs cannot show this effect.

Note about H1: the steady-state Kalman predictor of a linear model is equivalent to an ARMAX model.
Thus a linear Kalman filter cannot be better than a good linear regression on past levels.
The MLP contains that regression as a special case.

## 5. Work Packages

Do the work packages in sequence. Each work package has a gate.

### WP0 — Additions to the evaluation protocol (small effort)

The event catalog, the skill score, the phase lag and the rising and falling RMSE are available. Add these items:

1. Add a linear ARX baseline on past levels. Persistence is already the reference of the skill score.
2. Add the threshold-crossing score (CSI: hits, misses, false alarms) for each alert threshold.
3. Add the error of the crossing time in minutes, for events that cross.
4. Add the CRPS or the pinball score for the level fan and for τ.
5. Report each metric for each event. Calculate a bootstrap interval across events.
6. Calculate the phase lag with a sub-step interpolation of the cross-correlation peak.
7. Use leave-one-event-out validation, or validation by year, on the 65 events at Cento.

Gate: one table compares all models by horizon, with intervals. An A/B decision uses the intervals.

### WP1 — Diagnosis (small effort)

1. Calculate the cross-correlation lag from each upstream gauge to Cento for low, medium and high levels.

   A lag that decreases with level confirms H3. Record the smallest lag: it is the limit of the forecast horizon without rain.
2. Plot the level at Cento against the level at each upstream gauge for three floods. A loop shows storage and hysteresis.
3. Run the process model of the filter in open loop, without measurement updates. Record the flood error. This tests H1.
4. Simulate the step response of the RC model. Compare it with a measured flood wave. This tests H4.
5. Examine the filter innovations. Compare the variance in steady state and in flood. This tests H6.
6. Make sure that you compare predictions at 120 minutes, not filtered estimates.
7. For the EnKF, record the ensemble spread during floods. A spread that is too small shows filter collapse.
8. For H8, train the production MLP with the absolute-error loss or with the median quantile only. Compare the phase lag.
9. For H10, count the M9 decode operations on samples without a crossing. Calculate the RMSE with and without these samples.
10. For H11, select the nearest gauge downstream of Cento. Plot the production residuals against the downstream level and against the level difference (Cento minus downstream).

    A trend in the plot confirms H11. Do this separately for the rising limb.
11. For H11, examine the events with a second flood wave. A larger error on the second wave, when the downstream level is still high, also confirms H11.

Gate: for each hypothesis, write "confirmed", "rejected" or "not clear".

### WP2 — Improvement of the data-driven model (small effort)

1. Replace flood-only training with weights on flood samples. Keep the recession and the steady state in the training set.
2. Replace the pinball loss on τ with a discrete-time hazard model. For each 30 minute step, the model gives the probability of the first crossing.

   The censoring is then correct, and the model gives P(no crossing in 6 hours).
3. Operate the decode only when P(crossing) is above a limit. Adjust the limit on the CSI.
4. Convert Casalecchio chiusa to a discharge feature with the weir law `Q = c · (h − h_crest)^1.5`. Do the ablation again.
5. Add inputs: rate of change of each gauge, and level differences between adjacent gauges.
6. Test a loss with a shape term and a time term (DILATE or soft-DTW) for the trajectory output.
7. Test a generative decode: sample τ from the hazard model, then generate one trajectory for each sample.

   The fan is the set of trajectories. The "most probable" curve is the trajectory at the mode of τ.
8. Add rain inputs, if you can get them: rain gauges or radar, and the accumulated rain of the last 1 to 10 days.

   Rain is necessary for horizons longer than the smallest travel time.
9. Add the downstream gauge as an input: its level, its rate of change, and the level difference (Cento minus downstream).

   Do an ablation. Calculate the gain separately for events with a high and with a low downstream level at the start.
    The downstream level also follows the past level at Cento. A gain only in the high downstream class shows a real backwater effect.
10. If you know the state of the downstream structures (open or closed), add it as an input.

Gate: record the flood metrics of the best data-driven model. This is the reference for WP3 to WP5.

### WP3 — Physical model with correct structure (medium effort)

1. Convert level to discharge with a rating curve for each gauge: `Q = a · (h − h0)^b`.

   Use official rating curves if they are available. If not, make `a`, `h0`, `b` trainable parameters.
2. Model each reach as a cascade of `n` nonlinear reservoirs: `dS/dt = Q_in + Q_lat − Q_out`, with `Q_out = k · S^m`.

   The exponent `m > 1` makes the wave faster at high discharge. This is the answer to H3 and H4.
3. As an alternative, use Muskingum-Cunge routing with parameters that change with discharge.
4. Add the lateral inflow `Q_lat` as a state with random-walk dynamics. This is the answer to H5.
5. Write the model in a differentiable framework (PyTorch or JAX). Export the parameters to JSON for the JavaScript runtime.
6. Train the parameters on the multi-step forecast error, with the same horizons as the MLP. This is the answer to H7.
7. Keep the parameters in physical limits (positive values, limited exponents).
8. If WP1 confirms H11, replace the outflow law of the last reach with a law that uses the level difference:

   `Q_out = K(h) · sqrt(max(h − h_down + Δz, 0) / L)`, with `K(h) = a · (h − h0)^b`.
    `Δz` is the bed level difference and `L` is the reach length. This is a diffusive-wave model.
    The rating curve at Cento then has two inputs (stage-fall-discharge rating).
9. For the forecast, the model must know the future downstream level. Use one of these options, in sequence:

   persistence of the downstream level; one more reach downstream of Cento in the model; a small data-driven forecast of the downstream level.
10. Use geometry data to set initial values and limits of the parameters. Do not use it as a model input, because it is constant.
    - Gauge datum (elevation of the gauge zero): converts each level to an absolute water-surface elevation. It gives `Δz`.
      - River centerline: gives the reach length `L`.
      - DTM of the floodplain between the levees: gives the bankfull level and the storage volume as a function of level.
      - DTM of the levee crest: gives the maximum level before overflow.
11. If the DTM shows a floodplain, use a storage law with two parts: main channel below the bankfull level, main channel and floodplain above.

    Compare the bankfull level from the DTM with the level where the measured lag changes (WP1 item 1).
12. A LiDAR DTM does not give the bed elevation below the water surface. Do not calculate the channel section from the DTM only.

Gate: on flood events, the open-loop model has a phase lag near zero and an error smaller than the linear model.

### WP4 — State estimation on the new model (medium effort)

1. Use an EKF, a UKF or an EnKF on the WP3 model.
2. Make the process noise increase with discharge (multiplicative noise). This is the answer to H6.
3. Estimate the states in discharge or in log-discharge, to keep them positive.
4. For the EnKF, use 50 to 100 members and apply inflation. Monitor the spread.
5. Adjust the noise parameters on the multi-step forecast error or on the CRPS, not on the one-step likelihood.
6. Use the ensemble as the level fan. Compare the interval coverage with the production bands.

Gate: compare with the WP2 reference on flood metrics. Go to WP5 in all cases.

### WP5 — Hybrid models (medium effort)

Test these options in sequence. Stop when the gain becomes small.

1. Physical features: give the output of the routing model as an input to the production MLP.
2. Error correction: train a small ARX or MLP on the residuals of the physical model.
3. Neural lateral inflow: a small network calculates `Q_lat` from rain and from recent level changes.
4. Learned gain: replace the Kalman gain with a small recurrent network (KalmanNet approach).

Gate: select the model with the best flood metrics and with reliable intervals.

### WP6 — Final report (small effort)

1. Record the results of each work package in one table.
2. Write which hypotheses are confirmed.
3. Give a recommendation for the production model and for the `adopt` flag.

## 6. Decision Rules

| Condition | Decision |
|---|---|
| WP1 confirms H8 | The loss is a cause of the lag. Give priority to WP2 items 6 and 7. |
| WP1 confirms H10 | Repair the censoring and add the gate (WP2 items 2 and 3) before a new A/B test of the crossing time. |
| WP1 confirms H1, H2 or H3 | The linear model is the cause of the filter error. Continue with WP3. |
| WP1 confirms H11 | Add the downstream inputs (WP2 item 9). Use the level-difference outflow law in WP3 (items 8 and 9). |
| WP1 rejects H1 to H4 | Look for an implementation error in the filter before you continue. |
| WP3 open-loop model is not better than ARX on floods | Stop the physical approach. Continue with WP2 and WP5 item 1. |
| A model has a better CSI and crossing-time error, with an equal or worse RMSE | Do not reject it on the RMSE only. Show it as the "most probable" curve and keep the production median. |
| WP4 or WP5 is better than WP2 on flood metrics, with intervals that do not overlap | Use the physical or hybrid model. |

## 7. Risks

| Risk | Mitigation |
|---|---|
| 65 events at Cento are few | Report the result for each event, with bootstrap intervals. Use the other catalog events for pre-training. |
| Largest flood is above the training range | Physical structure extrapolates better than an MLP. Test with the largest event as the test set. |
| Rating curves are not known or change with time | Train the rating parameters. Examine the residual drift after each large flood. |
| Hydraulic structures or levee failures change the dynamics | Mark these periods. Remove them or model them separately. |
| Sensor faults during floods | Add a quality check. Let the filter operate with missing measurements. |
| Parameters are not identifiable | Fix `n` and `m` to physical values first. Release them one at a time. |

## 8. Expected Result

The most probable sequence of gains, from largest to smallest:

1. Correct censoring, gate on the decode, and metrics that do not give a double penalty (WP0, WP2).
2. Nonlinear routing in discharge space with multi-step training (WP3), used directly or as a feature (WP5 item 1).
3. Rain input, for horizons longer than the travel time (WP2).
4. State estimation with noise that changes with discharge (WP4).

The filter algorithm itself gives the smallest gain. The model structure, the loss and the inputs give the largest gain.
