'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const EP = require('../eval-protocol.js');

function seriesFrom(startMs, values, stepMs = EP.STEP_MS) {
    const times = [], vals = [], byTime = new Map();
    for (let i = 0; i < values.length; i++) {
        const t = startMs + i * stepMs;
        times.push(t);
        vals.push(values[i]);
        byTime.set(t, values[i]);
    }
    return { times, values: vals, byTime };
}

test('detectFloodEvents finds a clear rise-and-peak event', () => {
    const start = Date.parse('2026-01-01T00:00:00Z');
    const values = [];
    // 48h calm at 0, then rise to 3 over 12h, peak, fall
    for (let i = 0; i < 96; i++) values.push(0);
    for (let i = 1; i <= 24; i++) values.push(i * 0.125); // → 3.0
    for (let i = 1; i <= 24; i++) values.push(3 - i * 0.1);
    for (let i = 0; i < 48; i++) values.push(0.5);
    const series = seriesFrom(start, values);
    const events = EP.detectFloodEvents(series, {
        minRiseM: 0.5,
        minPeakLevel: 1.0,
        lookbackHours: 48,
        peakHalfWinHours: 4,
        mergeHours: 72
    });
    assert.ok(events.length >= 1);
    assert.ok(events[0].peakLevel >= 2.9);
    assert.ok(events[0].riseM >= 2.5);
});

test('detectFloodEvents merges peaks within 72h', () => {
    const start = Date.parse('2026-02-01T00:00:00Z');
    const values = [];
    for (let i = 0; i < 48; i++) values.push(0);
    // first peak
    for (let i = 1; i <= 10; i++) values.push(i * 0.3);
    for (let i = 1; i <= 10; i++) values.push(3 - i * 0.2);
    // second peak 24h later (within merge)
    for (let i = 0; i < 48; i++) values.push(1);
    for (let i = 1; i <= 10; i++) values.push(1 + i * 0.4);
    for (let i = 1; i <= 20; i++) values.push(5 - i * 0.2);
    for (let i = 0; i < 48; i++) values.push(0);
    const series = seriesFrom(start, values);
    const events = EP.detectFloodEvents(series, {
        minRiseM: 0.5,
        minPeakLevel: 2.0,
        mergeHours: 72
    });
    assert.equal(events.length, 1);
    assert.ok(events[0].peakLevel >= 4.5);
});

test('poly2At uses 6 points and extrapolates', () => {
    const t0 = Date.parse('2026-03-01T12:00:00Z');
    const byTime = new Map();
    // quadratic-ish: y = 0.01 x^2 at x=0..5 ending at t0
    for (let k = 5; k >= 0; k--) {
        const x = 5 - k;
        byTime.set(t0 - k * EP.STEP_MS, 0.01 * x * x);
    }
    const y1 = EP.poly2At(byTime, t0, 1, 6);
    const y4 = EP.poly2At(byTime, t0, 4, 6);
    assert.ok(Number.isFinite(y1));
    assert.ok(Number.isFinite(y4));
    assert.ok(y4 > y1);
});

test('bootstrapSkillCI is deterministic with fixed seed', () => {
    const events = [
        { n: 10, seModel: 2, seBase: 8 },
        { n: 12, seModel: 3, seBase: 9 },
        { n: 8, seModel: 1.5, seBase: 6 },
        { n: 15, seModel: 4, seBase: 10 }
    ];
    const a = EP.bootstrapSkillCI(events, { seed: 1, draws: 200 });
    const b = EP.bootstrapSkillCI(events, { seed: 1, draws: 200 });
    assert.deepEqual(a, b);
    assert.ok(Number.isFinite(a.skill));
    assert.ok(a.lo <= a.skill && a.skill <= a.hi);
    const c = EP.bootstrapSkillCI(events, { seed: 2, draws: 200 });
    // Different seed may differ; skill point estimate must match
    assert.equal(a.skill, c.skill);
});

test('formatEvalMarkdown includes station and model rows', () => {
    const md = EP.formatEvalMarkdown({
        from: '2026-01-01',
        to: '2026-11-01',
        seed: 1,
        stations: {
            Cento: {
                prod: {
                    120: {
                        skillAllPersist: 0.4,
                        skillEventPersist: 0.5,
                        skillEventPersistCI: { lo: 0.3, hi: 0.6 },
                        skillEventPoly2: 0.1,
                        skillEventRidge: 0.05,
                        skillEventArx: 0.2,
                        risingRmse: 0.1,
                        fallingRmse: 0.02,
                        phaseLagMin: 28.5,
                        csi: { yellow: 0.7, orange: 0.5, red: 0.2 },
                        crossingTimeErrMed: 15,
                        pinballEvent: 0.04,
                        peakLevelErrMed: -0.2,
                        peakTimeErrMed: 30,
                        nEvents: 3
                    }
                }
            }
        },
        generatedAt: '2026-11-13T00:00:00Z'
    });
    assert.match(md, /## Cento/);
    assert.match(md, /prod/);
    assert.match(md, /0\.50/);
    assert.match(md, /CSI/);
    assert.match(md, /NSE evt/);
    assert.match(md, /KGE evt/);
});

test('fitArx recovers a linear relation', () => {
    const x = [];
    const y = [];
    for (let i = 0; i < 40; i++) {
        const a = i * 0.1;
        const b = Math.sin(i);
        x.push([a, b]);
        y.push(2 + 3 * a - 1.5 * b);
    }
    const model = EP.fitArx({ x, y }, { ridge: 1e-9 });
    assert.ok(model);
    assert.ok(Math.abs(model.intercept - 2) < 0.05);
    assert.ok(Math.abs(model.coef[0] - 3) < 0.05);
    assert.ok(Math.abs(model.coef[1] + 1.5) < 0.05);
    const pred = EP.predictArx(model, [1, 0]);
    assert.ok(Math.abs(pred - 5) < 0.05);
});

test('csiFromCounts and pinballScore basics', () => {
    assert.equal(EP.csiFromCounts(2, 1, 1), 0.5);
    assert.equal(EP.csiFromCounts(0, 0, 0), null);
    assert.ok(EP.pinballScore(1, 0, 0.9) > EP.pinballScore(1, 0, 0.1));
    const mp = EP.meanPinball(1.0, [
        { tau: 0.1, q: 0.5 },
        { tau: 0.5, q: 0.9 },
        { tau: 0.9, q: 1.2 }
    ]);
    assert.ok(Number.isFinite(mp));
});

test('nseFromPairStats and kgeFromPairStats on perfect / scaled series', () => {
    const perfect = EP.emptyPairStats();
    const scaled = EP.emptyPairStats();
    for (let i = 0; i < 20; i++) {
        const o = i * 0.5;
        EP.accumulatePair(perfect, o, o);
        EP.accumulatePair(scaled, 2 * o, o);
    }
    assert.ok(Math.abs(EP.nseFromPairStats(perfect) - 1) < 1e-9);
    assert.ok(Math.abs(EP.kgeFromPairStats(perfect).kge - 1) < 1e-9);
    const kgeScaled = EP.kgeFromPairStats(scaled);
    assert.ok(Number.isFinite(kgeScaled.kge));
    assert.ok(Math.abs(kgeScaled.r - 1) < 1e-9);
    assert.ok(Math.abs(kgeScaled.alpha - 2) < 1e-9);
    assert.ok(Math.abs(kgeScaled.beta - 2) < 1e-9);
    const nseCi = EP.bootstrapHydroCI([perfect, scaled], 'nse', { seed: 1, draws: 100 });
    assert.ok(Number.isFinite(nseCi.value));
    assert.ok(nseCi.lo <= nseCi.value && nseCi.value <= nseCi.hi);
});

test('firstCrossingTime and crossingTimeErrorMin', () => {
    const t0 = Date.parse('2026-04-01T00:00:00Z');
    const obs = new Map();
    const fc = new Map();
    for (let k = 0; k <= 8; k++) {
        obs.set(t0 + k * EP.STEP_MS, 1 + k * 0.2);
        fc.set(t0 + k * EP.STEP_MS, 1 + k * 0.25);
    }
    const obsCross = EP.firstCrossingTime(obs, t0, 1.5, 8);
    const fcCross = EP.firstCrossingTime(fc, t0, 1.5, 8);
    assert.ok(fcCross < obsCross);
    const err = EP.crossingTimeErrorMin(obs, fc, t0, 1.5, 8);
    assert.ok(err < 0);
});

test('phaseLagSubStep interpolates around discrete minimum', () => {
    const seByShift = {
        '-1': { n: 50, se: 50 * 0.04 },
        '0': { n: 50, se: 50 * 0.01 },
        '1': { n: 50, se: 50 * 0.02 }
    };
    const lag = EP.phaseLagSubStep(seByShift, { minN: 20 });
    assert.ok(Number.isFinite(lag.phaseLagMin));
    assert.equal(lag.phaseLagMinDiscrete, 0);
    // Vertex should be slightly positive shift ⇒ negative lag minutes
    assert.ok(lag.phaseLagMin < 0);
});
