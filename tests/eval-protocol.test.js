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
});
