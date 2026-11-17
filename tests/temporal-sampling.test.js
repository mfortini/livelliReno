const test = require('node:test');
const assert = require('node:assert/strict');
const T = require('../temporal-sampling.js');

const MIN = 60000;
const series = [
    { time: new Date(0), value: 0 },
    { time: new Date(30 * MIN), value: 3 },
    { time: new Date(60 * MIN), value: 6 }
];

test('linear interpolation preserves observed nodes and midpoint', () => {
    assert.equal(T.sampleTemporal(series, 15 * MIN, { method: 'linear', cutoffMs: 60 * MIN }), 1.5);
    assert.equal(T.sampleTemporal(series, 30 * MIN, { method: 'linear', cutoffMs: 60 * MIN }), 3);
});

test('pchip is shape preserving on a monotone ramp', () => {
    const v = T.sampleTemporal(series, 45 * MIN, { method: 'pchip', cutoffMs: 60 * MIN });
    assert.ok(v > 3 && v < 6);
    assert.ok(Math.abs(v - 4.5) < 0.6);
});

test('no extrapolation beyond last observation at cutoff', () => {
    assert.equal(T.sampleTemporal(series, 90 * MIN, { method: 'linear', cutoffMs: 60 * MIN }), null);
    assert.equal(T.sampleTemporal(series, 30 * MIN, { method: 'pchip', cutoffMs: 20 * MIN }), null);
});

test('long gaps stay missing rather than bridged', () => {
    const gapped = [
        { time: new Date(0), value: 1 },
        { time: new Date(60 * MIN), value: 2 }
    ];
    assert.equal(T.sampleTemporal(gapped, 30 * MIN, { method: 'linear', cutoffMs: 60 * MIN, maxGapMinutes: 60 }), 1.5);
    const wider = [
        { time: new Date(0), value: 1 },
        { time: new Date(180 * MIN), value: 2 }
    ];
    assert.equal(T.sampleTemporal(wider, 90 * MIN, { method: 'linear', cutoffMs: 180 * MIN, maxGapMinutes: 60 }), null);
});

test('future observations do not change an emitted cutoff sample', () => {
    const cutoff = 60 * MIN;
    const before = T.sampleTemporal(series, 45 * MIN, { method: 'pchip', cutoffMs: cutoff });
    const extended = series.concat([{ time: new Date(90 * MIN), value: 20 }]);
    const after = T.sampleTemporal(extended, 45 * MIN, { method: 'pchip', cutoffMs: cutoff });
    assert.equal(after, before);
});

test('temporalContract defaults keep legacy 30 minute raw behaviour', () => {
    const c = T.temporalContract({});
    assert.equal(c.inputStepMinutes, 30);
    assert.equal(c.method, 'raw');
    const dense = T.temporalContract({ inputStepMinutes: 5, inputInterpolation: 'pchip' });
    assert.equal(dense.inputStepMinutes, 5);
    assert.equal(dense.method, 'pchip');
});

test('spectralFeatures: deterministic, correct length, dominant bin for known period', () => {
    const W = 48;
    const newestFirst = [];
    for (let i = W - 1; i >= 0; i--) newestFirst.push(Math.sin(2 * Math.PI * i / 12));
    const f = T.spectralFeatures(newestFirst);
    assert.equal(f.length, 28);
    assert.equal(f[0], 1); // full coverage
    // period 12 in W=48 -> DFT bin j = W/period = 4 -> dombin index 14 (1+12+1)
    assert.equal(f[14], 4);
    const again = T.spectralFeatures(newestFirst);
    assert.deepEqual(f, again);
});

test('spectralFeatures: NaN gaps reduce coverage and stay finite', () => {
    const W = 48;
    const win = [];
    for (let i = W - 1; i >= 0; i--) win.push(i % 9 === 0 ? NaN : Math.sin(i * 0.3));
    const f = T.spectralFeatures(win);
    assert.equal(f.length, 28);
    assert.ok(f[0] < 1);
    for (const v of f) assert.ok(Number.isFinite(v));
});

test('spectralFeatures: constant and short windows are safe', () => {
    const constWin = new Array(48).fill(2.5).reverse();
    const fc = T.spectralFeatures(constWin);
    assert.equal(fc.length, 28);
    for (const v of fc.slice(1)) assert.equal(v, 0);
    const short = T.spectralFeatures([1, 2, NaN]);
    assert.equal(short.length, 28);
    assert.ok(Number.isFinite(short[0]));
});

test('spectralFeatures: causality - appending future data does not change earlier window', () => {
    const base = [];
    for (let i = 63; i >= 0; i--) base.push(Math.sin(i * 0.2) + (i % 5) * 0.01);
    const f1 = T.spectralFeatures(base.slice(0, 48));
    const extended = base.concat([99, -50, 12345]);
    const f2 = T.spectralFeatures(extended.slice(0, 48));
    assert.deepEqual(f1, f2);
});

test('spectralFeatureNames matches spectralFeatures length', () => {
    for (const [fb, ws] of [[12, 5], [0, 5], [12, 0], [0, 0]]) {
        const names = T.spectralFeatureNames('Bagnetto_Reno', fb, ws);
        const f = T.spectralFeatures(new Array(48).fill(1).reverse(), fb, ws);
        assert.equal(names.length, f.length, `fb=${fb} ws=${ws}`);
    }
});
