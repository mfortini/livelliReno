#!/usr/bin/env node
/**
 * Fit causal peak-within-h amplitude prior: Cento max(y[t+Δ…t+h]) − y0 from upstream
 * levels/Δ. Used only while rising (soft-load + lift mediana).
 *
 *   node tools/fit-upstream-amplitude.js
 *     [--data storico] [--events models/events.json]
 *     [--out models/cento-upstream-amplitude.json]
 *     [--test-from 2026-01-01] [--min-peak 2] [--lambda 1]
 */
'use strict';

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
function opt(name, fallback) {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] != null ? process.argv[i + 1] : fallback;
}

const dataDir = path.resolve(root, opt('data', 'storico'));
const eventsPath = path.resolve(root, opt('events', 'models/events.json'));
const outPath = path.resolve(root, opt('out', 'models/cento-upstream-amplitude.json'));
const testFrom = Date.parse(opt('test-from', '2026-01-01T00:00:00Z'));
const minPeak = Number(opt('min-peak', '2'));
const lambda = Number(opt('lambda', '1'));
const STEP = 30 * 60 * 1000;
const UPSTREAMS = [
    'Bonconvento',
    'Forcelli',
    'Sostegno Reno',
    'Casalecchio tiro a volo',
    'Casalecchio chiusa'
];
const HORIZONS_MIN = [60, 120, 180, 240, 300, 360];

function loadSeries(name) {
    const file = path.join(dataDir, `${name}.csv`);
    if (!fs.existsSync(file)) return { times: [], values: [], byTime: new Map() };
    const times = [], values = [], byTime = new Map();
    for (const line of fs.readFileSync(file, 'utf8').split('\n').slice(1)) {
        const comma = line.indexOf(',');
        if (comma < 0) continue;
        const t = Date.parse(line.slice(0, comma));
        const v = Number(line.slice(comma + 1));
        if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
        times.push(t);
        values.push(v);
        byTime.set(t, v);
    }
    return { times, values, byTime };
}

function sample(series, t) {
    if (series.byTime.has(t)) return series.byTime.get(t);
    let lo = 0, hi = series.times.length - 1, idx = -1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (series.times[mid] <= t) {
            idx = mid;
            lo = mid + 1;
        } else hi = mid - 1;
    }
    if (idx < 0) return null;
    if (t - series.times[idx] > 45 * 60 * 1000) return null;
    return series.values[idx];
}

function peakDelta(series, t0, horizonMin) {
    const y0 = sample(series, t0);
    if (!Number.isFinite(y0)) return null;
    const end = t0 + horizonMin * 60 * 1000;
    let peak = y0;
    for (let t = t0 + STEP; t <= end; t += STEP) {
        const v = sample(series, t);
        if (Number.isFinite(v) && v > peak) peak = v;
    }
    return peak - y0;
}

function featureNames() {
    const names = ['bias', 'cento_y0', 'cento_d2h', 'cento_d6h'];
    for (const u of UPSTREAMS) {
        const tag = u.replace(/\s+/g, '_');
        names.push(`${tag}_y`, `${tag}_d2h`, `${tag}_d6h`);
    }
    return names;
}

function featuresAt(data, t0) {
    const cento = data.Cento;
    const y0 = sample(cento, t0);
    if (!Number.isFinite(y0)) return null;
    const y2 = sample(cento, t0 - 2 * 3600 * 1000);
    const y6 = sample(cento, t0 - 6 * 3600 * 1000);
    const x = [1, y0, Number.isFinite(y2) ? y0 - y2 : 0, Number.isFinite(y6) ? y0 - y6 : 0];
    for (const u of UPSTREAMS) {
        const s = data[u];
        const yu = sample(s, t0);
        if (!Number.isFinite(yu)) return null;
        const u2 = sample(s, t0 - 2 * 3600 * 1000);
        const u6 = sample(s, t0 - 6 * 3600 * 1000);
        x.push(yu, Number.isFinite(u2) ? yu - u2 : 0, Number.isFinite(u6) ? yu - u6 : 0);
    }
    return x;
}

function solveRidge(X, y, lam) {
    const n = X.length;
    const p = X[0].length;
    const A = Array.from({ length: p }, () => Array(p).fill(0));
    const b = Array(p).fill(0);
    for (let i = 0; i < n; i++) {
        const xi = X[i];
        const yi = y[i];
        for (let j = 0; j < p; j++) {
            b[j] += xi[j] * yi;
            for (let k = 0; k < p; k++) A[j][k] += xi[j] * xi[k];
        }
    }
    for (let j = 0; j < p; j++) A[j][j] += lam * (j === 0 ? 0 : 1); // no penalty on bias
    // Gaussian elimination
    for (let col = 0; col < p; col++) {
        let piv = col;
        for (let r = col + 1; r < p; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
        [A[col], A[piv]] = [A[piv], A[col]];
        [b[col], b[piv]] = [b[piv], b[col]];
        const div = A[col][col];
        if (Math.abs(div) < 1e-12) continue;
        for (let k = col; k < p; k++) A[col][k] /= div;
        b[col] /= div;
        for (let r = 0; r < p; r++) {
            if (r === col) continue;
            const f = A[r][col];
            for (let k = col; k < p; k++) A[r][k] -= f * A[col][k];
            b[r] -= f * b[col];
        }
    }
    return b;
}

function mae(pred, y) {
    let s = 0;
    for (let i = 0; i < y.length; i++) s += Math.abs(pred[i] - y[i]);
    return s / y.length;
}

function rmse(pred, y) {
    let s = 0;
    for (let i = 0; i < y.length; i++) s += (pred[i] - y[i]) ** 2;
    return Math.sqrt(s / y.length);
}

const names = ['Cento', ...UPSTREAMS];
const data = Object.fromEntries(names.map(n => [n, loadSeries(n)]));
if (!data.Cento.times.length) {
    console.error('Missing storico/Cento.csv');
    process.exit(1);
}

const catalog = JSON.parse(fs.readFileSync(eventsPath, 'utf8'));
const centoEvents = (catalog.events || []).filter(e => e.station === 'Cento' && e.peakLevel >= minPeak);
console.log(`Cento events peak≥${minPeak}: ${centoEvents.length}`);

const fnames = featureNames();
const horizons = {};
const stride = 2 * STEP; // 1 h

for (const hMin of HORIZONS_MIN) {
    const Xtr = [], ytr = [], Xte = [], yte = [];
    for (const ev of centoEvents) {
        const start = Date.parse(ev.start);
        const end = Date.parse(ev.end);
        const peakT = Date.parse(ev.peakTime);
        if (![start, end, peakT].every(Number.isFinite)) continue;
        // sample rising limb: start → peak
        for (let t = start; t <= peakT; t += stride) {
            const x = featuresAt(data, t);
            const d = peakDelta(data.Cento, t, hMin);
            if (!x || !Number.isFinite(d)) continue;
            if (t < testFrom) {
                Xtr.push(x);
                ytr.push(d);
            } else {
                Xte.push(x);
                yte.push(d);
            }
        }
    }
    if (Xtr.length < 50) {
        console.warn(`h=${hMin}: too few train rows (${Xtr.length}), skip`);
        continue;
    }
    const w = solveRidge(Xtr, ytr, lambda);
    const predTr = Xtr.map(x => x.reduce((s, v, i) => s + v * w[i], 0));
    const predTe = Xte.map(x => x.reduce((s, v, i) => s + v * w[i], 0));
    const persistTe = yte.map(() => 0); // baseline: peak delta 0
    horizons[String(hMin)] = {
        weights: w.map(v => Math.round(v * 1e6) / 1e6),
        train: { n: Xtr.length, mae: +mae(predTr, ytr).toFixed(4), rmse: +rmse(predTr, ytr).toFixed(4) },
        test: {
            n: Xte.length,
            mae: Xte.length ? +mae(predTe, yte).toFixed(4) : null,
            rmse: Xte.length ? +rmse(predTe, yte).toFixed(4) : null,
            persistMae: Xte.length ? +mae(persistTe, yte).toFixed(4) : null
        }
    };
    console.log(
        `h=${hMin}′ train n=${Xtr.length} mae=${horizons[String(hMin)].train.mae} | ` +
        `test n=${Xte.length} mae=${horizons[String(hMin)].test.mae} persistMae=${horizons[String(hMin)].test.persistMae}`
    );
}

const doc = {
    kind: 'cento_upstream_amplitude',
    version: 1,
    station: 'Cento',
    target: 'peak_within_h_delta',
    upstreams: UPSTREAMS,
    featureNames: fnames,
    stepMinutes: 30,
    horizons,
    testFrom: new Date(testFrom).toISOString(),
    minPeak,
    lambda,
    adopt: false,
    generatedAt: new Date().toISOString(),
    disclaimer: 'Sperimentale: prior di ampiezza monte→valle; non allertistico.'
};

fs.writeFileSync(outPath, JSON.stringify(doc, null, 2));
console.log(`Wrote ${path.relative(root, outPath)}`);
