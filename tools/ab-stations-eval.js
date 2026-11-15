#!/usr/bin/env node
// Rising/falling + phaseLag @120′ for A/B station variants (Cento MLP exports).
//   node tools/ab-stations-eval.js [--from 2026-01-01] [--every 120]
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const P = require('../predict-transfer.js');

const root = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const opt = (name, fallback) => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
};

const station = 'Cento';
const STEP = 30 * 60000;
const WINDOW = 72 * 3600000;
const H = 4; // 120′
const everyMin = Number(opt('every', '120'));
const issueStep = everyMin * 60000;
const dataDir = path.resolve(root, 'storico');
const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
const eventsDoc = JSON.parse(fs.readFileSync(path.join(root, 'models/events.json'), 'utf8'));
const names = Object.keys(config.sensors);
const rainGauges = ((config.rainfall && config.rainfall.gauges) || [])
    .map(e => (typeof e === 'string' ? e : e?.name))
    .filter(Boolean);

const MODELS = {
    baseline: 'models/cento-dnn-ab-baseline.json',
    bagnetto: 'models/cento-dnn-ab-bagnetto.json',
    far_monte: 'models/cento-dnn-ab-far-monte.json',
    tributaries: 'models/cento-dnn-ab-tributaries.json',
    dosso: 'models/cento-dnn-ab-dosso.json',
    bagnetto_dosso: 'models/cento-dnn-ab-bagnetto-dosso.json',
    bagnetto_dosso_chain: 'models/cento-dnn-ab-bagnetto-dosso-chain.json'
};

function loadSeries(name, dir = dataDir) {
    const file = path.join(dir, `${name}.csv`);
    if (!fs.existsSync(file)) return { times: [], values: [], byTime: new Map() };
    const times = [], values = [], byTime = new Map();
    for (const line of fs.readFileSync(file, 'utf8').split('\n').slice(1)) {
        const comma = line.indexOf(',');
        if (comma < 0) continue;
        const t = Date.parse(line.slice(0, comma));
        const v = Number(line.slice(comma + 1));
        if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
        times.push(t); values.push(v); byTime.set(t, v);
    }
    return { times, values, byTime };
}

function lowerBound(arr, target) {
    let lo = 0, hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

function windowSensor(name, series, now) {
    const start = lowerBound(series.times, now - WINDOW);
    const end = lowerBound(series.times, now + 1);
    const timeSeries = [];
    for (let i = start; i < end; i++) timeSeries.push({ time: new Date(series.times[i]), value: series.values[i] });
    return { id: name, name, timeSeries };
}

function phaseLagMin(predSeries, obsSeries, maxLagSteps = 6) {
    // predSeries/obsSeries: [{t,v}] aligned roughly; correlate at lag shifts of STEP
    if (!predSeries.length || !obsSeries.length) return null;
    const obsMap = new Map(obsSeries.map(p => [p.t, p.v]));
    let bestLag = 0, bestCorr = -Infinity;
    for (let lag = -maxLagSteps; lag <= maxLagSteps; lag++) {
        const xs = [], ys = [];
        for (const p of predSeries) {
            const o = obsMap.get(p.t + lag * STEP);
            if (!Number.isFinite(o) || !Number.isFinite(p.v)) continue;
            xs.push(p.v); ys.push(o);
        }
        if (xs.length < 8) continue;
        const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
        const my = ys.reduce((a, b) => a + b, 0) / xs.length;
        let num = 0, dx = 0, dy = 0;
        for (let i = 0; i < xs.length; i++) {
            const a = xs[i] - mx, b = ys[i] - my;
            num += a * b; dx += a * a; dy += b * b;
        }
        const den = Math.sqrt(dx * dy);
        if (!(den > 0)) continue;
        const corr = num / den;
        if (corr > bestCorr) {
            bestCorr = corr;
            bestLag = lag;
        }
    }
    if (!Number.isFinite(bestCorr) || bestCorr === -Infinity) return null;
    // Positive => prediction lags observation (late)
    return -bestLag * 30;
}

const data = Object.fromEntries(names.map(n => [n, loadSeries(n)]));
const rainData = Object.fromEntries(rainGauges.map(n => [n, loadSeries(n, path.join(dataDir, 'rain'))]));
const from = Date.parse(opt('from', '2026-01-01'));
const to = Date.parse(opt('to', new Date(data[station].times.at(-1)).toISOString()));
const deps = config.sensors[station]?.dependencies || [];
const events = (eventsDoc.events || []).filter(e =>
    e.station === station && Date.parse(e.end) >= from && Date.parse(e.start) <= to
);

function empty() {
    return { n: 0, se: 0, seP: 0, risingN: 0, risingSe: 0, fallingN: 0, fallingSe: 0, lags: [] };
}

function evalModel(mlpPath) {
    const node = JSON.parse(fs.readFileSync(path.resolve(root, mlpPath), 'utf8'));
    const forced = { ...node, reliable: true };
    const bucket = empty();
    const trajPred = [];
    const trajObs = [];
    for (const ev of events) {
        const peak = Date.parse(ev.peakTime);
        const start = Math.max(from, Date.parse(ev.start));
        const end = Math.min(to, Date.parse(ev.end) - H * STEP);
        for (let now = Math.ceil(start / issueStep) * issueStep; now <= end; now += issueStep) {
            const down = windowSensor(station, data[station], now);
            const anchor = P.latestFinite(down.timeSeries);
            if (!anchor || now - +anchor.time > STEP) continue;
            const y0 = data[station].byTime.get(now);
            const yH = data[station].byTime.get(now + H * STEP);
            if (!Number.isFinite(y0) || !Number.isFinite(yH)) continue;
            const sensorMap = Object.fromEntries(names.map(n => [n, windowSensor(n, data[n], now)]));
            sensorMap[station] = down;
            const ups = deps.map(u => sensorMap[u]);
            const rainSensors = rainGauges
                .map(n => windowSensor(n, rainData[n], now))
                .filter(s => s.timeSeries.length);
            const result = P.predictNode(down, ups, {
                hours: 24,
                now: new Date(now),
                coeffs: { nodes: { [station]: forced } },
                horizonHours: 2,
                downstreamId: station,
                rainSensors,
                sensorMap,
                smooth: 0
            });
            if (!result?.ok) continue;
            let best = null, bestDist = Infinity;
            for (const p of result.forecast || []) {
                if (p.regime === 'anchor') continue;
                const dist = Math.abs(+p.time - (now + H * STEP));
                if (dist < bestDist) { best = p; bestDist = dist; }
            }
            if (!best || bestDist > STEP) continue;
            const pred = best.med ?? best.value;
            if (!Number.isFinite(pred)) continue;
            const err = pred - yH;
            const errP = y0 - yH;
            bucket.n++;
            bucket.se += err * err;
            bucket.seP += errP * errP;
            if (now < peak) {
                bucket.risingN++;
                bucket.risingSe += err * err;
            } else {
                bucket.fallingN++;
                bucket.fallingSe += err * err;
            }
            trajPred.push({ t: now + H * STEP, v: pred });
            trajObs.push({ t: now + H * STEP, v: yH });
        }
    }
    const rmse = bucket.n ? Math.sqrt(bucket.se / bucket.n) : null;
    const rmseP = bucket.n ? Math.sqrt(bucket.seP / bucket.n) : null;
    return {
        path: mlpPath,
        n: bucket.n,
        rmse,
        skillPersist: rmseP > 0 ? 1 - rmse / rmseP : null,
        risingRmse: bucket.risingN ? Math.sqrt(bucket.risingSe / bucket.risingN) : null,
        risingN: bucket.risingN,
        fallingRmse: bucket.fallingN ? Math.sqrt(bucket.fallingSe / bucket.fallingN) : null,
        fallingN: bucket.fallingN,
        phaseLagMin: phaseLagMin(trajPred, trajObs)
    };
}

const report = {
    kind: 'ab_stations_eval',
    station,
    from: new Date(from).toISOString(),
    to: new Date(to).toISOString(),
    horizonMin: 120,
    everyMin,
    events: events.length,
    models: {}
};

for (const [id, mlpPath] of Object.entries(MODELS)) {
    if (!fs.existsSync(path.resolve(root, mlpPath))) {
        console.log(`skip ${id}: missing ${mlpPath}`);
        continue;
    }
    console.log(`eval ${id}…`);
    report.models[id] = evalModel(mlpPath);
    const m = report.models[id];
    console.log(
        `  n=${m.n} rmse=${m.rmse?.toFixed(3)} skill=${m.skillPersist?.toFixed(3)} ` +
        `rising=${m.risingRmse?.toFixed(3)} falling=${m.fallingRmse?.toFixed(3)} lag=${m.phaseLagMin}`
    );
}

const outJson = path.resolve(root, opt('json', 'models/ab-stations-monte-valle-eval.json'));
fs.writeFileSync(outJson, JSON.stringify(report, null, 2) + '\n');
console.log(`Wrote ${outJson}`);
