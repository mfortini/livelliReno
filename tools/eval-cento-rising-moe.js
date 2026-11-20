#!/usr/bin/env node
/**
 * M3: confronta prod vs MoE (prod + rising-only expert gated) su eventi Cento.
 * Uso:
 *   node tools/eval-cento-rising-moe.js [--rising models/cento-dnn-rising.json]
 */
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

const STEP = 30 * 60000;
const WINDOW = 72 * 3600000;
const HORIZONS = [4, 12];
const station = 'Cento';
const dataDir = path.resolve(root, opt('data', 'storico'));
const risingPath = path.resolve(root, opt('rising', 'models/cento-dnn-rising.json'));
const coeffsPath = path.resolve(root, opt('coeffs', 'models/transfer-coeffs-runtime.json'));
const eventsPath = path.resolve(root, opt('events', 'models/events.json'));
const fromMs = Date.parse(opt('from', '2026-01-01'));
const jsonOut = path.resolve(root, opt('json', 'models/eval-cento-rising-moe.json'));
const mdOut = path.resolve(root, opt('md', 'models/ab-rising-moe.md'));

function loadSeries(name, dir = dataDir) {
    const file = path.join(dir, `${name}.csv`);
    const times = [], values = [], byTime = new Map();
    if (!fs.existsSync(file)) return { times, values, byTime };
    for (const line of fs.readFileSync(file, 'utf8').split('\n').slice(1)) {
        const c = line.indexOf(',');
        if (c < 0) continue;
        const t = Date.parse(line.slice(0, c));
        const v = Number(line.slice(c + 1));
        if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
        times.push(t); values.push(v); byTime.set(t, v);
    }
    return { times, values, byTime };
}
function lowerBound(arr, t) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < t) lo = m + 1; else hi = m; }
    return lo;
}
function windowSensor(name, series, now) {
    const s = lowerBound(series.times, now - WINDOW);
    const e = lowerBound(series.times, now + 1);
    const timeSeries = [];
    for (let i = s; i < e; i++) timeSeries.push({ time: new Date(series.times[i]), value: series.values[i] });
    return { id: name, name, timeSeries };
}
function median(xs) {
    const a = xs.filter(Number.isFinite).sort((x, y) => x - y);
    if (!a.length) return null;
    const m = (a.length - 1) / 2;
    return (a[Math.floor(m)] + a[Math.ceil(m)]) / 2;
}
function mean(xs) {
    const a = xs.filter(Number.isFinite);
    return a.length ? a.reduce((s, v) => s + v, 0) / a.length : null;
}
function rmse(xs) {
    const a = xs.filter(Number.isFinite);
    return a.length ? Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length) : null;
}

if (!fs.existsSync(risingPath)) {
    console.error(`Missing ${risingPath}`);
    process.exit(1);
}

const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
const coeffs = JSON.parse(fs.readFileSync(coeffsPath, 'utf8'));
const rising = JSON.parse(fs.readFileSync(risingPath, 'utf8'));
const eventsDoc = JSON.parse(fs.readFileSync(eventsPath, 'utf8'));
const prodNode = { ...coeffs.nodes[station], reliable: true };
const moeCoeffs = {
    nodes: {
        [station]: { ...prodNode, moeRising: { ...rising, reliable: true } }
    }
};
const prodCoeffs = { nodes: { [station]: prodNode } };
const upstreams = prodNode.upstreams || [];
const names = [...new Set([station, ...upstreams])];
const data = Object.fromEntries(names.map(n => [n, loadSeries(n)]));
const rainGauges = ((config.rainfall && config.rainfall.gauges) || [])
    .map(e => (typeof e === 'string' ? e : e?.name)).filter(Boolean);
const rainData = Object.fromEntries(rainGauges.map(n => [n, loadSeries(n, path.join(dataDir, 'rain'))]));

const events = (eventsDoc.events || []).filter(e =>
    e.station === station && Date.parse(e.peakTime) >= fromMs
);

function run(coeffsObj, label) {
    const byH = Object.fromEntries(HORIZONS.map(h => [h, {
        risingErrs: [], fallingErrs: [], risingMinByEvent: [], peakErrs: []
    }]));
    for (const ev of events) {
        const peak = Date.parse(ev.peakTime);
        const start = Date.parse(ev.start);
        const end = Date.parse(ev.end) - Math.max(...HORIZONS) * STEP;
        const risingErrsEvt = Object.fromEntries(HORIZONS.map(h => [h, []]));
        for (const h of HORIZONS) {
            const issueAt = peak - h * STEP;
            const down = windowSensor(station, data[station], issueAt);
            const sensorMap = Object.fromEntries(names.map(n => [n, windowSensor(n, data[n], issueAt)]));
            sensorMap[station] = down;
            const ups = upstreams.map(u => sensorMap[u]);
            const rainSensors = rainGauges.map(n => windowSensor(n, rainData[n], issueAt)).filter(s => s.timeSeries.length);
            const result = P.predictNode(down, ups, {
                hours: 72, now: new Date(issueAt), coeffs: coeffsObj, horizonHours: 6,
                downstreamId: station, rainSensors, sensorMap, smooth: 0
            });
            if (!result?.ok) continue;
            let best = null, bestDist = Infinity;
            for (const p of result.forecast || []) {
                if (p.regime === 'anchor') continue;
                const dist = Math.abs(+p.time - peak);
                if (dist < bestDist) { best = p; bestDist = dist; }
            }
            const obs = data[station].byTime.get(peak);
            const pred = best?.med ?? best?.value;
            if (Number.isFinite(pred) && Number.isFinite(obs)) byH[h].peakErrs.push(pred - obs);
        }
        for (let now = Math.ceil(start / (2 * STEP)) * 2 * STEP; now <= end; now += 2 * STEP) {
            const down = windowSensor(station, data[station], now);
            const sensorMap = Object.fromEntries(names.map(n => [n, windowSensor(n, data[n], now)]));
            sensorMap[station] = down;
            const ups = upstreams.map(u => sensorMap[u]);
            const rainSensors = rainGauges.map(n => windowSensor(n, rainData[n], now)).filter(s => s.timeSeries.length);
            const result = P.predictNode(down, ups, {
                hours: 72, now: new Date(now), coeffs: coeffsObj, horizonHours: 6,
                downstreamId: station, rainSensors, sensorMap, smooth: 0
            });
            if (!result?.ok) continue;
            const limb = now < peak ? 'rising' : 'falling';
            for (const h of HORIZONS) {
                const target = now + h * STEP;
                const obs = data[station].byTime.get(target);
                if (!Number.isFinite(obs)) continue;
                let best = null, bestDist = Infinity;
                for (const p of result.forecast || []) {
                    if (p.regime === 'anchor') continue;
                    const dist = Math.abs(+p.time - target);
                    if (dist < bestDist) { best = p; bestDist = dist; }
                }
                const pred = best?.med ?? best?.value;
                if (!Number.isFinite(pred) || bestDist > STEP) continue;
                const err = pred - obs;
                byH[h][limb === 'rising' ? 'risingErrs' : 'fallingErrs'].push(err);
                if (limb === 'rising') risingErrsEvt[h].push(err);
            }
        }
        for (const h of HORIZONS) {
            if (risingErrsEvt[h].length) byH[h].risingMinByEvent.push(Math.min(...risingErrsEvt[h]));
        }
    }
    const out = { label, events: events.length };
    for (const h of HORIZONS) {
        out[h * 30] = {
            risingRmse: rmse(byH[h].risingErrs),
            fallingRmse: rmse(byH[h].fallingErrs),
            risingMinMed: median(byH[h].risingMinByEvent),
            risingBias: mean(byH[h].risingErrs),
            peakErrMed: median(byH[h].peakErrs),
            nRising: byH[h].risingErrs.length,
            nFalling: byH[h].fallingErrs.length
        };
    }
    return out;
}

console.error('[moe] scoring prod…');
const prod = run(prodCoeffs, 'prod');
console.error('[moe] scoring moe…');
const moe = run(moeCoeffs, 'moe');

const report = {
    kind: 'ab_rising_moe',
    generatedAt: new Date().toISOString(),
    risingPath: path.relative(root, risingPath),
    risingSkill120: rising.skill120,
    risingReliable: rising.reliable,
    risingOnly: rising.risingOnly === true,
    prod,
    moe,
    gate: {},
    disclaimer: 'Sperimentale: non operativo, non allertistico.'
};

for (const h of [120, 360]) {
    const dMin = (moe[h]?.risingMinMed ?? null) - (prod[h]?.risingMinMed ?? null);
    const dFall = prod[h]?.fallingRmse > 0
        ? (moe[h]?.fallingRmse - prod[h]?.fallingRmse) / prod[h].fallingRmse
        : null;
    report.gate[h] = {
        deltaRisingMin: dMin,
        fallingRel: dFall,
        passRisingMin: dMin != null && dMin >= 0.15,
        passFalling: dFall == null || dFall <= 0.10
    };
}

fs.writeFileSync(jsonOut, JSON.stringify(report, null, 2));
const md = [
    '# A/B M3 — MoE salita Cento (rising-only expert)',
    '',
    `**Expert:** \`${report.risingPath}\` · skill120=${report.risingSkill120} · risingOnly=${report.risingOnly}`,
    `**Gate:** Δ risingMin med ≥ +0.15 m vs prod; falling RMSE entro +10%.`,
    '',
    '| modello | h | rising RMSE | risingMin med | falling RMSE | peak err med |',
    '|---|---:|---:|---:|---:|---:|',
    ...[120, 360].flatMap(h => [
        `| prod | ${h} | ${prod[h]?.risingRmse?.toFixed(3)} | ${prod[h]?.risingMinMed?.toFixed(3)} | ${prod[h]?.fallingRmse?.toFixed(3)} | ${prod[h]?.peakErrMed?.toFixed(3)} |`,
        `| moe | ${h} | ${moe[h]?.risingRmse?.toFixed(3)} | ${moe[h]?.risingMinMed?.toFixed(3)} | ${moe[h]?.fallingRmse?.toFixed(3)} | ${moe[h]?.peakErrMed?.toFixed(3)} |`
    ]),
    '',
    '| h | Δ risingMin (moe−prod) | falling rel | gate |',
    '|---:|---:|---:|---|',
    ...[120, 360].map(h => {
        const g = report.gate[h];
        return `| ${h} | ${g.deltaRisingMin?.toFixed(3)} | ${g.fallingRel == null ? '—' : (100 * g.fallingRel).toFixed(1) + '%'} | ${g.passRisingMin && g.passFalling ? 'PASS' : 'FAIL'} |`;
    }),
    '',
    (report.gate[120]?.passRisingMin && report.gate[120]?.passFalling)
        ? '**Decisione:** soft-adopt MoE (`reliable` expert + `softMergeCentoRisingMoe`).'
        : '**Decisione:** non promuovere MoE; tenere prod + M2 UI/banda.',
    ''
].join('\n');
fs.writeFileSync(mdOut, md);
console.log(md);
console.log(`Wrote ${jsonOut}`);
