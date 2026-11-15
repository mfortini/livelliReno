#!/usr/bin/env node
// M9 A/B: prod vs crossing-time warp (± Casalecchio chiusa) on Cento.
// Metrics: rising/falling RMSE@120, phaseLagMin, skill vs persist (Nov event + test floods).
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
const dataDir = path.resolve(root, opt('data', 'storico'));
const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
const coeffs = JSON.parse(fs.readFileSync(path.join(root, 'models/transfer-coeffs.json'), 'utf8'));
const eventsDoc = JSON.parse(fs.readFileSync(path.join(root, 'models/events.json'), 'utf8'));
const crossWith = JSON.parse(fs.readFileSync(path.join(root, opt('crossing', 'models/cento-crossing-time.json')), 'utf8'));
const crossNo = fs.existsSync(path.join(root, 'models/cento-crossing-time-nochiusa.json'))
    ? JSON.parse(fs.readFileSync(path.join(root, 'models/cento-crossing-time-nochiusa.json'), 'utf8'))
    : null;

const STEP = 30 * 60000;
const WINDOW = 72 * 3600000;
const HORIZONS = [4, 8, 12]; // 120/240/360′
const PHASE = [-4, -2, -1, 0, 1, 2, 4];

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
function lb(a, t) {
    let lo = 0, hi = a.length;
    while (lo < hi) {
        const m = (lo + hi) >> 1;
        if (a[m] < t) lo = m + 1;
        else hi = m;
    }
    return lo;
}
function win(name, s, now) {
    const a = lb(s.times, now - WINDOW), b = lb(s.times, now + 1);
    const ts = [];
    for (let i = a; i < b; i++) ts.push({ time: new Date(s.times[i]), value: s.values[i] });
    return { id: name, name, timeSeries: ts };
}

const names = Object.keys(config.sensors);
const data = Object.fromEntries(names.map(n => [n, loadSeries(n)]));
const rainGauges = ((config.rainfall && config.rainfall.gauges) || []).map(e =>
    typeof e === 'string' ? e : e?.name
).filter(Boolean);
const rainData = Object.fromEntries(rainGauges.map(n => [n, loadSeries(n, path.join(dataDir, 'rain'))]));
const deps = config.sensors[station].dependencies || [];
const prodNode = { ...coeffs.nodes[station], reliable: true };

const stationEvents = (eventsDoc.events || []).filter(e => e.station === station);
const nov = stationEvents.filter(e => String(e.start).startsWith('2026-11'));
const testEv = stationEvents.filter(e => Date.parse(e.start) >= Date.parse('2026-01-01'));

function empty() {
    return Object.fromEntries(HORIZONS.map(h => [h, {
        n: 0, se: 0, seP: 0,
        rising: { n: 0, se: 0 }, falling: { n: 0, se: 0 },
        seShift: Object.fromEntries(PHASE.map(s => [s, { n: 0, se: 0 }]))
    }]));
}

function run(label, crossing, eventList, issueStep) {
    const node = crossing
        ? { ...prodNode, crossingTime: { ...crossing, adopt: true }, reliable: true }
        : prodNode;
    const stats = empty();
    let used = 0;
    for (const ev of eventList) {
        const peak = Date.parse(ev.peakTime);
        const start = Date.parse(ev.start);
        const end = Date.parse(ev.end) - Math.max(...HORIZONS) * STEP;
        for (let now = Math.ceil(start / issueStep) * issueStep; now <= end; now += issueStep) {
            const down = win(station, data[station], now);
            const anchor = P.latestFinite(down.timeSeries);
            if (!anchor || now - +anchor.time > STEP) continue;
            const y0 = data[station].byTime.get(now);
            if (!Number.isFinite(y0)) continue;
            const sensorMap = Object.fromEntries(names.map(n => [n, win(n, data[n], now)]));
            sensorMap[station] = down;
            // Ensure crossing stations present even if not in deps
            for (const up of (crossing?.stations || [])) {
                if (!sensorMap[up] && data[up]) sensorMap[up] = win(up, data[up], now);
            }
            const ups = deps.map(u => sensorMap[u]);
            const rainSensors = rainGauges.map(n => win(n, rainData[n], now)).filter(s => s.timeSeries.length);
            const result = P.predictNode(down, ups, {
                hours: 72,
                now: new Date(now),
                coeffs: { nodes: { [station]: node } },
                horizonHours: 6,
                downstreamId: station,
                rainSensors,
                sensorMap,
                smooth: 0,
                useCrossingTime: Boolean(crossing)
            });
            if (!result?.ok) continue;
            used++;
            const limb = now < peak ? 'rising' : 'falling';
            for (const h of HORIZONS) {
                const target = now + h * STEP;
                const obs = data[station].byTime.get(target);
                if (!Number.isFinite(obs)) continue;
                let best = null, bestDist = Infinity;
                for (const p of result.forecast || []) {
                    if (p.regime === 'anchor') continue;
                    const d = Math.abs(+p.time - target);
                    if (d < bestDist) { best = p; bestDist = d; }
                }
                if (!best || bestDist > STEP) continue;
                const pred = best.med ?? best.value;
                if (!Number.isFinite(pred)) continue;
                const err = pred - obs;
                const b = stats[h];
                b.n++; b.se += err * err; b.seP += (y0 - obs) ** 2;
                b[limb].n++; b[limb].se += err * err;
                for (const s of PHASE) {
                    const obsS = data[station].byTime.get(target + s * STEP);
                    if (!Number.isFinite(obsS)) continue;
                    const e = pred - obsS;
                    b.seShift[s].n++; b.seShift[s].se += e * e;
                }
            }
        }
    }
    const out = { label, used, events: eventList.length };
    for (const [h, b] of Object.entries(stats)) {
        if (!b.n) { out[h] = { n: 0 }; continue; }
        let bestS = null, bestR = Infinity;
        for (const [s, acc] of Object.entries(b.seShift)) {
            if (acc.n < 20) continue;
            const r = Math.sqrt(acc.se / acc.n);
            if (r < bestR) { bestR = r; bestS = Number(s); }
        }
        out[h] = {
            n: b.n,
            rmse: Math.sqrt(b.se / b.n),
            skill: 1 - Math.sqrt(b.se / b.n) / Math.sqrt(b.seP / b.n),
            phaseLagMin: bestS == null ? null : -bestS * 30,
            risingRmse: b.rising.n ? Math.sqrt(b.rising.se / b.rising.n) : null,
            risingN: b.rising.n,
            fallingRmse: b.falling.n ? Math.sqrt(b.falling.se / b.falling.n) : null,
            fallingN: b.falling.n
        };
    }
    return out;
}

const slices = [];
if (nov.length) {
    slices.push(run('nov/prod', null, nov, STEP));
    slices.push(run('nov/cross+chiusa', crossWith, nov, STEP));
    if (crossNo) slices.push(run('nov/cross-nochiusa', crossNo, nov, STEP));
}
if (testEv.length) {
    slices.push(run('test/prod', null, testEv, 4 * STEP));
    slices.push(run('test/cross+chiusa', crossWith, testEv, 4 * STEP));
    if (crossNo) slices.push(run('test/cross-nochiusa', crossNo, testEv, 4 * STEP));
}

const report = {
    kind: 'ab_m9_prob_crossing',
    station,
    generatedAt: new Date().toISOString(),
    slices
};
fs.writeFileSync(path.join(root, 'models/fase2-m9-eval.json'), JSON.stringify(report, null, 2));

function fmt(v, d = 3) {
    return v == null || !Number.isFinite(v) ? '—' : Number(v).toFixed(d);
}

const lines = [
    '# A/B M9 — crossing-time probabilistico (Cento)',
    '',
    '**Ipotesi:** warp della fan di livello con moda di τ (flood-only) riduce rising RMSE e phaseLagMin; Casalecchio chiusa aiuta vs ablazione.',
    '',
    '| Slice | h | n | RMSE | skill | phaseLag | rising RMSE | falling RMSE |',
    '|---|---:|---:|---:|---:|---:|---:|---:|'
];
for (const s of slices) {
    for (const h of HORIZONS) {
        const r = s[h] || { n: 0 };
        lines.push(
            `| ${s.label} | ${h * 30}′ | ${r.n || 0} | ${fmt(r.rmse)} | ${fmt(r.skill)} | ${r.phaseLagMin ?? '—'}′ | ${fmt(r.risingRmse)} | ${fmt(r.fallingRmse)} |`
        );
    }
}

const novProd = slices.find(s => s.label === 'nov/prod');
const novCross = slices.find(s => s.label === 'nov/cross+chiusa');
const novNo = slices.find(s => s.label === 'nov/cross-nochiusa');
const p120 = novProd?.['4'];
const c120 = novCross?.['4'];
const n120 = novNo?.['4'];
const risingWin = c120 && p120 && c120.risingRmse < p120.risingRmse;
const lagWin = c120 && p120 && (c120.phaseLagMin == null || c120.phaseLagMin < p120.phaseLagMin);
const chiusaHelps = c120 && n120 && c120.risingRmse <= n120.risingRmse;
const promote = risingWin && lagWin;

lines.push(
    '',
    '## Decisione',
    '',
    `| Voce | Esito |`,
    `|---|---|`,
    `| Rising@120 vs prod | ${risingWin ? 'MIGLIORA' : 'FAIL'} (${fmt(c120?.risingRmse)} vs ${fmt(p120?.risingRmse)}) |`,
    `| phaseLagMin@120 | ${lagWin ? 'MIGLIORA' : 'FAIL'} (${c120?.phaseLagMin ?? '—'}′ vs ${p120?.phaseLagMin ?? '—'}′) |`,
    `| Casalecchio chiusa vs nochiusa | ${chiusaHelps ? 'utile' : 'non meglio'} |`,
    `| Soft-adopt | **${promote ? 'sì (adopt=true)' : 'no'}** |`,
    '',
    promote
        ? 'Promozione: `adopt: true` in `cento-crossing-time.json`.'
        : 'Chiusura: crossing-time warp non batte prod sui criteri rising/fase; codice resta soft-load con adopt=false.',
    '',
    `Generato: ${report.generatedAt}`
);

fs.writeFileSync(path.join(root, 'models/ab-m9-prob-crossing.md'), lines.join('\n') + '\n');
console.log(lines.join('\n'));

if (promote) {
    crossWith.adopt = true;
    fs.writeFileSync(
        path.join(root, 'models/cento-crossing-time.json'),
        JSON.stringify(crossWith, null, 2) + '\n'
    );
    console.log('Set adopt=true on cento-crossing-time.json');
}
