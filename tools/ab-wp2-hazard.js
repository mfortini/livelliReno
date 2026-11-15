#!/usr/bin/env node
// WP2.2–2.3 A/B: prod vs hazard-gated decode (and optional M9 pinball).
//   node tools/ab-wp2-hazard.js [--gate-p 0.35] [--from 2026-01-01]
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
const H = 4;
const every = Number(opt('every', '120')) * 60000;
const gateP = Number(opt('gate-p', '0.35'));
const dataDir = path.resolve(root, 'storico');
const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
const coeffs = JSON.parse(fs.readFileSync(path.join(root, 'models/transfer-coeffs.json'), 'utf8'));
const eventsDoc = JSON.parse(fs.readFileSync(path.join(root, 'models/events.json'), 'utf8'));
const hazardPath = path.join(root, opt('hazard', 'models/cento-crossing-hazard.json'));
const pinballPath = path.join(root, 'models/cento-crossing-time.json');

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
const from = Date.parse(opt('from', '2026-01-01'));
const events = (eventsDoc.events || []).filter(e =>
    e.station === station && Date.parse(e.end) >= from
);

function empty() {
    return { n: 0, se: 0, seP: 0, risingN: 0, risingSe: 0, fallingN: 0, fallingSe: 0, decodeOn: 0 };
}

function finalize(b) {
    const rmse = b.n ? Math.sqrt(b.se / b.n) : null;
    const rmseP = b.n ? Math.sqrt(b.seP / b.n) : null;
    return {
        n: b.n,
        rmse,
        skillPersist: rmseP > 0 ? 1 - rmse / rmseP : null,
        risingRmse: b.risingN ? Math.sqrt(b.risingSe / b.risingN) : null,
        fallingRmse: b.fallingN ? Math.sqrt(b.fallingSe / b.fallingN) : null,
        decodeOn: b.decodeOn
    };
}

function run(label, crossing) {
    const node = crossing
        ? { ...prodNode, crossingTime: { ...crossing, adopt: true, decodeGateP: gateP }, reliable: true }
        : prodNode;
    const bucket = empty();
    for (const ev of events) {
        const peak = Date.parse(ev.peakTime);
        const start = Math.max(from, Date.parse(ev.start));
        const end = Date.parse(ev.end) - H * STEP;
        for (let now = Math.ceil(start / every) * every; now <= end; now += every) {
            const down = win(station, data[station], now);
            const anchor = P.latestFinite(down.timeSeries);
            if (!anchor || now - +anchor.time > STEP) continue;
            const y0 = data[station].byTime.get(now);
            const yH = data[station].byTime.get(now + H * STEP);
            if (!Number.isFinite(y0) || !Number.isFinite(yH)) continue;
            const sensorMap = Object.fromEntries(names.map(n => [n, win(n, data[n], now)]));
            sensorMap[station] = down;
            const ups = deps.map(u => sensorMap[u]);
            const rainSensors = rainGauges.map(n => win(n, rainData[n], now)).filter(s => s.timeSeries.length);
            const result = P.predictNode(down, ups, {
                hours: 24,
                now: new Date(now),
                coeffs: { nodes: { [station]: node } },
                horizonHours: 2,
                downstreamId: station,
                rainSensors,
                sensorMap,
                smooth: 0,
                useCrossingTime: !!crossing
            });
            if (!result?.ok) continue;
            if (result.crossing?.warped) bucket.decodeOn++;
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
            bucket.n++;
            bucket.se += err * err;
            bucket.seP += (y0 - yH) ** 2;
            if (now < peak) { bucket.risingN++; bucket.risingSe += err * err; }
            else { bucket.fallingN++; bucket.fallingSe += err * err; }
        }
    }
    return { label, ...finalize(bucket) };
}

const models = { prod: run('prod', null) };
if (fs.existsSync(hazardPath)) {
    const haz = JSON.parse(fs.readFileSync(hazardPath, 'utf8'));
    models.hazard_gated = run('hazard_gated', haz);
}
if (fs.existsSync(pinballPath)) {
    const pb = JSON.parse(fs.readFileSync(pinballPath, 'utf8'));
    models.pinball_m9 = run('pinball_m9', pb);
}

const report = {
    kind: 'wp2_hazard_gate',
    from: new Date(from).toISOString(),
    gateP,
    events: events.length,
    models,
    decision: null
};
const base = models.prod;
const haz = models.hazard_gated;
let promote = null;
let reason = 'hazard missing or worse than prod on rising RMSE @120';
if (haz && base?.risingRmse != null && haz.risingRmse != null) {
    if (haz.risingRmse <= base.risingRmse * 1.02 && (haz.skillPersist ?? 0) >= (base.skillPersist ?? 0) - 0.02) {
        promote = 'hazard_gated';
        reason = 'hazard gated matches/beats prod rising+skill @120';
    } else {
        reason = `hazard rising=${haz.risingRmse?.toFixed(3)} vs prod ${base.risingRmse?.toFixed(3)}; soft-adopt off`;
    }
}
report.decision = { promote, reason, adopt: false };

const outJson = path.join(root, 'models/ab-wp2-hazard.json');
const outMd = path.join(root, 'models/ab-wp2-hazard.md');
fs.writeFileSync(outJson, JSON.stringify(report, null, 2) + '\n');
const lines = [
    '# WP2 — hazard discreto + gate decode',
    '',
    `**Gate P(crossing):** ${gateP}`,
    `**Slice:** eventi Cento ≥ ${report.from} every ${every / 60000}′ @120′`,
    '',
    '| model | n | RMSE | skill | rising | falling | decodeOn |',
    '|---|---:|---:|---:|---:|---:|---:|'
];
for (const [k, m] of Object.entries(models)) {
    lines.push(
        `| ${k} | ${m.n} | ${m.rmse?.toFixed(3)} | ${m.skillPersist?.toFixed(3)} | ` +
        `${m.risingRmse?.toFixed(3)} | ${m.fallingRmse?.toFixed(3)} | ${m.decodeOn ?? 0} |`
    );
}
lines.push('', `**promote:** \`${promote}\``, `**motivo:** ${reason}`, `**adopt:** false`, '');
fs.writeFileSync(outMd, lines.join('\n'));
console.log(JSON.stringify(report.decision, null, 2));
console.log(`Wrote ${outMd}`);
