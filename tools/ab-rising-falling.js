#!/usr/bin/env node
// Rising vs falling limb RMSE for Cento MLP (prod ± onlineCorrection).
// Split by catalog peakTime: issue < peak → rising; issue >= peak → falling.
//
// Uso:
//   node tools/ab-rising-falling.js [--from 2026-01-01] [--to 2026-11-15]
//     [--events models/events.json] [--station Cento] [--json models/ab-rising-falling-nov.json]
//     [--md models/ab-rising-falling-nov.md]
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

const station = opt('station', 'Cento');
const dataDir = path.resolve(root, opt('data', 'storico'));
const eventsPath = path.resolve(root, opt('events', 'models/events.json'));
const coeffsPath = path.resolve(root, opt('coeffs', 'models/transfer-coeffs.json'));
const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
const coeffs = JSON.parse(fs.readFileSync(coeffsPath, 'utf8'));
const eventsDoc = JSON.parse(fs.readFileSync(eventsPath, 'utf8'));
const STEP = 30 * 60000;
const WINDOW = 72 * 3600000;
const HORIZONS = [1, 4, 8]; // 30′, 120′, 240′

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

function emptyBucket() {
    return Object.fromEntries(HORIZONS.map(h => [h, { n: 0, se: 0, seP: 0, ae: 0 }]));
}

function addErr(bucket, h, err, errP) {
    const b = bucket[h];
    b.n++;
    b.se += err * err;
    b.seP += errP * errP;
    b.ae += Math.abs(err);
}

function finalize(bucket) {
    const out = {};
    for (const [h, b] of Object.entries(bucket)) {
        if (!b.n) {
            out[h] = { n: 0, rmse: null, mae: null, skillPersist: null };
            continue;
        }
        const rmse = Math.sqrt(b.se / b.n);
        const rmseP = Math.sqrt(b.seP / b.n);
        out[h] = {
            n: b.n,
            rmse,
            mae: b.ae / b.n,
            skillPersist: rmseP > 0 ? 1 - rmse / rmseP : null
        };
    }
    return out;
}

const names = Object.keys(config.sensors);
const data = Object.fromEntries(names.map(name => [name, loadSeries(name)]));
const rainGauges = ((config.rainfall && config.rainfall.gauges) || []).map(e =>
    typeof e === 'string' ? e : e?.name
).filter(Boolean);
const rainDir = path.join(dataDir, 'rain');
const rainData = Object.fromEntries(rainGauges.map(name => [name, loadSeries(name, rainDir)]));

const allTimes = data[station].times;
const from = Date.parse(opt('from', '2026-01-01'));
const to = Date.parse(opt('to', new Date(allTimes.at(-1)).toISOString()));
const deps = config.sensors[station]?.dependencies || [];
const node = coeffs.nodes?.[station];
if (!node || node.kind !== 'window_delta_mlp') {
    console.error(`Need window_delta_mlp node for ${station}`);
    process.exit(1);
}
const forced = { ...node, reliable: true };

const stationEvents = (eventsDoc.events || []).filter(e =>
    e.station === station && Date.parse(e.end) >= from && Date.parse(e.start) <= to
);
const novEvents = stationEvents.filter(e => Date.parse(e.start) >= Date.parse('2026-11-01'));

const modes = [
    { key: 'prod', onlineCorrection: false },
    { key: 'mlp_online', onlineCorrection: true }
];

function runMode(mode, eventList, label, issueStep = STEP) {
    const rising = emptyBucket();
    const falling = emptyBucket();
    const all = emptyBucket();
    let used = 0;
    for (const ev of eventList) {
        const peak = Date.parse(ev.peakTime);
        const start = Math.max(from, Date.parse(ev.start));
        const end = Math.min(to, Date.parse(ev.end) - Math.max(...HORIZONS) * STEP);
        for (let now = Math.ceil(start / issueStep) * issueStep; now <= end; now += issueStep) {
            const down = windowSensor(station, data[station], now);
            const anchor = P.latestFinite(down.timeSeries);
            if (!anchor || now - +anchor.time > STEP) continue;
            const y0 = data[station].byTime.get(now);
            if (!Number.isFinite(y0)) continue;
            const sensorMap = Object.fromEntries(names.map(n => [n, windowSensor(n, data[n], now)]));
            sensorMap[station] = down;
            const ups = deps.map(u => sensorMap[u]);
            const rainSensors = rainGauges
                .map(n => windowSensor(n, rainData[n], now))
                .filter(s => s.timeSeries.length);
            const result = P.predictNode(down, ups, {
                hours: 72,
                now: new Date(now),
                coeffs: { nodes: { [station]: forced } },
                horizonHours: Math.max(...HORIZONS) * 0.5,
                downstreamId: station,
                rainSensors,
                sensorMap,
                smooth: 0,
                onlineCorrection: mode.onlineCorrection
            });
            if (!result?.ok) continue;
            used++;
            const limb = now < peak ? rising : falling;
            for (const h of HORIZONS) {
                const target = now + h * STEP;
                const obs = data[station].byTime.get(target);
                if (!Number.isFinite(obs)) continue;
                let best = null;
                let bestDist = Infinity;
                for (const p of result.forecast || []) {
                    if (p.regime === 'anchor') continue;
                    const dist = Math.abs(+p.time - target);
                    if (dist < bestDist) {
                        best = p;
                        bestDist = dist;
                    }
                }
                if (!best || bestDist > STEP) continue;
                const pred = best.med ?? best.value;
                if (!Number.isFinite(pred)) continue;
                const err = pred - obs;
                const errP = y0 - obs;
                addErr(limb, h, err, errP);
                addErr(all, h, err, errP);
            }
        }
    }
    return {
        label,
        mode: mode.key,
        usedIssues: used,
        events: eventList.length,
        rising: finalize(rising),
        falling: finalize(falling),
        all: finalize(all)
    };
}

const report = {
    kind: 'ab_rising_falling',
    station,
    from: new Date(from).toISOString(),
    to: new Date(to).toISOString(),
    horizonsMin: HORIZONS.map(h => h * 30),
    novEvent: novEvents[0] || null,
    slices: []
};

for (const mode of modes) {
    // Dense 30′ on the Nov flood; 2 h stride on the full test catalog (online corr is costly).
    if (novEvents.length) report.slices.push(runMode(mode, novEvents, 'nov_event', STEP));
    if (stationEvents.length) report.slices.push(runMode(mode, stationEvents, 'test_events', 4 * STEP));
}

function fmt(v, d = 3) {
    return v == null || !Number.isFinite(v) ? '—' : v.toFixed(d);
}

function mdTable(slice) {
    const lines = [
        `### ${slice.label} / \`${slice.mode}\` (issues=${slice.usedIssues}, events=${slice.events})`,
        '',
        '| Limb | h | n | RMSE | MAE | skill vs persist |',
        '|---|---:|---:|---:|---:|---:|'
    ];
    for (const limb of ['rising', 'falling', 'all']) {
        for (const h of HORIZONS) {
            const row = slice[limb][h];
            lines.push(
                `| ${limb} | ${h * 30}′ | ${row.n} | ${fmt(row.rmse)} | ${fmt(row.mae)} | ${fmt(row.skillPersist, 3)} |`
            );
        }
    }
    return lines.join('\n');
}

const mdPath = path.resolve(root, opt('md', 'models/ab-rising-falling-nov.md'));
const jsonPath = path.resolve(root, opt('json', 'models/ab-rising-falling-nov.json'));
const md = [
    '# Rising vs falling — Cento MLP',
    '',
    `**Station:** ${station}`,
    `**Window:** ${report.from} → ${report.to}`,
    report.novEvent
        ? `**Nov event:** peak ${report.novEvent.peakLevel} m @ ${report.novEvent.peakTime} (start ${report.novEvent.start})`
        : '**Nov event:** none',
    '',
    'Split: issue time `< peakTime` → rising; else falling. Candidates: `prod` vs `mlp_online` (phase+bias).',
    '',
    ...report.slices.map(mdTable),
    '',
    `Generato: ${new Date().toISOString()}`
].join('\n');

fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
fs.writeFileSync(mdPath, md);
console.log(`Wrote ${mdPath}`);
console.log(`Wrote ${jsonPath}`);
for (const slice of report.slices) {
    const r120 = slice.rising[4];
    const f120 = slice.falling[4];
    console.log(
        `[${slice.label}/${slice.mode}] rising@120 RMSE=${fmt(r120.rmse)} (n=${r120.n}) ` +
        `falling@120 RMSE=${fmt(f120.rmse)} (n=${f120.n})`
    );
}
