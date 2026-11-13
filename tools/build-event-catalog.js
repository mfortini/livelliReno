#!/usr/bin/env node
// Build flood event catalog from storico CSVs (Fase 0).
// Uso: node tools/build-event-catalog.js [--data storico] [--out models/events.json]
//                                        [--test-from 2026-01-01] [--min-rise 0.5]
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const EP = require('../eval-protocol.js');

const root = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const opt = (name, fallback) => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
};

const dataDir = path.resolve(root, opt('data', 'storico'));
const outPath = path.resolve(root, opt('out', 'models/events.json'));
const mdPath = path.resolve(root, opt('md', 'models/events-catalog.md'));
const testFrom = opt('test-from', '2026-01-01');
const minRiseM = Number(opt('min-rise', 0.5));
const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));

function loadSeries(name, dir = dataDir) {
    const file = path.join(dir, `${name}.csv`);
    if (!fs.existsSync(file)) return { times: [], values: [], byTime: new Map() };
    const times = [], values = [], byTime = new Map();
    const lines = fs.readFileSync(file, 'utf8').split('\n');
    for (let i = 1; i < lines.length; i++) {
        const comma = lines[i].indexOf(',');
        if (comma < 0) continue;
        const t = Date.parse(lines[i].slice(0, comma));
        const v = Number(lines[i].slice(comma + 1));
        if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
        // Snap to 30′ lattice for peak detection stability
        const snapped = Math.round(t / EP.STEP_MS) * EP.STEP_MS;
        if (byTime.has(snapped)) continue;
        times.push(snapped);
        values.push(v);
        byTime.set(snapped, v);
    }
    return { times, values, byTime };
}

const names = Object.keys(config.sensors);
const withDeps = names.filter(n => (config.sensors[n].dependencies || []).length);
const stationEvents = {};
const stationSummary = {};
let spanFrom = Infinity;
let spanTo = -Infinity;

for (const name of names) {
    const series = loadSeries(name);
    if (!series.times.length) continue;
    spanFrom = Math.min(spanFrom, series.times[0]);
    spanTo = Math.max(spanTo, series.times.at(-1));
    if (!withDeps.includes(name) && !['Cento', 'Gallo', 'Casalecchio chiusa'].includes(name)) {
        // Still catalog headwater stations that appear in report targets
    }
    const events = EP.detectFloodEvents(series, { minRiseM });
    stationEvents[name] = events;
    const testMs = Date.parse(testFrom);
    const inTest = events.filter(e => e.peakTime >= testMs);
    const inTrain = events.filter(e => e.peakTime < testMs);
    stationSummary[name] = {
        nEvents: events.length,
        nTrain: inTrain.length,
        nTest: inTest.length,
        maxPeak: events.length ? Math.max(...events.map(e => e.peakLevel)) : null,
        maxPeakTrain: inTrain.length ? Math.max(...inTrain.map(e => e.peakLevel)) : null,
        maxPeakTest: inTest.length ? Math.max(...inTest.map(e => e.peakLevel)) : null,
        from: new Date(series.times[0]).toISOString(),
        to: new Date(series.times.at(-1)).toISOString()
    };
}

const payload = EP.buildCatalogPayload(stationEvents, {
    minRiseM,
    testFrom: new Date(Date.parse(testFrom)).toISOString(),
    storico: {
        from: Number.isFinite(spanFrom) ? new Date(spanFrom).toISOString() : null,
        to: Number.isFinite(spanTo) ? new Date(spanTo).toISOString() : null,
        dataDir: path.relative(root, dataDir)
    },
    stationSummary
});

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(payload, null, 2) + '\n');
console.log(`Wrote ${outPath} (${payload.events.length} events)`);

const focus = ['Cento', 'Gallo', 'Casalecchio chiusa', 'Dosso', 'Bonconvento', 'Forcelli'];
const md = [];
md.push('# Catalogo eventi di piena (Fase 0)');
md.push('');
md.push(`Rilevamento automatico: picco locale (±4 h) con alzata ≥ ${minRiseM} m vs minimo nelle 48 h precedenti, picco ≥ p85 stazione; merge &lt; 72 h.`);
md.push('');
md.push(`**Storico:** ${payload.storico.from} → ${payload.storico.to}`);
md.push(`**Test congelato (da):** ${payload.testFrom}`);
md.push(`**Eventi totali:** ${payload.events.length}`);
md.push('');
md.push('## Riepilogo per stazione');
md.push('');
md.push('| Stazione | n | train | test | max peak | max train | max test |');
md.push('|---|---:|---:|---:|---:|---:|---:|');
for (const name of Object.keys(stationSummary).sort()) {
    const s = stationSummary[name];
    if (!s.nEvents) continue;
    md.push(
        `| ${name} | ${s.nEvents} | ${s.nTrain} | ${s.nTest} | ${fmt(s.maxPeak)} | ${fmt(s.maxPeakTrain)} | ${fmt(s.maxPeakTest)} |`
    );
}
md.push('');
for (const name of focus) {
    const list = payload.events.filter(e => e.station === name);
    if (!list.length) continue;
    md.push(`## ${name}`);
    md.push('');
    md.push('| Start | Peak | End | Peak (m) | Rise (m) | Split |');
    md.push('|---|---|---|---:|---:|---|');
    const testMs = Date.parse(testFrom);
    for (const e of list) {
        const split = Date.parse(e.peakTime) >= testMs ? 'test' : 'train';
        md.push(`| ${e.start} | ${e.peakTime} | ${e.end} | ${e.peakLevel} | ${e.riseM} | ${split} |`);
    }
    md.push('');
}
md.push('Il file `models/events.json` è editabile a mano per correzioni (validazione idrologo).');
md.push('');
fs.writeFileSync(mdPath, md.join('\n') + '\n');
console.log(`Wrote ${mdPath}`);

function fmt(v) {
    return Number.isFinite(v) ? v.toFixed(2) : 'n/d';
}
