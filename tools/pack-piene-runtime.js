#!/usr/bin/env node
'use strict';
/*
 * pack-piene-runtime.js — build compact runtime packs for piene.html.
 *
 * Inputs (read-only):
 *   models/events.json                      catalogo eventi per stazione
 *   models/eval-table.json                  metriche aggregate per stazione/modello/orizzonte
 *   models/eval-per-event.json              metriche per evento (solo Cento)
 *   models/report-piene-lag-align.json meta + serie monte-valle allineate al picco
 *
 * Outputs:
 *   models/piene-runtime.json        pack compatto per la pagina (core, target <= 0.5 MB raw)
 *   models/piene-cento-series.json   serie monte-valle Cento, caricate lazy dalla pagina
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const MODELS = path.join(ROOT, 'models');

function readJson(name) {
    return JSON.parse(fs.readFileSync(path.join(MODELS, name), 'utf8'));
}

function round(value, digits = 4) {
    if (value == null || !Number.isFinite(value)) return null;
    const f = 10 ** digits;
    return Math.round(value * f) / f;
}

const events = readJson('events.json');
const evalTable = readJson('eval-table.json');
const evalPerEvent = readJson('eval-per-event.json');
const report = readJson('report-piene-lag-align.json');

// ---------------------------------------------------------------------------
// Dictionaries
// ---------------------------------------------------------------------------
const MODEL_ORDER = ['persist', 'poly2', 'arx', 'ridge', 'prod', 'seq_gru', 'seq_cnn', 'muskingum_gru', 'rc_gru'];
const stationNames = Object.keys(evalTable.stations);
const stationIdx = new Map(stationNames.map((s, i) => [s, i]));

const modelSet = new Set();
for (const st of stationNames) for (const m of Object.keys(evalTable.stations[st])) modelSet.add(m);
const modelNames = [
    ...MODEL_ORDER.filter(m => modelSet.has(m)),
    ...[...modelSet].filter(m => !MODEL_ORDER.includes(m)).sort()
];
const modelIdx = new Map(modelNames.map((m, i) => [m, i]));

const horizons = (evalTable.protocol && evalTable.protocol.horizonsMinutes)
    ? [...evalTable.protocol.horizonsMinutes]
    : [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360];
const horizonIdx = new Map(horizons.map((h, i) => [h, i]));

// Fixed metric order for the compact aggregate rows. Keep in sync with piene.html.
const METRICS = [
    'nEvents', 'rmse', 'skillEventPersist', 'skillLo', 'skillHi',
    'nseEvent', 'kgeEvent', 'phaseLagMin', 'risingRmse', 'fallingRmse',
    'peakLevelErrMed', 'peakTimeErrMed',
    'podY', 'podO', 'podR', 'farY', 'farO', 'farR',
    'csiY', 'csiO', 'csiR', 'bandCoverageEvent'
];

// ---------------------------------------------------------------------------
// Cento per-event rows: pooled RMSE per model/horizon + compact rows
// ---------------------------------------------------------------------------
const centoRows = evalPerEvent.rows.filter(r => r.station === 'Cento');
const pooled = new Map(); // `${model}|${horizon}` -> {se, n}
for (const r of centoRows) {
    if (!Number.isFinite(r.rmse) || !Number.isFinite(r.n) || r.n <= 0) continue;
    const key = `${r.model}|${r.horizonMin}`;
    const acc = pooled.get(key) || { se: 0, n: 0 };
    acc.se += r.rmse * r.rmse * r.n;
    acc.n += r.n;
    pooled.set(key, acc);
}
function pooledRmse(model, horizonMin) {
    const acc = pooled.get(`${model}|${horizonMin}`);
    if (!acc || !acc.n) return null;
    return Math.sqrt(acc.se / acc.n);
}

// ---------------------------------------------------------------------------
// Cento event catalog (from events.json) + link to report meta by peakTime
// ---------------------------------------------------------------------------
const testFrom = events.testFrom || evalTable.from || null;
const metaByPeak = new Map();
for (const m of report.events || []) {
    if (m && m.peakTime) metaByPeak.set(m.peakTime, { id: Number(m.id), meta: m });
}

const catalog = Object.values(events.events)
    .filter(e => e.station === 'Cento')
    .sort((a, b) => a.peakTime < b.peakTime ? -1 : 1)
    .map(e => {
        const linked = metaByPeak.get(e.peakTime);
        const row = {
            start: e.start,
            end: e.end,
            peakTime: e.peakTime,
            peakLevel: round(e.peakLevel, 2),
            riseM: round(e.riseM, 2),
            split: testFrom && e.peakTime >= testFrom ? 'test' : 'train'
        };
        if (linked) {
            row.metaId = linked.id;
            if (linked.meta.perEventLag) row.perEventLag = linked.meta.perEventLag;
        }
        return row;
    });

// index of catalog events by eventKey used in eval-per-event ("Cento|<peakTime>")
const catalogIdxByKey = new Map(catalog.map((c, i) => [`Cento|${c.peakTime}`, i]));

// Compact per-event rows: [catalogIdx, modelIdx, horizonIdx, n, rmse, skillPersist, risingRmse, fallingRmse, nse, kge]
const centoPerEvent = [];
for (const r of centoRows) {
    const ci = catalogIdxByKey.get(r.eventKey);
    if (ci == null) continue;
    centoPerEvent.push([
        ci,
        modelIdx.get(r.model),
        horizonIdx.get(r.horizonMin),
        r.n ?? null,
        round(r.rmse),
        round(r.skillPersist),
        round(r.risingRmse),
        round(r.fallingRmse),
        round(r.nse),
        round(r.kge)
    ]);
}

// ---------------------------------------------------------------------------
// Aggregate metrics per station/model/horizon
// ---------------------------------------------------------------------------
function extractMetrics(model, row) {
    if (!row) return null;
    const skillCI = row.skillEventPersistCI || {};
    const out = [
        row.nEvents ?? null,
        round(pooledRmse(model, undefined)), // placeholder, replaced below per-horizon
        round(row.skillEventPersist),
        round(skillCI.lo),
        round(skillCI.hi),
        round(row.nseEvent),
        round(row.kgeEvent),
        round(row.phaseLagMin, 1),
        round(row.risingRmse),
        round(row.fallingRmse),
        round(row.peakLevelErrMed, 3),
        row.peakTimeErrMed == null ? null : Math.round(row.peakTimeErrMed),
        round(row.pod && row.pod.yellow),
        round(row.pod && row.pod.orange),
        round(row.pod && row.pod.red),
        round(row.far && row.far.yellow),
        round(row.far && row.far.orange),
        round(row.far && row.far.red),
        round(row.csi && row.csi.yellow),
        round(row.csi && row.csi.orange),
        round(row.csi && row.csi.red),
        round(row.bandCoverageEvent)
    ];
    return out;
}

// agg[stationIdx][modelIdx][horizonIdx] = metrics array (only when present)
const agg = {};
for (const st of stationNames) {
    const si = stationIdx.get(st);
    const models = evalTable.stations[st];
    for (const m of Object.keys(models)) {
        const mi = modelIdx.get(m);
        for (const hKey of Object.keys(models[m])) {
            const h = Number(hKey);
            const hi = horizonIdx.get(h);
            if (hi == null) continue;
            const arr = extractMetrics(m, models[m][hKey]);
            if (!arr) continue;
            // pooled rmse is computed per model/horizon (Cento only; null elsewhere)
            arr[1] = round(pooledRmse(m, h));
            if (!agg[si]) agg[si] = {};
            if (!agg[si][mi]) agg[si][mi] = {};
            agg[si][mi][hi] = arr;
        }
    }
}

// ---------------------------------------------------------------------------
// Station info (counts, peaks) from events.json
// ---------------------------------------------------------------------------
const stationInfo = {};
for (const st of Object.keys(events.stations || {})) {
    const s = events.stations[st];
    stationInfo[st] = {
        nEvents: s.nEvents ?? null,
        nTrain: s.nTrain ?? null,
        nTest: s.nTest ?? null,
        maxPeak: round(s.maxPeak, 2),
        maxPeakTest: round(s.maxPeakTest, 2)
    };
}

// Series availability: which event ids have charts
const seriesIds = (report.events || []).map(c => c.id).sort((a, b) => a - b);

const pack = {
    kind: 'piene-runtime',
    version: 1,
    generatedAt: new Date().toISOString(),
    disclaimer: events.disclaimer || evalTable.disclaimer || 'Sperimentale: catalogo automatico per valutazione; non allertistico.',
    caveat: evalTable.caveat || null,
    testFrom,
    detection: events.detection || null,
    stations: stationNames,
    models: modelNames,
    horizons,
    metrics: METRICS,
    stationInfo,
    catalog,
    agg,
    centoPerEvent,
    tau: report.tau || null,
    monteValleStations: (report.stations || []).map(s => s.name),
    seriesIds
};

// ---------------------------------------------------------------------------
// Series pack: shared hours axis (0.5 h step), segment offsets per event
// ---------------------------------------------------------------------------
const charts = report.events || [];
const mvStations = (report.stations || []).map(s => s.name);
let axisMin = Infinity;
let axisMax = -Infinity;
for (const c of charts) {
    axisMin = Math.min(axisMin, c.hours[0]);
    axisMax = Math.max(axisMax, c.hours[c.hours.length - 1]);
}
const STEP = (Number(report.sampleMinutes) || 60) / 60;
const nAxis = Math.round((axisMax - axisMin) / STEP) + 1;
const hoursAxis = Array.from({ length: nAxis }, (_, i) => round(axisMin + i * STEP, 1));

const seriesEvents = {};
for (const c of charts) {
    const meta = c;
    const off = Math.round((c.hours[0] - axisMin) / STEP);
    const levels = {};
    mvStations.forEach((name, i) => {
        const arr = c.levels[name];
        if (!arr) return;
        levels[i] = arr.map(v => (v == null || !Number.isFinite(v)) ? null : round(v, 2));
    });
    seriesEvents[c.id] = {
        off,
        n: c.hours.length,
        split: meta ? meta.split : null,
        peakTime: meta ? meta.peakTime : null,
        label: meta ? meta.label : null,
        levels
    };
}

const seriesPack = {
    kind: 'piene-cento-series',
    version: 1,
    generatedAt: pack.generatedAt,
    disclaimer: pack.disclaimer,
    stations: mvStations,
    axisStart: round(axisMin, 1),
    axisStep: STEP,
    hours: hoursAxis,
    events: seriesEvents
};

// ---------------------------------------------------------------------------
// Write outputs
// ---------------------------------------------------------------------------
const outCore = path.join(MODELS, 'piene-runtime.json');
const outSeries = path.join(MODELS, 'piene-cento-series.json');
fs.writeFileSync(outCore, JSON.stringify(pack));
fs.writeFileSync(outSeries, JSON.stringify(seriesPack));

const sizeCore = fs.statSync(outCore).size;
const sizeSeries = fs.statSync(outSeries).size;
console.log(`piene-runtime.json: ${(sizeCore / 1024).toFixed(1)} KiB (${sizeCore} bytes)`);
console.log(`piene-cento-series.json: ${(sizeSeries / 1024).toFixed(1)} KiB (${sizeSeries} bytes)`);
console.log(`catalogo Cento: ${catalog.length} eventi (${catalog.filter(c => c.split === 'test').length} test)`);
console.log(`righe per-evento Cento: ${centoPerEvent.length}`);
console.log(`serie monte-valle: ${Object.keys(seriesEvents).length} eventi, asse ${hoursAxis.length} punti (${hoursAxis[0]}..${hoursAxis[hoursAxis.length - 1]} h)`);
if (sizeCore > 512 * 1024) {
    console.warn('ATTENZIONE: pack core oltre 0.5 MB raw');
}
