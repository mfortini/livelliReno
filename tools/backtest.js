#!/usr/bin/env node
// Backtest del codice di predizione (predict-transfer.js) sullo storico.
// Campionamento: ogni --every minuti (default 120) + densificazione 30′ nelle finestre
// in cui |Δy| a +2h ≥ --event (default 0.10 m).
//
// Uso: node tools/backtest.js [--data storico] [--every 120] [--event 0.10]
//                             [--coeffs models/transfer-coeffs.json] [--no-coeffs]
//                             [--from 2025-01-01] [--to 2026-11-01] [--json out.json]
//                             [--no-update-reliable]
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
const dataDir = path.resolve(root, opt('data', 'storico'));
const everyMin = Number(opt('every', 120));
const eventThreshold = Number(opt('event', 0.10));
const coeffsPath = path.resolve(root, opt('coeffs', 'models/transfer-coeffs.json'));
const useCoeffs = !args.includes('--no-coeffs');
const jsonOut = opt('json', null);
const updateReliable = !args.includes('--no-update-reliable');
const SKILL_KEEP = P.SKILL_KEEP || 0.10;
const SKILL_DRIFT_WARN = 0.05;

const STEP = 30 * 60000;
const WINDOW = 72 * 3600000;
const MA_STEPS = 6; // 3-hour moving average baseline
const EVENT_PAD = 6 * 3600000; // densifica ±6h attorno a ogni evento

const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
const coeffs = useCoeffs && fs.existsSync(coeffsPath) ? JSON.parse(fs.readFileSync(coeffsPath, 'utf8')) : null;
const HORIZONS = (() => {
    const fromCoeffs = (coeffs?.horizonsMinutes || []).map(m => m / 30).filter(h => Number.isFinite(h) && h > 0);
    if (fromCoeffs.length) return [...new Set(fromCoeffs)].sort((a, b) => a - b);
    return [1, 2, 3, 4];
})();

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

function movingAverageAt(series, now, steps = MA_STEPS) {
    let sum = 0, n = 0;
    for (let i = 0; i < steps; i++) {
        const v = series.byTime.get(now - i * STEP);
        if (!Number.isFinite(v)) continue;
        sum += v;
        n++;
    }
    return n ? sum / n : null;
}

const names = Object.keys(config.sensors);
const data = Object.fromEntries(names.map(name => [name, loadSeries(name)]));
const rainGauges = (config.rainfall && config.rainfall.gauges) || [];
const rainDir = path.join(dataDir, 'rain');
const rainData = Object.fromEntries(rainGauges.map(name => [name, loadSeries(name, rainDir)]));
const allTimes = names.flatMap(name => [data[name].times[0], data[name].times.at(-1)]).filter(Number.isFinite);
if (!allTimes.length) {
    console.error(`Nessun CSV in ${dataDir}: esegui prima tools/fetch-storico.py`);
    process.exit(1);
}
const from = Date.parse(opt('from', new Date(Math.min(...allTimes) + WINDOW).toISOString()));
const to = Date.parse(opt('to', new Date(Math.max(...allTimes) - Math.max(...HORIZONS) * STEP).toISOString()));

const ranks = P.rankNodes(config);
const nodes = names
    .filter(name => (config.sensors[name].dependencies || []).length)
    .sort((a, b) => (ranks[b] || 0) - (ranks[a] || 0) || a.localeCompare(b));

function emptyStats() {
    return {
        n: 0, seModel: 0, sePersist: 0, seMa3: 0,
        aeModel: 0, aePersist: 0, aeMa3: 0,
        covered: 0, asked: 0
    };
}

function collectNows(node) {
    const series = data[node];
    const set = new Set();
    const first = Math.ceil(from / STEP) * STEP;
    for (let now = first; now <= to; now += everyMin * 60000) set.add(now);
    // densifica intorno agli eventi (Δ a +2h)
    for (let i = 0; i < series.times.length; i++) {
        const t = series.times[i];
        if (t < from || t > to) continue;
        const future = series.byTime.get(t + 4 * STEP);
        if (!Number.isFinite(future)) continue;
        if (Math.abs(future - series.values[i]) < eventThreshold) continue;
        const start = Math.max(from, t - EVENT_PAD);
        const end = Math.min(to, t + EVENT_PAD);
        for (let now = Math.ceil(start / STEP) * STEP; now <= end; now += STEP) set.add(now);
    }
    return [...set].sort((a, b) => a - b);
}

const report = {};
const t0 = Date.now();
for (const node of nodes) {
    const ups = config.sensors[node].dependencies;
    const nodeHorizonKeys = Object.keys(coeffs?.nodes?.[node]?.horizons || {})
        .map(Number).filter(Number.isFinite).sort((a, b) => a - b);
    const nodeHorizons = nodeHorizonKeys.length
        ? nodeHorizonKeys.map(min => min / 30)
        : HORIZONS.filter(h => h <= 4);
    const stats = {
        all: Object.fromEntries(nodeHorizons.map(h => [h, emptyStats()])),
        event: Object.fromEntries(nodeHorizons.map(h => [h, emptyStats()])),
        horizons: nodeHorizons,
        sources: {}, calls: 0, failed: 0
    };
    const nows = collectNows(node);
    const horizonHours = Math.max(...nodeHorizons) * 0.5;
    for (const now of nows) {
        const down = windowSensor(node, data[node], now);
        const anchor = P.latestFinite(down.timeSeries);
        if (!anchor || now - +anchor.time > STEP) continue;
        const rainSensors = rainGauges
            .map(name => windowSensor(name, rainData[name], now))
            .filter(s => s.timeSeries.length);
        const result = P.predictNode(down, ups.map(up => windowSensor(up, data[up], now)), {
            hours: 72, now: new Date(now), coeffs, horizonHours, downstreamId: node, rainSensors
        });
        stats.calls++;
        if (!result.ok) { stats.failed++; continue; }
        stats.sources[result.source] = (stats.sources[result.source] || 0) + 1;
        const byTime = new Map((result.forecast || []).map(point => [+point.time, point]));
        const ma3 = movingAverageAt(data[node], now);
        for (const h of nodeHorizons) {
            const actual = data[node].byTime.get(now + h * STEP);
            if (!Number.isFinite(actual)) continue;
            const isEvent = Math.abs(actual - anchor.value) >= eventThreshold;
            stats.all[h].asked++;
            if (isEvent) stats.event[h].asked++;
            const point = byTime.get(now + h * STEP);
            if (!point || !Number.isFinite(point.med)) continue;
            for (const bucket of isEvent ? [stats.all[h], stats.event[h]] : [stats.all[h]]) {
                bucket.n++;
                bucket.seModel += (point.med - actual) ** 2;
                bucket.sePersist += (anchor.value - actual) ** 2;
                bucket.aeModel += Math.abs(point.med - actual);
                bucket.aePersist += Math.abs(anchor.value - actual);
                if (Number.isFinite(ma3)) {
                    bucket.seMa3 += (ma3 - actual) ** 2;
                    bucket.aeMa3 += Math.abs(ma3 - actual);
                }
                if (actual >= point.min && actual <= point.max) bucket.covered++;
            }
        }
    }
    report[node] = stats;
}

const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : 'n/d';
const pct = value => Number.isFinite(value) ? `${Math.round(value * 100)}%` : 'n/d';
function skillOf(bucket, against = 'persist') {
    if (!bucket.n) return null;
    const rmseModel = Math.sqrt(bucket.seModel / bucket.n);
    const rmseBase = against === 'ma3'
        ? Math.sqrt(bucket.seMa3 / bucket.n)
        : Math.sqrt(bucket.sePersist / bucket.n);
    if (!rmseBase) return null;
    return 1 - rmseModel / rmseBase;
}
function line(label, bucket) {
    if (!bucket.asked) return `    ${label}: nessun campione`;
    const rmseModel = Math.sqrt(bucket.seModel / bucket.n);
    const rmsePersist = Math.sqrt(bucket.sePersist / bucket.n);
    const rmseMa3 = Math.sqrt(bucket.seMa3 / bucket.n);
    return `    ${label}: n=${String(bucket.n).padStart(5)} disp=${pct(bucket.n / bucket.asked).padStart(4)}`
        + ` RMSE mod=${fmt(rmseModel)} pers=${fmt(rmsePersist)} ma3=${fmt(rmseMa3)}`
        + ` skill/pers=${fmt(skillOf(bucket, 'persist'), 2).padStart(6)}`
        + ` skill/ma3=${fmt(skillOf(bucket, 'ma3'), 2).padStart(6)}`
        + ` MAE mod=${fmt(bucket.aeModel / bucket.n)}`
        + ` copertura=${pct(bucket.covered / bucket.n)}`;
}

console.log(`Backtest ${new Date(from).toISOString()} → ${new Date(to).toISOString()} · base ogni ${everyMin} min + eventi densificati · coeff offline: ${coeffs ? 'sì' : 'no'} · evento = |Δ| ≥ ${eventThreshold} m · orizzonti=${HORIZONS.map(h => h * 30).join(',')}′ · ${((Date.now() - t0) / 1000).toFixed(1)} s`);
const summary = {};
for (const node of nodes) {
    const stats = report[node];
    const sources = Object.entries(stats.sources).map(([key, value]) => `${key} ${pct(value / Math.max(1, stats.calls - stats.failed))}`).join(', ');
    console.log(`\n${node}  ← ${config.sensors[node].dependencies.join(' + ')}   (chiamate ${stats.calls}, fallite ${stats.failed}; fonte: ${sources || 'n/d'})`);
    for (const h of stats.horizons) console.log(line(`+${String(h * 30).padStart(3)}′ tutti `, stats.all[h]));
    for (const h of stats.horizons) console.log(line(`+${String(h * 30).padStart(3)}′ eventi`, stats.event[h]));
    const event120 = stats.event[4] || null;
    const all120 = stats.all[4] || null;
    summary[node] = {
        skill120: event120 ? skillOf(event120, 'persist') : null,
        skill120Ma3: event120 ? skillOf(event120, 'ma3') : null,
        skill120All: all120 ? skillOf(all120, 'persist') : null,
        coverage120: event120 && event120.n ? event120.covered / event120.n : null,
        sources: stats.sources,
        calls: stats.calls,
        failed: stats.failed
    };
}

// Skill drift vs previous summary
if (jsonOut && fs.existsSync(jsonOut)) {
    try {
        const prev = JSON.parse(fs.readFileSync(jsonOut, 'utf8'));
        for (const node of nodes) {
            const oldSkill = prev.summary?.[node]?.skill120;
            const newSkill = summary[node]?.skill120;
            if (Number.isFinite(oldSkill) && Number.isFinite(newSkill) && oldSkill - newSkill > SKILL_DRIFT_WARN) {
                console.warn(
                    `WARNING: ${node} skill dropped by ${(oldSkill - newSkill).toFixed(3)} `
                    + `(${oldSkill.toFixed(3)} → ${newSkill.toFixed(3)})`
                );
            }
        }
    } catch {
        // ignore unreadable previous summary
    }
}

if (jsonOut) {
    fs.writeFileSync(jsonOut, JSON.stringify({ from, to, everyMin, eventThreshold, horizons: HORIZONS, summary, report }, null, 2));
    console.log(`\nWrote ${jsonOut}`);
}

// Update reliable flags on transfer-coeffs from backtest event skill@120
if (updateReliable && coeffs?.nodes) {
    let changed = 0;
    for (const node of nodes) {
        const skill = summary[node]?.skill120;
        if (!Number.isFinite(skill) || !coeffs.nodes[node]) continue;
        const reliable = skill >= SKILL_KEEP;
        coeffs.nodes[node].backtestSkill120 = Math.round(skill * 1e6) / 1e6;
        if (coeffs.nodes[node].reliable !== reliable) changed++;
        coeffs.nodes[node].reliable = reliable;
        for (const up of coeffs.nodes[node].upstreams || []) {
            const edge = coeffs.edges?.[`${up}->${node}`];
            if (edge) {
                edge.reliable = reliable;
                edge.skill120 = coeffs.nodes[node].backtestSkill120;
            }
        }
        const flag = reliable ? 'OK' : 'no-skill';
        console.log(`reliable ${node}: backtest skill120=${skill.toFixed(4)} [${flag}]`);
    }
    coeffs.skillKeep = SKILL_KEEP;
    fs.writeFileSync(coeffsPath, JSON.stringify(coeffs, null, 2) + '\n');
    console.log(`Updated reliable flags in ${coeffsPath} (${changed} changed, threshold=${SKILL_KEEP})`);
}
