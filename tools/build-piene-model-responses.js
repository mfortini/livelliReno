#!/usr/bin/env node
'use strict';
/*
 * Build rolling hindcast responses for the Cento flood-event overlay.
 *
 * For each event sample at target time T, the stored prediction is the
 * production MLP forecast for T issued at T - horizonMin. This is causal:
 * only observations up to the issue time are visible to the model.
 */

const fs = require('node:fs');
const path = require('node:path');
const P = require('../predict-transfer.js');

const ROOT = path.join(__dirname, '..');
const STEP = 30 * 60 * 1000;
const WINDOW = 72 * 3600000;
const HORIZON_MIN = 120;

function argValue(name, fallback = null) {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : fallback;
}

function round(value, digits = 3) {
    if (!Number.isFinite(value)) return null;
    const f = 10 ** digits;
    return Math.round(value * f) / f;
}

function loadSeries(name, dir) {
    const file = path.join(dir, `${name}.csv`);
    if (!fs.existsSync(file)) return { times: [], values: [] };
    const times = [];
    const values = [];
    const lines = fs.readFileSync(file, 'utf8').split('\n');
    for (let i = 1; i < lines.length; i++) {
        const comma = lines[i].indexOf(',');
        if (comma < 0) continue;
        const t = Date.parse(lines[i].slice(0, comma));
        const v = Number(lines[i].slice(comma + 1));
        if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
        times.push(t);
        values.push(v);
    }
    // Sparse point cache: window construction can reuse point objects across
    // overlapping origins instead of allocating thousands of objects per step.
    return { times, values, points: new Array(times.length) };
}

function pointAt(series, index) {
    let point = series.points[index];
    if (point === undefined) {
        point = { time: new Date(series.times[index]), value: series.values[index] };
        series.points[index] = point;
    }
    return point;
}

function lowerBound(arr, target) {
    let lo = 0;
    let hi = arr.length;
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
    for (let i = start; i < end; i++) {
        timeSeries.push(pointAt(series, i));
    }
    return { id: name, name, timeSeries };
}

async function localFetch(url) {
    try {
        const buf = fs.readFileSync(path.join(ROOT, url));
        return {
            ok: true,
            status: 200,
            json: async () => JSON.parse(buf.toString('utf8')),
            arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
        };
    } catch {
        return { ok: false, status: 404, json: async () => null, arrayBuffer: async () => new ArrayBuffer(0) };
    }
}

function nearestForecast(forecast, targetMs, toleranceMs) {
    let best = null;
    let bestDist = Infinity;
    for (const point of forecast || []) {
        if (point.regime === 'anchor') continue;
        const dist = Math.abs(+point.time - targetMs);
        if (dist < bestDist) {
            best = point;
            bestDist = dist;
        }
    }
    return best && bestDist <= toleranceMs ? best : null;
}

async function main() {
    const horizonMin = Number(argValue('horizon', HORIZON_MIN)) || HORIZON_MIN;
    const limit = Number(argValue('limit', 0)) || 0;
    const outPath = path.resolve(ROOT, argValue('out', 'models/piene-cento-model-responses.json'));
    const report = JSON.parse(fs.readFileSync(path.join(ROOT, 'models/report-piene-lag-align.json'), 'utf8'));
    const config = JSON.parse(fs.readFileSync(path.join(ROOT, 'config.json'), 'utf8'));
    const coeffs = await P.loadCoeffs(localFetch);
    if (!coeffs?.nodes?.Cento) throw new Error('Modello Cento non disponibile nel runtime pack');
    if (horizonMin > 120) {
        await P.hydratePackedMlpNode(localFetch, coeffs, 'Cento', { part: 'tail' });
    }

    const waterNames = Object.keys(config.sensors || {});
    const water = Object.fromEntries(waterNames.map(name => [name, loadSeries(name, path.join(ROOT, 'storico'))]));
    const rainNames = ((config.rainfall && config.rainfall.gauges) || [])
        .map(entry => typeof entry === 'string' ? entry : entry?.name)
        .filter(Boolean);
    const rain = Object.fromEntries(rainNames.map(name => [name, loadSeries(name, path.join(ROOT, 'storico', 'rain'))]));

    const predictionCache = new Map();
    function predictionAtOrigin(originMs, targetMs) {
        if (predictionCache.has(originMs)) return predictionCache.get(originMs);
        const sensorMap = Object.fromEntries(waterNames.map(name => [name, windowSensor(name, water[name], originMs)]));
        const rainSensors = rainNames.map(name => windowSensor(name, rain[name], originMs));
        const down = sensorMap.Cento;
        const ups = (config.sensors.Cento.dependencies || []).map(up => sensorMap[up]).filter(Boolean);
        const anchor = P.latestFinite(down.timeSeries);
        let predicted = null;
        let persisted = Number.isFinite(anchor?.value) ? anchor.value : null;
        // Same freshness rule as the evaluation protocol: on the 30′ lattice the
        // anchor must not be older than one step.
        if (anchor && originMs - +anchor.time <= STEP) {
            const result = P.predictNode(down, ups, {
                hours: 72,
                now: new Date(originMs),
                coeffs,
                horizonHours: horizonMin / 60,
                downstreamId: 'Cento',
                sensorMap,
                rainSensors,
                useCrossingTime: false,
                useUpstreamAmplitude: false
            });
            const point = result?.ok ? nearestForecast(result.forecast, targetMs, STEP / 2 + 1) : null;
            if (point && Number.isFinite(point.med ?? point.value)) predicted = point.med ?? point.value;
        } else {
            persisted = null;
        }
        const out = { predicted, persisted };
        predictionCache.set(originMs, out);
        return out;
    }

    const sourceEvents = (report.events || []).slice(0, limit || undefined);
    const outEvents = {};
    let points = 0;
    let covered = 0;
    for (const [eventIndex, ev] of sourceEvents.entries()) {
        const peakMs = Date.parse(ev.peakTime);
        const pred = [];
        const base = [];
        for (const relHour of ev.hours || []) {
            const targetMs = Math.round(peakMs + relHour * 3600000);
            const originMs = targetMs - horizonMin * 60000;
            const row = predictionAtOrigin(originMs, targetMs);
            pred.push(round(row.predicted, 3));
            base.push(round(row.persisted, 3));
            points++;
            if (Number.isFinite(row.predicted)) covered++;
        }
        const n = pred.filter(Number.isFinite).length;
        if (n) {
            outEvents[String(ev.id)] = {
                peakTime: ev.peakTime,
                split: ev.split || null,
                n,
                coverage: round(n / Math.max(1, pred.length), 3),
                pred,
                base
            };
        }
        if (eventIndex === 0 || (eventIndex + 1) % 25 === 0 || eventIndex + 1 === sourceEvents.length) {
            console.error(`[responses] ${eventIndex + 1}/${sourceEvents.length} eventi · origini ${predictionCache.size}`);
        }
    }

    const out = {
        kind: 'piene-cento-model-responses',
        version: 1,
        generatedAt: new Date().toISOString(),
        disclaimer: 'Sperimentale: hindcast causale del modello prod MLP; non allertistico, non operativo.',
        downstream: 'Cento',
        model: 'prod',
        modelLabel: 'MLP produzione',
        horizonMin,
        semantics: `Al tempo T il valore è la previsione emessa a T-${horizonMin} minuti; non è una traiettoria singola.`,
        stepMinutes: 30,
        sampleHours: 1,
        events: outEvents,
        coverage: {
            points,
            covered,
            ratio: round(covered / Math.max(1, points), 3),
            eventsWithResponses: Object.keys(outEvents).length,
            eventsTotal: sourceEvents.length
        }
    };
    fs.writeFileSync(outPath, JSON.stringify(out));
    const size = fs.statSync(outPath).size;
    console.log(`[responses] wrote ${path.relative(ROOT, outPath)} (${(size / 1024).toFixed(1)} KiB)`);
    console.log(`[responses] eventi con risposta ${out.coverage.eventsWithResponses}/${out.coverage.eventsTotal} · punti coperti ${covered}/${points} (${out.coverage.ratio})`);
}

main().catch(error => {
    console.error(error && error.stack || error);
    process.exit(1);
});
