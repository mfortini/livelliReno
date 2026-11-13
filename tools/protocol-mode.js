'use strict';
/**
 * Fase 0 unified evaluation protocol.
 * Invoked from tools/backtest.js when --protocol is set.
 * Does NOT update reliable flags.
 */
const fs = require('node:fs');
const path = require('node:path');
const P = require('../predict-transfer.js');
const EP = require('../eval-protocol.js');

const STEP = EP.STEP_MS;
const WINDOW = 72 * 3600000;
const PROTOCOL_HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const PROTOCOL_TEST_FROM = '2026-01-01T00:00:00.000Z';

function loadSeries(name, dir) {
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

/** E5: hide observations newer than cutoffMs (simulates publish lag at each now). */
function seriesUntil(series, cutoffMs) {
    if (!series?.times?.length) return series;
    const times = [], values = [], byTime = new Map();
    for (let i = 0; i < series.times.length; i++) {
        const t = series.times[i];
        if (t > cutoffMs) break;
        times.push(t);
        values.push(series.values[i]);
        byTime.set(t, series.values[i]);
    }
    return { times, values, byTime };
}

function emptySeries() {
    return { times: [], values: [], byTime: new Map() };
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

function windowSensor(name, series, now, cutoffMs = null) {
    const start = lowerBound(series.times, now - WINDOW);
    const endLimit = cutoffMs == null ? now + 1 : Math.min(now + 1, cutoffMs + 1);
    const end = lowerBound(series.times, endLimit);
    const timeSeries = [];
    for (let i = start; i < end; i++) {
        timeSeries.push({ time: new Date(series.times[i]), value: series.values[i] });
    }
    return { id: name, name, timeSeries };
}

function valueAtOrBefore(series, t, cutoffMs = null) {
    if (cutoffMs != null && t > cutoffMs) t = cutoffMs;
    const v = series.byTime.get(t);
    if (Number.isFinite(v)) return v;
    // walk back one step on lattice
    if (cutoffMs != null) {
        const snapped = Math.floor(Math.min(t, cutoffMs) / STEP) * STEP;
        return series.byTime.get(snapped);
    }
    return v;
}

function forceReliable(node) {
    return node ? { ...node, reliable: true } : null;
}

function slugify(name) {
    return String(name || '').toLowerCase().replace(/\s+/g, '-');
}

function emptyHorizonAcc() {
    return {
        all: { n: 0, seModel: 0, sePersist: 0, sePoly2: 0, seRidge: 0 },
        event: { n: 0, seModel: 0, sePersist: 0, sePoly2: 0, seRidge: 0 },
        perEvent: new Map() // eventKey -> { n, seModel, sePersist, sePoly2, seRidge }
    };
}

function ensureEvent(acc, key) {
    if (!acc.perEvent.has(key)) {
        acc.perEvent.set(key, { n: 0, seModel: 0, sePersist: 0, sePoly2: 0, seRidge: 0 });
    }
    return acc.perEvent.get(key);
}

function accumulate(bucket, pred, actual, persist, poly2, ridge) {
    if (!Number.isFinite(pred) || !Number.isFinite(actual)) return;
    bucket.n++;
    bucket.seModel += (pred - actual) ** 2;
    bucket.sePersist += (persist - actual) ** 2;
    if (Number.isFinite(poly2)) bucket.sePoly2 += (poly2 - actual) ** 2;
    if (Number.isFinite(ridge)) bucket.seRidge += (ridge - actual) ** 2;
}

function predictModel(name, down, ups, now, horizonHours, sensorMap, rainSensors, payload) {
    if (name === 'persist') {
        const anchor = P.latestFinite(down.timeSeries);
        if (!anchor) return null;
        const points = [];
        for (let h = 1; h <= Math.round(horizonHours * 2); h++) {
            points.push({ time: new Date(+now + h * STEP), med: anchor.value });
        }
        return { ok: true, forecast: [{ time: anchor.time, med: anchor.value }, ...points] };
    }
    if (name === 'poly2') {
        const anchor = P.latestFinite(down.timeSeries);
        if (!anchor) return null;
        const points = [];
        for (let h = 1; h <= Math.round(horizonHours * 2); h++) {
            const med = EP.poly2At(payload.seriesByTime, +anchor.time, h, EP.POLY2_POINTS);
            if (!Number.isFinite(med)) {
                points.push({ time: new Date(+now + h * STEP), med: anchor.value });
            } else {
                points.push({ time: new Date(+now + h * STEP), med });
            }
        }
        return { ok: true, forecast: [{ time: anchor.time, med: anchor.value }, ...points] };
    }
    const result = P.predictNode(down, ups, {
        hours: 72,
        now: new Date(now),
        coeffs: payload.coeffs,
        horizonHours,
        downstreamId: payload.node,
        rainSensors,
        sensorMap,
        smooth: payload.smooth ?? 0,
        maxStep: payload.maxStep
    });
    return result?.ok ? result : null;
}

function loadJson(p) {
    return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null;
}

function buildCandidatesForNode(node, ctx) {
    const force = forceReliable;
    const out = {
        persist: { kind: 'baseline' },
        poly2: { kind: 'baseline' }
    };
    const ridgeNode = ctx.ridgeCoeffs?.nodes?.[node];
    if (ridgeNode && ridgeNode.kind === 'direct_horizon_delta_ridge') {
        out.ridge = {
            coeffs: { nodes: { [node]: force(ridgeNode) }, horizonsMinutes: ctx.ridgeCoeffs.horizonsMinutes }
        };
    }
    const prodNode = ctx.prodCoeffs?.nodes?.[node];
    if (prodNode) {
        out.prod = {
            coeffs: { nodes: { [node]: force(prodNode) }, horizonsMinutes: ctx.prodCoeffs.horizonsMinutes },
            smooth: prodNode.kind === 'direct_horizon_delta_ridge' ? 0.15 : 0,
            note: prodNode.kind
        };
    }
    const slug = slugify(node);
    const seqGru = loadJson(path.join(ctx.root, `models/seq-${slug}-gru.json`));
    if (seqGru && (seqGru.downstream === node || !seqGru.downstream)) {
        out.seq_gru = {
            coeffs: { nodes: { [node]: force(seqGru) } },
            note: 'forced reliable for protocol compare'
        };
    }
    if (node === 'Cento') {
        const seqCnn = loadJson(path.join(ctx.root, 'models/seq-cento-cnn.json'));
        if (seqCnn) {
            out.seq_cnn = { coeffs: { nodes: { [node]: force(seqCnn) } } };
        }
        const musk = loadJson(path.join(ctx.root, 'models/hybrid-muskingum-cento.json'));
        if (musk) {
            out.muskingum_gru = { coeffs: { nodes: { [node]: force(musk) } } };
        }
        const rc = loadJson(path.join(ctx.root, 'models/hybrid-rc-cento.json'));
        if (rc) {
            out.rc_gru = { coeffs: { nodes: { [node]: force(rc) } } };
        }
    }
    // Optional extra MLP weights for A/B (E3 block-split, E2 recipe winners)
    for (const [label, file] of Object.entries(ctx.extraMlp || {})) {
        const mlp = loadJson(file);
        if (!mlp) continue;
        const nodePayload = mlp.kind ? mlp : (mlp.nodes || {})[node];
        if (!nodePayload) continue;
        out[label] = {
            coeffs: { nodes: { [node]: force(nodePayload) } },
            smooth: 0,
            note: path.relative(ctx.root, file)
        };
    }
    return out;
}

function finalizeModel(accByH, seed) {
    const out = {};
    for (const [h, acc] of Object.entries(accByH)) {
        const hMin = String(Number(h) * 30);
        const skill = (seM, seB, n) => EP.skillFromRmse(EP.rmseOf(seM, n), EP.rmseOf(seB, n));
        const eventList = [...acc.perEvent.values()];
        const ciPersist = EP.bootstrapSkillCI(
            eventList.map(e => ({ n: e.n, seModel: e.seModel, seBase: e.sePersist })),
            { seed, draws: 1000, alpha: 0.10 }
        );
        out[hMin] = {
            nAll: acc.all.n,
            nEvent: acc.event.n,
            nEvents: acc.perEvent.size,
            skillAllPersist: skill(acc.all.seModel, acc.all.sePersist, acc.all.n),
            skillAllPoly2: skill(acc.all.seModel, acc.all.sePoly2, acc.all.n),
            skillAllRidge: skill(acc.all.seModel, acc.all.seRidge, acc.all.n),
            skillEventPersist: skill(acc.event.seModel, acc.event.sePersist, acc.event.n),
            skillEventPoly2: skill(acc.event.seModel, acc.event.sePoly2, acc.event.n),
            skillEventRidge: skill(acc.event.seModel, acc.event.seRidge, acc.event.n),
            skillEventPersistCI: ciPersist,
            peakLevelErrMed: EP.median(acc.peakLevelErrs || []),
            peakTimeErrMed: EP.median(acc.peakTimeErrs || []),
            pod: null,
            far: null,
            note: 'POD/FAR N/A: soglie ARPAE non in config statica'
        };
    }
    return out;
}

function run(options = {}) {
    const root = options.root || path.resolve(__dirname, '..');
    const args = options.args || process.argv.slice(2);
    const opt = (name, fallback) => {
        const i = args.indexOf(`--${name}`);
        return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
    };

    const dataDir = path.resolve(root, opt('data', 'storico'));
    const eventsPath = path.resolve(root, opt('events', 'models/events.json'));
    const coeffsPath = path.resolve(root, opt('coeffs', 'models/transfer-coeffs.json'));
    const ridgePath = path.resolve(root, opt('ridge', 'models/transfer-coeffs-ridge6h.json'));
    const jsonOut = path.resolve(root, opt('json', 'models/eval-table.json'));
    const mdOut = path.resolve(root, opt('md', 'models/eval-table.md'));
    const seed = Number(opt('seed', 1));
    const stationFilter = opt('stations', null); // comma-separated optional
    const publishLagMin = Number(opt('publish-lag-min', 0));
    const dropUpstream = opt('drop-upstream', null);
    // --extra-mlp label=path (repeatable): add MLP candidates for A/B
    const extraMlp = {};
    for (let i = 0; i < args.length; i++) {
        if (args[i] !== '--extra-mlp' || args[i + 1] === undefined) continue;
        const spec = args[++i];
        const eq = spec.indexOf('=');
        if (eq < 1) continue;
        extraMlp[spec.slice(0, eq)] = path.resolve(root, spec.slice(eq + 1));
    }

    if (!fs.existsSync(eventsPath)) {
        console.error(`Missing event catalog: ${eventsPath}. Run tools/build-event-catalog.js first.`);
        process.exit(1);
    }
    const catalog = JSON.parse(fs.readFileSync(eventsPath, 'utf8'));
    const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
    const prodCoeffs = loadJson(coeffsPath);
    const ridgeCoeffs = loadJson(ridgePath) || prodCoeffs;

    const names = Object.keys(config.sensors);
    const dataFull = Object.fromEntries(names.map(n => [n, loadSeries(n, dataDir)]));
    if (dropUpstream) {
        if (!dataFull[dropUpstream]) {
            console.error(`Unknown --drop-upstream: ${dropUpstream}`);
            process.exit(1);
        }
        dataFull[dropUpstream] = emptySeries();
        console.log(`[protocol] drop-upstream=${dropUpstream}`);
    }
    const data = dataFull;
    const rainGauges = ((config.rainfall && config.rainfall.gauges) || []).map(e =>
        typeof e === 'string' ? e : e?.name
    ).filter(Boolean);
    const rainDir = path.join(dataDir, 'rain');
    const rainDataFull = Object.fromEntries(rainGauges.map(n => [n, loadSeries(n, rainDir)]));
    const rainData = rainDataFull;
    if (publishLagMin > 0) {
        console.log(`[protocol] publish-lag-min=${publishLagMin}`);
    }

    const allTimes = names.flatMap(n => [data[n].times[0], data[n].times.at(-1)]).filter(Number.isFinite);
    if (!allTimes.length) {
        console.error(`Nessun CSV in ${dataDir}`);
        process.exit(1);
    }
    const from = Date.parse(opt('from', PROTOCOL_TEST_FROM));
    const to = Date.parse(opt('to', new Date(Math.max(...allTimes) - 12 * STEP).toISOString()));

    const ranks = P.rankNodes(config);
    let nodes = names
        .filter(n => (config.sensors[n].dependencies || []).length)
        .sort((a, b) => (ranks[b] || 0) - (ranks[a] || 0) || a.localeCompare(b));
    if (stationFilter) {
        const want = new Set(stationFilter.split(',').map(s => s.trim()).filter(Boolean));
        nodes = nodes.filter(n => want.has(n));
    }

    const horizons = PROTOCOL_HORIZONS;
    const horizonHours = Math.max(...horizons) * 0.5;
    const ctx = { root, prodCoeffs, ridgeCoeffs, extraMlp };

    const table = {
        kind: 'fase0_eval_table',
        version: 1,
        disclaimer: 'Sperimentale: non operativo, non allertistico.',
        protocol: {
            everyMin: 30,
            horizonsMinutes: horizons.map(h => h * 30),
            baselines: ['persist', 'poly2_6pt', 'ridge'],
            eventSource: path.relative(root, eventsPath),
            sampling: 'dense_30min',
            ma3: false,
            publishLagMin: publishLagMin > 0 ? publishLagMin : null,
            dropUpstream: dropUpstream || null,
            rawData: 'blocked: only validated storico CSV in repo (no distinct raw corpus)'
        },
        from: new Date(from).toISOString(),
        to: new Date(to).toISOString(),
        seed,
        caveat: 'Pesi fitted con mesi alterni che si sovrappongono al test; numeri assoluti possono essere ottimistici.',
        stations: {},
        generatedAt: null
    };

    const t0 = Date.now();
    console.log(`[protocol] ${new Date(from).toISOString()} → ${new Date(to).toISOString()} · nodes=${nodes.join(',')} · horizons=30–360′ · seed=${seed}`);

    const onlyCandOpt = opt('only-candidates', null);
    const onlyCand = onlyCandOpt
        ? new Set(onlyCandOpt.split(',').map(s => s.trim()).filter(Boolean))
        : null;

    for (const node of nodes) {
        let candidates = buildCandidatesForNode(node, ctx);
        if (onlyCand) {
            candidates = Object.fromEntries(
                Object.entries(candidates).filter(([k]) => onlyCand.has(k))
            );
        }
        const candKeys = Object.keys(candidates);
        const stationEvents = EP.eventsForStation(catalog, node);
        const eventWindows = stationEvents.map(e => ({
            key: `${e.station}|${e.peakTime}`,
            start: Date.parse(e.start),
            end: Date.parse(e.end),
            peakTime: Date.parse(e.peakTime),
            peakLevel: e.peakLevel
        }));

        const stats = {};
        for (const key of candKeys) {
            stats[key] = Object.fromEntries(horizons.map(h => [h, emptyHorizonAcc()]));
            for (const h of horizons) {
                stats[key][h].peakLevelErrs = [];
                stats[key][h].peakTimeErrs = [];
            }
        }

        const seriesTruth = data[node]; // full series for scoring actuals
        const first = Math.ceil(from / STEP) * STEP;
        const nows = [];
        for (let now = first; now <= to; now += STEP) nows.push(now);

        console.log(`[protocol] ${node}: ${nows.length} timestamps, candidates=${candKeys.join(',')}, events=${stationEvents.length}`);

        // Precompute ridge+persist+poly2 forecasts lazily per now via shared predict
        let done = 0;
        for (const now of nows) {
            done++;
            if (done === 1 || done % 2000 === 0 || done === nows.length) {
                console.log(`[protocol] ${node} progress ${done}/${nows.length} (~${((Date.now() - t0) / 1000).toFixed(1)}s)`);
            }
            // Publish lag: wall clock is `now`, but observations after cutoff are hidden
            // → recent lags become miss / last finite becomes anchor (staleMinutes-like).
            const cutoff = publishLagMin > 0 ? now - publishLagMin * 60000 : null;

            let anchorTime = now;
            let anchorVal = cutoff == null
                ? seriesTruth.byTime.get(now)
                : valueAtOrBefore(seriesTruth, now, cutoff);
            if (!Number.isFinite(anchorVal)) {
                const down0 = windowSensor(node, seriesTruth, now, cutoff);
                const a = P.latestFinite(down0.timeSeries);
                if (!a || now - +a.time > STEP + (publishLagMin > 0 ? publishLagMin * 60000 : 0)) continue;
                anchorTime = +a.time;
                anchorVal = a.value;
            } else if (cutoff != null) {
                anchorTime = Math.floor(Math.min(now, cutoff) / STEP) * STEP;
                while (anchorTime >= cutoff - WINDOW && !Number.isFinite(seriesTruth.byTime.get(anchorTime))) {
                    anchorTime -= STEP;
                }
                anchorVal = seriesTruth.byTime.get(anchorTime);
                if (!Number.isFinite(anchorVal)) continue;
            }

            const activeEvent = eventWindows.find(e => now >= e.start && now <= e.end) || null;

            const persistPred = {};
            const poly2Pred = {};
            let anyH = false;
            for (const h of horizons) {
                const actual = seriesTruth.byTime.get(now + h * STEP);
                if (!Number.isFinite(actual)) continue;
                anyH = true;
                persistPred[h] = anchorVal;
                poly2Pred[h] = EP.poly2At(seriesTruth.byTime, anchorTime, h, EP.POLY2_POINTS);
                if (!Number.isFinite(poly2Pred[h])) poly2Pred[h] = persistPred[h];
            }
            if (!anyH) continue;

            // Heavy models: windows at wall `now`, but truncated at cutoff
            const modelKeys = candKeys.filter(k => k !== 'persist' && k !== 'poly2');
            let down = null, ups = null, sensorMap = null, rainSensors = null;
            if (modelKeys.length) {
                down = windowSensor(node, seriesTruth, now, cutoff);
                rainSensors = rainGauges
                    .map(n => windowSensor(n, rainData[n], now, cutoff))
                    .filter(s => s.timeSeries.length);
                sensorMap = Object.fromEntries(
                    names.map(n => [n, windowSensor(n, data[n], now, cutoff)])
                );
                sensorMap[node] = down;
                ups = (config.sensors[node].dependencies || [])
                    .map(up => sensorMap[up] || windowSensor(up, data[up], now, cutoff));
            }

            const predByModel = { persist: persistPred, poly2: poly2Pred };
            for (const key of modelKeys) {
                const cfg = candidates[key];
                const result = predictModel(key, down, ups, now, horizonHours, sensorMap, rainSensors, {
                    node,
                    coeffs: cfg.coeffs,
                    smooth: cfg.smooth,
                    seriesByTime: seriesTruth.byTime
                });
                if (!result?.ok) continue;
                const byTime = new Map((result.forecast || []).map(p => [+p.time, p.med]));
                const map = {};
                for (const h of horizons) {
                    const v = byTime.get(now + h * STEP);
                    if (Number.isFinite(v)) map[h] = v;
                }
                predByModel[key] = map;
            }
            const ridgeMap = predByModel.ridge || null;

            for (const [key, map] of Object.entries(predByModel)) {
                for (const h of horizons) {
                    const actual = seriesTruth.byTime.get(now + h * STEP);
                    if (!Number.isFinite(actual)) continue;
                    const pred = map[h];
                    if (!Number.isFinite(pred)) continue;
                    const persist = persistPred[h];
                    const poly2 = poly2Pred[h];
                    const ridge = ridgeMap ? ridgeMap[h] : null;
                    if (!Number.isFinite(persist)) continue;

                    const acc = stats[key][h];
                    accumulate(acc.all, pred, actual, persist, poly2, ridge);
                    if (activeEvent) {
                        accumulate(acc.event, pred, actual, persist, poly2, ridge);
                        const pev = ensureEvent(acc, activeEvent.key);
                        accumulate(pev, pred, actual, persist, poly2, ridge);
                    }
                }
            }
        }

        // Peak metrics: for each catalog event in test window, issue at peak−h
        for (const ev of eventWindows) {
            if (ev.peakTime < from || ev.peakTime > to) continue;
            for (const [key, cfg] of Object.entries(candidates)) {
                for (const h of horizons) {
                    const issueAt = ev.peakTime - h * STEP;
                    if (issueAt < from) continue;
                    const peakCutoff = publishLagMin > 0 ? issueAt - publishLagMin * 60000 : null;
                    const down = windowSensor(node, seriesTruth, issueAt, peakCutoff);
                    const anchor = P.latestFinite(down.timeSeries);
                    if (!anchor || issueAt - +anchor.time > STEP + (publishLagMin > 0 ? publishLagMin * 60000 : 0)) continue;
                    const rainSensors = rainGauges
                        .map(n => windowSensor(n, rainData[n], issueAt, peakCutoff))
                        .filter(s => s.timeSeries.length);
                    const sensorMap = Object.fromEntries(
                        names.map(n => [n, windowSensor(n, data[n], issueAt, peakCutoff)])
                    );
                    sensorMap[node] = down;
                    const ups = (config.sensors[node].dependencies || [])
                        .map(up => sensorMap[up] || windowSensor(up, data[up], issueAt, peakCutoff));

                    let forecastByTime;
                    if (key === 'persist') {
                        forecastByTime = new Map();
                        for (let k = -12; k <= 12; k++) {
                            forecastByTime.set(ev.peakTime + k * STEP, anchor.value);
                        }
                    } else if (key === 'poly2') {
                        forecastByTime = new Map();
                        for (let k = -12; k <= 12; k++) {
                            const steps = Math.round((ev.peakTime + k * STEP - issueAt) / STEP);
                            if (steps < 1) continue;
                            let med = EP.poly2At(seriesTruth.byTime, +anchor.time, steps, EP.POLY2_POINTS);
                            if (!Number.isFinite(med)) med = anchor.value;
                            forecastByTime.set(ev.peakTime + k * STEP, med);
                        }
                    } else {
                        const result = predictModel(key, down, ups, issueAt, horizonHours, sensorMap, rainSensors, {
                            node,
                            coeffs: cfg.coeffs,
                            smooth: cfg.smooth,
                            seriesByTime: seriesTruth.byTime
                        });
                        if (!result?.ok) continue;
                        forecastByTime = new Map((result.forecast || []).map(p => [+p.time, p.med]));
                    }
                    const pe = EP.peakErrorsForIssue(seriesTruth.byTime, forecastByTime, ev.peakTime, h, 6);
                    if (pe) {
                        stats[key][h].peakLevelErrs.push(pe.levelErr);
                        stats[key][h].peakTimeErrs.push(pe.timeErrMin);
                    }
                }
            }
        }

        table.stations[node] = {};
        for (const key of candKeys) {
            table.stations[node][key] = finalizeModel(stats[key], seed);
        }
        console.log(`[protocol] done ${node}`);
    }

    table.generatedAt = new Date().toISOString();
    table.elapsedSec = (Date.now() - t0) / 1000;
    fs.mkdirSync(path.dirname(jsonOut), { recursive: true });
    fs.writeFileSync(jsonOut, JSON.stringify(table, null, 2) + '\n');
    fs.writeFileSync(mdOut, EP.formatEvalMarkdown(table));
    console.log(`[protocol] Wrote ${jsonOut}`);
    console.log(`[protocol] Wrote ${mdOut}`);
    console.log(`[protocol] elapsed ${table.elapsedSec.toFixed(1)}s`);
    return table;
}

module.exports = { run, PROTOCOL_HORIZONS, PROTOCOL_TEST_FROM };
