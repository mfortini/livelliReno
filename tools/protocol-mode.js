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
    for (let i = start; i < end; i++) {
        timeSeries.push({ time: new Date(series.times[i]), value: series.values[i] });
    }
    return { id: name, name, timeSeries };
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

    if (!fs.existsSync(eventsPath)) {
        console.error(`Missing event catalog: ${eventsPath}. Run tools/build-event-catalog.js first.`);
        process.exit(1);
    }
    const catalog = JSON.parse(fs.readFileSync(eventsPath, 'utf8'));
    const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
    const prodCoeffs = loadJson(coeffsPath);
    const ridgeCoeffs = loadJson(ridgePath) || prodCoeffs;

    const names = Object.keys(config.sensors);
    const data = Object.fromEntries(names.map(n => [n, loadSeries(n, dataDir)]));
    const rainGauges = ((config.rainfall && config.rainfall.gauges) || []).map(e =>
        typeof e === 'string' ? e : e?.name
    ).filter(Boolean);
    const rainDir = path.join(dataDir, 'rain');
    const rainData = Object.fromEntries(rainGauges.map(n => [n, loadSeries(n, rainDir)]));

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
    const ctx = { root, prodCoeffs, ridgeCoeffs };

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
            ma3: false
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

    for (const node of nodes) {
        const candidates = buildCandidatesForNode(node, ctx);
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

        const series = data[node];
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
            let anchorVal = series.byTime.get(now);
            let anchorTime = now;
            if (!Number.isFinite(anchorVal)) {
                // 15′ series may miss a lattice point; use latest finite within one step
                const down0 = windowSensor(node, series, now);
                const a = P.latestFinite(down0.timeSeries);
                if (!a || now - +a.time > STEP) continue;
                anchorTime = +a.time;
                anchorVal = a.value;
            }

            const activeEvent = eventWindows.find(e => now >= e.start && now <= e.end) || null;

            const persistPred = {};
            const poly2Pred = {};
            let anyH = false;
            for (const h of horizons) {
                const actual = series.byTime.get(now + h * STEP);
                if (!Number.isFinite(actual)) continue;
                anyH = true;
                persistPred[h] = series.byTime.get(anchorTime);
                if (!Number.isFinite(persistPred[h])) persistPred[h] = anchorVal;
                poly2Pred[h] = EP.poly2At(series.byTime, anchorTime, h, EP.POLY2_POINTS);
                if (!Number.isFinite(poly2Pred[h])) poly2Pred[h] = persistPred[h];
            }
            if (!anyH) continue;

            // Heavy models: build sensor windows once per timestamp
            const modelKeys = candKeys.filter(k => k !== 'persist' && k !== 'poly2');
            let down = null, ups = null, sensorMap = null, rainSensors = null;
            if (modelKeys.length) {
                down = windowSensor(node, series, now);
                rainSensors = rainGauges
                    .map(n => windowSensor(n, rainData[n], now))
                    .filter(s => s.timeSeries.length);
                sensorMap = Object.fromEntries(names.map(n => [n, windowSensor(n, data[n], now)]));
                sensorMap[node] = down;
                ups = (config.sensors[node].dependencies || [])
                    .map(up => sensorMap[up] || windowSensor(up, data[up], now));
            }

            const predByModel = { persist: persistPred, poly2: poly2Pred };
            for (const key of modelKeys) {
                const cfg = candidates[key];
                const result = predictModel(key, down, ups, now, horizonHours, sensorMap, rainSensors, {
                    node,
                    coeffs: cfg.coeffs,
                    smooth: cfg.smooth,
                    seriesByTime: series.byTime
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
                    const actual = series.byTime.get(now + h * STEP);
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
                    const down = windowSensor(node, series, issueAt);
                    const anchor = P.latestFinite(down.timeSeries);
                    if (!anchor || issueAt - +anchor.time > STEP) continue;
                    const rainSensors = rainGauges
                        .map(n => windowSensor(n, rainData[n], issueAt))
                        .filter(s => s.timeSeries.length);
                    const sensorMap = Object.fromEntries(names.map(n => [n, windowSensor(n, data[n], issueAt)]));
                    sensorMap[node] = down;
                    const ups = (config.sensors[node].dependencies || [])
                        .map(up => sensorMap[up] || windowSensor(up, data[up], issueAt));

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
                            let med = EP.poly2At(series.byTime, +anchor.time, steps, EP.POLY2_POINTS);
                            if (!Number.isFinite(med)) med = anchor.value;
                            forecastByTime.set(ev.peakTime + k * STEP, med);
                        }
                    } else {
                        const result = predictModel(key, down, ups, issueAt, horizonHours, sensorMap, rainSensors, {
                            node,
                            coeffs: cfg.coeffs,
                            smooth: cfg.smooth,
                            seriesByTime: series.byTime
                        });
                        if (!result?.ok) continue;
                        forecastByTime = new Map((result.forecast || []).map(p => [+p.time, p.med]));
                    }
                    const pe = EP.peakErrorsForIssue(series.byTime, forecastByTime, ev.peakTime, h, 6);
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
