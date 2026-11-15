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
const PHASE_SHIFTS = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
const ARX_ORDER = 3; // lags 0..3 of Cento + lag-0 upstreams
const CSI_COLORS = ['yellow', 'orange', 'red'];

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

function emptyCsi() {
    return Object.fromEntries(CSI_COLORS.map(c => [c, { hits: 0, misses: 0, falseAlarms: 0 }]));
}

function emptyHorizonAcc() {
    return {
        all: { n: 0, seModel: 0, sePersist: 0, sePoly2: 0, seRidge: 0, seArx: 0 },
        event: { n: 0, seModel: 0, sePersist: 0, sePoly2: 0, seRidge: 0, seArx: 0 },
        rising: { n: 0, se: 0 },
        falling: { n: 0, se: 0 },
        seShift: Object.fromEntries(PHASE_SHIFTS.map(s => [s, { n: 0, se: 0 }])),
        csi: emptyCsi(),
        pinballAll: { n: 0, sum: 0 },
        pinballEvent: { n: 0, sum: 0 },
        crossingTimeErrs: [],
        perEvent: new Map() // eventKey -> metrics
    };
}

function ensureEvent(acc, key) {
    if (!acc.perEvent.has(key)) {
        acc.perEvent.set(key, {
            n: 0, seModel: 0, sePersist: 0, sePoly2: 0, seRidge: 0, seArx: 0,
            rising: { n: 0, se: 0 }, falling: { n: 0, se: 0 },
            pinball: { n: 0, sum: 0 },
            crossingTimeErrs: []
        });
    }
    return acc.perEvent.get(key);
}

function accumulate(bucket, pred, actual, persist, poly2, ridge, arx) {
    if (!Number.isFinite(pred) || !Number.isFinite(actual)) return;
    bucket.n++;
    bucket.seModel += (pred - actual) ** 2;
    bucket.sePersist += (persist - actual) ** 2;
    if (Number.isFinite(poly2)) bucket.sePoly2 += (poly2 - actual) ** 2;
    if (Number.isFinite(ridge)) bucket.seRidge += (ridge - actual) ** 2;
    if (Number.isFinite(arx)) {
        bucket.seArx = (bucket.seArx || 0) + (arx - actual) ** 2;
    }
}

function accumulateCsi(csiAcc, thresholds, pred, actual) {
    if (!thresholds || !Number.isFinite(pred) || !Number.isFinite(actual)) return;
    for (const color of CSI_COLORS) {
        const thr = thresholds[color];
        if (!Number.isFinite(thr)) continue;
        const predHit = pred >= thr;
        const actualHit = actual >= thr;
        const bucket = csiAcc[color];
        if (predHit && actualHit) bucket.hits++;
        else if (!predHit && actualHit) bucket.misses++;
        else if (predHit && !actualHit) bucket.falseAlarms++;
    }
}

function accumulatePinball(bucket, actual, band) {
    if (!band || !Number.isFinite(actual)) return;
    const pairs = [];
    if (Number.isFinite(band.min)) pairs.push({ tau: 0.1, q: band.min });
    if (Number.isFinite(band.med)) pairs.push({ tau: 0.5, q: band.med });
    else if (Number.isFinite(band.max) && Number.isFinite(band.min)) {
        pairs.push({ tau: 0.5, q: 0.5 * (band.min + band.max) });
    }
    if (Number.isFinite(band.max)) pairs.push({ tau: 0.9, q: band.max });
    const v = EP.meanPinball(actual, pairs);
    if (!Number.isFinite(v)) return;
    bucket.n++;
    bucket.sum += v;
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
    if (name === 'arx') {
        const anchor = P.latestFinite(down.timeSeries);
        if (!anchor || !payload.arxByH) return null;
        const feats = EP.arxFeatures(
            payload.seriesByTime,
            payload.upSeriesByName,
            +anchor.time,
            payload.arOrder ?? ARX_ORDER,
            payload.upstreamNames || []
        );
        if (!feats) return null;
        const points = [];
        for (let h = 1; h <= Math.round(horizonHours * 2); h++) {
            const model = payload.arxByH[h];
            let med = model ? EP.predictArx(model, feats) : null;
            if (!Number.isFinite(med)) med = anchor.value;
            points.push({ time: new Date(+now + h * STEP), med });
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

/** Fit per-horizon ARX on lattice times in [trainFrom, trainTo], subsample every `stride` steps. */
function fitArxModels(downSeries, upSeriesByName, upstreamNames, horizons, trainFrom, trainTo, options = {}) {
    const arOrder = options.arOrder ?? ARX_ORDER;
    const stride = options.stride ?? 4; // every 2h on 30′ grid — keep fit cheap
    const maxH = Math.max(...horizons);
    const arxByH = {};
    const rowsByH = Object.fromEntries(horizons.map(h => [h, { x: [], y: [] }]));
    const t0 = Math.ceil(trainFrom / STEP) * STEP;
    const t1 = Math.floor(trainTo / STEP) * STEP - maxH * STEP;
    for (let t = t0; t <= t1; t += stride * STEP) {
        const feats = EP.arxFeatures(downSeries.byTime, upSeriesByName, t, arOrder, upstreamNames);
        if (!feats) continue;
        for (const h of horizons) {
            const y = downSeries.byTime.get(t + h * STEP);
            if (!Number.isFinite(y)) continue;
            rowsByH[h].x.push(feats);
            rowsByH[h].y.push(y);
        }
    }
    for (const h of horizons) {
        const model = EP.fitArx(rowsByH[h], { ridge: 1e-4 });
        if (model) arxByH[h] = model;
    }
    return { arxByH, arOrder, upstreamNames, nTrain: rowsByH[horizons[0]]?.x?.length || 0 };
}

function loadJson(p) {
    return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null;
}

function buildCandidatesForNode(node, ctx) {
    const force = forceReliable;
    const out = {
        persist: { kind: 'baseline' },
        poly2: { kind: 'baseline' },
        arx: { kind: 'baseline', arx: ctx.arxModels?.[node] || null }
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
    // M3: GRU≤2h + ridge long blend (prefer reliable-120 backup when present)
    if (node === 'Gallo' && seqGru && ridgeNode) {
        const gruBackup = loadJson(path.join(ctx.root, 'models/seq-gallo-gru.reliable-120.json')) || seqGru;
        out.blend_gru_ridge = {
            coeffs: {
                nodes: {
                    [node]: force({
                        kind: 'blend_gru_ridge',
                        downstream: node,
                        blendStartMin: 90,
                        blendEndMin: 150,
                        gru: { ...gruBackup, reliable: true },
                        ridge: { ...ridgeNode, reliable: true },
                        skill120: gruBackup.skill120 ?? seqGru.skill120 ?? ridgeNode.skill120
                    })
                }
            },
            smooth: 0.15,
            note: 'M3 GRU≤90′ → blend 90–150′ → ridge≥150′'
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
        const lag = EP.phaseLagSubStep(acc.seShift, { minN: 20, stepMin: 30 });
        const csi = {};
        const pod = {};
        const far = {};
        for (const color of CSI_COLORS) {
            const c = acc.csi?.[color] || { hits: 0, misses: 0, falseAlarms: 0 };
            csi[color] = EP.csiFromCounts(c.hits, c.misses, c.falseAlarms);
            const podDen = c.hits + c.misses;
            const farDen = c.hits + c.falseAlarms;
            pod[color] = podDen ? c.hits / podDen : null;
            far[color] = farDen ? c.falseAlarms / farDen : null;
            csi[`${color}Counts`] = c;
        }
        const risingRmse = EP.rmseOf(acc.rising?.se || 0, acc.rising?.n || 0);
        const fallingRmse = EP.rmseOf(acc.falling?.se || 0, acc.falling?.n || 0);
        const risingCi = EP.bootstrapMetricCI(
            eventList.map(e => ({
                value: e.rising?.n ? Math.sqrt(e.rising.se / e.rising.n) : null
            })),
            { seed, draws: 1000 }
        );
        const crossErrs = acc.crossingTimeErrs || [];
        const crossCi = EP.bootstrapMetricCI(
            crossErrs.map(v => ({ value: v })),
            { seed: seed + 1, draws: 1000 }
        );
        out[hMin] = {
            nAll: acc.all.n,
            nEvent: acc.event.n,
            nEvents: acc.perEvent.size,
            skillAllPersist: skill(acc.all.seModel, acc.all.sePersist, acc.all.n),
            skillAllPoly2: skill(acc.all.seModel, acc.all.sePoly2, acc.all.n),
            skillAllRidge: skill(acc.all.seModel, acc.all.seRidge, acc.all.n),
            skillAllArx: skill(acc.all.seModel, acc.all.seArx, acc.all.n),
            skillEventPersist: skill(acc.event.seModel, acc.event.sePersist, acc.event.n),
            skillEventPoly2: skill(acc.event.seModel, acc.event.sePoly2, acc.event.n),
            skillEventRidge: skill(acc.event.seModel, acc.event.seRidge, acc.event.n),
            skillEventArx: skill(acc.event.seModel, acc.event.seArx, acc.event.n),
            skillEventPersistCI: ciPersist,
            risingRmse,
            risingN: acc.rising?.n || 0,
            risingRmseCI: risingCi,
            fallingRmse,
            fallingN: acc.falling?.n || 0,
            phaseLagMin: lag.phaseLagMin,
            phaseLagMinDiscrete: lag.phaseLagMinDiscrete,
            crossingTimeErrMed: EP.median(crossErrs),
            crossingTimeErrCI: crossCi,
            crossingTimeN: crossErrs.length,
            pinballAll: acc.pinballAll?.n ? acc.pinballAll.sum / acc.pinballAll.n : null,
            pinballEvent: acc.pinballEvent?.n ? acc.pinballEvent.sum / acc.pinballEvent.n : null,
            peakLevelErrMed: EP.median(acc.peakLevelErrs || []),
            peakTimeErrMed: EP.median(acc.peakTimeErrs || []),
            bandCoverageAll: acc.band?.n ? acc.band.hit / acc.band.n : null,
            bandCoverageEvent: acc.bandEvent?.n ? acc.bandEvent.hit / acc.bandEvent.n : null,
            bandNAll: acc.band?.n || 0,
            bandNEvent: acc.bandEvent?.n || 0,
            csi,
            pod,
            far
        };
    }
    return out;
}

function exportPerEvent(accByH, node, model) {
    const rows = [];
    for (const [h, acc] of Object.entries(accByH)) {
        const hMin = Number(h) * 30;
        for (const [key, ev] of acc.perEvent.entries()) {
            rows.push({
                station: node,
                model,
                horizonMin: hMin,
                eventKey: key,
                n: ev.n,
                rmse: EP.rmseOf(ev.seModel, ev.n),
                skillPersist: EP.skillFromRmse(EP.rmseOf(ev.seModel, ev.n), EP.rmseOf(ev.sePersist, ev.n)),
                risingRmse: ev.rising?.n ? Math.sqrt(ev.rising.se / ev.rising.n) : null,
                fallingRmse: ev.falling?.n ? Math.sqrt(ev.falling.se / ev.falling.n) : null,
                pinball: ev.pinball?.n ? ev.pinball.sum / ev.pinball.n : null,
                crossingTimeErrMed: EP.median(ev.crossingTimeErrs || [])
            });
        }
    }
    return rows;
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
    const perEventOut = path.resolve(root, opt('per-event', 'models/eval-per-event.json'));
    const seed = Number(opt('seed', 1));
    const stationFilter = opt('stations', null); // comma-separated optional
    const publishLagMin = Number(opt('publish-lag-min', 0));
    const dropUpstream = opt('drop-upstream', null);
    const doYearFolds = args.includes('--year-folds') || opt('year-folds', '1') !== '0';
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
    const soglieDoc = loadJson(path.join(root, 'models/arpae-soglie.json')) || { stations: {} };
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
    const everyMin = Math.max(30, Number(opt('every', 30)) || 30);
    const everyMs = everyMin * 60000;

    // Fit ARX on pre-test history (frozen calendar cut) for each scored node.
    const arxModels = {};
    for (const node of nodes) {
        const ups = config.sensors[node].dependencies || [];
        const upMap = Object.fromEntries(ups.map(u => [u, data[u]]));
        const trainStart = data[node].times[0];
        arxModels[node] = fitArxModels(data[node], upMap, ups, horizons, trainStart, from - STEP, {
            arOrder: ARX_ORDER,
            stride: 4
        });
        console.log(`[protocol] ARX ${node}: nTrain≈${arxModels[node].nTrain}`);
    }
    const ctx = { root, prodCoeffs, ridgeCoeffs, extraMlp, arxModels };

    const table = {
        kind: 'fase0_eval_table',
        version: 2,
        disclaimer: 'Sperimentale: non operativo, non allertistico.',
        protocol: {
            everyMin,
            horizonsMinutes: horizons.map(h => h * 30),
            baselines: ['persist', 'poly2_6pt', 'ridge', 'arx'],
            eventSource: path.relative(root, eventsPath),
            sampling: 'dense_30min',
            ma3: false,
            publishLagMin: publishLagMin > 0 ? publishLagMin : null,
            dropUpstream: dropUpstream || null,
            yearFolds: doYearFolds,
            arxOrder: ARX_ORDER,
            soglie: 'models/arpae-soglie.json',
            rawData: 'blocked: only validated storico CSV in repo (no distinct raw corpus)'
        },
        from: new Date(from).toISOString(),
        to: new Date(to).toISOString(),
        seed,
        caveat: 'Pesi fitted con mesi alterni che si sovrappongono al test; numeri assoluti possono essere ottimistici.',
        stations: {},
        yearFolds: {},
        generatedAt: null
    };
    const perEventRows = [];

    const t0 = Date.now();
    console.log(`[protocol] ${new Date(from).toISOString()} → ${new Date(to).toISOString()} · nodes=${nodes.join(',')} · horizons=30–360′ · seed=${seed}`);

    const onlyCandOpt = opt('only-candidates', null);
    const onlyCand = onlyCandOpt
        ? new Set(onlyCandOpt.split(',').map(s => s.trim()).filter(Boolean))
        : null;

    for (const node of nodes) {
        let candidates = buildCandidatesForNode(node, ctx);
        if (onlyCand) {
            // Always keep persist/poly2/arx baselines for skill denominators / shared preds.
            candidates = Object.fromEntries(
                Object.entries(candidates).filter(([k]) =>
                    onlyCand.has(k) || k === 'persist' || k === 'poly2' || k === 'arx'
                )
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
        const thresholds = soglieDoc.stations?.[node] || null;
        const upstreamNames = config.sensors[node].dependencies || [];
        const upSeriesByName = Object.fromEntries(upstreamNames.map(u => [u, data[u]]));
        const nodeArx = arxModels[node];

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
        for (let now = first; now <= to; now += everyMs) nows.push(now);

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
            const bandByModel = {};
            for (const key of modelKeys) {
                const cfg = candidates[key];
                const result = predictModel(key, down, ups, now, horizonHours, sensorMap, rainSensors, {
                    node,
                    coeffs: cfg.coeffs,
                    smooth: cfg.smooth,
                    seriesByTime: seriesTruth.byTime,
                    arxByH: (cfg.arx || nodeArx)?.arxByH,
                    arOrder: (cfg.arx || nodeArx)?.arOrder ?? ARX_ORDER,
                    upstreamNames,
                    upSeriesByName
                });
                if (!result?.ok) continue;
                const byTime = new Map((result.forecast || []).map(p => [+p.time, p]));
                const map = {};
                const band = {};
                for (const h of horizons) {
                    const p = byTime.get(now + h * STEP);
                    if (!p) continue;
                    if (Number.isFinite(p.med)) map[h] = p.med;
                    if (Number.isFinite(p.min) && Number.isFinite(p.max)) {
                        band[h] = { min: p.min, max: p.max, med: Number.isFinite(p.med) ? p.med : null };
                    }
                }
                predByModel[key] = map;
                if (Object.keys(band).length) bandByModel[key] = band;
            }
            const ridgeMap = predByModel.ridge || null;
            const arxMap = predByModel.arx || null;

            for (const [key, map] of Object.entries(predByModel)) {
                for (const h of horizons) {
                    const actual = seriesTruth.byTime.get(now + h * STEP);
                    if (!Number.isFinite(actual)) continue;
                    const pred = map[h];
                    if (!Number.isFinite(pred)) continue;
                    const persist = persistPred[h];
                    const poly2 = poly2Pred[h];
                    const ridge = ridgeMap ? ridgeMap[h] : null;
                    const arx = arxMap ? arxMap[h] : null;
                    if (!Number.isFinite(persist)) continue;

                    const acc = stats[key][h];
                    accumulate(acc.all, pred, actual, persist, poly2, ridge, arx);
                    accumulateCsi(acc.csi, thresholds, pred, actual);
                    const band = bandByModel[key]?.[h];
                    if (band) {
                        acc.band = acc.band || { n: 0, hit: 0 };
                        acc.band.n++;
                        if (actual >= band.min && actual <= band.max) acc.band.hit++;
                        accumulatePinball(acc.pinballAll, actual, band);
                        if (activeEvent) {
                            acc.bandEvent = acc.bandEvent || { n: 0, hit: 0 };
                            acc.bandEvent.n++;
                            if (actual >= band.min && actual <= band.max) acc.bandEvent.hit++;
                            accumulatePinball(acc.pinballEvent, actual, band);
                        }
                    }
                    // Phase lag: compare pred at now+h to actual at now+h+s
                    for (const s of PHASE_SHIFTS) {
                        const shifted = seriesTruth.byTime.get(now + (h + s) * STEP);
                        if (!Number.isFinite(shifted)) continue;
                        const e = pred - shifted;
                        acc.seShift[s].n++;
                        acc.seShift[s].se += e * e;
                    }
                    if (activeEvent) {
                        accumulate(acc.event, pred, actual, persist, poly2, ridge, arx);
                        const pev = ensureEvent(acc, activeEvent.key);
                        accumulate(pev, pred, actual, persist, poly2, ridge, arx);
                        const limb = now < activeEvent.peakTime ? 'rising' : 'falling';
                        const err2 = (pred - actual) ** 2;
                        acc[limb].n++;
                        acc[limb].se += err2;
                        pev[limb].n++;
                        pev[limb].se += err2;
                        if (band) accumulatePinball(pev.pinball, actual, band);
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
                            seriesByTime: seriesTruth.byTime,
                            arxByH: (cfg.arx || nodeArx)?.arxByH,
                            arOrder: (cfg.arx || nodeArx)?.arOrder ?? ARX_ORDER,
                            upstreamNames,
                            upSeriesByName
                        });
                        if (!result?.ok) continue;
                        forecastByTime = new Map((result.forecast || []).map(p => [+p.time, p.med]));
                    }
                    const pe = EP.peakErrorsForIssue(seriesTruth.byTime, forecastByTime, ev.peakTime, h, 6);
                    if (pe) {
                        stats[key][h].peakLevelErrs.push(pe.levelErr);
                        stats[key][h].peakTimeErrs.push(pe.timeErrMin);
                    }
                    // Crossing-time error vs yellow (and orange/red if crossed) for events that cross
                    if (thresholds) {
                        for (const color of CSI_COLORS) {
                            const thr = thresholds[color];
                            if (!Number.isFinite(thr) || ev.peakLevel < thr) continue;
                            const err = EP.crossingTimeErrorMin(
                                seriesTruth.byTime, forecastByTime, issueAt, thr, h + 12
                            );
                            if (!Number.isFinite(err)) continue;
                            stats[key][h].crossingTimeErrs.push(err);
                            const pev = ensureEvent(stats[key][h], ev.key);
                            pev.crossingTimeErrs.push(err);
                            break; // one threshold per issue (lowest crossed = yellow first)
                        }
                    }
                }
            }
        }

        table.stations[node] = {};
        for (const key of candKeys) {
            table.stations[node][key] = finalizeModel(stats[key], seed);
            perEventRows.push(...exportPerEvent(stats[key], node, key));
        }
        console.log(`[protocol] done ${node}`);
    }

    // Year folds on Cento event windows (ARX refit; frozen models scored as-is).
    if (doYearFolds && (!stationFilter || stationFilter.split(',').includes('Cento')) && data.Cento) {
        table.yearFolds.Cento = runCentoYearFolds({
            catalog, config, data, rainData, rainGauges, names, soglieDoc,
            prodCoeffs, ridgeCoeffs, extraMlp, root, horizons, horizonHours, seed, onlyCand
        });
    }

    table.generatedAt = new Date().toISOString();
    table.elapsedSec = (Date.now() - t0) / 1000;

    // When --stations filters the run, keep other stations from a prior eval-table.
    if (stationFilter && fs.existsSync(jsonOut)) {
        try {
            const prev = JSON.parse(fs.readFileSync(jsonOut, 'utf8'));
            if (prev?.stations) {
                for (const [st, models] of Object.entries(prev.stations)) {
                    if (!table.stations[st]) table.stations[st] = models;
                }
            }
            if (prev?.yearFolds && !Object.keys(table.yearFolds || {}).length) {
                table.yearFolds = prev.yearFolds;
            }
        } catch (_) { /* ignore corrupt prior */ }
    }

    fs.mkdirSync(path.dirname(jsonOut), { recursive: true });
    fs.writeFileSync(jsonOut, JSON.stringify(table, null, 2) + '\n');
    fs.writeFileSync(mdOut, EP.formatEvalMarkdown(table));
    fs.writeFileSync(perEventOut, JSON.stringify({
        kind: 'fase0_eval_per_event',
        version: 1,
        disclaimer: 'Sperimentale: non operativo, non allertistico.',
        generatedAt: table.generatedAt,
        rows: perEventRows
    }, null, 2) + '\n');
    console.log(`[protocol] Wrote ${jsonOut}`);
    console.log(`[protocol] Wrote ${mdOut}`);
    console.log(`[protocol] Wrote ${perEventOut}`);
    console.log(`[protocol] elapsed ${table.elapsedSec.toFixed(1)}s`);
    return table;
}

/**
 * Leave-year-out on Cento catalog events. ARX refit on other years;
 * prod/ridge/seq scored with frozen weights on holdout event windows only.
 */
function runCentoYearFolds(ctx) {
    const node = 'Cento';
    const {
        catalog, config, data, rainData, rainGauges, names, soglieDoc,
        prodCoeffs, ridgeCoeffs, extraMlp, root, horizons, horizonHours, seed, onlyCand
    } = ctx;
    const stationEvents = EP.eventsForStation(catalog, node);
    const years = [...new Set(stationEvents.map(e => EP.yearOf(Date.parse(e.peakTime))))].sort();
    const upstreamNames = config.sensors[node].dependencies || [];
    const upSeriesByName = Object.fromEntries(upstreamNames.map(u => [u, data[u]]));
    const thresholds = soglieDoc.stations?.[node] || null;
    const folds = [];
    const issueEvery = 4 * STEP; // 2h issues inside events — cheaper than dense 30′

    for (const holdoutYear of years) {
        const holdout = stationEvents.filter(e => EP.yearOf(Date.parse(e.peakTime)) === holdoutYear);
        if (!holdout.length) continue;
        const trainFrom = data[node].times[0];
        // Leave-year-out: train on all years except holdout
        const arxRows = { arxByH: {}, arOrder: ARX_ORDER, upstreamNames, nTrain: 0 };
        const rowsByH = Object.fromEntries(horizons.map(h => [h, { x: [], y: [] }]));
        const maxH = Math.max(...horizons);
        const tEnd = data[node].times[data[node].times.length - 1] - maxH * STEP;
        for (let t = Math.ceil(trainFrom / STEP) * STEP; t <= tEnd; t += 4 * STEP) {
            if (EP.yearOf(t) === holdoutYear) continue;
            const feats = EP.arxFeatures(data[node].byTime, upSeriesByName, t, ARX_ORDER, upstreamNames);
            if (!feats) continue;
            for (const h of horizons) {
                const y = data[node].byTime.get(t + h * STEP);
                if (!Number.isFinite(y)) continue;
                rowsByH[h].x.push(feats);
                rowsByH[h].y.push(y);
            }
        }
        for (const h of horizons) {
            const model = EP.fitArx(rowsByH[h], { ridge: 1e-4 });
            if (model) arxRows.arxByH[h] = model;
        }
        arxRows.nTrain = rowsByH[horizons[0]]?.x?.length || 0;

        const arxModels = { [node]: arxRows };
        let candidates = buildCandidatesForNode(node, {
            root, prodCoeffs, ridgeCoeffs, extraMlp, arxModels
        });
        if (onlyCand) {
            candidates = Object.fromEntries(
                Object.entries(candidates).filter(([k]) =>
                    onlyCand.has(k) || k === 'persist' || k === 'poly2' || k === 'arx'
                )
            );
        }
        // Focus year-fold table on main baselines + prod
        const focus = new Set(['persist', 'poly2', 'arx', 'ridge', 'prod']);
        candidates = Object.fromEntries(Object.entries(candidates).filter(([k]) => focus.has(k)));
        const candKeys = Object.keys(candidates);
        const stats = {};
        for (const key of candKeys) {
            stats[key] = Object.fromEntries(horizons.map(h => [h, emptyHorizonAcc()]));
        }

        for (const ev of holdout) {
            const peak = Date.parse(ev.peakTime);
            const start = Date.parse(ev.start);
            const end = Date.parse(ev.end) - maxH * STEP;
            const evKey = `${ev.station}|${ev.peakTime}`;
            for (let now = Math.ceil(start / issueEvery) * issueEvery; now <= end; now += issueEvery) {
                const y0 = data[node].byTime.get(now);
                if (!Number.isFinite(y0)) continue;
                const down = windowSensor(node, data[node], now);
                const anchor = P.latestFinite(down.timeSeries);
                if (!anchor || now - +anchor.time > STEP) continue;
                const rainSensors = rainGauges
                    .map(n => windowSensor(n, rainData[n], now))
                    .filter(s => s.timeSeries.length);
                const sensorMap = Object.fromEntries(names.map(n => [n, windowSensor(n, data[n], now)]));
                sensorMap[node] = down;
                const ups = upstreamNames.map(u => sensorMap[u]);

                const persistPred = {};
                const poly2Pred = {};
                for (const h of horizons) {
                    const actual = data[node].byTime.get(now + h * STEP);
                    if (!Number.isFinite(actual)) continue;
                    persistPred[h] = y0;
                    poly2Pred[h] = EP.poly2At(data[node].byTime, +anchor.time, h, EP.POLY2_POINTS);
                    if (!Number.isFinite(poly2Pred[h])) poly2Pred[h] = y0;
                }
                const predByModel = { persist: persistPred, poly2: poly2Pred };
                const bandByModel = {};
                for (const key of candKeys.filter(k => k !== 'persist' && k !== 'poly2')) {
                    const cfg = candidates[key];
                    const result = predictModel(key, down, ups, now, horizonHours, sensorMap, rainSensors, {
                        node,
                        coeffs: cfg.coeffs,
                        smooth: cfg.smooth,
                        seriesByTime: data[node].byTime,
                        arxByH: arxRows.arxByH,
                        arOrder: ARX_ORDER,
                        upstreamNames,
                        upSeriesByName
                    });
                    if (!result?.ok) continue;
                    const byTime = new Map((result.forecast || []).map(p => [+p.time, p]));
                    const map = {};
                    const band = {};
                    for (const h of horizons) {
                        const p = byTime.get(now + h * STEP);
                        if (!p || !Number.isFinite(p.med)) continue;
                        map[h] = p.med;
                        if (Number.isFinite(p.min) && Number.isFinite(p.max)) {
                            band[h] = { min: p.min, max: p.max, med: p.med };
                        }
                    }
                    predByModel[key] = map;
                    if (Object.keys(band).length) bandByModel[key] = band;
                }
                const ridgeMap = predByModel.ridge || null;
                const arxMap = predByModel.arx || null;
                const limb = now < peak ? 'rising' : 'falling';

                for (const [key, map] of Object.entries(predByModel)) {
                    for (const h of horizons) {
                        const actual = data[node].byTime.get(now + h * STEP);
                        const pred = map[h];
                        if (!Number.isFinite(actual) || !Number.isFinite(pred)) continue;
                        const persist = persistPred[h];
                        const poly2 = poly2Pred[h];
                        const ridge = ridgeMap ? ridgeMap[h] : null;
                        const arx = arxMap ? arxMap[h] : null;
                        const acc = stats[key][h];
                        accumulate(acc.all, pred, actual, persist, poly2, ridge, arx);
                        accumulate(acc.event, pred, actual, persist, poly2, ridge, arx);
                        accumulateCsi(acc.csi, thresholds, pred, actual);
                        const pev = ensureEvent(acc, evKey);
                        accumulate(pev, pred, actual, persist, poly2, ridge, arx);
                        const err2 = (pred - actual) ** 2;
                        acc[limb].n++;
                        acc[limb].se += err2;
                        pev[limb].n++;
                        pev[limb].se += err2;
                        for (const s of PHASE_SHIFTS) {
                            const shifted = data[node].byTime.get(now + (h + s) * STEP);
                            if (!Number.isFinite(shifted)) continue;
                            acc.seShift[s].n++;
                            acc.seShift[s].se += (pred - shifted) ** 2;
                        }
                        const band = bandByModel[key]?.[h];
                        if (band) {
                            accumulatePinball(acc.pinballEvent, actual, band);
                            accumulatePinball(pev.pinball, actual, band);
                        }
                    }
                }
            }
        }

        const models = {};
        for (const key of candKeys) {
            models[key] = finalizeModel(stats[key], seed + holdoutYear);
        }
        folds.push({
            holdoutYear,
            nEvents: holdout.length,
            arxNTrain: arxRows.nTrain,
            models
        });
        console.log(`[protocol] year-fold holdout=${holdoutYear} events=${holdout.length}`);
    }
    return folds;
}

module.exports = { run, PROTOCOL_HORIZONS, PROTOCOL_TEST_FROM };
