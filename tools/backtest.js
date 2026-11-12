#!/usr/bin/env node
// Backtest del codice di predizione (predict-transfer.js) sullo storico.
// Campionamento: ogni --every minuti (default 120) + densificazione 30′ nelle finestre
// in cui |Δy| a +2h ≥ --event (default 0.10 m).
//
// Uso: node tools/backtest.js [--data storico] [--every 120] [--event 0.10]
//                             [--coeffs models/transfer-coeffs.json] [--no-coeffs]
//                             [--from 2025-01-01] [--to 2026-11-01] [--json out.json]
//                             [--no-update-reliable]
//                             [--compare-candidates Cento] [--mlp models/cento-dnn.json]
//                             [--muskingum models/hybrid-muskingum-cento.json]
//                             [--seq-gru models/seq-cento-gru.json] [--seq-cnn models/seq-cento-cnn.json]
//                             [--candidates-json models/cento-candidates-backtest.json]
//
// --compare-candidates runs an experimental side-by-side on one node without writing
// reliable flags. Default backtest still uses transfer-coeffs as before.
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
const compareNode = opt('compare-candidates', null);
const mlpPath = path.resolve(root, opt('mlp', 'models/cento-dnn.json'));
const muskPath = path.resolve(root, opt('muskingum', 'models/hybrid-muskingum-cento.json'));
const candidatesJson = opt('candidates-json', compareNode ? `models/${compareNode.toLowerCase()}-candidates-backtest.json` : null);
// Experimental compare never mutates reliable; plain --no-update-reliable still applies to the main path.
const updateReliable = !args.includes('--no-update-reliable') && !compareNode;
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
const rainGauges = ((config.rainfall && config.rainfall.gauges) || []).map(entry =>
    typeof entry === 'string' ? entry : entry?.name
).filter(Boolean);
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

const PHASE_SHIFTS = [-4, -2, -1, 0, 1, 2, 4]; // step da 30′; s<0 ⇒ obs anticipato

function emptyCandidateStats(horizons) {
    return Object.fromEntries(horizons.map(h => [h, {
        n: 0, se: 0, ae: 0, sePersist: 0,
        seShift: Object.fromEntries(PHASE_SHIFTS.map(s => [s, { n: 0, se: 0 }]))
    }]));
}

function finalizeCandidate(stats) {
    const out = {};
    for (const [h, bucket] of Object.entries(stats)) {
        if (!bucket.n) {
            out[h] = { n: 0, rmse: null, mae: null, skillPersist: null };
            continue;
        }
        const rmse = Math.sqrt(bucket.se / bucket.n);
        const rmseP = Math.sqrt(bucket.sePersist / bucket.n);
        let bestShift = null;
        let bestShiftRmse = Infinity;
        for (const [s, acc] of Object.entries(bucket.seShift || {})) {
            if (acc.n < 20) continue;
            const r = Math.sqrt(acc.se / acc.n);
            if (r < bestShiftRmse) { bestShiftRmse = r; bestShift = Number(s); }
        }
        out[h] = {
            n: bucket.n,
            rmse,
            mae: bucket.ae / bucket.n,
            skillPersist: rmseP ? 1 - rmse / rmseP : null,
            phaseLagMin: bestShift === null ? null : -bestShift * 30,
            phaseRmseGain: bestShift === null ? null : 1 - bestShiftRmse / rmse
        };
    }
    return out;
}

function predictCandidate(name, down, ups, now, horizonHours, sensorMap, rainSensors, payload) {
    if (name === 'persist') {
        const anchor = P.latestFinite(down.timeSeries);
        if (!anchor) return null;
        const points = [];
        for (let h = 1; h <= Math.round(horizonHours * 2); h++) {
            points.push({ time: new Date(+now + h * STEP), med: anchor.value, min: anchor.value, max: anchor.value });
        }
        return { ok: true, forecast: [{ time: anchor.time, med: anchor.value }, ...points], next: points };
    }
    if (name === 'local_trend') {
        const anchor = P.latestFinite(down.timeSeries);
        if (!anchor) return null;
        const prev = down.timeSeries.filter(p => +p.time <= +anchor.time - STEP && Number.isFinite(p.value)).at(-1);
        const dy = prev ? anchor.value - prev.value : 0;
        const points = [];
        for (let h = 1; h <= Math.round(horizonHours * 2); h++) {
            const med = anchor.value + dy * h;
            points.push({ time: new Date(+now + h * STEP), med, min: med, max: med });
        }
        return { ok: true, forecast: [{ time: anchor.time, med: anchor.value }, ...points], next: points };
    }
    if (name === 'poly2') {
        // Stupid predictor: quadratic OLS on the last 3–4 levels, extrapolate per horizon step.
        const anchor = P.latestFinite(down.timeSeries);
        if (!anchor) return null;
        const end = +anchor.time;
        const vals = [];
        for (let k = 3; k >= 0; k--) {
            vals.push(P.sampleAt(down.timeSeries, end - k * STEP));
        }
        let window = vals.every(Number.isFinite) ? vals : vals.slice(1);
        if (!window.every(Number.isFinite) || window.length < 3) {
            // Fall back to persistence if history is too thin.
            return predictCandidate('persist', down, ups, now, horizonHours, sensorMap, rainSensors, payload);
        }
        const n = window.length;
        const x = window.map((_, i) => i);
        // Solve least-squares for a + b x + c x² via normal equations (3×3).
        let sx = 0, sx2 = 0, sx3 = 0, sx4 = 0, sy = 0, sxy = 0, sx2y = 0;
        for (let i = 0; i < n; i++) {
            const xi = x[i], yi = window[i], xi2 = xi * xi;
            sx += xi; sx2 += xi2; sx3 += xi2 * xi; sx4 += xi2 * xi2;
            sy += yi; sxy += xi * yi; sx2y += xi2 * yi;
        }
        const A = [
            [n, sx, sx2],
            [sx, sx2, sx3],
            [sx2, sx3, sx4]
        ];
        const b = [sy, sxy, sx2y];
        // Gaussian elimination
        const M = A.map((row, i) => row.concat([b[i]]));
        for (let col = 0; col < 3; col++) {
            let piv = col;
            for (let r = col + 1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
            [M[col], M[piv]] = [M[piv], M[col]];
            const div = M[col][col];
            if (!Number.isFinite(div) || Math.abs(div) < 1e-12) {
                return predictCandidate('persist', down, ups, now, horizonHours, sensorMap, rainSensors, payload);
            }
            for (let c = col; c < 4; c++) M[col][c] /= div;
            for (let r = 0; r < 3; r++) {
                if (r === col) continue;
                const f = M[r][col];
                for (let c = col; c < 4; c++) M[r][c] -= f * M[col][c];
            }
        }
        const [a0, a1, a2] = [M[0][3], M[1][3], M[2][3]];
        const at = (xi) => a0 + a1 * xi + a2 * xi * xi;
        const points = [];
        for (let h = 1; h <= Math.round(horizonHours * 2); h++) {
            const med = at((n - 1) + h);
            points.push({ time: new Date(+now + h * STEP), med, min: med, max: med });
        }
        return { ok: true, forecast: [{ time: anchor.time, med: anchor.value }, ...points], next: points };
    }
    const result = P.predictNode(down, ups, {
        hours: 72,
        now: new Date(now),
        coeffs: payload.coeffs,
        horizonHours,
        downstreamId: payload.node,
        rainSensors,
        sensorMap,
        smooth: payload.smooth,
        maxStep: payload.maxStep
    });
    const phaseMs = Number(payload.phaseShiftMin) * 60000;
    if (result?.ok && Number.isFinite(phaseMs) && phaseMs !== 0) {
        // Display-level phase correction: same values, relabelled earlier.
        result.forecast = (result.forecast || []).map(p => ({ ...p, time: new Date(+p.time - phaseMs) }));
        result.next = (result.next || []).map(p => ({ ...p, time: new Date(+p.time - phaseMs) }));
    }
    return result;
}

if (compareNode) {
    if (!config.sensors[compareNode]) {
        console.error(`Unknown --compare-candidates node: ${compareNode}`);
        process.exit(1);
    }
    const mlp = fs.existsSync(mlpPath) ? JSON.parse(fs.readFileSync(mlpPath, 'utf8')) : null;
    const mlpVallePath = path.resolve(root, opt('mlp-valle', 'models/cento-dnn-valle.json'));
    const mlpNovallePath = path.resolve(root, opt('mlp-novalle', 'models/cento-dnn-novalle.json'));
    const mlpValle = fs.existsSync(mlpVallePath) ? JSON.parse(fs.readFileSync(mlpVallePath, 'utf8')) : null;
    const mlpNovalle = fs.existsSync(mlpNovallePath) ? JSON.parse(fs.readFileSync(mlpNovallePath, 'utf8')) : mlp;
    const musk = fs.existsSync(muskPath) ? JSON.parse(fs.readFileSync(muskPath, 'utf8')) : null;
    const rcPath = path.resolve(root, opt('rc', 'models/hybrid-rc-cento.json'));
    const rc = fs.existsSync(rcPath) ? JSON.parse(fs.readFileSync(rcPath, 'utf8')) : null;
    const seqGruPath = path.resolve(root, opt('seq-gru', 'models/seq-cento-gru.json'));
    const seqCnnPath = path.resolve(root, opt('seq-cnn', 'models/seq-cento-cnn.json'));
    const seqGru = fs.existsSync(seqGruPath) ? JSON.parse(fs.readFileSync(seqGruPath, 'utf8')) : null;
    const seqCnn = fs.existsSync(seqCnnPath) ? JSON.parse(fs.readFileSync(seqCnnPath, 'utf8')) : null;
    const mlpLagshiftPath = path.resolve(root, opt('mlp-lagshift', 'models/cento-dnn-lagalign.json'));
    const mlpLagaugmentPath = path.resolve(root, opt('mlp-lagaugment', 'models/cento-dnn-lagaugment.json'));
    const mlpLagshift = fs.existsSync(mlpLagshiftPath) ? JSON.parse(fs.readFileSync(mlpLagshiftPath, 'utf8')) : null;
    const mlpLagaugment = fs.existsSync(mlpLagaugmentPath) ? JSON.parse(fs.readFileSync(mlpLagaugmentPath, 'utf8')) : null;
    const prodNode = useCoeffs && fs.existsSync(coeffsPath)
        ? (JSON.parse(fs.readFileSync(coeffsPath, 'utf8')).nodes || {})[compareNode]
        : null;
    const chOpt = opt('chs', null);
    const horizons = chOpt
        ? chOpt.split(',').map(Number).filter(h => Number.isFinite(h) && h > 0)
        : [1, 2, 4, 8, 12];
    const horizonHours = Math.max(...horizons) * 0.5;
    const force = (node) => ({ ...node, reliable: true });
    const candidates = {
        persist: { kind: 'baseline' },
        local_trend: { kind: 'baseline' },
        poly2: { kind: 'baseline' },
        mlp_prod: prodNode ? { coeffs: { nodes: { [compareNode]: force(prodNode) } }, smooth: 0 } : null,
        mlp_novalle: mlpNovalle ? { coeffs: { nodes: { [compareNode]: force(mlpNovalle) } }, smooth: 0 } : null,
        mlp_valle: mlpValle ? { coeffs: { nodes: { [compareNode]: force(mlpValle) } }, smooth: 0 } : null,
        mlp_valle_ema065: mlpValle ? { coeffs: { nodes: { [compareNode]: force(mlpValle) } }, smooth: 0.65 } : null,
        mlp_lagshift: mlpLagshift ? { coeffs: { nodes: { [compareNode]: force(mlpLagshift) } }, smooth: 0 } : null,
        mlp_lagaugment: mlpLagaugment ? { coeffs: { nodes: { [compareNode]: force(mlpLagaugment) } }, smooth: 0 } : null,
        mlp_phase30: prodNode ? { coeffs: { nodes: { [compareNode]: force(prodNode) } }, smooth: 0, phaseShiftMin: 30 } : null,
        mlp_phase60: prodNode ? { coeffs: { nodes: { [compareNode]: force(prodNode) } }, smooth: 0, phaseShiftMin: 60 } : null,
        seq_gru: seqGru ? {
            coeffs: { nodes: { [compareNode]: force(seqGru) } },
            note: 'forced reliable for experimental compare only; does not promote runtime'
        } : null,
        seq_cnn: seqCnn ? {
            coeffs: { nodes: { [compareNode]: force(seqCnn) } },
            note: 'forced reliable for experimental compare only; does not promote runtime'
        } : null,
        muskingum_gru: musk ? {
            coeffs: { nodes: { [compareNode]: force(musk) } },
            note: 'forced reliable for experimental compare only; does not promote runtime'
        } : null,
        rc_gru: rc ? {
            coeffs: { nodes: { [compareNode]: force(rc) } },
            note: 'forced reliable for experimental compare only; does not promote runtime'
        } : null
    };
    function emptyBucketStats(horizons) {
        return {
            all: emptyCandidateStats(horizons),
            highAcceptor: emptyCandidateStats(horizons),
            lowAcceptor: emptyCandidateStats(horizons)
        };
    }
    const stats = {};
    for (const [key, cfg] of Object.entries(candidates)) {
        if (!cfg) continue;
        stats[key] = emptyBucketStats(horizons);
    }
    const ups = config.sensors[compareNode].dependencies || [];
    const acceptorNames = ['Dosso', 'Opera Reno Panfilia', 'Secondo Salto', 'Gallo'].filter(n => data[n]);
    const primaryAcceptor = acceptorNames[0] || null;
    let acceptorThr = null;
    if (primaryAcceptor && data[primaryAcceptor].values.length) {
        const vals = data[primaryAcceptor].values.filter(Number.isFinite).slice().sort((a, b) => a - b);
        acceptorThr = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.85))];
    }
    const nows = collectNows(compareNode);
    const t0 = Date.now();
    const candKeys = Object.keys(candidates).filter(k => candidates[k]);
    console.log(`[backtest] compare-candidates ${compareNode}: ${nows.length} timestamps, candidates=${candKeys.join(',')}`);
    let doneNows = 0;
    let usedNows = 0;
    for (const now of nows) {
        doneNows++;
        if (doneNows === 1 || doneNows % 250 === 0 || doneNows === nows.length) {
            const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
            console.log(`[backtest] progress ${doneNows}/${nows.length} (~${elapsed}s, used=${usedNows})`);
        }
        const down = windowSensor(compareNode, data[compareNode], now);
        const anchor = P.latestFinite(down.timeSeries);
        if (!anchor || now - +anchor.time > STEP) continue;
        usedNows++;
        const rainSensors = rainGauges
            .map(name => windowSensor(name, rainData[name], now))
            .filter(s => s.timeSeries.length);
        const sensorMap = Object.fromEntries(names.map(name => [name, windowSensor(name, data[name], now)]));
        sensorMap[compareNode] = down;
        const upSensors = ups.map(up => sensorMap[up] || windowSensor(up, data[up], now));
        let highAcc = false;
        if (primaryAcceptor && Number.isFinite(acceptorThr)) {
            const av = data[primaryAcceptor].byTime.get(now);
            highAcc = Number.isFinite(av) && av >= acceptorThr;
        }
        for (const [key, cfg] of Object.entries(candidates)) {
            if (!cfg) continue;
            const result = predictCandidate(key, down, upSensors, now, horizonHours, sensorMap, rainSensors, {
                node: compareNode,
                coeffs: cfg.coeffs,
                smooth: cfg.smooth,
                maxStep: cfg.maxStep,
                phaseShiftMin: cfg.phaseShiftMin
            });
            if (!result?.ok) continue;
            const byTime = new Map((result.forecast || []).map(point => [+point.time, point]));
            for (const h of horizons) {
                const actual = data[compareNode].byTime.get(now + h * STEP);
                if (!Number.isFinite(actual)) continue;
                const point = byTime.get(now + h * STEP);
                if (!point || !Number.isFinite(point.med)) continue;
                const targets = [stats[key].all[h], highAcc ? stats[key].highAcceptor[h] : stats[key].lowAcceptor[h]];
                for (const bucket of targets) {
                    bucket.n++;
                    bucket.se += (point.med - actual) ** 2;
                    bucket.ae += Math.abs(point.med - actual);
                    bucket.sePersist += (anchor.value - actual) ** 2;
                }
                for (const s of PHASE_SHIFTS) {
                    if (s === 0) continue;
                    const shifted = data[compareNode].byTime.get(now + (h + s) * STEP);
                    if (!Number.isFinite(shifted)) continue;
                    const acc = stats[key].all[h].seShift[s];
                    acc.n++;
                    acc.se += (point.med - shifted) ** 2;
                }
            }
        }
    }
    const summary = {};
    for (const [key, bucket] of Object.entries(stats)) {
        summary[key] = {
            all: finalizeCandidate(bucket.all),
            highAcceptor: finalizeCandidate(bucket.highAcceptor),
            lowAcceptor: finalizeCandidate(bucket.lowAcceptor)
        };
    }
    const out = {
        kind: 'experimental_candidates',
        node: compareNode,
        from,
        to,
        everyMin,
        eventThreshold,
        note: 'Does not write reliable flags or promote experimental models.',
        acceptors: acceptorNames,
        primaryAcceptor,
        acceptorThr85: acceptorThr,
        mlpPath: mlpNovalle ? path.relative(root, mlpNovallePath) : null,
        mlpVallePath: mlpValle ? path.relative(root, mlpVallePath) : null,
        mlpProdPath: prodNode ? path.relative(root, coeffsPath) : null,
        seqGruPath: seqGru ? path.relative(root, seqGruPath) : null,
        seqCnnPath: seqCnn ? path.relative(root, seqCnnPath) : null,
        muskingumPath: musk ? path.relative(root, muskPath) : null,
        rcPath: rc ? path.relative(root, rcPath) : null,
        summary,
        elapsedSec: (Date.now() - t0) / 1000
    };
    console.log(JSON.stringify({
        node: compareNode,
        acceptorThr85: acceptorThr,
        summary: Object.fromEntries(
            Object.entries(summary).map(([k, v]) => [k, {
                skill120: v.all?.[4]?.skillPersist,
                rmse120: v.all?.[4]?.rmse,
                n120: v.all?.[4]?.n,
                skill240: v.all?.[8]?.skillPersist,
                skill360: v.all?.[12]?.skillPersist,
                lag120: v.all?.[4]?.phaseLagMin,
                lag360: v.all?.[12]?.phaseLagMin,
                skill120HighAcc: v.highAcceptor?.[4]?.skillPersist,
                skill120LowAcc: v.lowAcceptor?.[4]?.skillPersist
            }])
        ),
        elapsedSec: out.elapsedSec
    }, null, 2));
    if (candidatesJson) {
        const dest = path.resolve(root, candidatesJson);
        fs.writeFileSync(dest, JSON.stringify(out, null, 2) + '\n');
        console.log(`Wrote ${dest}`);
    }
    process.exit(0);
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
        const sensorMap = Object.fromEntries(
            names.map(name => [name, windowSensor(name, data[name], now)])
        );
        sensorMap[node] = down;
        const result = P.predictNode(down, ups.map(up => sensorMap[up] || windowSensor(up, data[up], now)), {
            hours: 72, now: new Date(now), coeffs, horizonHours, downstreamId: node, rainSensors, sensorMap
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
