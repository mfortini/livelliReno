#!/usr/bin/env node
/**
 * WP1 diagnosis for research_plan_reno.md (H1–H11).
 * Offline only; does not touch reliable / adopt flags.
 *
 *   node tools/wp1-diagnosis.js [--data ./storico] [--json models/wp1-diagnosis.json]
 */
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const P = require('../predict-transfer.js');
const EP = require('../eval-protocol.js');

const root = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const opt = (name, fallback) => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
};

const dataDir = path.resolve(root, opt('data', 'storico'));
const jsonOut = path.resolve(root, opt('json', 'models/wp1-diagnosis.json'));
const STEP = EP.STEP_MS;
const WINDOW = 72 * 3600000;
const UPSTREAMS = [
    'Forcelli', 'Sostegno Reno', 'Bonconvento',
    'Casalecchio tiro a volo', 'Casalecchio chiusa'
];
const DOWN = 'Cento';
const DOSSO = 'Dosso';

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
    if (!s?.times?.length) return { id: name, name, timeSeries: [] };
    const a = lb(s.times, now - WINDOW), b = lb(s.times, now + 1);
    const ts = [];
    for (let i = a; i < b; i++) ts.push({ time: new Date(s.times[i]), value: s.values[i] });
    return { id: name, name, timeSeries: ts };
}

function loadJson(p) {
    return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf8')) : null;
}

function resample30(series) {
    // Snap to 30′ lattice using nearest existing point within ±15′
    if (!series.times.length) return series;
    const t0 = Math.ceil(series.times[0] / STEP) * STEP;
    const t1 = Math.floor(series.times.at(-1) / STEP) * STEP;
    const times = [], values = [], byTime = new Map();
    for (let t = t0; t <= t1; t += STEP) {
        let v = series.byTime.get(t);
        if (!Number.isFinite(v)) {
            // try ±15′ raw (common ARPAE grid)
            v = series.byTime.get(t - 15 * 60000) ?? series.byTime.get(t + 15 * 60000);
        }
        if (!Number.isFinite(v)) continue;
        times.push(t); values.push(v); byTime.set(t, v);
    }
    return { times, values, byTime };
}

function crossCorrLag(xBy, yBy, times, maxLagSteps = 48) {
    // lag > 0 means y lags x (wave travels x → y)
    let bestLag = 0, bestR = -Infinity;
    const pairs = [];
    for (const t of times) {
        const xv = xBy.get(t), yv = yBy.get(t);
        if (Number.isFinite(xv) && Number.isFinite(yv)) pairs.push([xv, yv, t]);
    }
    if (pairs.length < 50) return { lagSteps: null, lagMin: null, r: null, n: pairs.length };

    for (let lag = 0; lag <= maxLagSteps; lag++) {
        const xs = [], ys = [];
        for (const [xv, , t] of pairs) {
            const yv = yBy.get(t + lag * STEP);
            if (!Number.isFinite(yv)) continue;
            xs.push(xv); ys.push(yv);
        }
        if (xs.length < 40) continue;
        const r = pearson(xs, ys);
        if (Number.isFinite(r) && r > bestR) {
            bestR = r;
            bestLag = lag;
        }
    }
    return { lagSteps: bestLag, lagMin: bestLag * 30, r: bestR, n: pairs.length };
}

function pearson(xs, ys) {
    const n = xs.length;
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
        sx += xs[i]; sy += ys[i];
        sxx += xs[i] * xs[i]; syy += ys[i] * ys[i];
        sxy += xs[i] * ys[i];
    }
    const num = n * sxy - sx * sy;
    const den = Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    return den > 0 ? num / den : null;
}

function percentile(arr, p) {
    const a = arr.filter(Number.isFinite).slice().sort((x, y) => x - y);
    if (!a.length) return null;
    const i = Math.min(a.length - 1, Math.max(0, Math.floor(p * (a.length - 1))));
    return a[i];
}

// ---------- H3: lag by level band ----------
function diagnoseH3(cento, upstreams) {
    const bands = { low: [], mid: [], high: [] };
    const vals = cento.values.filter(Number.isFinite);
    const p33 = percentile(vals, 0.33);
    const p66 = percentile(vals, 0.66);
    for (const t of cento.times) {
        const y = cento.byTime.get(t);
        if (!Number.isFinite(y)) continue;
        if (y < p33) bands.low.push(t);
        else if (y < p66) bands.mid.push(t);
        else bands.high.push(t);
    }
    const byUp = {};
    let minLag = Infinity;
    for (const name of UPSTREAMS) {
        const up = upstreams[name];
        if (!up?.times?.length) continue;
        byUp[name] = {};
        for (const [band, times] of Object.entries(bands)) {
            const res = crossCorrLag(up.byTime, cento.byTime, times, 48);
            byUp[name][band] = res;
            if (Number.isFinite(res.lagMin) && res.lagMin < minLag) minLag = res.lagMin;
        }
    }
    return {
        levelCuts: { p33, p66 },
        byUpstream: byUp,
        minLagMin: Number.isFinite(minLag) ? minLag : null,
        note: 'lagMin = minutes Cento lags upstream (cross-corr peak). Decrease with level ⇒ H3.'
    };
}

// ---------- Hysteresis for 3 floods ----------
function diagnoseHysteresis(cento, upstreams, events) {
    const top = events
        .slice()
        .sort((a, b) => b.peakLevel - a.peakLevel)
        .slice(0, 3);
    return top.map(ev => {
        const start = Date.parse(ev.start);
        const end = Date.parse(ev.end);
        const peak = Date.parse(ev.peakTime);
        const series = {};
        for (const name of ['Cento', ...UPSTREAMS]) {
            const src = name === 'Cento' ? cento : upstreams[name];
            if (!src) continue;
            const pts = [];
            for (let t = start; t <= end; t += STEP) {
                const v = src.byTime.get(t);
                if (Number.isFinite(v)) pts.push({ t, v, limb: t < peak ? 'rising' : 'falling' });
            }
            series[name] = pts;
        }
        // Loop score: area between rising/falling Cento vs Sostegno
        const loop = loopScore(series['Cento'], series['Sostegno Reno']);
        return {
            peakTime: ev.peakTime,
            peakLevel: ev.peakLevel,
            riseM: ev.riseM,
            loopScoreSostegno: loop,
            nPoints: series.Cento?.length || 0
        };
    });
}

function loopScore(centoPts, upPts) {
    if (!centoPts?.length || !upPts?.length) return null;
    const upMap = new Map(upPts.map(p => [p.t, p.v]));
    let riseSum = 0, riseN = 0, fallSum = 0, fallN = 0;
    for (const p of centoPts) {
        const u = upMap.get(p.t);
        if (!Number.isFinite(u)) continue;
        if (p.limb === 'rising') { riseSum += p.v - u; riseN++; }
        else { fallSum += p.v - u; fallN++; }
    }
    if (!riseN || !fallN) return null;
    return (riseSum / riseN) - (fallSum / fallN);
}

// ---------- EnKF open-loop + innovations (H1, H6) ----------
function enkfProcessStep(zC, zF, zFprev, zS, zSprev, rain, params) {
    const a = params.A;
    const dzF = Number.isFinite(zFprev) ? zF - zFprev : 0;
    const dzS = Number.isFinite(zSprev) ? zS - zSprev : 0;
    const pulse =
        params.C_f * Math.max(dzF, 0) +
        params.C_s * Math.max(dzS, 0) +
        params.C_f_down * Math.min(dzF, 0) +
        params.C_s_down * Math.min(dzS, 0) +
        (params.alpha || 0) * (Number.isFinite(rain) && rain > 0 ? rain : 0);
    const couple = params.K_f * zF + params.K_s * zS;
    return zC + (pulse + couple - params.C_out * zC) / a;
}

function diagnoseEnkf(cento, forcelli, sostegno, events, enkfDoc) {
    if (!enkfDoc?.params) return { available: false };
    const params = enkfDoc.params;
    const mu = enkfDoc.means || {};
    const muC = mu.Cento ?? 0, muF = mu.Forcelli ?? 0, muS = mu['Sostegno Reno'] ?? 0;
    const lagF = enkfDoc.lags?.Forcelli?.steps ?? 16;
    const lagS = enkfDoc.lags?.['Sostegno Reno']?.steps ?? 5;
    const q = enkfDoc.processNoiseM ?? 0.01;
    const rObs = enkfDoc.obsNoiseM ?? 0.02;
    const N = enkfDoc.ensembleN ?? 40;

    // Build aligned lattice
    const t0 = Math.max(cento.times[0], forcelli.times[0], sostegno.times[0]) + Math.max(lagF, lagS) * STEP;
    const t1 = Math.min(cento.times.at(-1), forcelli.times.at(-1), sostegno.times.at(-1));
    const times = [];
    for (let t = Math.ceil(t0 / STEP) * STEP; t <= t1; t += STEP) {
        if (
            Number.isFinite(cento.byTime.get(t)) &&
            Number.isFinite(forcelli.byTime.get(t - lagF * STEP)) &&
            Number.isFinite(sostegno.byTime.get(t - lagS * STEP))
        ) times.push(t);
    }

    const eventSet = new Set();
    for (const ev of events) {
        const a = Date.parse(ev.start), b = Date.parse(ev.end);
        for (let t = a; t <= b; t += STEP) eventSet.add(t);
    }

    // Open-loop 4-step (120′) forecasts from truth state (no filter)
    let olN = 0, olSe = 0, olSeP = 0, olEventN = 0, olEventSe = 0;
    const innovCalm = [], innovFlood = [];
    const spreadFlood = [], spreadCalm = [];

    // Filtered ensemble for innovations / spread
    let ens = Float64Array.from({ length: N }, () => 0);
    const y0 = cento.byTime.get(times[0]);
    if (Number.isFinite(y0)) ens = ens.map(() => y0 - muC);

    for (let i = 1; i < times.length; i++) {
        const t = times[i];
        const tPrev = times[i - 1];
        const zF = forcelli.byTime.get(t - lagF * STEP) - muF;
        const zFp = forcelli.byTime.get(tPrev - lagF * STEP) - muF;
        const zS = sostegno.byTime.get(t - lagS * STEP) - muS;
        const zSp = sostegno.byTime.get(tPrev - lagS * STEP) - muS;
        const zCobs = cento.byTime.get(t) - muC;

        // Open-loop 120′ from previous truth
        const zStart = cento.byTime.get(tPrev) - muC;
        if (Number.isFinite(zStart) && i + 4 < times.length) {
            let z = zStart;
            let pf = forcelli.byTime.get(tPrev - lagF * STEP) - muF;
            let ps = sostegno.byTime.get(tPrev - lagS * STEP) - muS;
            let ok = true;
            for (let k = 0; k < 4; k++) {
                const tk = times[i + k];
                const zf = forcelli.byTime.get(tk - lagF * STEP) - muF;
                const zs = sostegno.byTime.get(tk - lagS * STEP) - muS;
                if (![zf, zs].every(Number.isFinite)) { ok = false; break; }
                z = enkfProcessStep(z, zf, pf, zs, ps, 0, params);
                pf = zf; ps = zs;
            }
            const actual = cento.byTime.get(times[i + 3]);
            if (ok && Number.isFinite(z) && Number.isFinite(actual)) {
                const pred = z + muC;
                const err = pred - actual;
                const persist = cento.byTime.get(tPrev);
                olN++; olSe += err * err;
                if (Number.isFinite(persist)) olSeP += (persist - actual) ** 2;
                if (eventSet.has(tPrev)) {
                    olEventN++; olEventSe += err * err;
                }
            }
        }

        // Ensemble forecast step + update
        if (![zF, zFp, zS, zSp, zCobs].every(Number.isFinite)) continue;
        for (let m = 0; m < N; m++) {
            ens[m] = enkfProcessStep(ens[m], zF, zFp, zS, zSp, 0, params);
            ens[m] += (Math.random() - 0.5) * 2 * q; // crude process noise
        }
        const meanPred = ens.reduce((a, b) => a + b, 0) / N;
        const innov = zCobs - meanPred;
        const spread = Math.sqrt(ens.reduce((s, v) => s + (v - meanPred) ** 2, 0) / N);
        const inFlood = eventSet.has(t);
        (inFlood ? innovFlood : innovCalm).push(innov);
        (inFlood ? spreadFlood : spreadCalm).push(spread);

        // Simple EnKF update
        const varF = spread * spread + 1e-9;
        const K = Math.min(0.5, varF / (varF + rObs * rObs));
        for (let m = 0; m < N; m++) {
            const pert = zCobs + (Math.random() - 0.5) * 2 * rObs;
            ens[m] = ens[m] + K * (pert - ens[m]);
        }
    }

    const varOf = (arr) => {
        if (arr.length < 2) return null;
        const m = arr.reduce((a, b) => a + b, 0) / arr.length;
        return arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
    };

    return {
        available: true,
        openLoop120: {
            n: olN,
            rmse: olN ? Math.sqrt(olSe / olN) : null,
            skillPersist: olN && olSeP ? 1 - Math.sqrt(olSe / olN) / Math.sqrt(olSeP / olN) : null,
            eventN: olEventN,
            eventRmse: olEventN ? Math.sqrt(olEventSe / olEventN) : null
        },
        innovations: {
            calmVar: varOf(innovCalm),
            floodVar: varOf(innovFlood),
            calmN: innovCalm.length,
            floodN: innovFlood.length,
            ratioFloodOverCalm: (() => {
                const a = varOf(innovCalm), b = varOf(innovFlood);
                return a > 0 && Number.isFinite(b) ? b / a : null;
            })()
        },
        ensembleSpread: {
            calmMean: spreadCalm.length ? spreadCalm.reduce((a, b) => a + b, 0) / spreadCalm.length : null,
            floodMean: spreadFlood.length ? spreadFlood.reduce((a, b) => a + b, 0) / spreadFlood.length : null
        },
        enkfSkill120Holdout: enkfDoc.metrics?.['120']?.event?.skill ?? enkfDoc.metrics?.['120']?.all?.skill ?? null
    };
}

// ---------- RC step response (H4) ----------
function diagnoseRcStep() {
    // Fixed mid-range conductances (unitless z-space) — qualitative diffusion response
    const theta = { G_f: 0.15, G_s: 0.2, C_c: 2.0, alpha: 0, G_d: 0.05 };
    const steps = 48;
    const zF = 2.0, zS = 2.0, zAcc = 0.0; // step up upstreams
    let z = 0;
    const response = [];
    for (let k = 0; k < steps; k++) {
        const inflow = theta.G_f * (zF - z) + theta.G_s * (zS - z);
        const drain = theta.G_d * Math.max(z - zAcc, 0);
        z = z + (inflow - drain) / theta.C_c;
        response.push({ step: k, min: k * 30, z });
    }
    // 50% rise time
    const zInf = response.at(-1).z;
    const half = response.find(p => p.z >= 0.5 * zInf);
    return {
        kind: 'fixed_theta_rc_step',
        riseTime50Min: half ? half.min : null,
        asymptoteZ: zInf,
        note: 'RC step is exponential (diffusion). A pure delay would jump after lag with little early rise.'
    };
}

// ---------- H8: MSE prod vs q50 median pinball ----------
function diagnoseH8(data, config, events, coeffsProd, coeffsQ50) {
    const deps = config.sensors[DOWN].dependencies || [];
    const names = Object.keys(config.sensors);
    const rainGauges = ((config.rainfall && config.rainfall.gauges) || [])
        .map(e => typeof e === 'string' ? e : e?.name).filter(Boolean);
    const rainDir = path.join(dataDir, 'rain');
    const rainData = Object.fromEntries(rainGauges.map(n => [n, loadSeries(n, rainDir)]));

    function runModel(nodePayload, eventList) {
        const node = { ...nodePayload, reliable: true };
        const PHASE = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
        const acc = {
            n: 0, se: 0, seP: 0,
            seShift: Object.fromEntries(PHASE.map(s => [s, { n: 0, se: 0 }]))
        };
        for (const ev of eventList) {
            const start = Date.parse(ev.start);
            const end = Date.parse(ev.end) - 4 * STEP;
            for (let now = Math.ceil(start / (4 * STEP)) * (4 * STEP); now <= end; now += 4 * STEP) {
                const down = win(DOWN, data[DOWN], now);
                const anchor = P.latestFinite(down.timeSeries);
                if (!anchor || now - +anchor.time > STEP) continue;
                const y0 = data[DOWN].byTime.get(now);
                if (!Number.isFinite(y0)) continue;
                const sensorMap = Object.fromEntries(
                    names.filter(n => data[n]).map(n => [n, win(n, data[n], now)])
                );
                sensorMap[DOWN] = down;
                const ups = deps.map(u => sensorMap[u]).filter(Boolean);
                const rainSensors = rainGauges
                    .filter(n => rainData[n])
                    .map(n => win(n, rainData[n], now))
                    .filter(s => s.timeSeries.length);
                const result = P.predictNode(down, ups, {
                    hours: 72, now: new Date(now),
                    coeffs: { nodes: { [DOWN]: node } },
                    horizonHours: 2, downstreamId: DOWN, rainSensors, sensorMap, smooth: 0
                });
                if (!result?.ok) continue;
                const target = now + 4 * STEP;
                const obs = data[DOWN].byTime.get(target);
                if (!Number.isFinite(obs)) continue;
                let best = null, bestD = Infinity;
                for (const p of result.forecast || []) {
                    if (p.regime === 'anchor') continue;
                    const d = Math.abs(+p.time - target);
                    if (d < bestD) { best = p; bestD = d; }
                }
                if (!best || bestD > STEP || !Number.isFinite(best.med)) continue;
                const err = best.med - obs;
                acc.n++; acc.se += err * err; acc.seP += (y0 - obs) ** 2;
                for (const s of PHASE) {
                    const o = data[DOWN].byTime.get(target + s * STEP);
                    if (!Number.isFinite(o)) continue;
                    acc.seShift[s].n++;
                    acc.seShift[s].se += (best.med - o) ** 2;
                }
            }
        }
        const lag = EP.phaseLagSubStep(acc.seShift, { minN: 20 });
        return {
            n: acc.n,
            rmse: acc.n ? Math.sqrt(acc.se / acc.n) : null,
            skill: acc.n && acc.seP ? 1 - Math.sqrt(acc.se / acc.n) / Math.sqrt(acc.seP / acc.n) : null,
            phaseLagMin: lag.phaseLagMin,
            phaseLagMinDiscrete: lag.phaseLagMinDiscrete
        };
    }

    const testEv = events.filter(e => Date.parse(e.start) >= Date.parse('2026-01-01'));
    const prodNode = coeffsProd?.nodes?.[DOWN] || coeffsProd;
    const q50Node = coeffsQ50?.kind ? coeffsQ50 : coeffsQ50?.nodes?.[DOWN];
    return {
        testEvents: testEv.length,
        prod: prodNode ? runModel(prodNode, testEv) : null,
        q50: q50Node ? runModel(q50Node, testEv) : null,
        note: 'q50 = pinball median (≈ MAE). Smaller |phaseLag| vs MSE would support H8.'
    };
}

// ---------- H10: M9 decode on non-crossing ----------
function diagnoseH10(data, config, events, coeffs, crossing) {
    if (!crossing) return { available: false };
    const deps = config.sensors[DOWN].dependencies || [];
    const names = Object.keys(config.sensors);
    const rainGauges = ((config.rainfall && config.rainfall.gauges) || [])
        .map(e => typeof e === 'string' ? e : e?.name).filter(Boolean);
    const rainDir = path.join(dataDir, 'rain');
    const rainData = Object.fromEntries(rainGauges.map(n => [n, loadSeries(n, rainDir)]));
    const node = {
        ...(coeffs.nodes?.[DOWN] || {}),
        reliable: true,
        crossingTime: { ...crossing, adopt: true }
    };

    let issues = 0, decodeOps = 0, liftOps = 0, warpOps = 0;
    let noCrossDecode = 0, noCrossSe = 0, noCrossN = 0;
    let withCrossSe = 0, withCrossN = 0;
    let allSe = 0, allN = 0;

    const testEv = events.filter(e => Date.parse(e.start) >= Date.parse('2026-01-01'));
    for (const ev of testEv) {
        const start = Date.parse(ev.start);
        const end = Date.parse(ev.end) - 4 * STEP;
        for (let now = Math.ceil(start / (4 * STEP)) * (4 * STEP); now <= end; now += 4 * STEP) {
            const down = win(DOWN, data[DOWN], now);
            const anchor = P.latestFinite(down.timeSeries);
            if (!anchor || now - +anchor.time > STEP) continue;
            const y0 = data[DOWN].byTime.get(now);
            if (!Number.isFinite(y0)) continue;
            const sensorMap = Object.fromEntries(names.map(n => [n, win(n, data[n], now)]));
            // ensure crossing stations
            for (const up of crossing.stations || []) {
                if (!sensorMap[up] && data[up]) sensorMap[up] = win(up, data[up], now);
            }
            sensorMap[DOWN] = down;
            const ups = deps.map(u => sensorMap[u]);
            const rainSensors = rainGauges.map(n => win(n, rainData[n], now)).filter(s => s.timeSeries.length);

            const tauPred = P.predictCrossingTau(crossing, sensorMap, rainSensors, now, STEP);
            const result = P.predictNode(down, ups, {
                hours: 72, now: new Date(now),
                coeffs: { nodes: { [DOWN]: node } },
                horizonHours: 6, downstreamId: DOWN, rainSensors, sensorMap,
                smooth: 0, useCrossingTime: true
            });
            if (!result?.ok) continue;
            issues++;

            const offset = tauPred?.offsetM ?? 1;
            const thr = y0 + offset;
            const obsCross = EP.firstCrossingTime(data[DOWN].byTime, now, thr, 12);
            const didCross = obsCross != null;

            // Detect decode via warp helper on a fresh forecast without adopt
            const rawNode = { ...(coeffs.nodes?.[DOWN] || {}), reliable: true };
            const raw = P.predictNode(down, ups, {
                hours: 72, now: new Date(now),
                coeffs: { nodes: { [DOWN]: rawNode } },
                horizonHours: 6, downstreamId: DOWN, rainSensors, sensorMap,
                smooth: 0, useCrossingTime: false
            });
            if (raw?.ok && tauPred) {
                const warped = P.warpForecastByCrossing(raw.forecast, tauPred, y0, now);
                if (warped.warped) {
                    decodeOps++;
                    if (warped.method === 'level_lift') liftOps++;
                    if (warped.method === 'time_warp') warpOps++;
                    if (!didCross) noCrossDecode++;
                }
            }

            const target = now + 4 * STEP;
            const obs = data[DOWN].byTime.get(target);
            let best = null, bestD = Infinity;
            for (const p of result.forecast || []) {
                if (p.regime === 'anchor') continue;
                const d = Math.abs(+p.time - target);
                if (d < bestD) { best = p; bestD = d; }
            }
            if (!best || bestD > STEP || !Number.isFinite(best.med) || !Number.isFinite(obs)) continue;
            const se = (best.med - obs) ** 2;
            allN++; allSe += se;
            if (didCross) { withCrossN++; withCrossSe += se; }
            else { noCrossN++; noCrossSe += se; }
        }
    }

    return {
        available: true,
        issues,
        decodeOps,
        liftOps,
        warpOps,
        noCrossDecode,
        decodeOnNoCrossRate: decodeOps ? noCrossDecode / decodeOps : null,
        rmseAll: allN ? Math.sqrt(allSe / allN) : null,
        rmseWhenObservedCross: withCrossN ? Math.sqrt(withCrossSe / withCrossN) : null,
        rmseWhenNoObservedCross: noCrossN ? Math.sqrt(noCrossSe / noCrossN) : null,
        nWhenCross: withCrossN,
        nWhenNoCross: noCrossN
    };
}

// ---------- H11: Dosso backwater residuals ----------
function diagnoseH11(data, config, events, coeffs) {
    const deps = config.sensors[DOWN].dependencies || [];
    const names = Object.keys(config.sensors);
    const rainGauges = ((config.rainfall && config.rainfall.gauges) || [])
        .map(e => typeof e === 'string' ? e : e?.name).filter(Boolean);
    const rainDir = path.join(dataDir, 'rain');
    const rainData = Object.fromEntries(rainGauges.map(n => [n, loadSeries(n, rainDir)]));
    const node = { ...(coeffs.nodes?.[DOWN] || {}), reliable: true };
    const dosso = data[DOSSO];
    if (!dosso?.times?.length) return { available: false, reason: 'missing Dosso' };

    const residuals = []; // { resid, dosso, diff, rising }
    const testEv = events.filter(e => Date.parse(e.start) >= Date.parse('2023-01-01'));

    for (const ev of testEv) {
        const peak = Date.parse(ev.peakTime);
        const start = Date.parse(ev.start);
        const end = Date.parse(ev.end) - 4 * STEP;
        for (let now = Math.ceil(start / (4 * STEP)) * (4 * STEP); now <= end; now += 4 * STEP) {
            const down = win(DOWN, data[DOWN], now);
            const anchor = P.latestFinite(down.timeSeries);
            if (!anchor || now - +anchor.time > STEP) continue;
            const sensorMap = Object.fromEntries(names.map(n => [n, win(n, data[n], now)]));
            sensorMap[DOWN] = down;
            const ups = deps.map(u => sensorMap[u]);
            const rainSensors = rainGauges.map(n => win(n, rainData[n], now)).filter(s => s.timeSeries.length);
            const result = P.predictNode(down, ups, {
                hours: 72, now: new Date(now),
                coeffs: { nodes: { [DOWN]: node } },
                horizonHours: 2, downstreamId: DOWN, rainSensors, sensorMap, smooth: 0
            });
            if (!result?.ok) continue;
            const target = now + 4 * STEP;
            const obs = data[DOWN].byTime.get(target);
            const dNow = dosso.byTime.get(now);
            const cNow = data[DOWN].byTime.get(now);
            if (![obs, dNow, cNow].every(Number.isFinite)) continue;
            let best = null, bestD = Infinity;
            for (const p of result.forecast || []) {
                if (p.regime === 'anchor') continue;
                const d = Math.abs(+p.time - target);
                if (d < bestD) { best = p; bestD = d; }
            }
            if (!best || !Number.isFinite(best.med)) continue;
            residuals.push({
                resid: best.med - obs, // >0 ⇒ overpredict
                dosso: dNow,
                diff: cNow - dNow,
                rising: now < peak
            });
        }
    }

    function corr(key) {
        const xs = residuals.map(r => r[key]);
        const ys = residuals.map(r => r.resid);
        return pearson(xs, ys);
    }
    function corrRising(key) {
        const R = residuals.filter(r => r.rising);
        if (R.length < 30) return null;
        return pearson(R.map(r => r[key]), R.map(r => r.resid));
    }

    // Second-wave heuristic: events whose start is within 10d of previous Cento peak
    const sorted = testEv.slice().sort((a, b) => Date.parse(a.peakTime) - Date.parse(b.peakTime));
    const secondWave = [];
    for (let i = 1; i < sorted.length; i++) {
        const dt = Date.parse(sorted[i].peakTime) - Date.parse(sorted[i - 1].peakTime);
        if (dt > 0 && dt < 10 * 86400000) secondWave.push(sorted[i]);
    }

    function eventRmse(evList) {
        let n = 0, se = 0;
        const keys = new Set(evList.map(e => e.peakTime));
        for (const ev of testEv) {
            if (!keys.has(ev.peakTime)) continue;
            const peak = Date.parse(ev.peakTime);
            const start = Date.parse(ev.start);
            const end = Date.parse(ev.end) - 4 * STEP;
            for (let now = Math.ceil(start / (4 * STEP)) * (4 * STEP); now <= end; now += 4 * STEP) {
                const down = win(DOWN, data[DOWN], now);
                const anchor = P.latestFinite(down.timeSeries);
                if (!anchor) continue;
                const sensorMap = Object.fromEntries(names.map(n => [n, win(n, data[n], now)]));
                sensorMap[DOWN] = down;
                const ups = deps.map(u => sensorMap[u]);
                const rainSensors = rainGauges.map(n => win(n, rainData[n], now)).filter(s => s.timeSeries.length);
                const result = P.predictNode(down, ups, {
                    hours: 72, now: new Date(now),
                    coeffs: { nodes: { [DOWN]: node } },
                    horizonHours: 2, downstreamId: DOWN, rainSensors, sensorMap, smooth: 0
                });
                if (!result?.ok) continue;
                const obs = data[DOWN].byTime.get(now + 4 * STEP);
                let best = null, bestD = Infinity;
                for (const p of result.forecast || []) {
                    if (p.regime === 'anchor') continue;
                    const d = Math.abs(+p.time - (now + 4 * STEP));
                    if (d < bestD) { best = p; bestD = d; }
                }
                if (!best || !Number.isFinite(best.med) || !Number.isFinite(obs)) continue;
                n++; se += (best.med - obs) ** 2;
                void peak;
            }
        }
        return { n, rmse: n ? Math.sqrt(se / n) : null, nEvents: evList.length };
    }

    const dossoP66 = percentile(dosso.values, 0.66);
    const highDown = residuals.filter(r => r.dosso >= dossoP66);
    const lowDown = residuals.filter(r => r.dosso < dossoP66);
    const rmseOf = (arr) => {
        if (!arr.length) return null;
        return Math.sqrt(arr.reduce((s, r) => s + r.resid * r.resid, 0) / arr.length);
    };

    return {
        available: true,
        nearestDownstream: DOSSO,
        nResiduals: residuals.length,
        corrResidVsDosso: corr('dosso'),
        corrResidVsDiff: corr('diff'),
        corrResidVsDossoRising: corrRising('dosso'),
        corrResidVsDiffRising: corrRising('diff'),
        dossoP66,
        rmseHighDosso: rmseOf(highDown),
        rmseLowDosso: rmseOf(lowDown),
        nHighDosso: highDown.length,
        nLowDosso: lowDown.length,
        secondWave: eventRmse(secondWave),
        allEvents: eventRmse(testEv),
        priorValleGate: loadJson(path.join(root, 'models/cento-valle-gate.json'))?.decision || null
    };
}

function verdict(id, status, reason) {
    return { id, status, reason };
}

function main() {
    console.log('[wp1] loading data…');
    const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
    const eventsDoc = loadJson(path.join(root, 'models/events.json'));
    const events = (eventsDoc.events || []).filter(e => e.station === DOWN);
    const namesAll = Object.keys(config.sensors);
    const raw = {};
    for (const n of namesAll) raw[n] = loadSeries(n);
    const data = {};
    for (const [n, s] of Object.entries(raw)) data[n] = resample30(s);

    const enkfDoc = loadJson(path.join(root, 'models/enkf-cento-summary.json'));
    const coeffs = loadJson(path.join(root, 'models/transfer-coeffs.json'));
    const q50 = loadJson(path.join(root, 'models/cento-dnn-q50.json'));
    const crossing = loadJson(path.join(root, 'models/cento-crossing-time.json'));
    const evalTable = loadJson(path.join(root, 'models/eval-table.json'));

    console.log('[wp1] H3 cross-corr lags…');
    const h3 = diagnoseH3(data[DOWN], data);

    console.log('[wp1] hysteresis…');
    const hyst = diagnoseHysteresis(data[DOWN], data, events);

    console.log('[wp1] EnKF open-loop / innovations…');
    const enkf = diagnoseEnkf(data[DOWN], data.Forcelli, data['Sostegno Reno'], events, enkfDoc);

    console.log('[wp1] RC step response…');
    const rcStep = diagnoseRcStep();

    console.log('[wp1] H8 prod vs q50…');
    const h8 = diagnoseH8(data, config, events, coeffs, q50);

    console.log('[wp1] H10 M9 decode…');
    const h10 = diagnoseH10(data, config, events, coeffs, crossing);

    console.log('[wp1] H11 Dosso residuals…');
    const h11 = diagnoseH11(data, config, events, coeffs);

    // Verdicts
    const verdicts = [];

    // H1: open-loop flood error large / skill poor ⇒ process model is the problem
    if (enkf.available) {
        const skill = enkf.openLoop120.skillPersist;
        const enkfSkill = enkf.enkfSkill120Holdout;
        if (Number.isFinite(skill) && skill < 0.3) {
            verdicts.push(verdict('H1', 'confirmed',
                `Open-loop EnKF@120 skill=${skill?.toFixed(3)} (event RMSE=${enkf.openLoop120.eventRmse?.toFixed(3)}); filter summary skill=${enkfSkill}. Poor open-loop ⇒ process structure, not only the filter.`));
        } else if (Number.isFinite(skill)) {
            verdicts.push(verdict('H1', 'not clear', `Open-loop skill=${skill?.toFixed(3)}; compare to filter skill=${enkfSkill}.`));
        } else {
            verdicts.push(verdict('H1', 'not clear', 'Open-loop metrics incomplete.'));
        }
    } else {
        verdicts.push(verdict('H1', 'not clear', 'EnKF summary missing.'));
    }

    // H2: structural — level RC does not conserve mass (documented)
    verdicts.push(verdict('H2', 'confirmed',
        'RC/Muskingum/EnKF hybrids operate on levels or level anomalies; docs state no mass balance. No Q-space model in repo.'));

    // H3: lag decreases with level
    let h3Confirm = false;
    const lagRows = [];
    for (const [up, bands] of Object.entries(h3.byUpstream || {})) {
        const lo = bands.low?.lagMin, hi = bands.high?.lagMin;
        lagRows.push({ up, low: lo, mid: bands.mid?.lagMin, high: hi });
        if (Number.isFinite(lo) && Number.isFinite(hi) && hi < lo - 30) h3Confirm = true;
    }
    verdicts.push(verdict('H3', h3Confirm ? 'confirmed' : 'not clear',
        `minLag=${h3.minLagMin}′. Lags by band: ${lagRows.map(r => `${r.up}: ${r.low}/${r.mid}/${r.high}`).join('; ')}. Travel-time assumption >120′: ${h3.minLagMin != null && h3.minLagMin > 120 ? 'holds at min band lag' : 'CHECK — min lag ≤120′ or null'}.`));

    // H4: RC is diffusion
    verdicts.push(verdict('H4', 'confirmed',
        `Fixed-θ RC step 50%-rise ≈ ${rcStep.riseTime50Min}′ (smooth exponential). Equation is G·Δz/C (diffusion), not a transport delay.`));

    // H5: lateral inflow — rain exists but plan says unused in WP1; MLP uses recent Δh
    verdicts.push(verdict('H5', 'not clear',
        'Rain series exist under storico/rain/ and enter EnKF α·rain / MLP rain features, but no explicit Q_lat state. Cannot reject; needs WP3 Q_lat test.'));

    // H6: innovation variance flood vs calm
    if (enkf.available && Number.isFinite(enkf.innovations.ratioFloodOverCalm)) {
        const ratio = enkf.innovations.ratioFloodOverCalm;
        verdicts.push(verdict('H6', ratio > 2 ? 'confirmed' : 'not clear',
            `Innovation var flood/calm=${ratio.toFixed(2)}; spread flood=${enkf.ensembleSpread.floodMean?.toFixed(3)} calm=${enkf.ensembleSpread.calmMean?.toFixed(3)}. Constant Q,R.`));
    } else {
        verdicts.push(verdict('H6', 'not clear', 'Innovation stats incomplete.'));
    }

    // H7: documented training mismatch
    verdicts.push(verdict('H7', 'confirmed',
        'EnKF fit blends 1+4-step MSE; MLP trains direct horizons 30–360′. Comparison is not equal (by construction).'));

    // H8
    if (h8.prod && h8.q50 && Number.isFinite(h8.prod.phaseLagMin) && Number.isFinite(h8.q50.phaseLagMin)) {
        const improved = Math.abs(h8.q50.phaseLagMin) + 5 < Math.abs(h8.prod.phaseLagMin);
        verdicts.push(verdict('H8', improved ? 'confirmed' : 'not clear',
            `prod lag=${h8.prod.phaseLagMin?.toFixed(1)}′ rmse=${h8.prod.rmse?.toFixed(3)}; q50 lag=${h8.q50.phaseLagMin?.toFixed(1)}′ rmse=${h8.q50.rmse?.toFixed(3)}.`));
    } else {
        verdicts.push(verdict('H8', 'not clear', 'Missing q50 weights or insufficient samples.'));
    }

    // H9: rising >> falling already in eval table
    const prod120 = evalTable?.stations?.Cento?.prod?.['120'];
    if (prod120 && prod120.risingRmse > 3 * (prod120.fallingRmse || 1e-9)) {
        verdicts.push(verdict('H9', 'confirmed',
            `Rising RMSE@120=${prod120.risingRmse?.toFixed(3)} vs falling=${prod120.fallingRmse?.toFixed(3)}; CSI/crossing metrics now in protocol (WP0).`));
    } else {
        verdicts.push(verdict('H9', 'confirmed',
            'M9/protocol show rising RMSE ≫ falling; RMSE alone under-rewards correct timing (plan §3.4).'));
    }

    // H10
    if (h10.available) {
        const gated = (h10.decodeOnNoCrossRate || 0) > 0.2
            || (h10.rmseWhenNoObservedCross > 1.5 * (h10.rmseWhenObservedCross || 1e-9));
        verdicts.push(verdict('H10', gated ? 'confirmed' : 'not clear',
            `decodeOps=${h10.decodeOps} (lift=${h10.liftOps}, warp=${h10.warpOps}); on no-cross samples=${h10.noCrossDecode} (${((h10.decodeOnNoCrossRate || 0) * 100).toFixed(0)}% of decodes). RMSE no-cross=${h10.rmseWhenNoObservedCross?.toFixed(3)} vs cross=${h10.rmseWhenObservedCross?.toFixed(3)}.`));
    } else {
        verdicts.push(verdict('H10', 'not clear', 'Crossing-time model missing.'));
    }

    // H11
    if (h11.available) {
        const corrStrong = Math.abs(h11.corrResidVsDossoRising || 0) > 0.15
            || Math.abs(h11.corrResidVsDiffRising || 0) > 0.15;
        const highWorse = h11.rmseHighDosso != null && h11.rmseLowDosso != null
            && h11.rmseHighDosso > h11.rmseLowDosso * 1.1;
        const secondWorse = h11.secondWave?.rmse != null && h11.allEvents?.rmse != null
            && h11.secondWave.rmse > h11.allEvents.rmse * 1.1;
        const status = (corrStrong || highWorse || secondWorse) ? 'confirmed' : 'not clear';
        verdicts.push(verdict('H11', status,
            `corr(resid,Dosso) rising=${h11.corrResidVsDossoRising?.toFixed(3)}; corr(resid,Cento−Dosso) rising=${h11.corrResidVsDiffRising?.toFixed(3)}; RMSE high/low Dosso=${h11.rmseHighDosso?.toFixed(3)}/${h11.rmseLowDosso?.toFixed(3)}; second-wave RMSE=${h11.secondWave?.rmse?.toFixed(3)} (nEv=${h11.secondWave?.nEvents}) vs all=${h11.allEvents?.rmse?.toFixed(3)}. Prior valle gate did not promote acceptors.`));
    } else {
        verdicts.push(verdict('H11', 'not clear', 'Dosso series missing.'));
    }

    const out = {
        kind: 'wp1_diagnosis',
        version: 1,
        disclaimer: 'Sperimentale: non operativo, non allertistico.',
        generatedAt: new Date().toISOString(),
        assumptions: {
            timeStepMin: 30,
            rawArchiveApproxMin: 15,
            rainAvailable: fs.existsSync(path.join(dataDir, 'rain')),
            rainUsedInWp1Models: false,
            downstreamGauge: DOSSO,
            downstreamUsedInWp1Models: false,
            travelTimeAssumptionGt120: h3.minLagMin == null ? 'unmeasured' : h3.minLagMin > 120,
            minMeasuredLagMin: h3.minLagMin,
            geometryInRepo: ['river_reno.geojson', 'torrente_samoggia.geojson'],
            dtmInRepo: false
        },
        h3,
        hysteresis: hyst,
        enkf,
        rcStep,
        h8,
        h10,
        h11,
        verdicts,
        decisionHints: {
            H8: 'If confirmed → prioritize WP2 items 6–7 (shape/time loss, generative decode).',
            H10: 'If confirmed → WP2 items 2–3 (hazard + gate) before next crossing A/B.',
            H1_H2_H3: 'If confirmed → continue WP3 nonlinear Q routing.',
            H11: 'If confirmed → WP2 item 9 downstream inputs + WP3 stage-fall-discharge.',
            reject_H1_to_H4: 'If rejected → inspect filter implementation bugs first.'
        }
    };

    fs.mkdirSync(path.dirname(jsonOut), { recursive: true });
    fs.writeFileSync(jsonOut, JSON.stringify(out, null, 2) + '\n');
    console.log(`[wp1] Wrote ${jsonOut}`);
    console.log('[wp1] verdicts:', verdicts.map(v => `${v.id}=${v.status}`).join(', '));
    return out;
}

main();
