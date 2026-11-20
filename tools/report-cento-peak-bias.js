#!/usr/bin/env node
/**
 * Analisi picco vs svuotamento — predittore prod Cento su tutte le piene del catalogo.
 * Scrive models/analisi-cento-picchi.{json,html}.
 *
 * Uso:
 *   node tools/report-cento-peak-bias.js
 *   node tools/report-cento-peak-bias.js --detail-n 20 --stride-all 60 --stride-detail 30
 */
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

const STEP = 30 * 60000;
const WINDOW = 72 * 3600000;
const HORIZONS = [2, 4, 8, 12]; // 60′, 120′, 240′, 360′
const MAX_SHIFT = 4; // ±2 h for phase search
const station = 'Cento';
const dataDir = path.resolve(root, opt('data', 'storico'));
const eventsPath = path.resolve(root, opt('events', 'models/events.json'));
const coeffsPath = path.resolve(root, opt('coeffs', 'models/transfer-coeffs-runtime.json'));
const detailN = Number(opt('detail-n', 25));
const strideAllMin = Number(opt('stride-all', 60));
const strideDetailMin = Number(opt('stride-detail', 30));
const outJson = path.resolve(root, opt('json', 'models/analisi-cento-picchi.json'));
const outHtml = path.resolve(root, opt('html', 'models/analisi-cento-picchi.html'));

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
        times.push(t);
        values.push(v);
        byTime.set(t, v);
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

function median(xs) {
    const a = xs.filter(Number.isFinite).sort((x, y) => x - y);
    if (!a.length) return null;
    const m = (a.length - 1) / 2;
    return (a[Math.floor(m)] + a[Math.ceil(m)]) / 2;
}

function mean(xs) {
    const a = xs.filter(Number.isFinite);
    if (!a.length) return null;
    return a.reduce((s, v) => s + v, 0) / a.length;
}

function rmse(errs) {
    const a = errs.filter(Number.isFinite);
    if (!a.length) return null;
    return Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);
}

function quantile(xs, q) {
    const a = xs.filter(Number.isFinite).sort((x, y) => x - y);
    if (!a.length) return null;
    const i = (a.length - 1) * q;
    const lo = Math.floor(i), hi = Math.ceil(i);
    return a[lo] + (a[hi] - a[lo]) * (i - lo);
}

function esc(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function emptyLimb() {
    return Object.fromEntries(HORIZONS.map((h) => [h, { n: 0, se: 0, bias: 0, ae: 0, seP: 0 }]));
}

function addLimb(bucket, h, err, errP) {
    const b = bucket[h];
    b.n++;
    b.se += err * err;
    b.bias += err;
    b.ae += Math.abs(err);
    b.seP += errP * errP;
}

function finalizeLimb(bucket) {
    const out = {};
    for (const [h, b] of Object.entries(bucket)) {
        out[h] = {
            n: b.n,
            rmse: b.n ? Math.sqrt(b.se / b.n) : null,
            mae: b.n ? b.ae / b.n : null,
            bias: b.n ? b.bias / b.n : null,
            skillPersist: b.n && b.seP > 0 ? 1 - Math.sqrt(b.se / b.n) / Math.sqrt(b.seP / b.n) : null
        };
    }
    return out;
}

function bestPhaseShift(obsArr, predArr) {
    let bestS = 0, bestSe = Infinity;
    for (let s = -MAX_SHIFT; s <= MAX_SHIFT; s++) {
        let se = 0, n = 0;
        for (let i = 0; i < obsArr.length; i++) {
            const j = i + s;
            if (j < 0 || j >= predArr.length) continue;
            const o = obsArr[i], p = predArr[j];
            if (!Number.isFinite(o) || !Number.isFinite(p)) continue;
            se += (p - o) * (p - o);
            n++;
        }
        if (n >= 8 && se / n < bestSe) {
            bestSe = se / n;
            bestS = s;
        }
    }
    const aligned = [];
    for (let i = 0; i < obsArr.length; i++) {
        const j = i + bestS;
        const o = obsArr[i];
        const p = j >= 0 && j < predArr.length ? predArr[j] : null;
        if (Number.isFinite(o) && Number.isFinite(p)) aligned.push(p - o);
    }
    return {
        shiftSteps: bestS,
        // s>0 ⇒ pred[i+s] aligns to obs[i] ⇒ the forecast wave arrives late.
        phaseLagMin: bestS * 30,
        rmseAligned: aligned.length ? Math.sqrt(aligned.reduce((s, e) => s + e * e, 0) / aligned.length) : null,
        ampBias: mean(aligned)
    };
}

function pickForecast(result, targetMs) {
    let best = null, bestDist = Infinity;
    for (const p of result.forecast || []) {
        if (p.regime === 'anchor') continue;
        const dist = Math.abs(+p.time - targetMs);
        if (dist < bestDist) {
            best = p;
            bestDist = dist;
        }
    }
    if (!best || bestDist > STEP) return null;
    const pred = best.med ?? best.value;
    return Number.isFinite(pred) ? pred : null;
}

console.error('[analisi] loading…');
const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));
const coeffs = JSON.parse(fs.readFileSync(coeffsPath, 'utf8'));
const eventsDoc = JSON.parse(fs.readFileSync(eventsPath, 'utf8'));
const node = coeffs.nodes?.[station];
if (!node || node.kind !== 'window_delta_mlp') {
    console.error('Need window_delta_mlp for Cento in coeffs');
    process.exit(1);
}
const forced = { ...node, reliable: true };
const upstreams = node.upstreams || config.sensors[station]?.dependencies || [];
const names = [...new Set([station, ...upstreams])];
const data = Object.fromEntries(names.map((n) => [n, loadSeries(n)]));
const rainGauges = ((config.rainfall && config.rainfall.gauges) || [])
    .map((e) => (typeof e === 'string' ? e : e?.name))
    .filter(Boolean);
const rainDir = path.join(dataDir, 'rain');
const rainData = Object.fromEntries(rainGauges.map((n) => [n, loadSeries(n, rainDir)]));

const testFromMs = Date.parse(eventsDoc.testFrom || '2026-01-01');
const allEvents = (eventsDoc.events || [])
    .filter((e) => e.station === station)
    .map((e) => ({
        ...e,
        split: Date.parse(e.peakTime) >= testFromMs ? 'test' : 'train'
    }))
    .sort((a, b) => Date.parse(a.peakTime) - Date.parse(b.peakTime));

const detailKeys = new Set();
for (const e of allEvents.filter((x) => x.split === 'test')) detailKeys.add(e.peakTime);
for (const e of [...allEvents].sort((a, b) => b.peakLevel - a.peakLevel).slice(0, detailN)) {
    detailKeys.add(e.peakTime);
}

const globalRising = emptyLimb();
const globalFalling = emptyLimb();
const byPeakBin = {
    lt2: emptyLimb(),
    '2to4': emptyLimb(),
    '4to6': emptyLimb(),
    ge6: emptyLimb()
};
const peakIssue = Object.fromEntries(HORIZONS.map((h) => [h, []]));
const eventRows = [];
const detailSeries = [];

function peakBin(level) {
    if (level < 2) return 'lt2';
    if (level < 4) return '2to4';
    if (level < 6) return '4to6';
    return 'ge6';
}

function predictAt(now) {
    const down = windowSensor(station, data[station], now);
    const anchor = P.latestFinite(down.timeSeries);
    if (!anchor || now - +anchor.time > STEP) return null;
    const y0 = data[station].byTime.get(now);
    if (!Number.isFinite(y0)) return null;
    const sensorMap = Object.fromEntries(names.map((n) => [n, windowSensor(n, data[n], now)]));
    sensorMap[station] = down;
    const ups = upstreams.map((u) => sensorMap[u]);
    const rainSensors = rainGauges
        .map((n) => windowSensor(n, rainData[n], now))
        .filter((s) => s.timeSeries.length);
    const result = P.predictNode(down, ups, {
        hours: 72,
        now: new Date(now),
        coeffs: { nodes: { [station]: forced } },
        horizonHours: 6,
        downstreamId: station,
        rainSensors,
        sensorMap,
        smooth: 0
    });
    if (!result?.ok) return null;
    return { result, y0 };
}

let done = 0;
const tStart = Date.now();
for (const ev of allEvents) {
    done++;
    if (done % 25 === 0 || done === allEvents.length) {
        const elapsed = ((Date.now() - tStart) / 1000).toFixed(0);
        console.error(`[analisi] ${done}/${allEvents.length} eventi (${elapsed}s)`);
    }
    const peak = Date.parse(ev.peakTime);
    const start = Date.parse(ev.start);
    const end = Date.parse(ev.end);
    const isDetail = detailKeys.has(ev.peakTime);
    const strideMs = (isDetail ? strideDetailMin : strideAllMin) * 60000;
    const maxH = Math.max(...HORIZONS);
    const issueEnd = end - maxH * STEP;

    const risingErrs = Object.fromEntries(HORIZONS.map((h) => [h, []]));
    const fallingErrs = Object.fromEntries(HORIZONS.map((h) => [h, []]));
    const risingObs = Object.fromEntries(HORIZONS.map((h) => [h, []]));
    const risingPred = Object.fromEntries(HORIZONS.map((h) => [h, []]));
    const seriesHours = [];
    const seriesHours360 = [];
    const seriesObs = [];
    const seriesObs360 = [];
    const seriesPred120 = [];
    const seriesPred360 = [];
    const seriesPersist120 = [];
    const seriesLead120 = []; // phase lead: use h=150′ knot as 120′ target proxy via h=5 if present, else scale
    const seriesBoost120 = [];

    // Peak-issued forecasts: issue at peak − h
    for (const h of HORIZONS) {
        const issueAt = peak - h * STEP;
        const got = predictAt(issueAt);
        if (!got) continue;
        const pred = pickForecast(got.result, peak);
        const obs = data[station].byTime.get(peak);
        if (!Number.isFinite(pred) || !Number.isFinite(obs)) continue;
        peakIssue[h].push({
            peakTime: ev.peakTime,
            split: ev.split,
            peakLevel: ev.peakLevel,
            riseM: ev.riseM,
            err: pred - obs,
            pred,
            obs,
            y0: got.y0
        });
    }

    for (let now = Math.ceil(start / strideMs) * strideMs; now <= issueEnd; now += strideMs) {
        const got = predictAt(now);
        if (!got) continue;
        const { result, y0 } = got;
        const limbRising = now < peak;
        const bin = peakBin(ev.peakLevel);

        for (const h of HORIZONS) {
            const target = now + h * STEP;
            const obs = data[station].byTime.get(target);
            if (!Number.isFinite(obs)) continue;
            const pred = pickForecast(result, target);
            if (!Number.isFinite(pred)) continue;
            const err = pred - obs;
            const errP = y0 - obs;
            const limb = limbRising ? globalRising : globalFalling;
            addLimb(limb, h, err, errP);
            if (limbRising) addLimb(byPeakBin[bin], h, err, errP);
            (limbRising ? risingErrs : fallingErrs)[h].push(err);
            if (limbRising) {
                risingObs[h].push(obs);
                risingPred[h].push(pred);
            }
        }

        if (isDetail) {
            const target120 = now + 4 * STEP;
            const target360 = now + 12 * STEP;
            const obsAtTarget = data[station].byTime.get(target120);
            const pred120 = pickForecast(result, target120);
            const pred360 = pickForecast(result, target360);
            // Phase-lead proxy: attribute the 150′ knot (h=5) to the 120′ target if available,
            // else fall back to 180′ (h=6) interpolated toward 120′.
            let lead120 = null;
            const p150 = pickForecast(result, now + 5 * STEP);
            const p180 = pickForecast(result, now + 6 * STEP);
            if (Number.isFinite(p150)) lead120 = y0 + (p150 - y0) * (4 / 5);
            else if (Number.isFinite(p180)) lead120 = y0 + (p180 - y0) * (4 / 6);

            // Amplitude boost on rising: inflate positive Δy by 1.35 (calibrated below on train peaks).
            let boost120 = pred120;
            if (Number.isFinite(pred120) && limbRising) {
                const dy = pred120 - y0;
                boost120 = y0 + (dy > 0 ? 1.35 * dy : dy);
            }

            const obs360 = data[station].byTime.get(target360);
            seriesHours.push((target120 - peak) / 3600000);
            seriesHours360.push((target360 - peak) / 3600000);
            seriesObs.push(Number.isFinite(obsAtTarget) ? obsAtTarget : null);
            seriesObs360.push(Number.isFinite(obs360) ? obs360 : null);
            seriesPred120.push(Number.isFinite(pred120) ? pred120 : null);
            seriesPred360.push(Number.isFinite(pred360) ? pred360 : null);
            seriesPersist120.push(Number.isFinite(y0) ? y0 : null);
            seriesLead120.push(Number.isFinite(lead120) ? lead120 : null);
            seriesBoost120.push(Number.isFinite(boost120) ? boost120 : null);
        }
    }

    const phase120 = bestPhaseShift(risingObs[4], risingPred[4]);
    const peakErr120 = (peakIssue[4].find((p) => p.peakTime === ev.peakTime) || {}).err ?? null;
    const risingMin120 = risingErrs[4].length ? Math.min(...risingErrs[4]) : null;

    eventRows.push({
        peakTime: ev.peakTime,
        start: ev.start,
        end: ev.end,
        peakLevel: ev.peakLevel,
        riseM: ev.riseM,
        split: ev.split,
        detail: isDetail,
        risingRmse120: rmse(risingErrs[4]),
        fallingRmse120: rmse(fallingErrs[4]),
        risingBias120: mean(risingErrs[4]),
        fallingBias120: mean(fallingErrs[4]),
        risingMin120,
        risingRmse360: rmse(risingErrs[12]),
        fallingRmse360: rmse(fallingErrs[12]),
        risingBias360: mean(risingErrs[12]),
        risingMin360: risingErrs[12].length ? Math.min(...risingErrs[12]) : null,
        peakErr120,
        peakErr360: (peakIssue[12].find((p) => p.peakTime === ev.peakTime) || {}).err ?? null,
        phaseLag120: phase120.phaseLagMin,
        ampBiasAligned120: phase120.ampBias,
        rmseAligned120: phase120.rmseAligned,
        nRising120: risingErrs[4].length,
        nFalling120: fallingErrs[4].length
    });

    if (isDetail) {
        const errLead = [];
        const errBoost = [];
        const errProd = [];
        let minRisingRes = null;
        let minRisingAt = null;
        for (let i = 0; i < seriesHours.length; i++) {
            if (seriesHours[i] > 0) continue; // rising+peak only for method score
            const o = seriesObs[i];
            if (!Number.isFinite(o)) continue;
            if (Number.isFinite(seriesPred120[i])) {
                const e = seriesPred120[i] - o;
                errProd.push(e);
                if (minRisingRes == null || e < minRisingRes) {
                    minRisingRes = e;
                    minRisingAt = seriesHours[i];
                }
            }
            if (Number.isFinite(seriesLead120[i])) errLead.push(seriesLead120[i] - o);
            if (Number.isFinite(seriesBoost120[i])) errBoost.push(seriesBoost120[i] - o);
        }
        detailSeries.push({
            peakTime: ev.peakTime,
            label: `${ev.peakTime.slice(0, 10)} · ${ev.peakLevel.toFixed(2)} m`,
            split: ev.split,
            peakLevel: ev.peakLevel,
            riseM: ev.riseM,
            hours: seriesHours,
            hours360: seriesHours360,
            obs: seriesObs,
            obs360: seriesObs360,
            pred120: seriesPred120,
            pred360: seriesPred360,
            persist120: seriesPersist120,
            lead120: seriesLead120,
            boost120: seriesBoost120,
            minRisingResidual120: minRisingRes,
            minRisingResidualAtH: minRisingAt,
            methodsRising: {
                prod: { rmse: rmse(errProd), bias: mean(errProd), n: errProd.length },
                phaseLead: { rmse: rmse(errLead), bias: mean(errLead), n: errLead.length },
                ampBoost135: { rmse: rmse(errBoost), bias: mean(errBoost), n: errBoost.length }
            }
        });
    }
}

// Calibrate amplitude boost on train peak-issued errors @120′
const trainPeakErrs = peakIssue[4].filter((p) => p.split === 'train').map((p) => p.err);
const medPeakUnder = median(trainPeakErrs); // typically negative
const meanRiseDy = (() => {
    // approximate: median (pred-y0) on peak issues
    const dys = peakIssue[4].filter((p) => p.split === 'train').map((p) => p.pred - p.y0);
    return median(dys);
})();
const suggestedBoost =
    Number.isFinite(medPeakUnder) && Number.isFinite(meanRiseDy) && meanRiseDy > 0.05
        ? 1 - medPeakUnder / meanRiseDy
        : null;

function summarizePeaks(list) {
    const errs = list.map((p) => p.err);
    const robust = errs.filter((e) => Math.abs(e) <= 1.5);
    return {
        n: list.length,
        bias: mean(errs),
        mae: mean(errs.map(Math.abs)),
        rmse: rmse(errs),
        p10: quantile(errs, 0.1),
        p50: median(errs),
        p90: quantile(errs, 0.9),
        underFrac: errs.length ? errs.filter((e) => e < -0.05).length / errs.length : null,
        robust: {
            n: robust.length,
            bias: mean(robust),
            p50: median(robust),
            underFrac: robust.length ? robust.filter((e) => e < -0.05).length / robust.length : null
        }
    };
}

const methodRollup = { prod: [], phaseLead: [], ampBoost135: [] };
for (const d of detailSeries) {
    for (const k of Object.keys(methodRollup)) {
        const m = d.methodsRising[k];
        if (m?.n) methodRollup[k].push(m);
    }
}
function roll(arr, key) {
    const xs = arr.map((m) => m[key]).filter(Number.isFinite);
    return { mean: mean(xs), median: median(xs), n: xs.length };
}

const report = {
    kind: 'analisi_cento_picchi',
    version: 1,
    disclaimer: 'Sperimentale: non operativo, non allertistico.',
    generatedAt: new Date().toISOString(),
    station,
    coeffs: path.relative(root, coeffsPath),
    testFrom: eventsDoc.testFrom,
    nEvents: allEvents.length,
    nTest: allEvents.filter((e) => e.split === 'test').length,
    nDetail: detailSeries.length,
    horizonsMin: HORIZONS.map((h) => h * 30),
    strideAllMin,
    strideDetailMin,
    diagnosis: {
        headline:
            'In salita la curva prod resta sotto e in ritardo (spesso −30…−70 cm nei momenti più ripidi); in discesa RMSE ≈ 3–5× più basso. L’emissione esatta a peak−2h sbaglia poco il livello di cresta: ciò che “manca il picco” è soprattutto il fronte.',
        mechanisms: [
            {
                id: 'front_lag',
                title: 'Fronte in ritardo (effetto visivo di picco basso)',
                detail:
                    'Durante la salita il knot @120′ insegue l’osservato con ritardo tipico ~30′. Finché l’onda non è arrivata al sensore, la previsione emessa ore prima resta sotto la cresta futura; a peak−2h il livello locale ha già incorporato quasi tutto e l’errore al colmo diventa piccolo.'
            },
            {
                id: 'mse_shrinkage',
                title: 'Shrinkage della media condizionata (MSE)',
                detail:
                    'La MLP ottimizza E[Δy|x]. Con tempo di arrivo incerto, la media è un fronte più basso e più tardo. In discesa la traiettoria è più monotona e localmente osservabile → molto meno shrinkage.'
            },
            {
                id: 'horizon_amplitude',
                title: 'Sottostima vera di ampiezza agli orizzonti lunghi',
                detail:
                    'A 6 h sulle piene grandi (test 2026 ≥6 m) il bias al picco torna negativo (~−15 cm). Rising bias @360′ ≈ −11 cm su tutto il catalogo: oltre al timing c’è estrapolazione/diffusione del Δy.'
            },
            {
                id: 'asymmetry',
                title: 'Asimmetria rising/falling strutturale',
                detail:
                    'Rising/falling RMSE cresce con h (≈2.7× @2 h → ≈5.4× @6 h). Lo svuotamento è un rilascio locale; la salita dipende da propagazione, celerità variabile e afflussi non visti a monte.'
            }
        ],
        methodsTried: [
            { name: 'lag-align feature shift/augment', adopted: false, note: 'Non riduce phaseLag; shift peggiora.' },
            { name: 'event-weight ×4 / quantile 0.75', adopted: false, note: 'RMSE marginale; stesso lag.' },
            { name: 'bande q10/q90 DNN', adopted: 'bands-only', note: 'Utili per incertezza; non correggono il mediano al picco.' },
            { name: 'GRU sequenziale', adopted: false, note: 'highAcc migliore in candidati; holdout fit debole → reliable false.' },
            { name: 'Muskingum/RC/EnKF ibridi', adopted: false, note: 'Skill evento ≪ MLP.' },
            { name: 'MLP 6 h estesa', adopted: true, note: 'Orizzonti 30–360′ in prod.' }
        ],
        nextLevers: [
            {
                id: 'asymmetric_loss',
                title: 'Loss asimmetrica / pinball alto solo in salita',
                why: 'Attacca direttamente lo shrinkage MSE senza cambiare feature.',
                risk: 'Può sovrastimare magre e peggiorare falling RMSE.'
            },
            {
                id: 'peak_issued_head',
                title: 'Testa dedicata “picco entro h” (livello e tempo)',
                why: 'La metrica operativa sui picchi non è RMSE su tutti i knot.',
                risk: 'Serve catalogo eventi denso; overfitting su poche piene grandi.'
            },
            {
                id: 'online_phase_gated',
                title: 'Correzione di fase online solo con P(crossing) alto',
                why: 'H10: il decode lift scatta troppo spesso senza crossing reale.',
                risk: 'Gate troppo stretto → nessun beneficio; troppo largo → RMSE.'
            },
            {
                id: 'amp_boost_rising',
                title: 'Boost ampiezza Δy>0 calibrato su errori al picco (post-hoc)',
                why: 'Corregge il bias negativo residuo dopo il timing; banale da A/B in pagina.',
                risk: 'Fattore globale; eventi piccoli possono essere gonfiati.'
            },
            {
                id: 'discharge_routing',
                title: 'Routing in portata (rating curve + Muskingum nonlineare)',
                why: 'H1/H2: i filtri su livello non conservano massa; la salita è un problema di Q.',
                risk: 'Niente Q(h) ufficiale in repo; serve calibrazione da livelli.'
            }
        ],
        suggestedBoostFromTrainPeak:
            suggestedBoost != null ? Math.round(suggestedBoost * 100) / 100 : null,
        medTrainPeakErr120: medPeakUnder
    },
    aggregate: {
        rising: finalizeLimb(globalRising),
        falling: finalizeLimb(globalFalling),
        byPeakBin: Object.fromEntries(
            Object.entries(byPeakBin).map(([k, v]) => [k, finalizeLimb(v)])
        ),
        peakIssued: Object.fromEntries(
            HORIZONS.map((h) => [
                h,
                {
                    all: summarizePeaks(peakIssue[h]),
                    test: summarizePeaks(peakIssue[h].filter((p) => p.split === 'test')),
                    train: summarizePeaks(peakIssue[h].filter((p) => p.split === 'train'))
                }
            ])
        )
    },
    methodCompareDetail: {
        note: 'Su eventi dettaglio, solo limb rising (issue < peak), target @120′. phaseLead≈anticipo knot; ampBoost135=Δy+ ×1.35.',
        prod: { rmse: roll(methodRollup.prod, 'rmse'), bias: roll(methodRollup.prod, 'bias') },
        phaseLead: { rmse: roll(methodRollup.phaseLead, 'rmse'), bias: roll(methodRollup.phaseLead, 'bias') },
        ampBoost135: { rmse: roll(methodRollup.ampBoost135, 'rmse'), bias: roll(methodRollup.ampBoost135, 'bias') }
    },
    events: eventRows,
    details: detailSeries
};

fs.writeFileSync(outJson, JSON.stringify(report));
console.error(`[analisi] wrote ${outJson} (${(fs.statSync(outJson).size / 1e6).toFixed(2)} MB)`);
fs.writeFileSync(outHtml, buildHtml(report));
console.error(`[analisi] wrote ${outHtml}`);
console.error(
    `[analisi] rising@120 RMSE=${report.aggregate.rising[4]?.rmse?.toFixed(3)} ` +
        `falling@120=${report.aggregate.falling[4]?.rmse?.toFixed(3)} ` +
        `peakBias@120=${report.aggregate.peakIssued[4]?.all?.bias?.toFixed(3)} m`
);

function buildHtml(doc) {
    const r120 = doc.aggregate.rising[4] || {};
    const f120 = doc.aggregate.falling[4] || {};
    const r360 = doc.aggregate.rising[12] || {};
    const f360 = doc.aggregate.falling[12] || {};
    const peak360t = doc.aggregate.peakIssued[12]?.test || {};
    const mc = doc.methodCompareDetail;
    const detailMins = (doc.details || [])
        .map((d) => d.minRisingResidual120)
        .filter((v) => Number.isFinite(v) && v > -2);
    const medMinRising = detailMins.length
        ? detailMins.slice().sort((a, b) => a - b)[Math.floor(detailMins.length / 2)]
        : null;

    const mechHtml = doc.diagnosis.mechanisms
        .map((m) => `<li><strong>${esc(m.title)}</strong> — ${esc(m.detail)}</li>`)
        .join('');
    const triedHtml = doc.diagnosis.methodsTried
        .map(
            (m) =>
                `<tr><td>${esc(m.name)}</td><td>${esc(String(m.adopted))}</td><td>${esc(m.note)}</td></tr>`
        )
        .join('');
    const nextHtml = doc.diagnosis.nextLevers
        .map(
            (m) =>
                `<tr><td>${esc(m.title)}</td><td>${esc(m.why)}</td><td>${esc(m.risk)}</td></tr>`
        )
        .join('');

    const binRows = Object.entries(doc.aggregate.byPeakBin)
        .map(([k, v]) => {
            const row = v[4] || {};
            return `<tr><td>${esc(k)}</td><td>${row.n ?? 0}</td><td>${fmt(row.rmse)}</td><td>${fmt(row.bias)}</td><td>${fmt(row.skillPersist)}</td></tr>`;
        })
        .join('');

    const payload = JSON.stringify({
        events: doc.events.map((e) => ({
            peakTime: e.peakTime,
            peakLevel: e.peakLevel,
            riseM: e.riseM,
            split: e.split,
            detail: e.detail,
            risingRmse120: e.risingRmse120,
            fallingRmse120: e.fallingRmse120,
            risingBias120: e.risingBias120,
            fallingBias120: e.fallingBias120,
            risingMin120: e.risingMin120,
            risingMin360: e.risingMin360,
            peakErr120: e.peakErr120,
            peakErr360: e.peakErr360,
            phaseLag120: e.phaseLag120,
            ampBiasAligned120: e.ampBiasAligned120
        })),
        details: doc.details.map((d) => ({
            peakTime: d.peakTime,
            label: d.label,
            split: d.split,
            peakLevel: d.peakLevel,
            riseM: d.riseM,
            hours: d.hours,
            hours360: d.hours360,
            obs: d.obs,
            pred120: d.pred120,
            pred360: d.pred360,
            persist120: d.persist120,
            lead120: d.lead120,
            boost120: d.boost120,
            minRisingResidual120: d.minRisingResidual120,
            minRisingResidualAtH: d.minRisingResidualAtH,
            methodsRising: d.methodsRising
        }))
    });

    return `<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Analisi picchi Cento — predittore prod</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg:#f4f1ea; --ink:#1c1b19; --muted:#5c574f; --line:#d5cfc3; --card:#fff; --bad:#8b2e1a; --ok:#1f5c3a; --accent:#1d4e89; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.45 "Source Sans 3", system-ui, sans-serif; color:var(--ink); background:var(--bg); }
  header { padding:1.2rem 1.5rem; background:var(--card); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:3; }
  h1 { margin:0 0 .35rem; font-size:1.35rem; font-family: "IBM Plex Serif", Georgia, serif; }
  .meta { color:var(--muted); font-size:.9rem; max-width:70rem; }
  main { padding:1rem 1.5rem 3rem; max-width:1120px; margin:0 auto; }
  h2 { font-size:1.15rem; margin:1.6rem 0 .6rem; font-family:"IBM Plex Serif", Georgia, serif; }
  h3 { font-size:1rem; margin:1rem 0 .4rem; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:.75rem; margin:1rem 0; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:4px; padding:.75rem .9rem; }
  .card .k { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
  .card .v { font-size:1.35rem; font-variant-numeric:tabular-nums; margin-top:.15rem; }
  .card .v.bad { color:var(--bad); }
  .card .v.ok { color:var(--ok); }
  table { border-collapse:collapse; width:100%; background:var(--card); margin:.5rem 0 1rem; font-size:.9rem; }
  th,td { border:1px solid var(--line); padding:.4rem .55rem; text-align:left; }
  th { background:#ebe6dc; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; }
  ul.mech { margin:.4rem 0 1rem; padding-left:1.2rem; }
  ul.mech li { margin:.35rem 0; }
  .toolbar { display:flex; flex-wrap:wrap; gap:.75rem; align-items:center; margin:1rem 0; }
  .toolbar label { color:var(--muted); font-size:.9rem; }
  .event { background:var(--card); border:1px solid var(--line); border-radius:4px; padding:.75rem 1rem .25rem; margin:0 0 1rem; }
  .event h3 { margin:0 0 .2rem; }
  .sub { color:var(--muted); font-size:.85rem; margin-bottom:.4rem; }
  .plot { width:100%; height:360px; }
  .note { font-size:.85rem; color:var(--muted); }
  #scatter, #hist { width:100%; height:320px; background:var(--card); border:1px solid var(--line); border-radius:4px; margin:.5rem 0 1rem; }
</style>
</head>
<body>
<header>
  <h1>Cento — perché i picchi sono bassi e lo svuotamento no</h1>
  <div class="meta">${esc(doc.diagnosis.headline)} Replay causale del <code>window_delta_mlp</code> prod su ${doc.nEvents} piene del catalogo (${doc.nTest} test ≥ ${esc(String(doc.testFrom).slice(0, 10))}). ${esc(doc.disclaimer)} Generato ${esc(doc.generatedAt)}.</div>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="k">Rising RMSE @120′</div><div class="v bad">${fmt(r120.rmse)} m</div></div>
    <div class="card"><div class="k">Falling RMSE @120′</div><div class="v ok">${fmt(f120.rmse)} m</div></div>
    <div class="card"><div class="k">R/F @120′ → @360′</div><div class="v">${r120.rmse && f120.rmse ? (r120.rmse / f120.rmse).toFixed(1) : '—'}× → ${r360.rmse && f360.rmse ? (r360.rmse / f360.rmse).toFixed(1) : '—'}×</div></div>
    <div class="card"><div class="k">Bias rising @360′</div><div class="v bad">${fmt(r360.bias)} m</div></div>
    <div class="card"><div class="k">Min residuo salita (mediana dettaglio)</div><div class="v bad">${fmt(medMinRising)} m</div></div>
    <div class="card"><div class="k">Bias picco test @360′</div><div class="v bad">${fmt(peak360t.bias)} m</div></div>
  </div>

  <h2>Diagnosi</h2>
  <ul class="mech">${mechHtml}</ul>

  <h2>Errore vs ampiezza piena (tutti gli issue)</h2>
  <table>
    <thead><tr><th>Classe picco Cento</th><th>n @120′</th><th>RMSE</th><th>Bias (pred−obs)</th><th>Skill vs persist</th></tr></thead>
    <tbody>${binRows}</tbody>
  </table>
  <p class="note">Bias negativo = sottostima. Le piene ≥6 m concentrano il problema di ampiezza.</p>

  <h2>Distribuzione errori al picco (emissione a peak−h)</h2>
  <div id="hist"></div>
  <div id="scatter"></div>

  <h2>Cosa è già stato provato</h2>
  <table>
    <thead><tr><th>Metodo</th><th>Adopt</th><th>Esito</th></tr></thead>
    <tbody>${triedHtml}</tbody>
  </table>

  <h2>Correzioni post-hoc sul replay (eventi dettaglio)</h2>
  <table>
    <thead><tr><th>Metodo</th><th>RMSE rising medio</th><th>Bias rising medio</th></tr></thead>
    <tbody>
      <tr><td>prod</td><td class="num">${fmt(mc.prod.rmse.mean)}</td><td class="num">${fmt(mc.prod.bias.mean)}</td></tr>
      <tr><td>phaseLead (anticipo knot)</td><td class="num">${fmt(mc.phaseLead.rmse.mean)}</td><td class="num">${fmt(mc.phaseLead.bias.mean)}</td></tr>
      <tr><td>ampBoost ×1.35 su Δy&gt;0</td><td class="num">${fmt(mc.ampBoost135.rmse.mean)}</td><td class="num">${fmt(mc.ampBoost135.bias.mean)}</td></tr>
    </tbody>
  </table>
  <p class="note">Boost suggerito da mediana errore picco train @120′: <strong>${doc.diagnosis.suggestedBoostFromTrainPeak ?? 'n/d'}</strong> (med peak err ${fmt(doc.diagnosis.medTrainPeakErr120)} m). Non adottare in prod senza protocollo A/B.</p>

  <h2>Leve più promettenti</h2>
  <table>
    <thead><tr><th>Leva</th><th>Perché</th><th>Rischio</th></tr></thead>
    <tbody>${nextHtml}</tbody>
  </table>

  <h2>Piene — tabella e idrogrammi</h2>
  <div class="toolbar">
    <label>Filtro <select id="split"><option value="detail">dettaglio (test + top picchi)</option><option value="test">solo test</option><option value="all">tutti in tabella</option></select></label>
    <label>Ordina <select id="sort"><option value="peak">picco</option><option value="peakErr">errore picco @120</option><option value="rising">rising RMSE</option><option value="date">data</option></select></label>
    <span class="meta" id="count"></span>
  </div>
  <div id="table-wrap"></div>
  <div id="events"></div>
  <p class="note">Curve dettaglio: osservato al target +2 h, previsione prod @120′/@360′, persistenza, phaseLead e ampBoost. Asse X = ore dal picco Cento. Stride dettaglio ${doc.strideDetailMin}′, aggregati ${doc.strideAllMin}′.</p>
</main>
<script>
const DATA = ${payload};
function fmt(v,d=3){return v==null||!Number.isFinite(v)?'—':Number(v).toFixed(d);}
function renderOverview(){
  const errs = DATA.events.map(e=>e.peakErr120).filter(Number.isFinite);
  Plotly.newPlot('hist',[{
    x:errs, type:'histogram', nbinsx:30, marker:{color:'#1d4e89'}, name:'err picco @120′'
  }],{
    margin:{t:28,r:16,b:40,l:48},
    title:'Errore livello al picco (pred−obs) @120′, tutti gli eventi',
    xaxis:{title:'pred − obs (m)'}, yaxis:{title:'Conteggio'}
  },{responsive:true,displayModeBar:false});
  Plotly.newPlot('scatter',[{
    x:DATA.events.map(e=>e.peakLevel),
    y:DATA.events.map(e=>e.peakErr120),
    mode:'markers',
    marker:{size:7, color:DATA.events.map(e=>e.split==='test'?'#8b2e1a':'#1d4e89'), opacity:0.75},
    text:DATA.events.map(e=>e.peakTime.slice(0,10)+' '+e.split),
    name:'eventi'
  },{
    x:[0,10], y:[0,0], mode:'lines', line:{dash:'dot', color:'#999', width:1}, hoverinfo:'skip', showlegend:false
  }],{
    margin:{t:28,r:16,b:40,l:48},
    title:'Sottostima al picco vs livello di picco',
    xaxis:{title:'Picco osservato Cento (m)'}, yaxis:{title:'pred − obs @120′ (m)'}
  },{responsive:true,displayModeBar:false});
}
function render(){
  const split=document.getElementById('split').value;
  const sort=document.getElementById('sort').value;
  let rows=DATA.events.slice();
  if(split==='test') rows=rows.filter(e=>e.split==='test');
  if(split==='detail') rows=rows.filter(e=>e.detail);
  rows.sort((a,b)=>{
    if(sort==='peak') return b.peakLevel-a.peakLevel;
    if(sort==='peakErr') return (a.peakErr120??0)-(b.peakErr120??0);
    if(sort==='rising') return (b.risingRmse120??0)-(a.risingRmse120??0);
    return Date.parse(a.peakTime)-Date.parse(b.peakTime);
  });
  document.getElementById('count').textContent=rows.length+' eventi';
  let html='<table><thead><tr><th>Picco</th><th>Split</th><th class="num">Livello</th><th class="num">Rise</th><th class="num">Rising RMSE</th><th class="num">Falling RMSE</th><th class="num">Min residuo salita</th><th class="num">Err picco@120</th><th class="num">Err picco@360</th><th class="num">Lag fase′</th></tr></thead><tbody>';
  for(const e of rows){
    html+='<tr>'+
      '<td>'+e.peakTime.slice(0,16).replace('T',' ')+'</td>'+
      '<td>'+e.split+'</td>'+
      '<td class="num">'+fmt(e.peakLevel,2)+'</td>'+
      '<td class="num">'+fmt(e.riseM,2)+'</td>'+
      '<td class="num">'+fmt(e.risingRmse120)+'</td>'+
      '<td class="num">'+fmt(e.fallingRmse120)+'</td>'+
      '<td class="num">'+fmt(e.risingMin120)+'</td>'+
      '<td class="num">'+fmt(e.peakErr120)+'</td>'+
      '<td class="num">'+fmt(e.peakErr360)+'</td>'+
      '<td class="num">'+fmt(e.phaseLag120,0)+'</td></tr>';
  }
  html+='</tbody></table>';
  document.getElementById('table-wrap').innerHTML=html;

  const root=document.getElementById('events');
  root.innerHTML='';
  const details=DATA.details.filter(d=>rows.some(r=>r.peakTime===d.peakTime));
  for(const e of details){
    const wrap=document.createElement('section');
    wrap.className='event';
    const m=e.methodsRising||{};
    wrap.innerHTML='<h3>'+e.label+(e.split==='test'?' [test]':'')+'</h3>'+
      '<div class="sub">Rise '+fmt(e.riseM,2)+' m · min residuo salita '+fmt(e.minRisingResidual120)+
      ' m @ '+fmt(e.minRisingResidualAtH,1)+' h dal picco · RMSE rising prod '+fmt(m.prod&&m.prod.rmse)+
      ' · phaseLead '+fmt(m.phaseLead&&m.phaseLead.rmse)+
      ' · boost '+fmt(m.ampBoost135&&m.ampBoost135.rmse)+'</div>'+
      '<div class="plot" id="p'+e.peakTime+'"></div>';
    root.appendChild(wrap);
    const traces=[
      {x:e.hours,y:e.obs,name:'Osservato',mode:'lines',line:{width:2.4,color:'#1c1b19'}},
      {x:e.hours,y:e.pred120,name:'Prod @120′',mode:'lines',line:{width:2,color:'#1d4e89'}},
      {x:e.hours360||e.hours,y:e.pred360,name:'Prod @360′',mode:'lines',line:{width:1.2,color:'#5b8ab5',dash:'dot'}},
      {x:e.hours,y:e.lead120,name:'PhaseLead @120′',mode:'lines',line:{width:1.5,color:'#b35c00'}},
      {x:e.hours,y:e.boost120,name:'AmpBoost×1.35 @120′',mode:'lines',line:{width:1.5,color:'#0b6e4f'}},
      {x:e.hours,y:e.persist120,name:'Persistenza',mode:'lines',line:{width:1,color:'#999',dash:'dash'}}
    ];
    Plotly.newPlot('p'+e.peakTime,traces,{
      margin:{t:20,r:16,b:40,l:48},
      xaxis:{title:'Ore da picco Cento',zeroline:true},
      yaxis:{title:'Livello (m)'},
      legend:{orientation:'h',y:1.15},
      shapes:[{type:'line',x0:0,x1:0,y0:0,y1:1,yref:'paper',line:{color:'#999',width:1,dash:'dot'}}]
    },{responsive:true,displayModeBar:false});
  }
}
document.getElementById('split').addEventListener('change',render);
document.getElementById('sort').addEventListener('change',render);
renderOverview();
render();
</script>
</body>
</html>`;
}

function fmt(v, d = 3) {
    return v == null || !Number.isFinite(v) ? '—' : Number(v).toFixed(d);
}
