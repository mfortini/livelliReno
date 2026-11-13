'use strict';
/**
 * Fase 0 — shared evaluation protocol helpers.
 * Event catalog detection, poly2(6) baseline, skill/bootstrap metrics.
 * Used by tools/build-event-catalog.js and tools/backtest.js --protocol.
 */

const STEP_MS = 30 * 60000;
const POLY2_POINTS = 6; // 2.5 h lookback, aligned with fit-ml-cento POLY_NAIVE_POINTS

function mulberry32(seed) {
    let t = seed >>> 0;
    return function next() {
        t += 0x6D2B79F5;
        let r = Math.imul(t ^ (t >>> 15), 1 | t);
        r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
}

/**
 * Detect flood events: local peaks (±peakHalfWin steps) with rise ≥ minRiseM
 * vs min in the previous lookbackHours. Merge peaks closer than mergeHours.
 */
function detectFloodEvents(series, options = {}) {
    const minRiseM = options.minRiseM ?? 0.5;
    const lookbackHours = options.lookbackHours ?? 48;
    const peakHalfWinHours = options.peakHalfWinHours ?? 4;
    const mergeHours = options.mergeHours ?? 72;
    const stepMs = options.stepMs ?? STEP_MS;

    const lookbackSteps = Math.round((lookbackHours * 3600000) / stepMs);
    const halfWin = Math.round((peakHalfWinHours * 3600000) / stepMs);
    const mergeMs = mergeHours * 3600000;

    const { times, values, byTime } = series;
    if (!times?.length) return [];

    // Ignore peaks in the quiet bulk of the series (use p85 unless overridden).
    const finiteVals = values.filter(Number.isFinite).slice().sort((a, b) => a - b);
    const p85 = finiteVals.length
        ? finiteVals[Math.min(finiteVals.length - 1, Math.floor(finiteVals.length * 0.85))]
        : -Infinity;
    const minPeakLevel = options.minPeakLevel ?? p85;

    const candidates = [];
    for (let i = 0; i < times.length; i++) {
        const t = times[i];
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        if (v < minPeakLevel) continue;

        let isPeak = true;
        for (let k = -halfWin; k <= halfWin; k++) {
            if (k === 0) continue;
            const vv = byTime.get(t + k * stepMs);
            if (Number.isFinite(vv) && vv > v) {
                isPeak = false;
                break;
            }
        }
        if (!isPeak) continue;

        let minPrev = v;
        for (let k = 1; k <= lookbackSteps; k++) {
            const vv = byTime.get(t - k * stepMs);
            if (Number.isFinite(vv) && vv < minPrev) minPrev = vv;
        }
        const riseM = v - minPrev;
        if (riseM < minRiseM) continue;

        // Start: last time level was within 10% of rise above minPrev before peak
        const startThr = minPrev + 0.1 * riseM;
        let start = t - lookbackSteps * stepMs;
        for (let k = lookbackSteps; k >= 1; k--) {
            const tt = t - k * stepMs;
            const vv = byTime.get(tt);
            if (Number.isFinite(vv) && vv <= startThr) {
                start = tt;
                break;
            }
        }
        // End: first return below startThr after peak, or +lookback
        let end = t + lookbackSteps * stepMs;
        for (let k = 1; k <= lookbackSteps; k++) {
            const tt = t + k * stepMs;
            const vv = byTime.get(tt);
            if (Number.isFinite(vv) && vv <= startThr) {
                end = tt;
                break;
            }
        }

        candidates.push({
            start,
            end,
            peakTime: t,
            peakLevel: v,
            riseM,
            minPrev
        });
    }

    // Merge peaks within mergeHours — keep highest peak
    candidates.sort((a, b) => a.peakTime - b.peakTime);
    const merged = [];
    for (const ev of candidates) {
        const last = merged[merged.length - 1];
        if (last && ev.peakTime - last.peakTime < mergeMs) {
            if (ev.peakLevel > last.peakLevel) {
                last.peakTime = ev.peakTime;
                last.peakLevel = ev.peakLevel;
                last.riseM = ev.riseM;
                last.minPrev = ev.minPrev;
            }
            last.start = Math.min(last.start, ev.start);
            last.end = Math.max(last.end, ev.end);
        } else {
            merged.push({ ...ev });
        }
    }
    return merged;
}

function iso(ms) {
    return new Date(ms).toISOString();
}

function buildCatalogPayload(stationEvents, meta = {}) {
    const events = [];
    for (const [station, list] of Object.entries(stationEvents)) {
        for (const ev of list) {
            events.push({
                station,
                start: iso(ev.start),
                end: iso(ev.end),
                peakTime: iso(ev.peakTime),
                peakLevel: Math.round(ev.peakLevel * 1000) / 1000,
                riseM: Math.round(ev.riseM * 1000) / 1000
            });
        }
    }
    events.sort((a, b) => a.peakTime.localeCompare(b.peakTime) || a.station.localeCompare(b.station));
    return {
        version: 1,
        kind: 'flood_event_catalog',
        disclaimer: 'Sperimentale: catalogo automatico per valutazione; non allertistico.',
        detection: {
            minRiseM: meta.minRiseM ?? 0.5,
            minPeakRule: meta.minPeakRule ?? 'station_p85',
            lookbackHours: meta.lookbackHours ?? 48,
            peakHalfWinHours: meta.peakHalfWinHours ?? 4,
            mergeHours: meta.mergeHours ?? 72
        },
        storico: meta.storico || null,
        testFrom: meta.testFrom || null,
        stations: meta.stationSummary || {},
        events
    };
}

function eventsForStation(catalog, station) {
    return (catalog.events || []).filter(e => e.station === station);
}

function inEventWindow(catalogEvents, nowMs) {
    for (const ev of catalogEvents) {
        const a = Date.parse(ev.start);
        const b = Date.parse(ev.end);
        if (Number.isFinite(a) && Number.isFinite(b) && nowMs >= a && nowMs <= b) return ev;
    }
    return null;
}

/** Quadratic OLS on last `points` levels (default 6); extrapolate per horizon step. */
function poly2At(seriesByTime, anchorTime, horizonSteps, points = POLY2_POINTS) {
    const vals = [];
    for (let k = points - 1; k >= 0; k--) {
        const v = seriesByTime.get(anchorTime - k * STEP_MS);
        vals.push(v);
    }
    if (!vals.every(Number.isFinite) || vals.length < 3) return null;
    const n = vals.length;
    const x = vals.map((_, i) => i);
    let sx = 0, sx2 = 0, sx3 = 0, sx4 = 0, sy = 0, sxy = 0, sx2y = 0;
    for (let i = 0; i < n; i++) {
        const xi = x[i], yi = vals[i], xi2 = xi * xi;
        sx += xi; sx2 += xi2; sx3 += xi2 * xi; sx4 += xi2 * xi2;
        sy += yi; sxy += xi * yi; sx2y += xi2 * yi;
    }
    const A = [
        [n, sx, sx2],
        [sx, sx2, sx3],
        [sx2, sx3, sx4]
    ];
    const b = [sy, sxy, sx2y];
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let col = 0; col < 3; col++) {
        let piv = col;
        for (let r = col + 1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
        [M[col], M[piv]] = [M[piv], M[col]];
        const div = M[col][col];
        if (!Number.isFinite(div) || Math.abs(div) < 1e-12) return null;
        for (let c = col; c < 4; c++) M[col][c] /= div;
        for (let r = 0; r < 3; r++) {
            if (r === col) continue;
            const f = M[r][col];
            for (let c = col; c < 4; c++) M[r][c] -= f * M[col][c];
        }
    }
    const [a0, a1, a2] = [M[0][3], M[1][3], M[2][3]];
    const xi = (n - 1) + horizonSteps;
    return a0 + a1 * xi + a2 * xi * xi;
}

function skillFromRmse(rmseModel, rmseBase) {
    if (!Number.isFinite(rmseModel) || !Number.isFinite(rmseBase) || rmseBase === 0) return null;
    return 1 - rmseModel / rmseBase;
}

function rmseOf(se, n) {
    if (!n) return null;
    return Math.sqrt(se / n);
}

/**
 * Bootstrap 90% CI of skill = 1 - rmse_model/rmse_base over event-level aggregates.
 * Each event contributes { seModel, seBase, n }.
 * Returns { skill, lo, hi, nEvents } or nulls.
 */
function bootstrapSkillCI(eventStats, options = {}) {
    const B = options.draws ?? 1000;
    const seed = options.seed ?? 1;
    const alpha = options.alpha ?? 0.10; // 90% CI
    const usable = (eventStats || []).filter(e => e && e.n > 0);
    if (!usable.length) return { skill: null, lo: null, hi: null, nEvents: 0 };

    const pooled = (list) => {
        let seM = 0, seB = 0, n = 0;
        for (const e of list) {
            seM += e.seModel;
            seB += e.seBase;
            n += e.n;
        }
        return skillFromRmse(rmseOf(seM, n), rmseOf(seB, n));
    };

    const point = pooled(usable);
    if (usable.length === 1) {
        return { skill: point, lo: point, hi: point, nEvents: 1 };
    }

    const rand = mulberry32(seed);
    const skills = [];
    for (let b = 0; b < B; b++) {
        const sample = [];
        for (let i = 0; i < usable.length; i++) {
            sample.push(usable[Math.floor(rand() * usable.length)]);
        }
        const s = pooled(sample);
        if (Number.isFinite(s)) skills.push(s);
    }
    skills.sort((a, b) => a - b);
    if (!skills.length) return { skill: point, lo: null, hi: null, nEvents: usable.length };
    const loIdx = Math.max(0, Math.floor((alpha / 2) * skills.length));
    const hiIdx = Math.min(skills.length - 1, Math.ceil((1 - alpha / 2) * skills.length) - 1);
    return {
        skill: point,
        lo: skills[loIdx],
        hi: skills[hiIdx],
        nEvents: usable.length
    };
}

function median(arr) {
    const a = arr.filter(Number.isFinite).slice().sort((x, y) => x - y);
    if (!a.length) return null;
    const m = Math.floor(a.length / 2);
    return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

/**
 * Peak error for one event: compare forecast peak in [peak±searchHours] issued at
 * (peakTime - horizon) vs observed peak. Returns { levelErr, timeErrMin } or null.
 */
function peakErrorsForIssue(obsByTime, forecastByTime, peakTime, horizonSteps, searchHours = 6) {
    const issueAt = peakTime - horizonSteps * STEP_MS;
    const searchSteps = Math.round((searchHours * 3600000) / STEP_MS);
    let obsPeak = null, obsPeakT = null;
    let fcPeak = null, fcPeakT = null;
    for (let k = -searchSteps; k <= searchSteps; k++) {
        const t = peakTime + k * STEP_MS;
        const ov = obsByTime.get(t);
        if (Number.isFinite(ov) && (obsPeak === null || ov > obsPeak)) {
            obsPeak = ov;
            obsPeakT = t;
        }
        // forecast values keyed by absolute time in the forecast map for this issue
        const fv = forecastByTime.get(t);
        if (Number.isFinite(fv) && (fcPeak === null || fv > fcPeak)) {
            fcPeak = fv;
            fcPeakT = t;
        }
    }
    if (obsPeak === null || fcPeak === null) return null;
    return {
        issueAt,
        levelErr: fcPeak - obsPeak,
        timeErrMin: (fcPeakT - obsPeakT) / 60000
    };
}

function formatEvalMarkdown(table) {
    const lines = [];
    lines.push('# Tabella unica di valutazione (Fase 0)');
    lines.push('');
    lines.push(`**Protocollo:** campionamento 30′, orizzonti 30–360′, eventi da catalogo, baseline persist / poly2(6) / ridge.`);
    lines.push(`**Test congelato:** ${table.from} → ${table.to}`);
    lines.push(`**Seed bootstrap:** ${table.seed}`);
    lines.push('');
    lines.push('> Caveat: i pesi attuali sono stati fitted con mesi alterni che si sovrappongono al test; i numeri assoluti possono essere ottimistici. Usare per ranking omogeneo; retrain con split pulito = Fase 1.');
    lines.push('');

    for (const [station, models] of Object.entries(table.stations || {})) {
        lines.push(`## ${station}`);
        lines.push('');
        lines.push('| Modello | h (′) | skill all /pers | skill evt /pers | CI90 evt | skill evt /poly2 | skill evt /ridge | peak err med (m) | peak time med (′) | n evt |');
        lines.push('|---|---:|---:|---:|---|---:|---:|---:|---:|---:|');
        for (const [model, horizons] of Object.entries(models)) {
            for (const [hMin, m] of Object.entries(horizons)) {
                const ci = m.skillEventPersistCI;
                const ciStr = ci && Number.isFinite(ci.lo)
                    ? `[${fmt(ci.lo)}, ${fmt(ci.hi)}]`
                    : 'n/d';
                lines.push(
                    `| ${model} | ${hMin} | ${fmt(m.skillAllPersist)} | ${fmt(m.skillEventPersist)} | ${ciStr} | ${fmt(m.skillEventPoly2)} | ${fmt(m.skillEventRidge)} | ${fmt(m.peakLevelErrMed)} | ${fmt(m.peakTimeErrMed, 0)} | ${m.nEvents ?? 'n/d'} |`
                );
            }
        }
        lines.push('');
    }
    lines.push('---');
    lines.push(`Generato: ${table.generatedAt || new Date().toISOString()}`);
    return lines.join('\n') + '\n';
}

function fmt(v, digits = 2) {
    return Number.isFinite(v) ? v.toFixed(digits) : 'n/d';
}

module.exports = {
    STEP_MS,
    POLY2_POINTS,
    mulberry32,
    detectFloodEvents,
    buildCatalogPayload,
    eventsForStation,
    inEventWindow,
    poly2At,
    skillFromRmse,
    rmseOf,
    bootstrapSkillCI,
    median,
    peakErrorsForIssue,
    formatEvalMarkdown
};
