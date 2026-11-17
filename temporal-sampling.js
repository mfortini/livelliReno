const finite = value => typeof value === 'number' && Number.isFinite(value);
const finiteMs = value => Number.isFinite(Number(value));
const MINUTE = 60000;

function temporalContract(recipe) {
    const inputStep = Number(recipe?.inputStepMinutes || recipe?.stepMinutes || 30);
    const lookbackMin = Number(recipe?.lookbackMinutes || inputStep * (recipe?.lookbackSteps || 0));
    return {
        inputStepMinutes: finite(inputStep) && inputStep > 0 ? inputStep : 30,
        lookbackMinutes: finite(lookbackMin) && lookbackMin > 0 ? lookbackMin : 1440,
        derivWindowMinutes: Number(recipe?.derivWindowMinutes || 150),
        maxGapMinutes: Number(recipe?.maxGapMinutes || 60),
        method: recipe?.inputInterpolation || (recipe?.inputStepMinutes === 5 ? 'linear' : 'raw')
    };
}

function usablePoint(point, cutoffMs) {
    if (!point || !finite(point.value)) return null;
    const t = Number(point.time);
    if (!finiteMs(t) || t > cutoffMs) return null;
    return { time: t, value: point.value };
}

function cleanedSeries(series, cutoffMs) {
    const points = [];
    for (const point of series || []) {
        const p = usablePoint(point, cutoffMs);
        if (p) points.push(p);
    }
    points.sort((a, b) => a.time - b.time || a.value - b.value);
    return points;
}

function interpolateLinear(points, targetMs, maxGapMinutes = Infinity) {
    if (!points.length || targetMs < points[0].time || targetMs > points.at(-1).time) return null;
    let lo = 0, hi = points.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (points[mid].time < targetMs) lo = mid + 1;
        else hi = mid;
    }
    const right = points[lo];
    const left = points[lo - 1];
    if (!left) return right.value;
    if (!right || right.time === targetMs) return right ? right.value : null;
    if ((right.time - left.time) > maxGapMinutes * MINUTE) return null;
    const w = (targetMs - left.time) / (right.time - left.time);
    return left.value * (1 - w) + right.value * w;
}

function pchipSlopes(points) {
    const n = points.length;
    const h = [], d = [], m = new Array(n).fill(0);
    for (let i = 0; i < n - 1; i++) {
        h[i] = points[i + 1].time - points[i].time;
        d[i] = (points[i + 1].value - points[i].value) / h[i];
    }
    if (n === 2) { m[0] = d[0]; m[1] = d[0]; return m; }
    m[0] = ((2 * h[0] + h[1]) * d[0] - h[0] * d[1]) / (h[0] + h[1]);
    if (m[0] * d[0] <= 0) m[0] = 0;
    else if (d[0] * d[1] < 0 && Math.abs(m[0]) > Math.abs(3 * d[0])) m[0] = 3 * d[0];
    m[n - 1] = ((2 * h[n - 2] + h[n - 3]) * d[n - 2] - h[n - 2] * d[n - 3]) / (h[n - 2] + h[n - 3]);
    if (m[n - 1] * d[n - 2] <= 0) m[n - 1] = 0;
    else if (d[n - 2] * d[n - 3] < 0 && Math.abs(m[n - 1]) > Math.abs(3 * d[n - 2])) m[n - 1] = 3 * d[n - 2];
    for (let i = 1; i < n - 1; i++) {
        if (d[i - 1] * d[i] <= 0) { m[i] = 0; continue; }
        const w1 = 2 * h[i] + h[i - 1];
        const w2 = h[i] + 2 * h[i - 1];
        m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]);
    }
    return m;
}

function interpolatePchip(points, targetMs, maxGapMinutes = Infinity, slopes = null) {
    if (!points.length || targetMs < points[0].time || targetMs > points.at(-1).time) return null;
    if (points.length === 1) return points[0].value;
    let lo = 0, hi = points.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (points[mid].time < targetMs) lo = mid + 1;
        else hi = mid;
    }
    const j = points[lo].time === targetMs ? lo : lo - 1;
    const i = Math.max(0, j);
    const p0 = points[i], p1 = points[Math.min(i + 1, points.length - 1)];
    if (p0.time === targetMs) return p0.value;
    if (p1.time === targetMs) return p1.value;
    if (p0.time === p1.time) return p0.value;
    const h = p1.time - p0.time;
    if (h > maxGapMinutes * MINUTE) return null;
    const t = (targetMs - p0.time) / h;
    const m = slopes || pchipSlopes(points);
    const m0 = m[i], m1 = m[Math.min(i + 1, m.length - 1)];
    const t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * p0.value
        + (t3 - 2 * t2 + t) * h * m0
        + (-2 * t3 + 3 * t2) * p1.value
        + (t3 - t2) * h * m1;
}

function createTemporalView(series, cutoffMs, options = {}) {
    const maxGap = options.maxGapMinutes ?? 60;
    const method = options.method || 'raw';
    const points = cleanedSeries(series, cutoffMs, maxGap);
    const slopes = method === 'pchip' ? pchipSlopes(points) : null;
    return {
        points,
        sample(targetMs) {
            if (!points.length) return null;
            if (method === 'raw' || method === 'linear') return interpolateLinear(points, targetMs, maxGap);
            if (method === 'pchip') return interpolatePchip(points, targetMs, maxGap, slopes);
            return interpolateLinear(points, targetMs, maxGap);
        }
    };
}

function sampleTemporal(series, targetMs, options = {}) {
    const cutoff = options.cutoffMs == null ? targetMs : Number(options.cutoffMs);
    return createTemporalView(series, cutoff, options).sample(targetMs);
}

function temporalSeriesPoints(series, endMs, contract, options = {}) {
    const method = contract.method === 'pchip' ? 'pchip' : (contract.method === 'linear' ? 'linear' : 'raw');
    const maxGap = options.maxGapMinutes ?? contract.maxGapMinutes;
    const cutoff = options.cutoffMs == null ? endMs : Number(options.cutoffMs);
    const points = cleanedSeries(series, cutoff, maxGap);
    const last = points.at(-1);
    if (!last || last.time > cutoff) return [];
    const startMs = endMs - contract.lookbackMinutes * MINUTE;
    const out = [];
    for (let t = startMs; t <= endMs; t += contract.inputStepMinutes * MINUTE) {
        const value = method === 'raw'
            ? interpolateLinear(points, t, maxGap)
            : sampleTemporal(series, t, { cutoffMs: cutoff, maxGapMinutes: maxGap, method });
        out.push({ time: new Date(t), value });
    }
    return out;
}

function temporalContext(series, nowMs, recipe = {}, options = {}) {
    const contract = temporalContract(recipe);
    if (contract.inputStepMinutes >= 30 && contract.method === 'raw') return null;
    return {
        contract,
        sample: (targetMs) => sampleTemporal(series, targetMs, {
            cutoffMs: options.cutoffMs == null ? nowMs : options.cutoffMs,
            maxGapMinutes: options.maxGapMinutes ?? contract.maxGapMinutes,
            method: contract.method
        })
    };
}

function temporalSampleAt(series, targetMs, recipe, options = {}) {
    const ctx = temporalContext(series, options.nowMs ?? targetMs, recipe, options);
    return ctx ? ctx.sample(targetMs) : null;
}

// Causal spectral features, mirrored 1:1 by tools/spectral_features.py.
// Input: newest-first window of W samples (NaN allowed). Output order:
// cov, fft1..J, centroid, dombin, bandlo, bandmid, bandhi, wav{1..K}rms,
// wav{1..K}last  -> 27 floats for J=12, K=5.
function spectralFillNaN(values) {
    const n = values.length;
    const finiteIdx = [];
    const finiteVal = [];
    for (let i = 0; i < n; i++) {
        const v = Number(values[i]);
        if (Number.isFinite(v)) { finiteIdx.push(i); finiteVal.push(v); }
    }
    if (finiteVal.length < 2) return null;
    if (finiteVal.length === n) return values.slice();
    const out = new Array(n);
    const lastIdx = finiteIdx.length - 1;
    let p = 0;
    for (let i = 0; i < n; i++) {
        if (i <= finiteIdx[0]) { out[i] = finiteVal[0]; continue; }
        if (i >= finiteIdx[lastIdx]) { out[i] = finiteVal[lastIdx]; continue; }
        while (p + 1 <= lastIdx && finiteIdx[p + 1] < i) p++;
        const i0 = finiteIdx[p], i1 = finiteIdx[p + 1];
        out[i] = finiteVal[p] + ((i - i0) / (i1 - i0)) * (finiteVal[p + 1] - finiteVal[p]);
    }
    return out;
}

function spectralFeatureNames(stationTag, fftBins = 12, waveletScales = 5) {
    const names = [`spec_${stationTag}_cov`];
    for (let j = 1; j <= fftBins; j++) names.push(`spec_${stationTag}_fft${j}`);
    names.push(`spec_${stationTag}_centroid`, `spec_${stationTag}_dombin`,
        `spec_${stationTag}_bandlo`, `spec_${stationTag}_bandmid`, `spec_${stationTag}_bandhi`);
    for (let k = 1; k <= waveletScales; k++) names.push(`spec_${stationTag}_wav${k}rms`);
    for (let k = 1; k <= waveletScales; k++) names.push(`spec_${stationTag}_wav${k}last`);
    return names;
}

function spectralFeatures(valuesNewestFirst, fftBins = 12, waveletScales = 5) {
    const nFeat = 1 + fftBins + 5 + 2 * waveletScales;
    const n = valuesNewestFirst.length;
    const zeros = new Array(nFeat).fill(0);
    if (n < 4) {
        if (n > 0) {
            let c = 0;
            for (const v of valuesNewestFirst) if (Number.isFinite(Number(v))) c++;
            zeros[0] = c / n;
        }
        return zeros;
    }
    let covCount = 0;
    for (const v of valuesNewestFirst) if (Number.isFinite(Number(v))) covCount++;
    const cov = covCount / n;
    // chronological: reverse newest-first input
    const chron = valuesNewestFirst.slice().reverse().map((v) => Number(v));
    const filled = spectralFillNaN(chron);
    if (!filled) { zeros[0] = cov; return zeros; }
    let mean = 0;
    for (const v of filled) mean += v;
    mean /= n;
    const x = filled.map((v) => v - mean);
    const hann = new Array(n);
    for (let i = 0; i < n; i++) hann[i] = n <= 1 ? 1 : 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
    const xw = x.map((v, i) => v * hann[i]);
    const mags = [];
    for (let j = 1; j <= fftBins; j++) {
        let re = 0, im = 0;
        for (let i = 0; i < n; i++) {
            const ang = 2 * Math.PI * j * i / n;
            re += xw[i] * Math.cos(ang);
            im -= xw[i] * Math.sin(ang);
        }
        mags.push(Math.hypot(re / n, im / n));
    }
    const powers = mags.map((m) => m * m);
    let total = 0;
    for (const p of powers) total += p;
    const out = [cov];
    for (const m of mags) out.push(Math.log1p(m));
    const col = 1 + fftBins;
    if (!(total > 0) || !Number.isFinite(total)) {
        for (let k = 0; k < 5; k++) out.push(0);
    } else {
        let centroid = 0;
        for (let j = 0; j < fftBins; j++) centroid += (j + 1) * powers[j];
        centroid /= total;
        let dombin = 1, best = -1;
        for (let j = 0; j < fftBins; j++) if (powers[j] > best) { best = powers[j]; dombin = j + 1; }
        const bandSum = (a, b) => { let s = 0; for (let j = a; j <= b; j++) s += powers[j]; return s / total; };
        out.push(centroid, dombin, bandSum(0, 2), bandSum(3, 7), bandSum(8, fftBins - 1));
    }
    // Haar à trous on detrended, unwindowed series; rms + last detail per scale
    let a = x.slice();
    const rmsOut = [], lastOut = [];
    for (let k = 0; k < waveletScales; k++) {
        const s = 1 << k;
        const aNext = a.slice();
        for (let i = s; i < n; i++) aNext[i] = 0.5 * (a[i] + a[i - s]);
        let sumSq = 0, lastD = 0;
        for (let i = 0; i < n; i++) {
            const d = a[i] - aNext[i];
            sumSq += d * d;
            lastD = d;
        }
        rmsOut.push(Math.sqrt(sumSq / n));
        lastOut.push(lastD);
        a = aNext;
    }
    for (const v of rmsOut) out.push(v);
    for (const v of lastOut) out.push(v);
    return out.map((v) => (Number.isFinite(v) ? v : 0));
}

const api = {
    temporalContract,
    cleanedSeries,
    interpolateLinear,
    interpolatePchip,
    sampleTemporal,
    createTemporalView,
    temporalSeriesPoints,
    temporalContext,
    temporalSampleAt,
    spectralFeatures,
    spectralFeatureNames,
    MINUTE
};
if (typeof module !== 'undefined' && module.exports) module.exports = api;
else globalThis.TemporalSampling = api;
