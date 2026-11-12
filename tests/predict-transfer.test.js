const { test } = require('node:test');
const assert = require('node:assert/strict');
const P = require('../predict-transfer.js');
const { readFileSync } = require('node:fs');

const STEP = P.DEFAULT_STEP_MS;

function seriesFrom(start, values, step = STEP) {
    return values.map((value, i) => ({ time: new Date(+start + i * step), value }));
}

function sensor(id, values, start = new Date('2026-11-11T00:00:00Z')) {
    return { id, name: id, levels: { yellow: 5, orange: 7, red: 9 }, timeSeries: seriesFrom(start, values) };
}

const sampleConfig = {
    sensors: {
        A: { arpaeId: '1', url: 'https://example.test/1', dependencies: [] },
        B: { arpaeId: '2', url: 'https://example.test/2', dependencies: ['A'] },
        C: { arpaeId: '3', url: 'https://example.test/3', dependencies: ['B'] },
        D: { arpaeId: '4', url: 'https://example.test/4', dependencies: ['A', 'C'] }
    }
};

test('buildGraph emits upstream→downstream edges from dependencies', () => {
    const graph = P.buildGraph(sampleConfig);
    assert.deepEqual(graph.nodes.sort(), ['A', 'B', 'C', 'D']);
    assert.deepEqual(graph.edges.map(e => e.id).sort(), ['A->B', 'A->D', 'B->C', 'C->D']);
});

test('rankNodes takes max depth across diamond DAGs', () => {
    const ranks = P.rankNodes(sampleConfig);
    assert.equal(ranks.D, 0);
    assert.equal(ranks.C, 1);
    assert.equal(ranks.B, 2);
    assert.equal(ranks.A, 3);
});

test('estimateLag recovers a known shift', () => {
    const base = Array.from({ length: 40 }, (_, i) => Math.sin(i / 3));
    const lagSteps = 4;
    const upstream = base.slice(0, base.length - lagSteps);
    const downstream = base.slice(lagSteps);
    const est = P.estimateLag(
        [...upstream, ...Array(lagSteps).fill(0)],
        [...Array(lagSteps).fill(0), ...downstream],
        10
    );
    // Build proper lagged pair series
    const xs = Array.from({ length: 48 }, (_, i) => Math.sin(i / 3));
    const ys = Array.from({ length: 48 }, (_, i) => Math.sin((i - lagSteps) / 3));
    const recovered = P.estimateLag(xs, ys, 12);
    assert.equal(recovered.lagSteps, lagSteps);
    assert.ok(Math.abs(recovered.r) > 0.99);
});

test('fitAffine recovers a and b', () => {
    const xs = [1, 2, 3, 4, 5];
    const ys = xs.map(x => 2 * x + 3);
    const fit = P.fitAffine(xs, ys);
    assert.ok(Math.abs(fit.a - 2) < 1e-9);
    assert.ok(Math.abs(fit.b - 3) < 1e-9);
});

test('predictEdge returns half-hour forecast forks min/med/max', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const now = new Date('2026-11-11T20:00:00Z');
    const n = 48;
    const upValues = Array.from({ length: n }, (_, i) => 1 + 0.05 * i + Math.sin(i / 4));
    const lag = 3;
    const a = 1.2, b = -0.4;
    const downValues = Array.from({ length: n }, (_, i) => {
        const src = i >= lag ? upValues[i - lag] : upValues[0];
        return a * src + b;
    });
    const up = sensor('Sostegno Reno', upValues, start);
    const down = sensor('Cento', downValues, start);
    const result = P.predictEdge(up, down, { hours: 24, now, horizonHours: 3, maxLagHours: 6 });
    assert.equal(result.ok, true);
    assert.ok(result.forecast.length >= 4);
    assert.ok(result.next.length >= 1);
    for (const point of result.next) {
        assert.ok(point.time instanceof Date);
        assert.equal(point.time.getUTCMinutes() % 30, 0);
        assert.ok(Number.isFinite(point.min));
        assert.ok(Number.isFinite(point.med));
        assert.ok(Number.isFinite(point.max));
        assert.ok(point.min <= point.med && point.med <= point.max);
    }
    assert.ok(Math.abs(result.lagMinutes - lag * 30) <= 30);
    const lastDown = down.timeSeries.filter(p => Number.isFinite(p.value)).at(-1).value;
    assert.ok(Math.abs(result.forecast[0].med - lastDown) < 1e-6, 'forecast must start from last observed downstream');
});

test('forecast rate-limit prevents impossible jumps from affine raw output', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const now = new Date('2026-11-11T10:00:00Z');
    // Quiet downstream history → tiny maxStep; upstream would imply a huge jump without anchoring/limit
    const up = sensor('A', Array.from({ length: 24 }, (_, i) => i < 20 ? 1 : 20), start);
    const down = sensor('B', Array.from({ length: 24 }, () => 0.5), start);
    const model = { a: 1, b: 0, lagMinutes: 30 };
    const anchor = P.latestFinite(down.timeSeries);
    const maxStep = P.maxStepFromSeries(down.timeSeries);
    const points = P.forecastForward(up, model, 2, now, P.DEFAULT_STEP_MS, { anchor, maxStep });
    assert.ok(points.length >= 2);
    assert.ok(Math.abs(points[0].value - anchor.value) < 1e-9);
    for (let i = 1; i < points.length; i++) {
        assert.ok(Math.abs(points[i].value - points[i - 1].value) <= maxStep + 1e-9);
    }
});

test('offline coeffs are preferred when live correlation is weak', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    // Nearly flat / uncorrelated series so |r| stays below WEAK_R
    const up = sensor('A', Array.from({ length: 30 }, (_, i) => 1 + (i % 2) * 0.001), start);
    const down = sensor('B', Array.from({ length: 30 }, (_, i) => 2 + ((i * 3) % 5) * 0.001), start);
    const coeffs = {
        edges: {
            'A->B': { lagMinutes: 60, a: 0.8, b: 0.1, r: 0.9, rmse: 0.15, trainedAt: '2026-01-01' }
        }
    };
    const result = P.predictEdge(up, down, { hours: 12, now, coeffs, horizonHours: 2 });
    assert.equal(result.ok, true);
    assert.equal(result.source, 'offline');
    assert.equal(result.lagMinutes, 60);
    assert.equal(result.a, 0.8);
    assert.ok(result.next[0].max - result.next[0].min > 0);
});

test('forkAround builds symmetric min/med/max', () => {
    assert.deepEqual(P.forkAround(2, 0.5), { min: 1.5, med: 2, max: 2.5 });
    assert.deepEqual(P.forkAround(2, 0.2, 0.8), { min: 1.8, med: 2, max: 2.8 });
    assert.equal(P.forkAround(null, 1), null);
});

test('offline dual-regime node picks high regime above threshold', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    const upA = sensor('A', Array.from({ length: 40 }, () => 5), start);
    const upB = sensor('B', Array.from({ length: 40 }, () => 5), start);
    const down = sensor('C', Array.from({ length: 40 }, () => 1), start);
    const coeffs = {
        nodes: {
            C: {
                upstreams: ['A', 'B'],
                lagMinutes: [30, 60],
                highThreshold: 2,
                regimes: {
                    base: { a: [0.1, 0.1], b: 0, c: [0, 0], rmse: 0.1, spreadLow: 0.1, spreadHigh: 0.1 },
                    high: { a: [0.5, 0.5], b: 1, c: [0, 0], rmse: 0.2, spreadLow: 0.2, spreadHigh: 0.4 }
                },
                r: 0.9,
                rmse: 0.15,
                kind: 'dual_regime_multi_lag_ridge',
                trainedAt: '2026-01-01'
            }
        },
        edges: {}
    };
    // Flat uncorrelated live series → prefer offline
    const upFlat = sensor('A', Array.from({ length: 40 }, (_, i) => 1 + (i % 2) * 0.001), start);
    const upFlatB = sensor('B', Array.from({ length: 40 }, (_, i) => 1 + ((i * 3) % 5) * 0.001), start);
    const downFlat = sensor('C', Array.from({ length: 40 }, (_, i) => 2 + ((i * 5) % 7) * 0.001), start);
    const weak = P.predictNode(downFlat, [upFlat, upFlatB], { hours: 12, now, coeffs, horizonHours: 2 });
    assert.equal(weak.ok, true);
    assert.equal(weak.source, 'offline');

    const active = P.pickRegime(coeffs.nodes.C.regimes ? {
        ...coeffs.nodes.C,
        a: coeffs.nodes.C.regimes.base.a,
        b: coeffs.nodes.C.regimes.base.b,
        c: coeffs.nodes.C.regimes.base.c,
        regimes: {
            base: coeffs.nodes.C.regimes.base,
            high: coeffs.nodes.C.regimes.high
        }
    } : null, 5);
    assert.equal(active.activeRegime, 'high');
    assert.equal(active.b, 1);

    const quiet = P.pickRegime({
        a: [0.1, 0.1], b: 0, c: [0, 0],
        highThreshold: 2,
        regimes: coeffs.nodes.C.regimes
    }, 1);
    assert.equal(quiet.activeRegime, 'base');

    // smoke: high upstream series still forecasts
    const hi = P.predictNode(down, [upA, upB], { hours: 12, now, coeffs, horizonHours: 2 });
    assert.equal(hi.ok, true);
    assert.ok(hi.forecast.length >= 1);
});

test('direct-horizon offline forecasts deltas and rejects unreliable nodes', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const up = sensor('A', Array.from({ length: 48 }, (_, i) => 1 + 0.02 * i), start);
    const down = sensor('B', Array.from({ length: 48 }, (_, i) => 2 + 0.01 * i), start);
    const now = down.timeSeries.at(-1).time;
    const horizon = (extra = {}) => ({
        b: 0, wLocal: 0.2, wWindow: [0.5], wRate: [0],
        wWindow2: [0.1], wRate2: [0],
        spreadLow: 0.08, spreadHigh: 0.12,
        spreadLowHigh: 0.25, spreadHighHigh: 0.4,
        ...extra
    });
    const good = {
        nodes: {
            B: {
                upstreams: ['A'],
                lagMinutes: [60],
                reliable: true,
                skill120: 0.25,
                kind: 'direct_horizon_delta_ridge',
                thresholds: { A: { mean: 0, std: 1, z85: 1.0 } },
                horizons: {
                    '30': horizon({ validate: { rmse: 0.1, skill: 0.1 } }),
                    '60': horizon({ validate: { rmse: 0.12, skill: 0.15 } }),
                    '90': horizon({ validate: { rmse: 0.14, skill: 0.2 } }),
                    '120': horizon({ validateEvents: { rmse: 0.15, skill: 0.25 } })
                },
                trainedAt: '2026-01-01'
            }
        },
        edges: {}
    };
    const ok = P.predictNode(down, [up], { hours: 24, now, coeffs: good, horizonHours: 2 });
    assert.equal(ok.ok, true);
    assert.equal(ok.kind, 'direct_horizon_delta_ridge');
    assert.equal(ok.source, 'offline');
    assert.ok(ok.forecast.length >= 4);
    assert.ok(Math.abs(ok.forecast[0].med - down.timeSeries.at(-1).value) < 1e-9);
    assert.ok(Number.isFinite(ok.bias));

    // High-regime upstream → wider fork (use next[], not the anchor point)
    const upHigh = sensor('A', Array.from({ length: 48 }, () => 5), start);
    const hi = P.predictNode(down, [upHigh], { hours: 24, now, coeffs: good, horizonHours: 2 });
    assert.equal(hi.ok, true);
    assert.ok(hi.next.length >= 1);
    assert.ok(hi.next[0].max - hi.next[0].min >= 0.5);

    const bad = {
        nodes: {
            B: {
                ...good.nodes.B,
                reliable: false,
                skill120: 0.01
            }
        },
        edges: {}
    };
    const rejected = P.predictNode(down, [up], { hours: 24, now, coeffs: bad, horizonHours: 2 });
    assert.equal(rejected.ok, false);
    assert.equal(rejected.reliable, false);
    assert.match(rejected.reason, /persistenza/i);
});

test('mlpForward applies ReLU then linear', () => {
    const layers = [
        { W: [[1, 0], [0, 1]], b: [0, -2], activation: 'relu' },
        { W: [[1, 1]], b: [0.5], activation: 'linear' }
    ];
    // relu(I·[1,3]+[0,-2]) = [1,1] → linear sum + 0.5 = 2.5
    assert.equal(P.mlpForward([1, 3], layers), 2.5);
    assert.equal(P.mlpForward([0, 0], layers), 0.5);
});

test('window_delta_mlp forecasts Cento from feature window', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    const n = 60;
    const values = Array.from({ length: n }, (_, i) => 1 + 0.01 * i);
    const cento = sensor('Cento', values, start);
    const forcelli = sensor('Forcelli', values.map(v => v + 0.2), start);
    const sostegno = sensor('Sostegno Reno', values.map(v => v + 0.5), start);
    const featureNames = [];
    const mean = [];
    const std = [];
    for (const station of ['Cento', 'Forcelli', 'Sostegno Reno']) {
        const tag = station.replace(/ /g, '_');
        for (let k = 0; k < 4; k++) {
            featureNames.push(`level_${tag}_lag${k}`, `delta_${tag}_lag${k}`, `miss_${tag}_lag${k}`);
            mean.push(0, 0, 0);
            std.push(1, 1, 1);
        }
    }
    featureNames.push('rain6h', 'rain24h');
    mean.push(0, 0);
    std.push(1, 1);
    const nIn = featureNames.length;
    const layers = [
        {
            W: Array.from({ length: 2 }, () => Array.from({ length: nIn }, () => 0)),
            b: [0, 0],
            activation: 'relu'
        },
        { W: [[0, 0]], b: [0.05], activation: 'linear' }
    ];
    const horizonBlock = (extra = {}) => ({
        mean, std, layers,
        spreadLow: 0.02, spreadHigh: 0.03,
        spreadLowHigh: 0.05, spreadHighHigh: 0.08,
        ...extra
    });
    const coeffs = {
        nodes: {
            Cento: {
                kind: 'window_delta_mlp',
                stations: ['Cento', 'Forcelli', 'Sostegno Reno'],
                upstreams: ['Forcelli', 'Sostegno Reno'],
                lookbackSteps: 4,
                stepMinutes: 30,
                featureNames,
                useRain: true,
                rainGauges: ['Anzola'],
                reliable: true,
                skill120: 0.4,
                horizons: {
                    '30': horizonBlock(),
                    '60': horizonBlock(),
                    '90': horizonBlock(),
                    '120': horizonBlock({ validate: { skill: 0.4, rmse: 0.1 } })
                }
            }
        }
    };
    const sensorMap = {
        Cento: cento,
        Forcelli: forcelli,
        'Sostegno Reno': sostegno
    };
    const result = P.predictNode(cento, [forcelli, sostegno], {
        hours: 24, now, coeffs, horizonHours: 2, downstreamId: 'Cento', sensorMap, rainSensors: []
    });
    assert.equal(result.ok, true);
    assert.equal(result.kind, 'window_delta_mlp');
    assert.ok(result.forecast.length >= 4);
    assert.ok(Math.abs(result.forecast[0].med - cento.timeSeries.at(-1).value) < 1e-9);
    // Default smooth=0: each independent horizon uses the raw y0+Δ knot.
    const y0 = cento.timeSeries.at(-1).value;
    assert.ok(Math.abs(result.next[0].med - (y0 + 0.05)) < 1e-6);
    assert.ok(Math.abs(result.next[1].med - (y0 + 0.05)) < 1e-6);

    // Legacy EMA 0.65 shrinks the first increment to 35% of the raw Δ (documented brake).
    const ema = P.forecastWindowMlp(P.parseWindowMlpNode(coeffs.nodes.Cento), sensorMap, [], 2, now, STEP, {
        anchor: P.latestFinite(cento.timeSeries),
        downstream: cento,
        smooth: 0.65,
        maxStep: null
    });
    assert.ok(Math.abs(ema[1].value - (y0 + 0.35 * 0.05)) < 1e-6);
});

test('muskingum stable decode keeps non-negative coeffs and rising-step response', () => {
    for (let i = 0; i < 40; i++) {
        const raw = Array.from({ length: 9 }, () => (i - 20) * 0.35 + (i % 3) * 0.1);
        const theta = P.decodeMuskingumTheta(raw, { stableKX: true });
        for (const [K, X] of [[theta.K_f, theta.X_f], [theta.K_s, theta.X_s]]) {
            assert.ok(2 * K * X <= 1 + 1e-6, `2KX=${2 * K * X}`);
            assert.ok(1 <= 2 * K * (1 - X) + 1e-6, `2K(1-X)=${2 * K * (1 - X)}`);
            const { c0, c1, c2 } = P.muskingumCoeffs(K, X);
            assert.ok(c0 >= -1e-6 && c1 >= -1e-6 && c2 >= -1e-6);
        }
        // Rising inflow step must not produce a negative first outlet when O1=0.
        const O = P.muskingumStep(1, 0, 0, theta.K_f, theta.X_f);
        assert.ok(O >= -1e-6, `inverse step O=${O}`);
    }
    // Legacy decode still allows negative c0 (documented structural defect of old weights).
    const legacy = P.decodeMuskingumTheta([2, 2, 2, 2, 0, 0, 0, 0, 0], { stableKX: false });
    assert.ok(P.muskingumCoeffs(legacy.K_f, legacy.X_f).c0 < 0);
});

test('muskingumStep conserves convex combination for mid X inside stability box', () => {
    const { K, X } = P.decodeKX(0, -1); // mid K for X≈0.15-ish
    const O = P.muskingumStep(2, 1, 1.5, K, X);
    assert.ok(Number.isFinite(O));
    assert.ok(O > 0.5 && O < 3);
});

test('muskingum warmup changes state vs cold start on a ramp', () => {
    const I = Array.from({ length: 8 }, (_, i) => i * 0.2);
    const { K, X } = P.decodeKX(0, 0);
    const warm = P.muskingumWarmup(I, K, X);
    const cold = { I_prev: I.at(-1), O: I.at(-1) };
    assert.ok(Math.abs(warm.O - cold.O) > 1e-6);
    assert.equal(warm.I_prev, I.at(-1));
});

test('maxStep floor preserves observed local slope on rising limb', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    // Quiet history then a sharp rise in the last step.
    const values = Array.from({ length: 60 }, (_, i) => (i < 58 ? 1.0 : 1.0 + (i - 57) * 0.4));
    const cento = sensor('Cento', values, start);
    const forcelli = sensor('Forcelli', values.map(v => v + 0.2), start);
    const sostegno = sensor('Sostegno Reno', values.map(v => v + 0.5), start);
    const featureNames = [];
    const mean = [];
    const std = [];
    for (const station of ['Cento', 'Forcelli', 'Sostegno Reno']) {
        const tag = station.replace(/ /g, '_');
        for (let k = 0; k < 4; k++) {
            featureNames.push(`level_${tag}_lag${k}`, `delta_${tag}_lag${k}`, `miss_${tag}_lag${k}`);
            mean.push(0, 0, 0);
            std.push(1, 1, 1);
        }
    }
    featureNames.push('rain6h', 'rain24h');
    mean.push(0, 0);
    std.push(1, 1);
    const nIn = featureNames.length;
    // MLP wants +0.5 m at +30′ — larger than quiet-history maxStep (~0.05) but ≈ dyLocal.
    const layers = [
        { W: Array.from({ length: 2 }, () => Array.from({ length: nIn }, () => 0)), b: [0, 0], activation: 'relu' },
        { W: [[0, 0]], b: [0.5], activation: 'linear' }
    ];
    const block = { mean, std, layers, spreadLow: 0.02, spreadHigh: 0.03 };
    const model = P.parseWindowMlpNode({
        kind: 'window_delta_mlp',
        stations: ['Cento', 'Forcelli', 'Sostegno Reno'],
        upstreams: ['Forcelli', 'Sostegno Reno'],
        lookbackSteps: 4,
        stepMinutes: 30,
        featureNames,
        useRain: true,
        horizons: { '30': block, '60': block, '90': block, '120': block },
        reliable: true,
        skill120: 0.4
    });
    const sensorMap = { Cento: cento, Forcelli: forcelli, 'Sostegno Reno': sostegno };
    const quietMax = P.maxStepFromSeries(cento.timeSeries);
    assert.ok(quietMax < 0.2, `expected quiet maxStep, got ${quietMax}`);
    const pts = P.forecastWindowMlp(model, sensorMap, [], 2, now, STEP, {
        anchor: P.latestFinite(cento.timeSeries),
        downstream: cento,
        maxStep: quietMax,
        smooth: 0
    });
    const y0 = values.at(-1);
    // With dyLocal floor, first knot can rise by ~0.4 (observed) rather than quietMax alone.
    assert.ok(pts[1].value - y0 >= 0.39, `got Δ=${pts[1].value - y0}`);
});

test('muskingum_gru rejects unreliable model and rolls dense when reliable', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    const n = 80;
    const values = Array.from({ length: n }, (_, i) => 1 + 0.01 * i);
    const cento = sensor('Cento', values, start);
    const forcelli = sensor('Forcelli', values.map(v => v + 0.2), start);
    const sostegno = sensor('Sostegno Reno', values.map(v => v + 0.5), start);
    const H = 4;
    const F = 10;
    const zeroIh = Array.from({ length: 3 * H }, () => Array(F).fill(0));
    const zeroHh = Array.from({ length: 3 * H }, () => Array(H).fill(0));
    const node = {
        kind: 'muskingum_gru',
        downstream: 'Cento',
        upstreams: ['Forcelli', 'Sostegno Reno'],
        lookbackSteps: 8,
        stepMinutes: 30,
        featPerStep: F,
        means: { Cento: 0, Forcelli: 0, 'Sostegno Reno': 0, rain: 0 },
        stds: { Cento: 1, Forcelli: 1, 'Sostegno Reno': 1, rain: 1 },
        lags: { Forcelli: { steps: 0, minutes: 0 }, 'Sostegno Reno': { steps: 0, minutes: 0 } },
        gru: {
            inputSize: F,
            hiddenSize: H,
            weightIh: zeroIh,
            weightHh: zeroHh,
            biasIh: Array(3 * H).fill(0),
            biasHh: Array(3 * H).fill(0)
        },
        head: {
            W: Array.from({ length: 9 }, () => Array(H).fill(0)),
            b: [0, 0, 0, 0, 0, 0, 0, 0, 0]
        },
        spreadLow: 0.05,
        spreadHigh: 0.05,
        reliable: false,
        skill120: 0.05
    };
    const rejected = P.predictNode(cento, [forcelli, sostegno], {
        hours: 24, now, coeffs: { nodes: { Cento: node } }, horizonHours: 2,
        downstreamId: 'Cento',
        sensorMap: { Cento: cento, Forcelli: forcelli, 'Sostegno Reno': sostegno }
    });
    assert.equal(rejected.ok, false);
    assert.equal(rejected.kind, 'muskingum_gru');

    node.reliable = true;
    const ok = P.predictNode(cento, [forcelli, sostegno], {
        hours: 24, now, coeffs: { nodes: { Cento: node } }, horizonHours: 2,
        downstreamId: 'Cento',
        sensorMap: { Cento: cento, Forcelli: forcelli, 'Sostegno Reno': sostegno }
    });
    assert.equal(ok.ok, true);
    assert.equal(ok.kind, 'muskingum_gru');
    // dense 30′ points over 2 h (+ anchor) → at least 5
    assert.ok(ok.forecast.length >= 5);
    // coherent: no huge jump from last obs
    assert.ok(Math.abs(ok.forecast[0].med - values.at(-1)) < 1e-6);
    for (let i = 1; i < ok.forecast.length; i++) {
        assert.ok(Math.abs(ok.forecast[i].med - ok.forecast[i - 1].med) < 2);
    }
});

test('Python/JS Muskingum decode+warmup parity fixture', () => {
    const fixtures = [
        {
            raw: [0, 0, 0, 0, 0, 0, 0, 0, 0],
            K_f: 1.3333333333333335, X_f: 0.25,
            stepO: 0.11111111111111109,
            warmI: [0, 1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7, 1],
            warmO: 0.8096109042611099
        },
        {
            raw: [1.5, -0.8, 0.2, 1.1, -1, 0.5, 0, 0, 0],
            K_f: 2.4596425657880046, X_f: 0.17401020754895502,
            stepO: 0.02843891568936031,
            warmI: [0, 1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7, 1],
            warmO: 0.6590470316455795
        },
        {
            raw: [-2, 2, 3, -1, 0, 0, 0, 0, 0],
            K_f: 0.8849901099267018, X_f: 0.4023188311911529,
            stepO: 0.1399027584559314,
            warmI: [0, 1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7, 1],
            warmO: 0.8735728414408037
        }
    ];
    for (const fx of fixtures) {
        const theta = P.decodeMuskingumTheta(fx.raw, { stableKX: true });
        assert.ok(Math.abs(theta.K_f - fx.K_f) < 1e-9);
        assert.ok(Math.abs(theta.X_f - fx.X_f) < 1e-9);
        const step = P.muskingumStep(1, 0, 0, theta.K_f, theta.X_f);
        assert.ok(Math.abs(step - fx.stepO) < 1e-9);
        const warm = P.muskingumWarmup(fx.warmI, theta.K_f, theta.X_f);
        assert.ok(Math.abs(warm.O - fx.warmO) < 1e-9);
    }
});

test('constant upstream keeps finite muskingum rollout without inverse first step', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    const n = 80;
    const flat = Array.from({ length: n }, () => 1.2);
    const cento = sensor('Cento', flat, start);
    const forcelli = sensor('Forcelli', flat.map(v => v + 0.2), start);
    const sostegno = sensor('Sostegno Reno', flat.map(v => v + 0.5), start);
    const H = 4, F = 10;
    const node = {
        kind: 'muskingum_gru',
        downstream: 'Cento',
        upstreams: ['Forcelli', 'Sostegno Reno'],
        lookbackSteps: 8,
        stepMinutes: 30,
        featPerStep: F,
        means: { Cento: 0, Forcelli: 0, 'Sostegno Reno': 0, rain: 0 },
        stds: { Cento: 1, Forcelli: 1, 'Sostegno Reno': 1, rain: 1 },
        lags: { Forcelli: { steps: 0, minutes: 0 }, 'Sostegno Reno': { steps: 0, minutes: 0 } },
        paramDecode: { stableKX: true, dt: 1, xMin: 0.05, xMax: 0.45 },
        gru: {
            inputSize: F, hiddenSize: H,
            weightIh: Array.from({ length: 3 * H }, () => Array(F).fill(0)),
            weightHh: Array.from({ length: 3 * H }, () => Array(H).fill(0)),
            biasIh: Array(3 * H).fill(0),
            biasHh: Array(3 * H).fill(0)
        },
        head: { W: Array.from({ length: 9 }, () => Array(H).fill(0)), b: Array(9).fill(0) },
        reliable: true,
        skill120: 0.2
    };
    const ok = P.predictNode(cento, [forcelli, sostegno], {
        hours: 24, now, coeffs: { nodes: { Cento: node } }, horizonHours: 2,
        downstreamId: 'Cento',
        sensorMap: { Cento: cento, Forcelli: forcelli, 'Sostegno Reno': sostegno }
    });
    assert.equal(ok.ok, true);
    for (const p of ok.forecast) assert.ok(Number.isFinite(p.med));
    // Flat inputs: forecast stays near anchor (no runaway / inverse spike).
    assert.ok(Math.abs(ok.next[0].med - 1.2) < 0.5);
});

test('rcStep: high acceptor drains less than low acceptor', () => {
    const theta = P.decodeRcTheta([0, 0, 0, 0, 0, 0, 0, 0, 0], {});
    const z0 = 0;
    const zf = 1, zs = 1, rain = 0;
    let zHigh = z0, zLow = z0;
    const high = [2, 2, 2, 2];
    const low = [-2, -2, -2, -2];
    for (let i = 0; i < 8; i++) {
        zHigh = P.rcStep(zHigh, zf, zs, high, rain, theta);
        zLow = P.rcStep(zLow, zf, zs, low, rain, theta);
    }
    assert.ok(Number.isFinite(zHigh) && Number.isFinite(zLow));
    assert.ok(zHigh >= zLow - 1e-6, `highAcc=${zHigh} lowAcc=${zLow}`);
});

test('rc_gru rejects unreliable and rolls when reliable with acceptors', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    const n = 80;
    const values = Array.from({ length: n }, (_, i) => 1 + 0.01 * i);
    const cento = sensor('Cento', values, start);
    const forcelli = sensor('Forcelli', values.map(v => v + 0.2), start);
    const sostegno = sensor('Sostegno Reno', values.map(v => v + 0.5), start);
    const dosso = sensor('Dosso', values.map(v => v - 0.1), start);
    const panfilia = sensor('Opera Reno Panfilia', values.map(v => v - 0.2), start);
    const secondo = sensor('Secondo Salto', values.map(v => v - 0.3), start);
    const gallo = sensor('Gallo', values.map(v => v - 0.15), start);
    const H = 4, F = 22;
    const node = {
        kind: 'rc_gru',
        downstream: 'Cento',
        upstreams: ['Forcelli', 'Sostegno Reno'],
        acceptors: ['Dosso', 'Opera Reno Panfilia', 'Secondo Salto', 'Gallo'],
        lookbackSteps: 8,
        stepMinutes: 30,
        featPerStep: F,
        means: {
            Cento: 0, Forcelli: 0, 'Sostegno Reno': 0, rain: 0,
            Dosso: 0, 'Opera Reno Panfilia': 0, 'Secondo Salto': 0, Gallo: 0
        },
        stds: {
            Cento: 1, Forcelli: 1, 'Sostegno Reno': 1, rain: 1,
            Dosso: 1, 'Opera Reno Panfilia': 1, 'Secondo Salto': 1, Gallo: 1
        },
        lags: { Forcelli: { steps: 0, minutes: 0 }, 'Sostegno Reno': { steps: 0, minutes: 0 } },
        paramDecode: { gMin: 0.02, gMax: 2.5, cMin: 0.5, cMax: 12, dt: 1 },
        gru: {
            inputSize: F, hiddenSize: H,
            weightIh: Array.from({ length: 3 * H }, () => Array(F).fill(0)),
            weightHh: Array.from({ length: 3 * H }, () => Array(H).fill(0)),
            biasIh: Array(3 * H).fill(0),
            biasHh: Array(3 * H).fill(0)
        },
        head: { W: Array.from({ length: 9 }, () => Array(H).fill(0)), b: Array(9).fill(0) },
        reliable: false,
        skill120: 0.05
    };
    const sensorMap = {
        Cento: cento, Forcelli: forcelli, 'Sostegno Reno': sostegno,
        Dosso: dosso, 'Opera Reno Panfilia': panfilia, 'Secondo Salto': secondo, Gallo: gallo
    };
    const rejected = P.predictNode(cento, [forcelli, sostegno], {
        hours: 24, now, coeffs: { nodes: { Cento: node } }, horizonHours: 2,
        downstreamId: 'Cento', sensorMap
    });
    assert.equal(rejected.ok, false);
    assert.equal(rejected.kind, 'rc_gru');
    node.reliable = true;
    const ok = P.predictNode(cento, [forcelli, sostegno], {
        hours: 24, now, coeffs: { nodes: { Cento: node } }, horizonHours: 2,
        downstreamId: 'Cento', sensorMap
    });
    assert.equal(ok.ok, true);
    assert.equal(ok.kind, 'rc_gru');
    assert.ok(ok.forecast.length >= 5);
    assert.ok(Math.abs(ok.forecast[0].med - values.at(-1)) < 1e-6);
});

test('window_delta_mlp parse keeps acceptors list', () => {
    const node = P.parseWindowMlpNode({
        kind: 'window_delta_mlp',
        stations: ['Cento', 'Forcelli', 'Dosso'],
        upstreams: ['Forcelli'],
        acceptors: ['Dosso'],
        lookbackSteps: 2,
        featureNames: ['level_Cento_lag0', 'delta_Cento_lag0', 'miss_Cento_lag0'],
        horizons: {
            '30': {
                mean: [0, 0, 0], std: [1, 1, 1],
                layers: [{ W: [[0, 0, 0]], b: [0], activation: 'linear' }],
                spreadLow: 0.1, spreadHigh: 0.1
            }
        },
        reliable: true,
        skill120: 0.2
    });
    assert.deepEqual(node.acceptors, ['Dosso']);
});

test('loadCoeffs soft-fails on missing file', async () => {
    const missing = await P.loadCoeffs(async () => ({ ok: false, status: 404 }));
    assert.equal(missing, null);
    const ok = await P.loadCoeffs(async () => ({
        ok: true,
        json: async () => ({ edges: { 'A->B': { lagMinutes: 30, a: 1, b: 0 } } })
    }));
    assert.ok(ok.edges['A->B']);
});

test('config dependencies form a connected DAG used by rankNodes', () => {
    const cfg = JSON.parse(readFileSync(new URL('../config.json', `file://${__filename}`), 'utf8'));
    const ranks = P.rankNodes(cfg);
    assert.equal(ranks.Cento > ranks.Dosso, true);
    assert.equal(ranks['Sostegno Reno'] > ranks.Cento, true);
    assert.equal(ranks['Vergato DX'] > ranks['Casalecchio chiusa'], true);
});
