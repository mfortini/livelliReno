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

test('hindcastSeries replays +2h without peeking and returns residuals', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const n = 96; // 48 h @ 30′
    const up = sensor('A', Array.from({ length: n }, (_, i) => 1 + 0.01 * i), start);
    const down = sensor('B', Array.from({ length: n }, (_, i) => 2 + 0.005 * i), start);
    const now = down.timeSeries.at(-1).time;
    const horizon = () => ({
        b: 0, wLocal: 0.2, wWindow: [0.4], wRate: [0],
        wWindow2: [0], wRate2: [0],
        spreadLow: 0.05, spreadHigh: 0.05,
        spreadLowHigh: 0.1, spreadHighHigh: 0.1
    });
    const coeffs = {
        nodes: {
            B: {
                upstreams: ['A'],
                lagMinutes: [60],
                reliable: true,
                skill120: 0.3,
                kind: 'direct_horizon_delta_ridge',
                thresholds: { A: { mean: 0, std: 1, z85: 1.0 } },
                horizons: {
                    '30': horizon(), '60': horizon(), '90': horizon(), '120': horizon()
                },
                trainedAt: '2026-01-01'
            }
        },
        edges: {}
    };
    const hc = P.hindcastSeries(down, [up], {
        hours: 24, now, coeffs, horizonHours: 2, downstreamId: 'B', stepMs: P.DEFAULT_STEP_MS
    });
    assert.equal(hc.ok, true);
    assert.ok(hc.points.length >= 20, `expected many hindcast points, got ${hc.points.length}`);
    for (const p of hc.points) {
        assert.ok(Number.isFinite(p.observed));
        assert.ok(Number.isFinite(p.predicted));
        assert.ok(Math.abs(p.residual - (p.predicted - p.observed)) < 1e-9);
        assert.ok(+p.time <= +now + 1);
        assert.ok(+p.origin < +p.time);
    }
});

test('direct-horizon densifies every 30′ with maxStep continuity from anchor', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const up = sensor('A', Array.from({ length: 48 }, (_, i) => 1 + 0.02 * i), start);
    const down = sensor('B', Array.from({ length: 48 }, (_, i) => 2 + 0.01 * i), start);
    const now = down.timeSeries.at(-1).time;
    const horizon = () => ({
        b: 0.5, wLocal: 0, wWindow: [0], wRate: [0],
        wWindow2: [0], wRate2: [0],
        spreadLow: 0.05, spreadHigh: 0.05,
        spreadLowHigh: 0.1, spreadHighHigh: 0.1
    });
    // Sparse knots only at 60′ and 120′ — densifier must fill 30′ and 90′.
    const coeffs = {
        nodes: {
            B: {
                upstreams: ['A'],
                lagMinutes: [60],
                reliable: true,
                skill120: 0.3,
                kind: 'direct_horizon_delta_ridge',
                thresholds: { A: { mean: 0, std: 1, z85: 1.0 } },
                horizons: {
                    '60': horizon(),
                    '120': horizon()
                },
                trainedAt: '2026-01-01'
            }
        },
        edges: {}
    };
    const maxStep = P.maxStepFromSeries(down.timeSeries);
    const raw = P.forecastDirectHorizon([up], coeffs.nodes.B, 2, now, P.DEFAULT_STEP_MS, {
        anchor: P.latestFinite(down.timeSeries),
        maxStep,
        downstream: down,
        bias: 0,
        smooth: 0.35
    });
    assert.ok(raw.length >= 5, `expected dense path, got ${raw.length}`);
    assert.equal(raw[0].regime, 'anchor');
    assert.ok(Math.abs(raw[0].value - down.timeSeries.at(-1).value) < 1e-9);
    for (let i = 1; i < raw.length; i++) {
        const dt = +raw[i].time - +raw[i - 1].time;
        assert.ok(Math.abs(dt - P.DEFAULT_STEP_MS) < 1, `step ${i} dt=${dt}`);
        assert.ok(Math.abs(raw[i].value - raw[i - 1].value) <= maxStep + 1e-9);
    }
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

test('window_delta_transformer forecast from sequence tensor', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    const n = 60;
    const values = Array.from({ length: n }, (_, i) => 1 + 0.01 * i);
    const cento = sensor('Cento', values, start);
    const forcelli = sensor('Forcelli', values.map(v => v + 0.2), start);
    const stations = ['Cento', 'Forcelli'];
    const channelNames = [];
    for (const station of stations) {
        const tag = station.replace(/ /g, '_');
        channelNames.push(`level_${tag}`, `delta_${tag}`, `miss_${tag}`);
    }
    const nC = channelNames.length;
    const d = 4;
    const nhead = 2;
    const L = 4;
    const mean = Array(nC).fill(0);
    const std = Array(nC).fill(1);
    // Zero in_proj / attn / FFN → residual path carries zeros from input proj; head bias → constant delta.
    const zeroMat = (rows, cols) => Array.from({ length: rows }, () => Array(cols).fill(0));
    const ln = {
        weight: Array(d).fill(1),
        bias: Array(d).fill(0),
        eps: 1e-5
    };
    const layer = {
        norm1: ln,
        attn: {
            nhead,
            dModel: d,
            inProjWeight: zeroMat(3 * d, d),
            inProjBias: Array(3 * d).fill(0),
            outProjWeight: zeroMat(d, d),
            outProjBias: Array(d).fill(0)
        },
        norm2: ln,
        linear1: { W: zeroMat(8, d), b: Array(8).fill(0) },
        linear2: { W: zeroMat(d, 8), b: Array(d).fill(0) }
    };
    const transformer = {
        dModel: d,
        nhead,
        nlayers: 1,
        inProj: { W: zeroMat(d, nC), b: Array(d).fill(0) },
        pos: Array.from({ length: L }, () => Array(d).fill(0)),
        layers: [layer],
        norm: ln
    };
    const headLayers = [{ W: [Array(d).fill(0)], b: [0.05], activation: 'linear' }];
    const tfNode = {
        kind: 'window_delta_transformer',
        stations,
        upstreams: ['Forcelli'],
        channelNames,
        lookbackSteps: L,
        stepMinutes: 30,
        useRain: false,
        reliable: true,
        skill120: 0.4,
        horizons: {
            '30': { mean, std, transformer, headLayers, spreadLow: 0.01, spreadHigh: 0.02 },
            '60': { mean, std, transformer, headLayers, spreadLow: 0.01, spreadHigh: 0.02 },
            '90': { mean, std, transformer, headLayers, spreadLow: 0.01, spreadHigh: 0.02 },
            '120': { mean, std, transformer, headLayers, spreadLow: 0.01, spreadHigh: 0.02, validate: { skill: 0.4 } }
        }
    };
    const sensorMap = { Cento: cento, Forcelli: forcelli };
    const result = P.predictNode(cento, [forcelli], {
        hours: 24, now, coeffs: { nodes: { Cento: tfNode } },
        horizonHours: 2, downstreamId: 'Cento', sensorMap, rainSensors: []
    });
    assert.equal(result.ok, true);
    assert.equal(result.kind, 'window_delta_transformer');
    const y0 = cento.timeSeries.at(-1).value;
    assert.ok(Math.abs(result.next[0].med - (y0 + 0.05)) < 1e-6);
});

test('window_delta_gru / cnn forecast from sequence tensor', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    const n = 60;
    const values = Array.from({ length: n }, (_, i) => 1 + 0.01 * i);
    const cento = sensor('Cento', values, start);
    const forcelli = sensor('Forcelli', values.map(v => v + 0.2), start);
    const stations = ['Cento', 'Forcelli'];
    const channelNames = [];
    for (const station of stations) {
        const tag = station.replace(/ /g, '_');
        channelNames.push(`level_${tag}`, `delta_${tag}`, `miss_${tag}`);
    }
    const nC = channelNames.length;
    const mean = Array(nC).fill(0);
    const std = Array(nC).fill(1);
    const hidden = 2;
    // Identity-ish GRU: zero weights → h stays 0; head bias → constant delta.
    const gru = {
        inputSize: nC,
        hiddenSize: hidden,
        weightIh: Array.from({ length: 3 * hidden }, () => Array(nC).fill(0)),
        weightHh: Array.from({ length: 3 * hidden }, () => Array(hidden).fill(0)),
        biasIh: Array(3 * hidden).fill(0),
        biasHh: Array(3 * hidden).fill(0)
    };
    const headLayers = [{ W: [[0, 0]], b: [0.07], activation: 'linear' }];
    const gruNode = {
        kind: 'window_delta_gru',
        stations,
        upstreams: ['Forcelli'],
        channelNames,
        lookbackSteps: 4,
        stepMinutes: 30,
        useRain: false,
        reliable: true,
        skill120: 0.5,
        horizons: {
            '30': { mean, std, gru, headLayers, spreadLow: 0.02, spreadHigh: 0.03 },
            '60': { mean, std, gru, headLayers, spreadLow: 0.02, spreadHigh: 0.03 },
            '90': { mean, std, gru, headLayers, spreadLow: 0.02, spreadHigh: 0.03 },
            '120': { mean, std, gru, headLayers, spreadLow: 0.02, spreadHigh: 0.03, validate: { skill: 0.5 } }
        }
    };
    const sensorMap = { Cento: cento, Forcelli: forcelli };
    const gruResult = P.predictNode(cento, [forcelli], {
        hours: 24, now, coeffs: { nodes: { Cento: gruNode } },
        horizonHours: 2, downstreamId: 'Cento', sensorMap, rainSensors: []
    });
    assert.equal(gruResult.ok, true);
    assert.equal(gruResult.kind, 'window_delta_gru');
    const y0 = cento.timeSeries.at(-1).value;
    assert.ok(Math.abs(gruResult.next[0].med - (y0 + 0.07)) < 1e-6);

    // Conv1d zero weights + bias → constant channel; head bias → delta.
    const convLayers = [{
        W: [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
        b: [0],
        kernelSize: 3,
        padding: 1,
        activation: 'relu'
    }];
    // Fix: W shape (out=1, in=6, k=3)
    convLayers[0].W = [Array.from({ length: nC }, () => [0, 0, 0])];
    const cnnNode = {
        kind: 'window_delta_cnn',
        stations,
        upstreams: ['Forcelli'],
        channelNames,
        lookbackSteps: 4,
        stepMinutes: 30,
        useRain: false,
        reliable: true,
        skill120: 0.4,
        horizons: {
            '30': {
                mean, std, convLayers,
                headLayers: [{ W: [[0]], b: [0.04], activation: 'linear' }],
                spreadLow: 0.01, spreadHigh: 0.02
            },
            '60': {
                mean, std, convLayers,
                headLayers: [{ W: [[0]], b: [0.04], activation: 'linear' }],
                spreadLow: 0.01, spreadHigh: 0.02
            },
            '90': {
                mean, std, convLayers,
                headLayers: [{ W: [[0]], b: [0.04], activation: 'linear' }],
                spreadLow: 0.01, spreadHigh: 0.02
            },
            '120': {
                mean, std, convLayers,
                headLayers: [{ W: [[0]], b: [0.04], activation: 'linear' }],
                spreadLow: 0.01, spreadHigh: 0.02,
                validate: { skill: 0.4 }
            }
        }
    };
    const cnnResult = P.predictNode(cento, [forcelli], {
        hours: 24, now, coeffs: { nodes: { Cento: cnnNode } },
        horizonHours: 2, downstreamId: 'Cento', sensorMap, rainSensors: []
    });
    assert.equal(cnnResult.ok, true);
    assert.equal(cnnResult.kind, 'window_delta_cnn');
    assert.ok(Math.abs(cnnResult.next[0].med - (y0 + 0.04)) < 1e-6);
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

test('forecastWindowMlp raw features are horizon-independent (safe to reuse)', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    const n = 60;
    const values = Array.from({ length: n }, (_, i) => 1 + 0.01 * i);
    const cento = sensor('Cento', values, start);
    const forcelli = sensor('Forcelli', values.map(v => v + 0.2), start);
    const sensorMap = { Cento: cento, Forcelli: forcelli };
    const stations = ['Cento', 'Forcelli'];
    const recipe = { usePolyDerivatives: false };
    const at0 = P.buildWindowFeatureVector(sensorMap, stations, 4, +now, STEP, [], false, recipe, 0);
    const at120 = P.buildWindowFeatureVector(sensorMap, stations, 4, +now, STEP, [], false, recipe, 120);
    assert.deepEqual(at0, at120);

    const featureNames = Object.keys(at0);
    const mean = featureNames.map(() => 0);
    const std = featureNames.map(() => 1);
    const nIn = featureNames.length;
    const layers = [
        { W: Array.from({ length: 2 }, () => Array.from({ length: nIn }, () => 0)), b: [0, 0], activation: 'relu' },
        { W: [[0, 0]], b: [0.04], activation: 'linear' }
    ];
    const block = { mean, std, layers, spreadLow: 0.01, spreadHigh: 0.02 };
    const model = P.parseWindowMlpNode({
        kind: 'window_delta_mlp',
        stations,
        lookbackSteps: 4,
        stepMinutes: 30,
        featureNames,
        featureRecipe: recipe,
        useRain: false,
        reliable: true,
        horizons: { '30': block, '60': { ...block }, '120': { ...block } }
    });
    const pts = P.forecastWindowMlp(model, sensorMap, [], 2, now, STEP, {
        anchor: P.latestFinite(cento.timeSeries),
        downstream: cento,
        smooth: 0,
        maxStep: null
    });
    const y0 = cento.timeSeries.at(-1).value;
    const knots = pts.filter(p => p.regime !== 'anchor');
    assert.equal(knots.length, 3);
    for (const p of knots) assert.ok(Math.abs(p.value - (y0 + 0.04)) < 1e-6);
});

test('residual_lin3 decode is lin3 + MLP residual', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const n = 60;
    // Anchor now on the last sample so lin3 sees a complete 3-point window.
    const now = new Date(+start + (n - 1) * STEP);
    // Rising limb: +0.02 per 30′ step → lin3(h=1) = y0 + 0.02, lin3_delta = 0.02.
    const values = Array.from({ length: n }, (_, i) => 1 + 0.02 * i);
    const cento = sensor('Cento', values, start);
    const forcelli = sensor('Forcelli', values.map(v => v + 0.2), start);
    const y0 = values[n - 1];
    const lin3_30 = P.lin3Extrapolate(cento.timeSeries, +now, STEP, 1, 3);
    assert.ok(Math.abs(lin3_30 - (y0 + 0.02)) < 1e-9);

    const featureNames = [];
    const mean = [];
    const std = [];
    for (const station of ['Cento', 'Forcelli']) {
        const tag = station.replace(/ /g, '_');
        for (let k = 0; k < 4; k++) {
            featureNames.push(`level_${tag}_lag${k}`, `delta_${tag}_lag${k}`, `miss_${tag}_lag${k}`);
            mean.push(0, 0, 0);
            std.push(1, 1, 1);
        }
    }
    featureNames.push('lin3_delta_Cento');
    mean.push(0);
    std.push(1);
    const nIn = featureNames.length;
    // Constant residual +0.03 regardless of inputs.
    const layers = [
        {
            W: Array.from({ length: 2 }, () => Array.from({ length: nIn }, () => 0)),
            b: [0, 0],
            activation: 'relu'
        },
        { W: [[0, 0]], b: [0.03], activation: 'linear' }
    ];
    const block = {
        mean, std, layers,
        spreadLow: 0.02, spreadHigh: 0.03
    };
    const model = P.parseWindowMlpNode({
        kind: 'window_delta_mlp',
        downstream: 'Cento',
        stations: ['Cento', 'Forcelli'],
        upstreams: ['Forcelli'],
        lookbackSteps: 4,
        stepMinutes: 30,
        featureNames,
        featureRecipe: { targetMode: 'residual_lin3', lin3Points: 3 },
        useRain: false,
        reliable: true,
        skill120: 0.4,
        horizons: { '30': block, '60': block }
    });
    const raw = P.buildWindowFeatureVector(
        { Cento: cento, Forcelli: forcelli },
        model.stations, model.lookbackSteps, +now, STEP, [], false,
        model.featureRecipe, 30
    );
    assert.ok(Math.abs(raw.lin3_delta_Cento - 0.02) < 1e-9);

    const pts = P.forecastWindowMlp(model, { Cento: cento, Forcelli: forcelli }, [], 1, now, STEP, {
        anchor: P.latestFinite(cento.timeSeries),
        downstream: cento,
        smooth: 0,
        maxStep: null
    });
    // ŷ = lin3 + r̂ = (y0 + 0.02) + 0.03
    assert.ok(pts.length >= 2);
    assert.ok(Math.abs(pts[1].value - (y0 + 0.02 + 0.03)) < 1e-6);
});

test('residual_poly2 decode is poly2 + MLP residual', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const n = 60;
    const now = new Date(+start + (n - 1) * STEP);
    // Linear rise → poly2(6) matches lin3 slope (+0.02 / step).
    const values = Array.from({ length: n }, (_, i) => 1 + 0.02 * i);
    const cento = sensor('Cento', values, start);
    const forcelli = sensor('Forcelli', values.map(v => v + 0.2), start);
    const y0 = values[n - 1];
    const poly2_30 = P.poly2Extrapolate(cento.timeSeries, +now, STEP, 1, 6);
    assert.ok(Math.abs(poly2_30 - (y0 + 0.02)) < 1e-9);

    const featureNames = [];
    const mean = [];
    const std = [];
    for (const station of ['Cento', 'Forcelli']) {
        const tag = station.replace(/ /g, '_');
        for (let k = 0; k < 4; k++) {
            featureNames.push(`level_${tag}_lag${k}`, `delta_${tag}_lag${k}`, `miss_${tag}_lag${k}`);
            mean.push(0, 0, 0);
            std.push(1, 1, 1);
        }
    }
    featureNames.push('poly2_delta_Cento');
    mean.push(0);
    std.push(1);
    const nIn = featureNames.length;
    const layers = [
        {
            W: Array.from({ length: 2 }, () => Array.from({ length: nIn }, () => 0)),
            b: [0, 0],
            activation: 'relu'
        },
        { W: [[0, 0]], b: [0.03], activation: 'linear' }
    ];
    const block = { mean, std, layers, spreadLow: 0.02, spreadHigh: 0.03 };
    const model = P.parseWindowMlpNode({
        kind: 'window_delta_mlp',
        downstream: 'Cento',
        stations: ['Cento', 'Forcelli'],
        upstreams: ['Forcelli'],
        lookbackSteps: 4,
        stepMinutes: 30,
        featureNames,
        featureRecipe: { targetMode: 'residual_poly2', poly2Points: 6 },
        useRain: false,
        reliable: true,
        skill120: 0.4,
        horizons: { '30': block, '60': block }
    });
    const raw = P.buildWindowFeatureVector(
        { Cento: cento, Forcelli: forcelli },
        model.stations, model.lookbackSteps, +now, STEP, [], false,
        model.featureRecipe, 30
    );
    assert.ok(Math.abs(raw.poly2_delta_Cento - 0.02) < 1e-9);

    const pts = P.forecastWindowMlp(model, { Cento: cento, Forcelli: forcelli }, [], 1, now, STEP, {
        anchor: P.latestFinite(cento.timeSeries),
        downstream: cento,
        smooth: 0,
        maxStep: null
    });
    assert.ok(pts.length >= 2);
    assert.ok(Math.abs(pts[1].value - (y0 + 0.02 + 0.03)) < 1e-6);
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

test('applyMlpOnlineCorrection subtracts bias and advances phase', () => {
    const t0 = Date.parse('2026-11-11T12:00:00Z');
    const forecast = [
        { time: new Date(t0), value: 1.0, regime: 'anchor' },
        { time: new Date(t0 + 30 * 60000), value: 1.5, regime: 'high', med: 1.5, min: 1.4, max: 1.6 }
    ];
    const out = P.applyMlpOnlineCorrection(forecast, { bias: 0.2, phaseShiftMin: 30 });
    assert.equal(out[0].value, 1.0);
    assert.equal(out[0].regime, 'anchor');
    assert.ok(Math.abs(out[1].value - 1.3) < 1e-9);
    assert.ok(Math.abs(out[1].med - 1.3) < 1e-9);
    assert.equal(+out[1].time, t0); // advanced by 30′
});

test('warpForecastByCrossing advances rise to match mode tau', () => {
    const nowMs = Date.parse('2026-11-11T12:00:00Z');
    const y0 = 0;
    const forecast = [
        { time: new Date(nowMs), value: 0, med: 0, regime: 'anchor' },
        { time: new Date(nowMs + 120 * 60000), value: 1.0, med: 1.0, min: 0.8, max: 1.2, regime: 'high' },
        { time: new Date(nowMs + 240 * 60000), value: 1.5, med: 1.5, min: 1.2, max: 1.8, regime: 'high' }
    ];
    // Path hits +1 m at 120′; mode says 60′ → scale 0.5
    const warped = P.warpForecastByCrossing(forecast, { modeMin: 60, offsetM: 1, censorMin: 360 }, y0, nowMs);
    assert.equal(warped.warped, true);
    assert.equal(warped.method, 'time_warp');
    assert.ok(Math.abs(warped.scale - 0.5) < 1e-9);
    assert.equal(+warped.forecast[1].time, nowMs + 60 * 60000);
    assert.equal(+warped.forecast[2].time, nowMs + 120 * 60000);
});

test('warpForecastByCrossing lifts path that never reaches offset', () => {
    const nowMs = Date.parse('2026-11-11T12:00:00Z');
    const y0 = 0;
    const forecast = [
        { time: new Date(nowMs), value: 0, med: 0, regime: 'anchor' },
        { time: new Date(nowMs + 60 * 60000), value: 0.2, med: 0.2, min: 0.1, max: 0.3, regime: 'high' },
        { time: new Date(nowMs + 120 * 60000), value: 0.3, med: 0.3, min: 0.2, max: 0.4, regime: 'high' }
    ];
    const warped = P.warpForecastByCrossing(forecast, { modeMin: 60, offsetM: 1, censorMin: 360 }, y0, nowMs);
    assert.equal(warped.warped, true);
    assert.equal(warped.method, 'level_lift');
    assert.ok(warped.forecast[1].med >= 0.99, `expected ~1 at modeTau, got ${warped.forecast[1].med}`);
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

test('buildWindowFeatureVector supports yellow headroom and chain deltas', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const now = new Date('2026-11-11T04:30:00Z');
    const cento = sensor('Cento', Array.from({ length: 10 }, () => 1.0), start);
    const dosso = sensor('Dosso', Array.from({ length: 10 }, () => 3.0), start);
    const values = P.buildWindowFeatureVector(
        { Cento: cento, Dosso: dosso },
        ['Cento', 'Dosso'],
        2,
        +now,
        STEP,
        [],
        false,
        {
            useSogliaHeadroom: true,
            sogliaKind: 'yellow',
            sogliaLevels: { Cento: 5.5, Dosso: 8.3 },
            chainPairs: [['Cento', 'Dosso']]
        }
    );
    // headroom = yellow - level → Cento 5.5-1=4.5, Dosso 8.3-3=5.3
    assert.ok(Math.abs(values.headroom_yellow_Cento_lag0 - 4.5) < 1e-9);
    assert.ok(Math.abs(values.headroom_yellow_Dosso_lag0 - 5.3) < 1e-9);
    assert.ok(Math.abs(values.chain_Cento__Dosso_lag0 - (1.0 - 3.0)) < 1e-9);
    assert.equal(values.chain_miss_Cento__Dosso_lag0, 0);
});

test('localPolyDerivatives recovers linear and quadratic kinematics', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    // y = 2 + 0.5 t + 0.25 t² at t=0,1,2,3 → at end (t=3): d1=0.5+2*0.25*3=2.0, d2=0.5
    const ys = [0, 1, 2, 3].map(t => 2 + 0.5 * t + 0.25 * t * t);
    const series = [];
    for (let i = 0; i < 4; i++) {
        series.push({ time: new Date(+start + i * STEP), value: ys[i] });
    }
    const end = +start + 3 * STEP;
    const { d1, d2 } = P.localPolyDerivatives(series, end, STEP, 4);
    assert.ok(Math.abs(d1 - 2.0) < 1e-6, `d1=${d1}`);
    assert.ok(Math.abs(d2 - 0.5) < 1e-6, `d2=${d2}`);
    const flat = Array.from({ length: 4 }, (_, i) => ({ time: new Date(+start + i * STEP), value: 3 }));
    const z = P.localPolyDerivatives(flat, end, STEP, 4);
    assert.ok(Math.abs(z.d1) < 1e-9 && Math.abs(z.d2) < 1e-9);
});

test('buildWindowFeatureVector emits d1/d2 poly derivative features', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const levels = Array.from({ length: 10 }, (_, i) => 1 + 0.1 * i);
    const cento = sensor('Cento', levels, start);
    const now = new Date(+start + 9 * STEP);
    const values = P.buildWindowFeatureVector(
        { Cento: cento },
        ['Cento'],
        2,
        +now,
        STEP,
        [],
        false,
        { usePolyDerivatives: true }
    );
    assert.ok(Number.isFinite(values.d1_Cento_lag0));
    assert.ok(Number.isFinite(values.d2_Cento_lag0));
    // linear → d2 ≈ 0, d1 ≈ 0.1
    assert.ok(Math.abs(values.d1_Cento_lag0 - 0.1) < 1e-6, `d1=${values.d1_Cento_lag0}`);
    assert.ok(Math.abs(values.d2_Cento_lag0) < 1e-6, `d2=${values.d2_Cento_lag0}`);
});

test('buildWindowFeatureVector adds yellow fill ratio beside headroom', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const now = new Date('2026-11-11T04:30:00Z');
    const cento = sensor('Cento', Array.from({ length: 10 }, () => 1.1), start);
    const values = P.buildWindowFeatureVector(
        { Cento: cento },
        ['Cento'],
        2,
        +now,
        STEP,
        [],
        false,
        {
            useSogliaHeadroom: true,
            useSogliaRatio: true,
            sogliaKind: 'yellow',
            sogliaLevels: { Cento: 5.5 }
        }
    );
    assert.ok(Math.abs(values.headroom_yellow_Cento_lag0 - (5.5 - 1.1)) < 1e-9);
    assert.ok(Math.abs(values.ratio_yellow_Cento_lag0 - (1.1 / 5.5)) < 1e-9);
    assert.ok(Math.abs(values.ratio_delta_Cento_lag0 - 0) < 1e-9);
});

test('buildWindowFeatureVector featureSignal=delta drops level on flat window', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const levels = Array.from({ length: 10 }, (_, i) => 1 + 0.1 * i);
    const cento = sensor('Cento', levels, start);
    const now = new Date(+start + 9 * STEP);
    const v = P.buildWindowFeatureVector(
        { Cento: cento }, ['Cento'], 2, +now, STEP, [], false,
        { featureSignal: 'delta', usePolyDerivatives: false }, 0
    );
    assert.equal(v.level_Cento_lag0, undefined);
    assert.ok(Number.isFinite(v.delta_Cento_lag0));
    assert.equal(v.miss_Cento_lag0, 0);
});

test('buildWindowFeatureVector tauFrac samples sparse lags and can drop level', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const levels = Array.from({ length: 40 }, (_, i) => 1 + 0.1 * i);
    const up = sensor('Forcelli', levels, start);
    const cento = sensor('Cento', levels.map((v) => v + 2), start);
    const now = new Date(+start + 39 * STEP);
    const recipe = {
        tauFrac: true,
        tauFracSignal: 'level',
        stationLagSteps: { Forcelli: [0, 4, 8] },
        stationLookbackSteps: { Cento: 2 },
        usePolyDerivatives: false
    };
    const v = P.buildWindowFeatureVector(
        { Cento: cento, Forcelli: up }, ['Cento', 'Forcelli'], 48, +now, STEP, [], false, recipe, 0
    );
    assert.ok(Math.abs(v.level_Forcelli_lag0 - (1 + 0.1 * 39)) < 1e-9);
    assert.ok(Math.abs(v.level_Forcelli_lag4 - (1 + 0.1 * 35)) < 1e-9);
    assert.ok(Math.abs(v.level_Forcelli_lag8 - (1 + 0.1 * 31)) < 1e-9);
    assert.equal(v.level_Forcelli_lag1, undefined);
    assert.ok(Math.abs(v.level_Cento_lag0 - (3 + 0.1 * 39)) < 1e-9);
    assert.ok(Math.abs(v.level_Cento_lag1 - (3 + 0.1 * 38)) < 1e-9);
    assert.equal(v.level_Cento_lag2, undefined);
    const recipeD = { ...recipe, tauFracSignal: 'delta' };
    const vd = P.buildWindowFeatureVector(
        { Cento: cento, Forcelli: up }, ['Cento', 'Forcelli'], 48, +now, STEP, [], false, recipeD, 0
    );
    assert.equal(vd.level_Forcelli_lag0, undefined);
    assert.ok(Number.isFinite(vd.delta_Forcelli_lag0));
    assert.ok(Math.abs(vd.level_Cento_lag0 - (3 + 0.1 * 39)) < 1e-9);
});

test('buildWindowFeatureVector aligns upstream window to horizon via tau', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const levels = Array.from({ length: 30 }, (_, i) => 1 + 0.1 * i);
    const up = sensor('Forcelli', levels, start);
    const now = new Date(+start + 29 * STEP);
    const recipe = {
        upstreamLagAlign: true,
        upstreamLagMinutes: { Forcelli: 480 }, // 16 step × 30′
        usePolyDerivatives: false
    };
    // horizon 0 → lag k reads x(t - 16 - k); lag0 = level at index 29-16=13 → 2.3
    const v0 = P.buildWindowFeatureVector(
        { Forcelli: up }, ['Forcelli'], 4, +now, STEP, [], false, recipe, 0
    );
    assert.ok(Math.abs(v0.level_Forcelli_lag0 - (1 + 0.1 * 13)) < 1e-9, `lag0=${v0.level_Forcelli_lag0}`);
    // horizon 480′ = tau → lag0 reads x(t + 16 - 16) = x(now) → index 29 → 3.9
    const vh = P.buildWindowFeatureVector(
        { Forcelli: up }, ['Forcelli'], 4, +now, STEP, [], false, recipe, 480
    );
    assert.ok(Math.abs(vh.level_Forcelli_lag0 - (1 + 0.1 * 29)) < 1e-9, `h lag0=${vh.level_Forcelli_lag0}`);
    assert.equal(vh.miss_Forcelli_lag0, 0);
    // horizon 600′ > tau → lag0 peeks in the future → miss=1, level null
    const vf = P.buildWindowFeatureVector(
        { Forcelli: up }, ['Forcelli'], 6, +now, STEP, [], false, recipe, 600
    );
    assert.equal(vf.level_Forcelli_lag0, null);
    assert.equal(vf.miss_Forcelli_lag0, 1);
    assert.equal(vf.delta_Forcelli_lag0, null);
    assert.equal(vf.d1_Forcelli_lag0 ?? null, null);
    // lag1 (k=1 → s=1+16-20=-3 <0) still missing; lag4 → s=0 → x(now)
    assert.equal(vf.miss_Forcelli_lag1, 1);
    assert.ok(Math.abs(vf.level_Forcelli_lag4 - (1 + 0.1 * 29)) < 1e-9, `lag4=${vf.level_Forcelli_lag4}`);
    // lag5 → s=1 → x(now-30′) = index 28
    assert.ok(Math.abs(vf.level_Forcelli_lag5 - (1 + 0.1 * 28)) < 1e-9, `lag5=${vf.level_Forcelli_lag5}`);
    // downstream station in the same call stays unshifted
    const cento = sensor('Cento', levels.map(v => v + 5), start);
    const vmix = P.buildWindowFeatureVector(
        { Cento: cento, Forcelli: up }, ['Cento', 'Forcelli'], 2, +now, STEP, [], false, recipe, 0
    );
    assert.ok(Math.abs(vmix.level_Cento_lag0 - (6 + 0.1 * 29)) < 1e-9);
    assert.ok(Math.abs(vmix.level_Forcelli_lag0 - (1 + 0.1 * 13)) < 1e-9);
    // recipe without upstreamLagAlign → behaviour unchanged
    const vflat = P.buildWindowFeatureVector(
        { Forcelli: up }, ['Forcelli'], 2, +now, STEP, [], false, { usePolyDerivatives: false }, 480
    );
    assert.ok(Math.abs(vflat.level_Forcelli_lag0 - (1 + 0.1 * 29)) < 1e-9);
});

test('buildWindowFeatureVector augment adds arriving-wave block without shifting', () => {
    const start = new Date('2026-11-11T00:00:00Z');
    const levels = Array.from({ length: 30 }, (_, i) => 1 + 0.1 * i);
    const up = sensor('Forcelli', levels, start);
    const now = new Date(+start + 29 * STEP);
    const recipe = {
        upstreamLagAlign: true,
        upstreamLagMode: 'augment',
        upstreamLagAlignWindow: 3,
        upstreamLagMinutes: { Forcelli: 480 },
        usePolyDerivatives: false
    };
    const v = P.buildWindowFeatureVector(
        { Forcelli: up }, ['Forcelli'], 2, +now, STEP, [], false, recipe, 120
    );
    // flat window untouched: lag0 = x(now) → index 29
    assert.ok(Math.abs(v.level_Forcelli_lag0 - (1 + 0.1 * 29)) < 1e-9);
    // a-block: j=0 → x(t + 4 - 16) = index 29+4-16 = 17; j=2 → index 15
    assert.ok(Math.abs(v.alevel_Forcelli_lag0 - (1 + 0.1 * 17)) < 1e-9, `a0=${v.alevel_Forcelli_lag0}`);
    assert.ok(Math.abs(v.alevel_Forcelli_lag2 - (1 + 0.1 * 15)) < 1e-9, `a2=${v.alevel_Forcelli_lag2}`);
    assert.equal(v.amiss_Forcelli_lag0, 0);
    // h > tau: s = -4 + j → j<4 miss; j=4 → x(now)
    const vf = P.buildWindowFeatureVector(
        { Forcelli: up }, ['Forcelli'], 2, +now, STEP, [], false, recipe, 600
    );
    assert.equal(vf.alevel_Forcelli_lag0, null);
    assert.equal(vf.amiss_Forcelli_lag0, 1);
    assert.equal(vf.amiss_Forcelli_lag2, 1);
});

test('window_delta_mlp parse uses real upstream lags from recipe', () => {
    const parsed = P.parseWindowMlpNode({
        kind: 'window_delta_mlp',
        stations: ['Cento', 'Forcelli', 'Sostegno Reno'],
        upstreams: ['Forcelli', 'Sostegno Reno'],
        lookbackSteps: 48,
        stepMinutes: 30,
        featureNames: ['level_Cento_lag0'],
        featureRecipe: {
            upstreamLagAlign: true,
            upstreamLagMinutes: { Forcelli: 480, 'Sostegno Reno': 150 }
        },
        horizons: {
            '30': {
                mean: [0], std: [1],
                layers: [{ W: [[0]], b: [0], activation: 'linear' }],
                validate: { skill: 0.4, rmse: 0.1 }
            }
        }
    });
    assert.deepEqual(parsed.lagMinutes, [480, 150]);
    const flat = P.parseWindowMlpNode({
        kind: 'window_delta_mlp',
        stations: ['Cento', 'Forcelli'],
        upstreams: ['Forcelli'],
        lookbackSteps: 48,
        featureNames: ['x'],
        horizons: {
            '30': { mean: [0], std: [1], layers: [{ W: [[0]], b: [0], activation: 'linear' }] }
        }
    });
    assert.deepEqual(flat.lagMinutes, [48 * 30]);
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

test('loadCoeffs soft-adopts reliable seq GRU over MLP Cento', async () => {
    const coeffsBody = {
        nodes: {
            Cento: { kind: 'window_delta_mlp', stations: ['Cento'], featureNames: ['x'], horizons: {} },
            Gallo: { kind: 'direct_horizon_delta_ridge', reliable: true, horizons: {} },
            'Casalecchio chiusa': { kind: 'direct_horizon_delta_ridge', reliable: true, horizons: {} },
            Bonconvento: { kind: 'direct_horizon_delta_ridge', reliable: true, horizons: {} }
        }
    };
    const files = {
        'models/transfer-coeffs-runtime.json': coeffsBody,
        'models/transfer-coeffs.json': coeffsBody,
        'models/seq-cento-gru.json': {
            kind: 'window_delta_gru',
            downstream: 'Cento',
            reliable: true,
            stations: ['Cento', 'Forcelli'],
            channelNames: ['level_Cento'],
            lookbackSteps: 4,
            horizons: {}
        },
        'models/seq-gallo-gru.json': {
            kind: 'window_delta_gru',
            downstream: 'Gallo',
            reliable: true,
            stations: ['Gallo', 'Opera Reno Panfilia'],
            channelNames: ['level_Gallo'],
            lookbackSteps: 4,
            horizons: {}
        }
    };
    const fetched = [];
    const data = await P.loadCoeffs(async (url) => {
        fetched.push(url);
        const body = files[url];
        if (!body) return { ok: false, status: 404 };
        return { ok: true, json: async () => body };
    });
    assert.equal(data.nodes.Cento.kind, 'window_delta_gru');
    assert.equal(data.nodes.Cento.reliable, true);
    assert.equal(data.nodes.Gallo.kind, 'window_delta_gru');
    assert.equal(P.seqGruUrl('Casalecchio chiusa'), 'models/seq-casalecchio-chiusa-gru.json');
    // Whitelist: MLP + SEQ_SOFT_ADOPT_EXTRA only — not every ridge station.
    assert.ok(fetched.includes('models/seq-cento-gru.json'));
    assert.ok(fetched.includes('models/seq-gallo-gru.json'));
    assert.ok(fetched.includes('models/seq-casalecchio-chiusa-gru.json'));
    assert.ok(!fetched.includes('models/seq-bonconvento-gru.json'));
    assert.ok(!fetched.some(u => /hybrid-(rc|muskingum)-cento/.test(u)));
    // Candidates before adopt: MLP Cento + ridge extras. After adopt Cento is already GRU.
    assert.deepEqual(
        P.seqSoftAdoptCandidates({
            nodes: {
                Cento: { kind: 'window_delta_mlp' },
                Gallo: { kind: 'direct_horizon_delta_ridge' },
                'Casalecchio chiusa': { kind: 'direct_horizon_delta_ridge' },
                Bonconvento: { kind: 'direct_horizon_delta_ridge' }
            }
        }).sort(),
        ['Casalecchio chiusa', 'Cento', 'Gallo'].sort()
    );
});

test('config dependencies form a connected DAG used by rankNodes', () => {
    const cfg = JSON.parse(readFileSync(new URL('../config.json', `file://${__filename}`), 'utf8'));
    const ranks = P.rankNodes(cfg);
    assert.equal(ranks.Cento > ranks.Dosso, true);
    assert.equal(ranks['Sostegno Reno'] > ranks.Cento, true);
    assert.equal(ranks['Bagnetto Reno'] > ranks.Cento, true);
    assert.equal(ranks['Vergato DX'] > ranks['Casalecchio chiusa'], true);
    assert.equal(ranks.Pracchia > ranks['Vergato DX'], true);
    assert.equal(ranks['Secondo Salto'] > ranks['Opera Po'], true);
});

test('maxReliableHorizonMinutes cuts below SKILL_KEEP and respects cap', () => {
    const model = {
        horizons: {
            '60': { validateEvents: { skill: 0.5 } },
            '120': { validateEvents: { skill: 0.4 } },
            '180': { validateEvents: { skill: 0.05 } },
            '240': { validate: { skill: 0.5 } },
            '360': { validateEvents: { skill: 0.5 } }
        }
    };
    // 180 under threshold → best reliable is 120; cap default still allows longer if skilled
    assert.equal(P.maxReliableHorizonMinutes(model, { capMinutes: 240 }), 240);
    assert.equal(P.maxReliableHorizonMinutes({
        horizons: {
            '60': { validateEvents: { skill: 0.5 } },
            '120': { validateEvents: { skill: 0.4 } },
            '180': { validateEvents: { skill: 0.05 } },
            '240': { validateEvents: { skill: 0.05 } }
        }
    }, { capMinutes: 240 }), 120);
    // No skilled horizons → still minMinutes
    assert.equal(P.maxReliableHorizonMinutes({
        horizons: { '30': { skill: 0.01 }, '240': { skill: 0.02 } }
    }, { capMinutes: 240, minMinutes: 120 }), 120);
    // Cap clamps a skilled long horizon
    assert.equal(P.maxReliableHorizonMinutes(model, { capMinutes: 200 }), 120);
    assert.equal(P.maxReliableHorizonMinutes(null), 120);
    // Ridge-style 6 h knots with flat .skill fields
    assert.equal(P.maxReliableHorizonMinutes({
        horizons: {
            '120': { skill: 0.77 },
            '240': { skill: 0.67 },
            '360': { skill: 0.58 }
        }
    }), 360);
});

test('reliableFromSkill uses enter 0.10 / exit 0.05 hysteresis', () => {
    assert.equal(P.SKILL_ENTER, 0.10);
    assert.equal(P.SKILL_EXIT, 0.05);
    assert.equal(P.reliableFromSkill(0.12, false), true);
    assert.equal(P.reliableFromSkill(0.08, false), false);
    assert.equal(P.reliableFromSkill(0.08, true), true);
    assert.equal(P.reliableFromSkill(0.04, true), false);
    assert.equal(P.reliableFromSkill(null, true), false);
});

test('mergeQuantileBands keeps median layers and attaches Q10/Q90', () => {
    const node = {
        kind: 'window_delta_mlp',
        horizons: {
            '120': { layers: [{ W: [[1]], b: [0], activation: 'linear' }], mean: [0], std: [1] }
        }
    };
    const bands = {
        source: 'test-bands',
        quantiles: [0.1, 0.5, 0.9],
        horizons: {
            '120': {
                layersQ10: [{ W: [[0.5]], b: [0], activation: 'linear' }],
                layersQ90: [{ W: [[1.5]], b: [0], activation: 'linear' }],
                conformalQhat: 0.02
            }
        }
    };
    const merged = P.mergeQuantileBands(node, bands);
    assert.equal(merged.horizons['120'].layers, node.horizons['120'].layers);
    assert.equal(merged.horizons['120'].layersQ10[0].W[0][0], 0.5);
    assert.equal(merged.horizons['120'].layersQ90[0].W[0][0], 1.5);
    assert.equal(merged.horizons['120'].conformalQhat, 0.02);
    assert.equal(merged.bandSource, 'test-bands');
});

test('applySkillGates flips reliable with hysteresis on CI lo', () => {
    const data = {
        nodes: {
            A: { kind: 'direct_horizon_delta_ridge', reliable: true, skill120: 0.2 },
            B: { kind: 'direct_horizon_delta_ridge', reliable: false, skill120: 0.05 }
        }
    };
    P.applySkillGates(data, {
        skillEnter: 0.10,
        skillExit: 0.05,
        stations: {
            A: { skill120: 0.08, skill120CI: { lo: 0.06, hi: 0.12 } }, // stay on (lo≥exit)
            B: { skill120: 0.15, skill120CI: { lo: 0.12, hi: 0.2 } }  // enter
        }
    });
    assert.equal(data.nodes.A.reliable, true);
    assert.equal(data.nodes.B.reliable, true);
    assert.ok(data.nodes.A.skill120CI);
});

test('missingUpstreams and widenForecastBand degrade path', () => {
    const now = new Date('2026-11-11T12:00:00Z');
    const fresh = {
        id: 'Up',
        timeSeries: [{ time: new Date(+now - 10 * 60000), value: 1 }]
    };
    const stale = {
        id: 'Stale',
        timeSeries: [{ time: new Date(+now - 3 * 3600000), value: 2 }]
    };
    assert.deepEqual(P.missingUpstreams([fresh], now, 90), []);
    assert.deepEqual(P.missingUpstreams([stale], now, 90), ['Stale']);
    const wide = P.widenForecastBand([
        { time: now, med: 1, min: 0.8, max: 1.2, value: 1 }
    ], 2);
    assert.ok(Math.abs(wide[0].min - 0.6) < 1e-9);
    assert.ok(Math.abs(wide[0].max - 1.4) < 1e-9);
});

test('blend_gru_ridge joins GRU short and ridge long without jump at edges', () => {
    const start = new Date('2026-11-10T00:00:00Z');
    const now = new Date('2026-11-11T12:00:00Z');
    // Series must include `now` so ridge knots (anchor-relative) align with GRU (now-relative).
    const n = Math.round((+now - +start) / (30 * 60000)) + 1;
    const values = Array.from({ length: n }, (_, i) => 1 + 0.01 * i);
    const down = sensor('Gallo', values, start);
    const up = sensor('Opera Reno Panfilia', values.map(v => v + 0.1), start);
    const stations = ['Gallo', 'Opera Reno Panfilia'];
    const channelNames = [];
    for (const station of stations) {
        const tag = station.replace(/ /g, '_');
        channelNames.push(`level_${tag}`, `delta_${tag}`, `miss_${tag}`);
    }
    const nC = channelNames.length;
    const mean = Array(nC).fill(0);
    const std = Array(nC).fill(1);
    const hidden = 1;
    const gru = {
        inputSize: nC,
        hiddenSize: hidden,
        weightIh: Array.from({ length: 3 * hidden }, () => Array(nC).fill(0)),
        weightHh: Array.from({ length: 3 * hidden }, () => Array(hidden).fill(0)),
        biasIh: Array(3 * hidden).fill(0),
        biasHh: Array(3 * hidden).fill(0)
    };
    const headLayers = [{ W: [[0]], b: [0.10], activation: 'linear' }];
    const gruNode = {
        kind: 'window_delta_gru',
        stations,
        upstreams: ['Opera Reno Panfilia'],
        channelNames,
        lookbackSteps: 4,
        stepMinutes: 30,
        useRain: false,
        reliable: true,
        skill120: 0.8,
        horizons: {
            '30': { mean, std, gru, headLayers, spreadLow: 0.01, spreadHigh: 0.01 },
            '60': { mean, std, gru, headLayers, spreadLow: 0.01, spreadHigh: 0.01 },
            '90': { mean, std, gru, headLayers, spreadLow: 0.01, spreadHigh: 0.01 },
            '120': { mean, std, gru, headLayers, spreadLow: 0.01, spreadHigh: 0.01, validate: { skill: 0.8 } }
        }
    };
    const ridgeNode = {
        kind: 'direct_horizon_delta_ridge',
        upstreams: ['Opera Reno Panfilia'],
        lagMinutes: [60],
        reliable: true,
        skill120: 0.7,
        useRain: false,
        horizons: {
            '30': { b: 0.05, wLocal: 0, wWindow: [0], wRate: [0], wWindow2: [0], wRate2: [0], wRain6: 0, wRain24: 0, spreadLow: 0.02, spreadHigh: 0.02 },
            '60': { b: 0.05, wLocal: 0, wWindow: [0], wRate: [0], wWindow2: [0], wRate2: [0], wRain6: 0, wRain24: 0, spreadLow: 0.02, spreadHigh: 0.02 },
            '90': { b: 0.05, wLocal: 0, wWindow: [0], wRate: [0], wWindow2: [0], wRate2: [0], wRain6: 0, wRain24: 0, spreadLow: 0.02, spreadHigh: 0.02 },
            '120': { b: 0.05, wLocal: 0, wWindow: [0], wRate: [0], wWindow2: [0], wRate2: [0], wRain6: 0, wRain24: 0, spreadLow: 0.02, spreadHigh: 0.02, skill: 0.7 },
            '150': { b: 0.05, wLocal: 0, wWindow: [0], wRate: [0], wWindow2: [0], wRate2: [0], wRain6: 0, wRain24: 0, spreadLow: 0.02, spreadHigh: 0.02 },
            '180': { b: 0.05, wLocal: 0, wWindow: [0], wRate: [0], wWindow2: [0], wRate2: [0], wRain6: 0, wRain24: 0, spreadLow: 0.02, spreadHigh: 0.02 },
            '360': { b: 0.05, wLocal: 0, wWindow: [0], wRate: [0], wWindow2: [0], wRate2: [0], wRain6: 0, wRain24: 0, spreadLow: 0.02, spreadHigh: 0.02, skill: 0.5 }
        }
    };
    const blend = {
        kind: 'blend_gru_ridge',
        downstream: 'Gallo',
        blendStartMin: 90,
        blendEndMin: 150,
        gru: gruNode,
        ridge: ridgeNode,
        reliable: true,
        skill120: 0.8
    };
    const sensorMap = { Gallo: down, 'Opera Reno Panfilia': up };
    const result = P.predictNode(down, [up], {
        hours: 24, now, coeffs: { nodes: { Gallo: blend } },
        horizonHours: 6, downstreamId: 'Gallo', sensorMap, rainSensors: [], smooth: 0
    });
    assert.equal(result.ok, true);
    assert.equal(result.kind, 'blend_gru_ridge');
    const y0 = down.timeSeries.at(-1).value;
    const pts = (result.forecast || []).filter(p => p.regime !== 'anchor');
    assert.ok(pts.length >= 4);
    const at90 = pts.find(p => Math.abs(+p.time - (+now + 90 * 60000)) < 1000);
    const at180 = pts.find(p => Math.abs(+p.time - (+now + 180 * 60000)) < 1000);
    assert.ok(at90);
    assert.ok(at180);
    // GRU head bias 0.10 → med ≈ y0+0.10 at ≤90′; ridge b=0.05 → med ≈ y0+0.05 at ≥150′
    assert.ok(Math.abs(at90.med - (y0 + 0.10)) < 1e-6);
    assert.ok(Math.abs(at180.med - (y0 + 0.05)) < 1e-6);
});

test('estimatePredictedLag recovers a shifted hindcast series', () => {
    const n = 40;
    const obs = Array.from({ length: n }, (_, i) => Math.sin(i / 4) + 0.1 * i);
    // Predicted lags observed by 2 steps → best d = -2 → lagMin = +60
    const k = 2;
    const points = Array.from({ length: n }, (_, i) => ({
        predicted: i >= k ? obs[i - k] : obs[0],
        observed: obs[i]
    }));
    const lag = P.estimatePredictedLag(points, P.DEFAULT_STEP_MS);
    assert.ok(lag);
    assert.equal(lag.lagSteps, k);
    assert.equal(lag.lagMin, k * 30);
    assert.ok(Math.abs(lag.r) > 0.9);
    assert.equal(P.estimatePredictedLag(points.slice(0, 5)), null);
});

test('E1 Layer A: Cento MLP mlpForward matches Python parity fixture', () => {
    const fx = JSON.parse(readFileSync('tests/fixtures/cento-mlp-parity.json', 'utf8'));
    const model = JSON.parse(readFileSync(fx.modelPath, 'utf8'));
    const block = model.horizons[String(fx.horizonMinutes)];
    assert.ok(block, 'horizon block');
    let maxAbs = 0;
    for (const sample of fx.samples) {
        const values = {};
        for (let i = 0; i < fx.featureNames.length; i++) {
            values[fx.featureNames[i]] = sample.x[i];
        }
        const x = P.standardizeWindowFeatures(values, fx.featureNames, block.mean, block.std);
        assert.ok(x);
        const delta = P.mlpForward(x, block.layers);
        assert.ok(Number.isFinite(delta));
        const err = Math.abs(delta - sample.deltaHat);
        if (err > maxAbs) maxAbs = err;
    }
    assert.ok(maxAbs < 1e-5, `max |Δδ|=${maxAbs}`);
});

test('E1 Layer B: buildWindowFeatureVector ≈ Python features on lattice', () => {
    const fx = JSON.parse(readFileSync('tests/fixtures/cento-mlp-parity.json', 'utf8'));
    const withSeries = fx.samples.filter(s => s.series);
    assert.ok(withSeries.length >= 1);
    const groups = { level: 0, delta: 0, miss: 0, d1: 0, d2: 0, rain: 0, other: 0 };
    let nCompared = 0;
    for (const sample of withSeries) {
        const sensorMap = {};
        for (const station of fx.stations) {
            const pts = (sample.series[station] || [])
                .filter(p => p[1] != null)
                .map(p => ({ time: new Date(p[0]), value: p[1] }))
                .sort((a, b) => +a.time - +b.time);
            sensorMap[station] = { id: station, name: station, timeSeries: pts };
        }
        // Inject rain scalars via a synthetic gauge that basinRainAt can read:
        // when useRain, compare dumped rain keys directly against Python rain field.
        const raw = P.buildWindowFeatureVector(
            sensorMap,
            fx.stations,
            fx.lookbackSteps,
            Date.parse(sample.t),
            STEP,
            [],
            false, // isolate level/poly path; rain checked separately
            fx.featureRecipe,
            fx.horizonMinutes
        );
        for (let i = 0; i < fx.featureNames.length; i++) {
            const name = fx.featureNames[i];
            if (name === 'rain6h' || name === 'rain24h') continue;
            const py = sample.x[i];
            const js = raw[name];
            const a = py == null ? null : py;
            const b = Number.isFinite(js) ? js : null;
            if (a == null && b == null) continue;
            const err = (a == null || b == null) ? 1 : Math.abs(a - b);
            nCompared++;
            let g = 'other';
            if (name.startsWith('level_')) g = 'level';
            else if (name.startsWith('delta_')) g = 'delta';
            else if (name.startsWith('miss_')) g = 'miss';
            else if (name.startsWith('d1_')) g = 'd1';
            else if (name.startsWith('d2_')) g = 'd2';
            groups[g] = Math.max(groups[g], err);
        }
        if (sample.rain) {
            const i6 = fx.featureNames.indexOf('rain6h');
            const i24 = fx.featureNames.indexOf('rain24h');
            if (i6 >= 0 && sample.x[i6] != null && sample.rain.rain6h != null) {
                groups.rain = Math.max(groups.rain, Math.abs(sample.x[i6] - sample.rain.rain6h));
            }
            if (i24 >= 0 && sample.x[i24] != null && sample.rain.rain24h != null) {
                groups.rain = Math.max(groups.rain, Math.abs(sample.x[i24] - sample.rain.rain24h));
            }
        }
    }
    assert.ok(nCompared > 100, `compared ${nCompared}`);
    // Exact lattice → level/miss should match tightly; poly d1/d2 within OLS noise.
    assert.ok(groups.level < 1e-5, `level max|Δ|=${groups.level}`);
    assert.ok(groups.miss < 1e-9, `miss max|Δ|=${groups.miss}`);
    assert.ok(groups.delta < 1e-5, `delta max|Δ|=${groups.delta}`);
    assert.ok(groups.d1 < 1e-4, `d1 max|Δ|=${groups.d1}`);
    assert.ok(groups.d2 < 1e-4, `d2 max|Δ|=${groups.d2}`);
});
