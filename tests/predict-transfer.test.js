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
