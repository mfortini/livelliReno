const { test } = require('node:test');
const assert = require('node:assert/strict');
const D = require('../dashboard-data.js');
const { readFileSync } = require('node:fs');
const { Script } = require('node:vm');

function item(id = '1') {
    return {
        _id: id,
        anagrafica: { nome: 'Cento', geometry: { coordinates: [11.3, 44.7] }, sensori: { livello_idro: { soglie: [5.5, 7, 8.7] } } },
        dati: { '20261111': { '1500': { livello_idro: -1.2 }, '1530': { livello_idro: 0 }, '1600': { livello_idro: null } } }
    };
}

function sensor(values, levels = { yellow: 1, orange: 2, red: 3 }) {
    return { levels, timeSeries: values.map((value, i) => ({ time: new Date(Date.UTC(2026, 10, 11, 12, i * 30)), value })) };
}

test('parsing preserves negative levels and nulls without mutating coordinates', () => {
    const raw = item();
    const original = structuredClone(raw);
    const parsed = D.parseSensorData('Cento', raw);
    assert.deepEqual(raw, original);
    assert.deepEqual(parsed.coordinates, [44.7, 11.3]);
    assert.deepEqual(parsed.timeSeries.map(p => p.value), [-1.2, 0, null]);
    assert.equal(parsed.timeSeries[0].time.toISOString(), '2026-11-11T15:00:00.000Z');
    assert.equal(D.latestPoint(parsed).time.toISOString(), '2026-11-11T15:30:00.000Z');
    assert.equal(D.latestPoint(parsed).value, 0);
});

test('invalid structure, coordinates, timestamps and non-numeric readings are handled', () => {
    assert.equal(D.parseSensorData('Cento', null), null);
    const raw = item();
    raw.anagrafica.geometry.coordinates = [200, 95];
    raw.dati['20261111']['1700'] = { livello_idro: '4' };
    raw.dati['20261111']['9999'] = { livello_idro: 4 };
    raw.dati['20260230'] = { '1200': { livello_idro: 5 } };
    const parsed = D.parseSensorData('Cento', raw);
    assert.equal(parsed.coordinates, null);
    assert.equal(parsed.timeSeries.length, 4);
    assert.equal(parsed.timeSeries.at(-1).value, null);
});

test('partial thresholds never act as zero and equality counts as exceeded', () => {
    assert.equal(D.status(sensor([9.56], { yellow: 10.5, orange: null, red: null })).key, 'normal');
    assert.equal(D.status(sensor([11], { yellow: 10.5, orange: null, red: null })).key, 'yellow');
    assert.equal(D.status(sensor([1.01], { yellow: null, orange: 3, red: 5 })).key, 'unknown');
    assert.equal(D.status(sensor([3], { yellow: null, orange: 3, red: 5 })).key, 'orange');
    assert.equal(D.status(sensor([-1], { yellow: 0, orange: 2, red: 3 })).key, 'normal');
    assert.equal(D.status(sensor([0], { yellow: 0, orange: 2, red: 3 })).key, 'yellow');
    assert.equal(D.status(sensor([null])).key, 'missing');
    assert.equal(D.status(sensor([4], { yellow: null, orange: null, red: null })).key, 'unknown');
});

test('freshness is separate from hydrometric state and future values are not fresh', () => {
    const s = sensor([4]);
    const now = new Date('2026-11-11T14:01:00Z');
    assert.equal(D.status(s).key, 'red');
    assert.equal(D.isStale(s, now, 90), true);
    assert.equal(D.isStale(s, new Date('2026-11-11T12:30:00Z'), 90), false);
    assert.equal(D.isStale(s, new Date('2026-11-10T12:00:00Z'), 90), true);
});

test('normalization preserves null, supports negatives and requires a positive threshold', () => {
    const s = sensor([]);
    assert.equal(D.normalize(s, null), null);
    assert.equal(D.normalize(s, -3), -1);
    assert.equal(D.normalize(sensor([], { yellow: null, orange: null, red: null }), 4), null);
    assert.equal(D.normalize(sensor([], { yellow: 0, orange: null, red: null }), 4), null);
});

test('weighted averages ignore nulls and large gaps, including normalized values', () => {
    const s = sensor([2, 4, null, 8]);
    assert.equal(D.weightedAverage(s, s.timeSeries, false), 3);
    assert.equal(D.weightedAverage(s, s.timeSeries, true), 1);
    assert.equal(D.weightedAverage(sensor([]), [], false), null);
    assert.equal(D.weightedAverage(sensor([1]), sensor([1]).timeSeries, false), null);
    const gap = [{ time: new Date(0), value: 2 }, { time: new Date(4 * 3600000), value: 4 }];
    assert.equal(D.weightedAverage(s, gap, false), null);
    assert.equal(D.withGaps(gap).length, 3);
    assert.equal(D.withGaps(gap)[1].value, null);
});

test('time window uses absolute instants, excludes future samples and works across DST', () => {
    const s = { timeSeries: ['2026-10-24T01:00:00Z', '2026-10-25T01:00:00Z', '2026-10-26T01:00:00Z'].map(t => ({ time: new Date(t), value: 1 })) };
    assert.equal(D.filter(s, 24, new Date('2026-10-25T01:00:00Z')).length, 2);
    assert.equal(D.romeTime(new Date('2026-10-25T00:30:00Z')), '2026-10-25 02:30:00');
    assert.equal(D.romeTime(new Date('2026-10-25T01:30:00Z')), '2026-10-25 02:30:00');
});

const config = { sensors: { Cento: { arpaeId: '1', url: 'https://example.test/1' }, Forcelli: { arpaeId: '2', url: 'https://example.test/2' } } };
const response = data => ({ ok: true, json: async () => data });

test('batch maps results by ID, independently of order, using one request', async () => {
    const calls = [];
    const client = D.createClient(async url => { calls.push(url); return response({ _items: [item('2'), item('1')], _meta: { total: 2 } }); });
    const result = await client.load(config);
    assert.equal(calls.length, 1);
    assert.equal(result.items.Cento._id, '1');
    assert.equal(result.items.Forcelli._id, '2');
    assert.equal(result.failed.length, 0);
    assert.deepEqual(JSON.parse(new URL(calls[0]).searchParams.get('where'))._id.$in, ['1', '2']);
});

test('incomplete batch falls back only for missing station', async () => {
    const calls = [];
    const client = D.createClient(async url => {
        calls.push(url);
        return response(url.includes('example.test/2') ? { _items: [item('2')] } : { _items: [item('1')], _meta: { total: 2 } });
    });
    const result = await client.load(config);
    assert.equal(calls.length, 2);
    assert.equal(result.items.Forcelli._id, '2');
});

test('HTTP errors, malformed payloads and failed fallback leave explicit missing results', async () => {
    const client = D.createClient(async () => ({ ok: false, status: 503 }));
    assert.deepEqual((await client.load(config)).failed, ['Cento', 'Forcelli']);
    const malformed = D.createClient(async () => response({ unexpected: [] }));
    assert.equal((await malformed.load(config)).failed.length, 2);
});

test('a malformed station is retried individually and a wrong fallback ID is rejected', async () => {
    const client = D.createClient(async url => response(url.includes('example.test') ? { _items: [item('wrong')] } : { _items: [{ _id: '1' }, item('2')], _meta: { total: 2 } }));
    const result = await client.load(config);
    assert.deepEqual(result.failed, ['Cento']);
    assert.equal(result.items.Forcelli._id, '2');
});

test('paginated results are collected without individually re-fetching stations', async () => {
    const calls = [];
    const client = D.createClient(async url => {
        calls.push(url);
        return response({ _items: [item(new URL(url).searchParams.get('page') === '2' ? '2' : '1')], _meta: { total: 2, max_results: 1 } });
    });
    const result = await client.load(config);
    assert.equal(calls.length, 2);
    assert.equal(result.failed.length, 0);
});

test('fallback concurrency is bounded to three requests', async () => {
    const many = { sensors: Object.fromEntries(Array.from({ length: 8 }, (_, i) => [`s${i}`, { arpaeId: String(i), url: `https://example.test/${i}` }])) };
    let active = 0, maximum = 0;
    const client = D.createClient(async url => {
        if (!url.includes('example.test')) return response({ _items: [] });
        maximum = Math.max(maximum, ++active);
        await new Promise(resolve => setTimeout(resolve, 2));
        active--;
        return response({ _items: [item(url.split('/').at(-1))] });
    });
    assert.equal((await client.load(many)).failed.length, 0);
    assert.equal(maximum, 3);
});

test('inline application compiles and all configured IDs are unique and match fallback URLs', () => {
    const html = readFileSync(new URL('../index.html', `file://${__filename}`), 'utf8');
    for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) assert.doesNotThrow(() => new Script(match[1]));
    assert.doesNotMatch(html, /id="dependency-graph"/);
    assert.doesNotMatch(html, /<script[^>]+cytoscape/);
    assert.ok(html.indexOf('<details id="map-container"') < html.indexOf('<main id="main-content"'));
    const cfg = JSON.parse(readFileSync(new URL('../config.json', `file://${__filename}`), 'utf8'));
    const ranks = require('../predict-transfer.js').rankNodes(cfg);
    const ordered = Object.keys(cfg.sensors).sort((a, b) => ranks[b] - ranks[a]);
    for (const [downstream, sensor] of Object.entries(cfg.sensors)) for (const upstream of sensor.dependencies) assert.ok(ordered.indexOf(upstream) < ordered.indexOf(downstream));
    const ids = Object.values(cfg.sensors).map(sensor => sensor.arpaeId);
    assert.equal(new Set(ids).size, ids.length);
    for (const sensor of Object.values(cfg.sensors)) assert.equal(JSON.parse(new URL(sensor.url).searchParams.get('where'))._id, sensor.arpaeId);
});

test('request timeout aborts its fetch', async () => {
    const client = D.createClient((url, options) => new Promise((resolve, reject) => {
        options.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
    }), 5);
    await assert.rejects(client.json('https://example.test/timeout'), /aborted/);
});

test('cache validates version, age, station ID and survives unavailable storage', () => {
    const now = Date.now();
    const records = { Cento: { item: item('1'), fetchedAt: now } };
    let value;
    const storage = { getItem: () => value, setItem: (key, content) => { value = content; } };
    D.saveCache(storage, records);
    assert.equal(D.readCache(storage, config, now).Cento.item._id, '1');
    assert.deepEqual(D.readCache(storage, config, now + 25 * 3600000), {});
    assert.deepEqual(D.readCache(storage, { sensors: { Cento: { arpaeId: '99' } } }, now), {});
    value = '{broken';
    assert.deepEqual(D.readCache(storage, config, now), {});
    const unavailable = { getItem() { throw Error('denied'); }, setItem() { throw Error('full'); } };
    assert.deepEqual(D.readCache(unavailable, config, now), {});
    assert.doesNotThrow(() => D.saveCache(unavailable, records));
});
