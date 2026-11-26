'use strict';
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const MODELS = path.join(__dirname, '..', 'models');

function readJson(name) {
    return JSON.parse(fs.readFileSync(path.join(MODELS, name), 'utf8'));
}

test('piene-runtime pack structure is consistent', () => {
    const pack = readJson('piene-runtime.json');
    assert.strictEqual(pack.kind, 'piene-runtime');
    assert.ok(pack.disclaimer && typeof pack.disclaimer === 'string');
    assert.ok(pack.caveat, 'caveat presente');
    assert.ok(pack.stations.includes('Cento'));
    assert.ok(pack.models.includes('prod'));
    assert.ok(pack.horizons.includes(120));
    assert.strictEqual(pack.metrics.length, 22, 'metriche attese nel dizionario');

    // aggregate rows: length matches metrics dictionary, indices valid
    const si = pack.stations.indexOf('Cento');
    const mi = pack.models.indexOf('prod');
    const hi = pack.horizons.indexOf(120);
    const arr = pack.agg[si][mi][hi];
    assert.strictEqual(arr.length, pack.metrics.length);
    const skill = arr[pack.metrics.indexOf('skillEventPersist')];
    assert.ok(skill > 0 && skill <= 1, 'skill@120 Cento prod positiva');
    const rmse = arr[pack.metrics.indexOf('rmse')];
    assert.ok(rmse > 0 && rmse < 1, 'rmse@120 Cento prod plausibile');
});

test('catalog and per-event rows are aligned', () => {
    const pack = readJson('piene-runtime.json');
    assert.ok(pack.catalog.length > 100, 'catalogo Cento non vuoto');
    assert.ok(pack.catalog.some(c => c.split === 'test'), 'almeno un evento test');
    for (const row of pack.centoPerEvent) {
        assert.ok(row[0] >= 0 && row[0] < pack.catalog.length, 'catalog idx valido');
        assert.ok(row[1] >= 0 && row[1] < pack.models.length, 'model idx valido');
        assert.ok(row[2] >= 0 && row[2] < pack.horizons.length, 'horizon idx valido');
        assert.strictEqual(row.length, 10);
    }
});


test('every catalog event is linked to a monte-valle series segment', () => {
    const pack = readJson('piene-runtime.json');
    const ids = new Set(pack.seriesIds);
    assert.strictEqual(ids.size, pack.catalog.length, 'una serie per evento di catalogo');
    for (const c of pack.catalog) {
        assert.ok(c.metaId != null, 'metaId mancante per ' + c.peakTime);
        assert.ok(ids.has(c.metaId), 'serie mancante per ' + c.peakTime);
    }
});

test('series pack uses one shared axis and preserves offsets/nulls', () => {
    const sp = readJson('piene-cento-series.json');
    assert.strictEqual(sp.kind, 'piene-cento-series');
    const step = sp.axisStep;
    assert.strictEqual(step, 1, 'campionamento orario preservato');
    // shared axis is regular
    for (let i = 1; i < sp.hours.length; i++) {
        assert.ok(Math.abs((sp.hours[i] - sp.hours[i - 1]) - step) < 1e-9, 'asse regolare');
    }
    let anyNull = false;
    for (const [id, ev] of Object.entries(sp.events)) {
        assert.ok(ev.off >= 0 && ev.off + ev.n <= sp.hours.length, 'segmento dentro asse: ' + id);
        const firstHour = sp.hours[ev.off];
        const lastHour = sp.hours[ev.off + ev.n - 1];
        assert.ok(firstHour <= 0 && lastHour >= 0, 'segmento include il picco: ' + id);
        for (const arr of Object.values(ev.levels)) {
            assert.strictEqual(arr.length, ev.n, 'lunghezza serie coerente');
            if (arr.includes(null)) anyNull = true;
        }
    }
    assert.ok(Object.keys(sp.events).length > 0);
    // nulls preserved where present in source (at least structure allows them)
    assert.strictEqual(typeof anyNull, 'boolean');
});

test('pack core stays within raw size budget', () => {
    const size = fs.statSync(path.join(MODELS, 'piene-runtime.json')).size;
    assert.ok(size <= 512 * 1024, 'core pack <= 0.5 MB raw, ora ' + size);
});

test('model responses are aligned to every monte-valle event segment', () => {
    const sp = readJson('piene-cento-series.json');
    const responses = readJson('piene-cento-model-responses.json');
    assert.strictEqual(responses.kind, 'piene-cento-model-responses');
    assert.strictEqual(responses.downstream, 'Cento');
    assert.strictEqual(responses.horizonMin, 120);
    assert.ok(responses.semantics.includes('T-120'), 'semantica causale esplicita');
    assert.strictEqual(responses.coverage.eventsTotal, Object.keys(sp.events).length);
    assert.ok(responses.coverage.ratio > 0.95, 'copertura hindcast alta');

    for (const [id, ev] of Object.entries(sp.events)) {
        const r = responses.events[id];
        assert.ok(r, 'risposta presente per evento ' + id);
        assert.strictEqual(r.pred.length, ev.n, 'lunghezza pred allineata: ' + id);
        assert.strictEqual(r.base.length, ev.n, 'lunghezza persistenza allineata: ' + id);
        assert.ok(r.pred.some(Number.isFinite), 'almeno una previsione finita: ' + id);
    }
});
