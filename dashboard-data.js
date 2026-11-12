(function(root) {
    'use strict';
    const HOUR = 3600000;
    const MAX_GAP = HOUR;
    const CACHE_KEY = 'reno-observations-v2';
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const RAIN_VALUE_KEYS = ['precipitazione_cumulata_1h', 'precipitazione', 'precipitazione_cumulata'];

    function rainfallGauges(config) {
        const gauges = config?.rainfall?.gauges || [];
        return gauges.map(entry => {
            if (typeof entry === 'string') return { name: entry, impacts: [], dagColumn: 1 };
            if (!entry || typeof entry.name !== 'string') return null;
            return {
                name: entry.name,
                arpaeId: entry.arpaeId != null ? String(entry.arpaeId) : null,
                url: typeof entry.url === 'string' ? entry.url : null,
                impacts: Array.isArray(entry.impacts) ? entry.impacts.slice() : [],
                dagColumn: Number.isFinite(Number(entry.dagColumn)) ? Number(entry.dagColumn) : 1
            };
        }).filter(Boolean);
    }

    function readingValue(reading, kind) {
        if (!reading || typeof reading !== 'object') return null;
        if (kind === 'rain') {
            for (const key of RAIN_VALUE_KEYS) {
                if (finite(reading[key])) return reading[key];
            }
            return null;
        }
        return finite(reading.livello_idro) ? reading.livello_idro : null;
    }
    const romeFormatter = new Intl.DateTimeFormat('sv-SE', {
        timeZone: 'Europe/Rome', year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23'
    });

    function parseSensorData(id, item, options = {}) {
        if (!item?.anagrafica || !item.dati || typeof item.dati !== 'object') return null;
        const kind = options.kind === 'rain' ? 'rain' : 'level';
        const meta = item.anagrafica;
        const coords = meta.geometry?.coordinates;
        const coordinates = Array.isArray(coords) && finite(coords[0]) && finite(coords[1]) &&
            Math.abs(coords[0]) <= 180 && Math.abs(coords[1]) <= 90 ? [coords[1], coords[0]] : null;
        const thresholds = kind === 'rain' ? [] : (meta.sensori?.livello_idro?.soglie || []);
        const timeSeries = [];
        for (const [date, times] of Object.entries(item.dati)) {
            if (!/^\d{8}$/.test(date) || !times || typeof times !== 'object') continue;
            for (const [clock, reading] of Object.entries(times)) {
                if (!/^\d{4}$/.test(clock)) continue;
                const year = Number(date.slice(0, 4)), month = Number(date.slice(4, 6)), day = Number(date.slice(6, 8));
                const hour = Number(clock.slice(0, 2)), minute = Number(clock.slice(2));
                const time = new Date(Date.UTC(year, month - 1, day, hour, minute));
                if (time.getUTCFullYear() !== year || time.getUTCMonth() !== month - 1 ||
                    time.getUTCDate() !== day || hour > 23 || minute > 59) continue;
                timeSeries.push({ time, value: readingValue(reading, kind) });
            }
        }
        timeSeries.sort((a, b) => a.time - b.time);
        // Rain series must contain at least one finite precip reading.
        if (kind === 'rain' && !timeSeries.some(point => finite(point.value))) return null;
        return {
            id, name: typeof meta.nome === 'string' ? meta.nome : id,
            kind,
            unit: kind === 'rain' ? 'mm' : 'm',
            basin: typeof meta.sottobacino === 'string' ? meta.sottobacino : '', coordinates,
            levels: Object.fromEntries(['yellow', 'orange', 'red'].map((key, i) => [key, finite(thresholds[i]) ? thresholds[i] : null])),
            impacts: Array.isArray(options.impacts) ? options.impacts.slice() : [],
            dagColumn: Number.isFinite(Number(options.dagColumn)) ? Number(options.dagColumn) : 1,
            timeSeries
        };
    }

    function latestPoint(sensor) {
        return sensor?.timeSeries?.findLast(point => finite(point.value)) || null;
    }

    function status(sensor) {
        const point = latestPoint(sensor);
        if (!point) return { key: 'missing', label: 'Nessuna misura valida', color: '#64748b' };
        if (sensor?.kind === 'rain') {
            return { key: 'rain', label: 'Precipitazione', color: '#1d6f9f' };
        }
        for (const [key, label, color] of [
            ['red', 'Soglia 3 superata', '#b42332'], ['orange', 'Soglia 2 superata', '#a64b00'], ['yellow', 'Soglia 1 superata', '#806000']
        ]) {
            if (finite(sensor.levels[key]) && point.value >= sensor.levels[key]) return { key, label, color };
        }
        return finite(sensor.levels.yellow)
            ? { key: 'normal', label: 'Sotto soglia 1', color: '#13745a' }
            : { key: 'unknown', label: 'Soglie incomplete', color: '#64748b' };
    }

    function isStale(sensor, now = new Date(), minutes = 90) {
        const point = latestPoint(sensor);
        const age = point ? now - point.time : Infinity;
        return age > minutes * 60000 || age < -5 * 60000;
    }

    function highestThreshold(sensor) {
        return Math.max(0, ...Object.values(sensor?.levels || {}).filter(finite));
    }

    // Yellow soglia is present on all level stations; preferred reference (as in ML features).
    function yellowThreshold(sensor) {
        const yellow = sensor?.levels?.yellow;
        return finite(yellow) && yellow > 0 ? yellow : null;
    }

    // Scarto rispetto al giallo: (livello − giallo) / giallo.
    // 0 ⇒ al giallo; <0 ⇒ sotto (più sicuro); >0 ⇒ sopra (verso arancio/rosso).
    function normalize(sensor, value) {
        const yellow = yellowThreshold(sensor);
        return finite(value) && yellow != null ? (value - yellow) / yellow : null;
    }

    function filter(sensor, hours, now = new Date()) {
        return sensor.timeSeries.filter(point => point.time >= now - hours * HOUR && point.time <= now);
    }

    function weightedAverage(sensor, points, normalized) {
        let sum = 0, duration = 0;
        for (let i = 1; i < points.length; i++) {
            const previous = points[i - 1], current = points[i];
            const dt = current.time - previous.time;
            const a = normalized ? normalize(sensor, previous.value) : previous.value;
            const b = normalized ? normalize(sensor, current.value) : current.value;
            if (!finite(a) || !finite(b) || dt <= 0 || dt > MAX_GAP) continue;
            sum += (a + b) / 2 * dt;
            duration += dt;
        }
        return duration ? sum / duration : null;
    }

    function withGaps(points) {
        const result = [];
        points.forEach((point, i) => {
            if (i && point.time - points[i - 1].time > MAX_GAP) {
                result.push({ time: new Date(+points[i - 1].time + 1), value: null });
            }
            result.push(point);
        });
        return result;
    }

    function romeTime(date) {
        const p = Object.fromEntries(romeFormatter.formatToParts(date).map(part => [part.type, part.value]));
        return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second}`;
    }

    function thresholdShapes(sensor, { normalized = false } = {}) {
        const shapes = [];
        const yellow = yellowThreshold(sensor);
        const colors = { yellow: '#a68000', orange: '#be5900', red: '#b42332' };
        for (const key of ['yellow', 'orange', 'red']) {
            const raw = sensor?.levels?.[key];
            if (!finite(raw)) continue;
            let y = raw;
            if (normalized) {
                if (yellow == null) continue;
                y = (raw - yellow) / yellow;
            }
            shapes.push({
                type: 'line', xref: 'paper', x0: 0, x1: 1,
                y0: y, y1: y,
                line: { color: colors[key], width: 1, dash: 'dash' }
            });
        }
        return shapes;
    }

    function averageShape(value) {
        if (!finite(value)) return null;
        return {
            type: 'line', xref: 'paper', x0: 0, x1: 1, y0: value, y1: value,
            line: { color: '#64748b', width: 1.5, dash: 'dot' }
        };
    }

    function yRangeFor(sensor, values, { normalized = false, padding = 0.5 } = {}) {
        const finiteValues = (values || []).filter(finite);
        const minValue = finiteValues.length ? Math.min(...finiteValues) : 0;
        const maxValue = finiteValues.length ? Math.max(...finiteValues) : 1;
        if (normalized) {
            const yellow = yellowThreshold(sensor);
            const extras = [0, -1];
            if (yellow != null) {
                for (const key of ['orange', 'red']) {
                    const raw = sensor?.levels?.[key];
                    if (finite(raw)) extras.push((raw - yellow) / yellow);
                }
            }
            const lo = Math.min(minValue, ...extras);
            const hi = Math.max(maxValue, ...extras);
            return [lo - 0.05, hi + 0.05];
        }
        const thresholds = ['yellow', 'orange', 'red']
            .map(k => sensor?.levels?.[k])
            .filter(finite);
        const lowest = thresholds.length ? Math.min(...thresholds) : null;
        const highest = thresholds.length ? Math.max(...thresholds) : null;
        const yMin = Math.min(-1, minValue - padding, lowest != null ? lowest - padding : minValue - padding);
        const yMax = Math.max(maxValue + padding, highest != null ? highest + padding : maxValue + padding);
        return [yMin, yMax];
    }

    function romeTickVals(startMs, endMs, tickCount = 4) {
        const span = endMs - startMs;
        const ticks = Array.from({ length: tickCount }, (_, i) => startMs + i * span / (tickCount - 1));
        const texts = ticks.map(t => {
            const r = romeTime(new Date(t));
            return `${r.slice(8, 10)}/${r.slice(5, 7)}<br>${r.slice(11, 16)}`;
        });
        return { tickvals: ticks, ticktext: texts };
    }

    function createClient(fetcher, timeout = 15000) {
        async function json(url) {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), timeout);
            try {
                const response = await fetcher(url, { signal: controller.signal });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return await response.json();
            } finally {
                clearTimeout(timer);
            }
        }

        async function load(config) {
            const levelEntries = Object.entries(config.sensors || {}).map(([name, cfg]) => ({
                name, arpaeId: String(cfg.arpaeId), url: cfg.url, kind: 'level'
            }));
            const rainEntries = rainfallGauges(config)
                .filter(g => g.arpaeId && g.url)
                .map(g => ({ name: g.name, arpaeId: String(g.arpaeId), url: g.url, kind: 'rain', impacts: g.impacts, dagColumn: g.dagColumn }));
            const entries = [...levelEntries, ...rainEntries];
            const metaByName = Object.fromEntries(entries.map(e => [e.name, e]));
            const byId = new Map();
            for (const entry of entries) {
                if (!byId.has(entry.arpaeId)) byId.set(entry.arpaeId, []);
                byId.get(entry.arpaeId).push(entry.name);
            }
            const items = {};
            const kinds = {};
            const url = new URL('https://apps.arpae.it/REST/meteo_osservati/');
            url.searchParams.set('where', JSON.stringify({ _id: { $in: [...byId.keys()] } }));
            url.searchParams.set('max_results', String(Math.min(100, byId.size)));
            const parseOk = (name, item) => {
                const meta = metaByName[name];
                return parseSensorData(name, item, {
                    kind: meta?.kind,
                    impacts: meta?.impacts,
                    dagColumn: meta?.dagColumn
                });
            };
            try {
                let page = 1, received = 0;
                while (page <= Math.ceil(byId.size / 25) + 1) {
                    url.searchParams.set('page', String(page));
                    const payload = await json(url.href);
                    if (!Array.isArray(payload?._items)) throw new Error('Risposta ARPAE non valida');
                    for (const item of payload._items) {
                        const names = byId.get(String(item?._id)) || [];
                        for (const name of names) {
                            if (parseOk(name, item)) {
                                items[name] = item;
                                kinds[name] = metaByName[name].kind;
                            }
                        }
                    }
                    received += payload._items.length;
                    const total = payload._meta?.total;
                    const pageSize = payload._meta?.max_results || Math.min(100, byId.size);
                    if (!finite(total) || received >= total || payload._items.length < pageSize || Object.keys(items).length === entries.length) break;
                    page++;
                }
            } catch (error) {
                if (error.name !== 'AbortError') console.warn('Recupero aggregato ARPAE non disponibile:', error.message);
            }
            const missing = entries.filter(entry => !items[entry.name]);
            let cursor = 0;
            await Promise.all(Array.from({ length: Math.min(3, missing.length) }, async () => {
                while (cursor < missing.length) {
                    const entry = missing[cursor++];
                    try {
                        const payload = await json(entry.url);
                        const item = payload?._items?.find(value => String(value?._id) === String(entry.arpaeId));
                        if (parseOk(entry.name, item)) {
                            items[entry.name] = item;
                            kinds[entry.name] = entry.kind;
                        }
                    } catch (error) {
                        if (error.name !== 'AbortError') console.warn(`Stazione ${entry.name} non disponibile:`, error.message);
                    }
                }
            }));
            return {
                items,
                kinds,
                failed: entries.filter(entry => !items[entry.name]).map(entry => entry.name)
            };
        }
        return { json, load };
    }

    function readCache(storage, config, now = Date.now()) {
        try {
            const cached = JSON.parse(storage?.getItem(CACHE_KEY) || 'null');
            if (cached?.version !== 2 || !cached.records) return {};
            const rainByName = Object.fromEntries(rainfallGauges(config).map(g => [g.name, g]));
            return Object.fromEntries(Object.entries(cached.records).filter(([name, record]) => {
                if (!record || !finite(record.fetchedAt) || now - record.fetchedAt < 0 || now - record.fetchedAt >= 24 * HOUR) return false;
                if (config.sensors[name]) {
                    return String(record.item?._id) === String(config.sensors[name]?.arpaeId)
                        && parseSensorData(name, record.item, { kind: 'level' });
                }
                const gauge = rainByName[name];
                if (!gauge?.arpaeId) return false;
                return String(record.item?._id) === String(gauge.arpaeId)
                    && parseSensorData(name, record.item, { kind: 'rain', impacts: gauge.impacts, dagColumn: gauge.dagColumn });
            }));
        } catch {
            return {};
        }
    }

    function saveCache(storage, records) {
        try {
            storage?.setItem(CACHE_KEY, JSON.stringify({ version: 2, records }));
        } catch {}
    }

    const api = {
        parseSensorData, rainfallGauges, latestPoint, status, isStale, highestThreshold, yellowThreshold, normalize,
        filter, weightedAverage, withGaps, romeTime, thresholdShapes, averageShape, yRangeFor, romeTickVals,
        createClient, readCache, saveCache
    };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.RenoData = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
