(function(root) {
    'use strict';
    const HOUR = 3600000;
    const MAX_GAP = HOUR;
    const CACHE_KEY = 'reno-observations-v1';
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const romeFormatter = new Intl.DateTimeFormat('sv-SE', {
        timeZone: 'Europe/Rome', year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23'
    });

    function parseSensorData(id, item) {
        if (!item?.anagrafica || !item.dati || typeof item.dati !== 'object') return null;
        const meta = item.anagrafica;
        const coords = meta.geometry?.coordinates;
        const coordinates = Array.isArray(coords) && finite(coords[0]) && finite(coords[1]) &&
            Math.abs(coords[0]) <= 180 && Math.abs(coords[1]) <= 90 ? [coords[1], coords[0]] : null;
        const thresholds = meta.sensori?.livello_idro?.soglie || [];
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
                timeSeries.push({ time, value: finite(reading?.livello_idro) ? reading.livello_idro : null });
            }
        }
        timeSeries.sort((a, b) => a.time - b.time);
        return {
            id, name: typeof meta.nome === 'string' ? meta.nome : id,
            basin: typeof meta.sottobacino === 'string' ? meta.sottobacino : '', coordinates,
            levels: Object.fromEntries(['yellow', 'orange', 'red'].map((key, i) => [key, finite(thresholds[i]) ? thresholds[i] : null])),
            timeSeries
        };
    }

    function latestPoint(sensor) {
        return sensor?.timeSeries?.findLast(point => finite(point.value)) || null;
    }

    function status(sensor) {
        const point = latestPoint(sensor);
        if (!point) return { key: 'missing', label: 'Nessuna misura valida', color: '#64748b' };
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
        return Math.max(0, ...Object.values(sensor.levels).filter(finite));
    }

    function normalize(sensor, value) {
        const threshold = highestThreshold(sensor);
        return finite(value) && threshold > 0 ? value / threshold : null;
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
            const entries = Object.entries(config.sensors);
            const byId = new Map(entries.map(([name, cfg]) => [String(cfg.arpaeId), name]));
            const items = {};
            const url = new URL('https://apps.arpae.it/REST/meteo_osservati/');
            url.searchParams.set('where', JSON.stringify({ _id: { $in: [...byId.keys()] } }));
            url.searchParams.set('max_results', String(Math.min(100, entries.length)));
            try {
                let page = 1, received = 0;
                while (page <= Math.ceil(entries.length / 25) + 1) {
                    url.searchParams.set('page', String(page));
                    const payload = await json(url.href);
                    if (!Array.isArray(payload?._items)) throw new Error('Risposta ARPAE non valida');
                    for (const item of payload._items) {
                        const name = byId.get(String(item?._id));
                        if (name && parseSensorData(name, item)) items[name] = item;
                    }
                    received += payload._items.length;
                    const total = payload._meta?.total;
                    const pageSize = payload._meta?.max_results || Math.min(100, entries.length);
                    if (!finite(total) || received >= total || payload._items.length < pageSize || Object.keys(items).length === entries.length) break;
                    page++;
                }
            } catch (error) {
                if (error.name !== 'AbortError') console.warn('Recupero aggregato ARPAE non disponibile:', error.message);
            }
            const missing = entries.filter(([name]) => !items[name]);
            let cursor = 0;
            await Promise.all(Array.from({ length: Math.min(3, missing.length) }, async () => {
                while (cursor < missing.length) {
                    const [name, cfg] = missing[cursor++];
                    try {
                        const payload = await json(cfg.url);
                        const item = payload?._items?.find(value => String(value?._id) === String(cfg.arpaeId));
                        if (parseSensorData(name, item)) items[name] = item;
                    } catch (error) {
                        if (error.name !== 'AbortError') console.warn(`Stazione ${name} non disponibile:`, error.message);
                    }
                }
            }));
            return { items, failed: entries.filter(([name]) => !items[name]).map(([name]) => name) };
        }
        return { json, load };
    }

    function readCache(storage, config, now = Date.now()) {
        try {
            const cached = JSON.parse(storage?.getItem(CACHE_KEY) || 'null');
            if (cached?.version !== 1 || !cached.records) return {};
            return Object.fromEntries(Object.entries(cached.records).filter(([name, record]) =>
                config.sensors[name] && record && finite(record.fetchedAt) && now - record.fetchedAt >= 0 && now - record.fetchedAt < 24 * HOUR &&
                String(record.item?._id) === String(config.sensors[name]?.arpaeId) && parseSensorData(name, record.item)));
        } catch {
            return {};
        }
    }

    function saveCache(storage, records) {
        try {
            storage?.setItem(CACHE_KEY, JSON.stringify({ version: 1, records }));
        } catch {}
    }

    const api = { parseSensorData, latestPoint, status, isStale, highestThreshold, normalize, filter, weightedAverage, withGaps, romeTime, createClient, readCache, saveCache };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.RenoData = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
