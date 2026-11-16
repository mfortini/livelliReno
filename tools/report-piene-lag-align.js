#!/usr/bin/env node
/**
 * Report: tutte le piene Cento con serie monte allineate (τ globali, fino a 2 hop).
 * Scrive models/report-piene-lag-align.{json,html} e stampa path canvas-ready.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const STEP_MS = 30 * 60 * 1000;
const STEP_MIN = 30;
const MAX_LAG_STEPS = 96; // 48 h
const SAMPLE_EVERY = 2; // 1 h categories for charts
const PAD_H = 12;
const WINDOW_H = 48; // ± around peak if event is shorter/longer clamp

const DOWN = 'Cento';

function loadCsv(name) {
    const p = path.join(ROOT, 'storico', `${name}.csv`);
    const text = fs.readFileSync(p, 'utf8');
    const raw = new Map();
    for (const line of text.split(/\r?\n/)) {
        const row = line.trim();
        if (!row || row.startsWith('time')) continue;
        const c = row.indexOf(',');
        if (c < 0) continue;
        const t = Date.parse(row.slice(0, c));
        const v = Number(row.slice(c + 1));
        if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
        raw.set(t, v);
    }
    // Resample to 30′ grid (storico may be 15′); nearest within ±15′.
    const times = [...raw.keys()].sort((a, b) => a - b);
    if (!times.length) return new Map();
    const byTime = new Map();
    const t0 = Math.ceil(times[0] / STEP_MS) * STEP_MS;
    const t1 = Math.floor(times[times.length - 1] / STEP_MS) * STEP_MS;
    let j = 0;
    for (let t = t0; t <= t1; t += STEP_MS) {
        while (j + 1 < times.length && Math.abs(times[j + 1] - t) <= Math.abs(times[j] - t)) j++;
        const nearest = times[j];
        if (Math.abs(nearest - t) <= 15 * 60000) byTime.set(t, raw.get(nearest));
    }
    return byTime;
}

function coverageGaps(byTime, minGapHours = 24) {
    const times = [...byTime.keys()].sort((a, b) => a - b);
    const gaps = [];
    for (let i = 1; i < times.length; i++) {
        const dt = times[i] - times[i - 1];
        if (dt >= minGapHours * 3600000) {
            gaps.push({
                from: new Date(times[i - 1]).toISOString(),
                to: new Date(times[i]).toISOString(),
                days: Math.round((dt / 86400000) * 10) / 10
            });
        }
    }
    return gaps;
}

function hopsUpTo(config, station, maxHop) {
    const sensors = config.sensors || {};
    const out = new Map(); // name -> hop
    const q = [[station, 0]];
    const seen = new Set([station]);
    while (q.length) {
        const [cur, hop] = q.shift();
        if (hop >= maxHop) continue;
        const deps = sensors[cur]?.dependencies || [];
        for (const d of deps) {
            if (!sensors[d] || seen.has(d)) continue;
            seen.add(d);
            out.set(d, hop + 1);
            q.push([d, hop + 1]);
        }
    }
    return out;
}

function pearson(xs, ys) {
    const n = xs.length;
    if (n < 40) return null;
    let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
        sx += xs[i]; sy += ys[i];
        sxx += xs[i] * xs[i]; syy += ys[i] * ys[i];
        sxy += xs[i] * ys[i];
    }
    const den = Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    if (!(den > 0)) return null;
    return (n * sxy - sx * sy) / den;
}

/** lag>0: y lags x (onda x→y). Stima su train (< testFrom). */
function estimateTau(upBy, downBy, testFromMs, maxLagSteps) {
    const pairs = [];
    for (const [t, yv] of downBy) {
        if (t >= testFromMs) continue;
        const xv = upBy.get(t);
        if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
        pairs.push([t, xv, yv]);
    }
    if (pairs.length < 200) return { lagSteps: null, lagMin: null, r: null, n: pairs.length };

    let bestLag = 0, bestR = -Infinity;
    for (let lag = 0; lag <= maxLagSteps; lag++) {
        const xs = [], ys = [];
        for (const [t, xv] of pairs) {
            const yv = downBy.get(t + lag * STEP_MS);
            if (!Number.isFinite(yv)) continue;
            xs.push(xv);
            ys.push(yv);
        }
        const r = pearson(xs, ys);
        if (r != null && r > bestR) {
            bestR = r;
            bestLag = lag;
        }
    }
    return {
        lagSteps: bestLag,
        lagMin: bestLag * STEP_MIN,
        r: bestR === -Infinity ? null : Math.round(bestR * 1000) / 1000,
        n: pairs.length
    };
}

/** Peak lag per evento: argmax up in [peak-maxLag, peak], vs Cento peak. */
function eventPeakLag(upBy, peakMs, peakLevel, maxLagSteps) {
    let bestT = null, bestV = -Infinity;
    for (let k = 0; k <= maxLagSteps; k++) {
        const t = peakMs - k * STEP_MS;
        const v = upBy.get(t);
        if (!Number.isFinite(v)) continue;
        if (v > bestV) {
            bestV = v;
            bestT = t;
        }
    }
    if (bestT == null) return { lagMin: null, upPeak: null };
    return {
        lagMin: Math.round((peakMs - bestT) / 60000),
        upPeak: Math.round(bestV * 100) / 100
    };
}

function sampleEvent(seriesMap, tauMinMap, startMs, endMs, peakMs) {
    const t0 = Math.min(startMs - PAD_H * 3600000, peakMs - WINDOW_H * 3600000);
    const t1 = Math.max(endMs + PAD_H * 3600000, peakMs + WINDOW_H * 3600000);
    const hours = [];
    const levels = {};
    for (const name of Object.keys(seriesMap)) levels[name] = [];

    for (let t = t0; t <= t1; t += STEP_MS * SAMPLE_EVERY) {
        const h = Math.round(((t - peakMs) / 3600000) * 10) / 10;
        hours.push(h);
        for (const [name, byTime] of Object.entries(seriesMap)) {
            const tauMs = (tauMinMap[name] || 0) * 60000;
            // Shift upstream forward by τ so wave aligns with Cento time.
            const srcT = name === DOWN ? t : t - tauMs;
            const v = byTime.get(srcT) ?? byTime.get(srcT - STEP_MS) ?? byTime.get(srcT + STEP_MS);
            levels[name].push(Number.isFinite(v) ? Math.round(v * 100) / 100 : null);
        }
    }
    return { hours, levels };
}

function main() {
    const config = JSON.parse(fs.readFileSync(path.join(ROOT, 'config.json'), 'utf8'));
    const eventsDoc = JSON.parse(fs.readFileSync(path.join(ROOT, 'models/events.json'), 'utf8'));
    const testFrom = Date.parse(eventsDoc.testFrom || '2026-01-01T00:00:00.000Z');

    const hopMap = hopsUpTo(config, DOWN, 2);
    const ups = [...hopMap.entries()].sort((a, b) => a[1] - b[1] || a[0].localeCompare(b[0]));
    const stationNames = [DOWN, ...ups.map(([n]) => n)];

    console.error(`[report] stazioni: ${stationNames.join(', ')}`);
    const series = {};
    const gapsByStation = {};
    for (const n of stationNames) {
        console.error(`[report] load ${n}…`);
        series[n] = loadCsv(n);
        const gaps = coverageGaps(series[n], 7 * 24);
        gapsByStation[n] = gaps;
        if (gaps.length) {
            console.error(
                `[report] ${n}: ${gaps.length} gap≥7g (max ${Math.max(...gaps.map(g => g.days))}d)`
            );
        }
    }

    const tau = {};
    tau[DOWN] = { lagSteps: 0, lagMin: 0, r: 1, n: null, hop: 0 };
    for (const [name, hop] of ups) {
        const est = estimateTau(series[name], series[DOWN], testFrom, MAX_LAG_STEPS);
        tau[name] = { ...est, hop };
        console.error(
            `[report] τ(${name}→${DOWN}) hop=${hop} ${est.lagMin}′ r=${est.r} n=${est.n}`
        );
    }
    const tauMinMap = Object.fromEntries(
        Object.entries(tau).map(([k, v]) => [k, v.lagMin || 0])
    );

    const centoEvents = (eventsDoc.events || []).filter(e => e.station === DOWN);
    const events = centoEvents.map((e, idx) => {
        const startMs = Date.parse(e.start);
        const endMs = Date.parse(e.end);
        const peakMs = Date.parse(e.peakTime);
        const sampled = sampleEvent(series, tauMinMap, startMs, endMs, peakMs);
        const perEventLag = {};
        for (const [name] of ups) {
            perEventLag[name] = eventPeakLag(series[name], peakMs, e.peakLevel, MAX_LAG_STEPS);
        }
        const split = peakMs >= testFrom ? 'test' : 'train';
        return {
            id: idx,
            label: `${e.peakTime.slice(0, 10)} · picco ${e.peakLevel} m · Δ${e.riseM} m`,
            start: e.start,
            end: e.end,
            peakTime: e.peakTime,
            peakLevel: e.peakLevel,
            riseM: e.riseM,
            split,
            perEventLag,
            hours: sampled.hours,
            levels: sampled.levels
        };
    });

    const out = {
        version: 1,
        kind: 'cento_flood_lag_align_2hop',
        generatedAt: new Date().toISOString(),
        disclaimer: 'Sperimentale: allineamento τ cross-corr train; non allertistico.',
        downstream: DOWN,
        maxHop: 2,
        stepMinutes: STEP_MIN,
        sampleMinutes: STEP_MIN * SAMPLE_EVERY,
        testFrom: new Date(testFrom).toISOString(),
        stations: stationNames.map(n => ({
            name: n,
            hop: n === DOWN ? 0 : hopMap.get(n),
            tauMin: tau[n].lagMin,
            tauR: tau[n].r,
            nPoints: series[n].size,
            gaps: gapsByStation[n]
        })),
        tau,
        gapsByStation,
        nEvents: events.length,
        nTest: events.filter(e => e.split === 'test').length,
        events
    };

    const jsonPath = path.join(ROOT, 'models/report-piene-lag-align.json');
    fs.writeFileSync(jsonPath, JSON.stringify(out));
    console.error(`[report] wrote ${jsonPath} (${(fs.statSync(jsonPath).size / 1e6).toFixed(2)} MB)`);

    const htmlPath = path.join(ROOT, 'models/report-piene-lag-align.html');
    fs.writeFileSync(htmlPath, buildHtml(out));
    console.error(`[report] wrote ${htmlPath}`);
}

function buildHtml(doc) {
    const stationOrder = doc.stations.map(s => s.name);
    const tauRows = doc.stations
        .map(s =>
            `<tr><td>${esc(s.name)}</td><td>${s.hop}</td><td>${s.tauMin ?? 'n/d'}</td><td>${s.tauR ?? 'n/d'}</td></tr>`
        )
        .join('');
    // Embed slim payload: drop null-heavy by keeping as-is; Plotly in page.
    const payload = JSON.stringify({
        stations: stationOrder,
        tau: doc.tau,
        events: doc.events.map(e => ({
            id: e.id,
            label: e.label,
            split: e.split,
            peakTime: e.peakTime,
            peakLevel: e.peakLevel,
            riseM: e.riseM,
            perEventLag: e.perEventLag,
            hours: e.hours,
            levels: e.levels
        }))
    });

    return `<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Piene Cento — lag allineati (2 hop)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg:#f7f5f1; --ink:#1c1b19; --muted:#5c574f; --line:#d9d3c8; --card:#fff; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.45 system-ui,sans-serif; color:var(--ink); background:var(--bg); }
  header { padding:1.25rem 1.5rem; border-bottom:1px solid var(--line); background:var(--card); position:sticky; top:0; z-index:2; }
  h1 { margin:0 0 .35rem; font-size:1.35rem; }
  .meta { color:var(--muted); font-size:.9rem; }
  main { padding:1rem 1.5rem 3rem; max-width:1100px; margin:0 auto; }
  table { border-collapse:collapse; width:100%; margin:1rem 0 1.5rem; background:var(--card); }
  th,td { border:1px solid var(--line); padding:.4rem .55rem; text-align:left; font-size:.9rem; }
  th { background:#efebe3; }
  .toolbar { display:flex; flex-wrap:wrap; gap:.75rem; align-items:center; margin:1rem 0; }
  .toolbar label { color:var(--muted); font-size:.9rem; }
  .event { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:.75rem 1rem 0; margin:0 0 1rem; }
  .event h2 { margin:0 0 .25rem; font-size:1.05rem; }
  .event .sub { color:var(--muted); font-size:.85rem; margin-bottom:.5rem; }
  .plot { width:100%; height:340px; }
  .note { font-size:.85rem; color:var(--muted); margin-top:1rem; }
</style>
</head>
<body>
<header>
  <h1>Piene Cento — livelli monte allineati (≤2 hop)</h1>
  <div class="meta">τ da cross-corr sui mesi train (&lt; ${esc(doc.testFrom.slice(0,10))}).
  Upstream shiftati avanti di τ. Campione ${doc.sampleMinutes}′ · ${doc.nEvents} eventi (${doc.nTest} test).
  ${esc(doc.disclaimer)}
  Gap storici ≥7g: ${esc(
    doc.stations
      .filter(s => (s.gaps || []).length)
      .map(s => s.name + ' ' + s.gaps.map(g => g.days + 'd').join(','))
      .join(' · ') || 'nessuno'
  )}.</div>
</header>
<main>
  <h2>Ritardi globali τ (train)</h2>
  <table>
    <thead><tr><th>Stazione</th><th>Hop</th><th>τ (′)</th><th>r</th></tr></thead>
    <tbody>${tauRows}</tbody>
  </table>
  <div class="toolbar">
    <label>Filtro <select id="split"><option value="all">tutti</option><option value="test">solo test 2026+</option><option value="train">solo train</option></select></label>
    <label>Scala <select id="scale"><option value="raw">livello (m)</option><option value="norm">normalizzato [0,1] su finestra</option></select></label>
    <span class="meta" id="count"></span>
  </div>
  <div id="events"></div>
  <p class="note">Asse X: ore rispetto al picco Cento. Serie monte: x_allineata(t)=x(t−τ).
  Lag per-evento in didascalia = ritardo picco monte→picco Cento (argmax su [picco−48h, picco]).</p>
</main>
<script>
const DATA = ${payload};
const colors = {
  Cento: '#1c1b19',
  Forcelli: '#0b6e4f',
  'Bagnetto Reno': '#1d4e89',
  'Sostegno Reno': '#3d5a80',
  'Lavino di Sotto': '#b35c00',
  'Paltrone Samoggia': '#8b3a62'
};
function normSeries(arr) {
  const vals = arr.filter(v => Number.isFinite(v));
  if (!vals.length) return arr.map(() => null);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const d = hi - lo || 1;
  return arr.map(v => Number.isFinite(v) ? (v - lo) / d : null);
}
function render() {
  const split = document.getElementById('split').value;
  const scale = document.getElementById('scale').value;
  const root = document.getElementById('events');
  root.innerHTML = '';
  const list = DATA.events.filter(e => split === 'all' || e.split === split);
  document.getElementById('count').textContent = list.length + ' grafici';
  for (const e of list) {
    const wrap = document.createElement('section');
    wrap.className = 'event';
    const lagBits = Object.entries(e.perEventLag || {}).map(([k,v]) =>
      k + ' ' + (v.lagMin != null ? v.lagMin + "′" : 'n/d')).join(' · ');
    wrap.innerHTML = '<h2>' + e.label + (e.split === 'test' ? ' [test]' : '') + '</h2>' +
      '<div class="sub">Finestra ' + e.peakTime.slice(0,10) + ' · lag picco osservati: ' + lagBits + '</div>' +
      '<div class="plot" id="p' + e.id + '"></div>';
    root.appendChild(wrap);
    const traces = DATA.stations.map(name => {
      let y = e.levels[name] || [];
      if (scale === 'norm') y = normSeries(y);
      return {
        x: e.hours,
        y,
        name: name + (name === 'Cento' ? '' : ' (+τ ' + (DATA.tau[name]?.lagMin ?? '?') + "′)"),
        type: 'scatter',
        mode: 'lines',
        line: { width: name === 'Cento' ? 2.4 : 1.4, color: colors[name] || undefined }
      };
    });
    Plotly.newPlot('p' + e.id, traces, {
      margin: { t: 24, r: 16, b: 40, l: 48 },
      xaxis: { title: 'Ore da picco Cento', zeroline: true },
      yaxis: { title: scale === 'norm' ? 'Normalizzato' : 'Livello (m)', zeroline: false },
      legend: { orientation: 'h', y: 1.12 },
      shapes: [{ type: 'line', x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
        line: { color: '#999', width: 1, dash: 'dot' } }]
    }, { responsive: true, displayModeBar: false });
  }
}
document.getElementById('split').addEventListener('change', render);
document.getElementById('scale').addEventListener('change', render);
render();
</script>
</body>
</html>`;
}

function esc(s) {
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

main();
