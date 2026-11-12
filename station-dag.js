(function (root) {
    'use strict';

    const DAG_COLUMN = {
        'Sasso Marconi': 0, 'Casalecchio chiusa': 0, 'Casalecchio tiro a volo': 0,
        'Bonconvento': 0, 'Sostegno Reno': 0, 'Gallo': 0,
        'Vergato DX': 1, 'Lavino di Sotto': 1, 'Cento': 1, 'Dosso': 1,
        'Opera Reno Panfilia': 1, 'Secondo Salto': 1,
        'Paltrone Samoggia': 2, 'Forcelli': 2
    };

    /**
     * Render the station dependency chain into `container`.
     * options: {
     *   config, buildGraph, rankNodes, getSensor, blankSensor,
     *   status(sensor), latestValue(sensor), formatNumber(v),
     *   onSelect(sensorId), debounce(fn, ms)
     * }
     */
    function renderStationDag(container, options = {}) {
        const config = options.config;
        if (!container || !config) return;
        const P = {
            buildGraph: options.buildGraph,
            rankNodes: options.rankNodes
        };
        if (typeof P.buildGraph !== 'function' || typeof P.rankNodes !== 'function') return;

        if (!container._dagBound && typeof options.onSelect === 'function') {
            container._dagBound = true;
            const select = id => options.onSelect(id);
            container.addEventListener('click', evt => {
                const node = evt.target.closest('[data-sensor]');
                if (!node) return;
                select(node.getAttribute('data-sensor'));
            });
            container.addEventListener('keydown', evt => {
                if (evt.key !== 'Enter' && evt.key !== ' ') return;
                const node = evt.target.closest('[data-sensor]');
                if (!node) return;
                evt.preventDefault();
                select(node.getAttribute('data-sensor'));
            });
            if (typeof options.debounce === 'function') {
                window.addEventListener('resize', options.debounce(() => {
                    renderStationDag(container, options);
                }, 200));
            }
        }

        const svgNS = 'http://www.w3.org/2000/svg';
        const graph = P.buildGraph(config);
        const ranks = P.rankNodes(config);
        const maxRank = Math.max(0, ...Object.values(ranks));
        const rows = Array.from({ length: maxRank + 1 }, () => []);
        for (const id of graph.nodes) rows[maxRank - (ranks[id] || 0)].push(id);
        rows.forEach(row => row.sort());

        const getSensor = options.getSensor || (() => null);
        const blankSensor = options.blankSensor || (id => ({ id, name: id, levels: {}, timeSeries: [] }));
        const statusFn = options.status || (() => ({ color: '#94a3b8', label: 'n/d' }));
        const latestValue = options.latestValue || (() => null);
        const fmt = options.formatNumber || (v => String(v));

        const meta = new Map();
        for (const id of graph.nodes) {
            const sensor = getSensor(id) || blankSensor(id);
            meta.set(id, { sensor, status: statusFn(sensor), latest: latestValue(sensor) });
        }

        const ctx = renderStationDag.ctx || (renderStationDag.ctx = document.createElement('canvas').getContext('2d'));
        const valueText = id => {
            const latest = meta.get(id).latest;
            return latest === null || latest === undefined ? '—' : `${fmt(latest)} m`;
        };
        const PADX = 5, DOT = 12, H = 42, ROWH = 58, COLGAP = 10, TOP = 10, BOTTOM = 10;
        const containerWidth = Math.max(container.clientWidth || 0, 240);
        const columns = [[], [], []];
        for (const id of graph.nodes) columns[DAG_COLUMN[id] ?? 1].push(id);

        const wrapText = (text, maxW) => {
            if (Math.ceil(ctx.measureText(text).width) <= maxW) return [text];
            const lines = [];
            let current = '';
            for (const word of text.split(' ')) {
                const candidate = current ? `${current} ${word}` : word;
                if (current && Math.ceil(ctx.measureText(candidate).width) > maxW) {
                    lines.push(current);
                    current = word;
                } else current = candidate;
            }
            if (current) lines.push(current);
            if (lines.length > 2) lines[1] = `${lines[1]} ${lines.splice(2).join(' ')}`;
            return lines.slice(0, 2);
        };

        let widthOf, colW, needed, nameFontPx = 11, nameLines = new Map();
        for (const px of [11, 10]) {
            ctx.font = `600 ${px}px system-ui, -apple-system, "Segoe UI", sans-serif`;
            const maxName = Math.round(64 * 11 / px);
            const lines = new Map();
            const nw = new Map();
            for (const id of graph.nodes) {
                const wrapped = wrapText(id, maxName);
                lines.set(id, wrapped);
                nw.set(id, Math.max(...wrapped.map(line => Math.ceil(ctx.measureText(line).width))));
            }
            ctx.font = `600 ${Math.max(9, px - 1)}px system-ui, -apple-system, "Segoe UI", sans-serif`;
            const vw = new Map(graph.nodes.map(id => [id, Math.ceil(ctx.measureText(valueText(id)).width)]));
            const wOf = id => PADX * 2 + Math.max(DOT + nw.get(id), vw.get(id));
            const cW = [0, 1, 2].map(c => Math.max(0, ...columns[c].map(wOf)));
            const total = cW[0] + cW[1] + cW[2] + 2 * COLGAP;
            if (total <= containerWidth || px === 10) {
                widthOf = wOf; colW = cW; needed = total; nameFontPx = px; nameLines = lines;
                break;
            }
        }
        const width = Math.max(containerWidth, needed, 240);
        const colX = [];
        let bandX = (width - needed) / 2;
        for (let c = 0; c < 3; c++) { colX[c] = bandX; bandX += colW[c] + COLGAP; }

        const pos = new Map();
        rows.forEach((row, ri) => {
            const y = TOP + ri * ROWH;
            for (const id of row) {
                const c = DAG_COLUMN[id] ?? 1;
                const w = widthOf(id);
                pos.set(id, { x: colX[c] + (colW[c] - w) / 2, y, w, h: H });
            }
        });
        const height = TOP + Math.max(0, rows.length - 1) * ROWH + H + BOTTOM;

        const svg = document.createElementNS(svgNS, 'svg');
        svg.setAttribute('width', width);
        svg.setAttribute('height', height);
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

        for (const edge of graph.edges) {
            const from = pos.get(edge.upstream), to = pos.get(edge.downstream);
            if (!from || !to) continue;
            const x1 = from.x + from.w / 2, y1 = from.y + from.h;
            const x2 = to.x + to.w / 2, y2 = to.y;
            const midY = (y1 + y2) / 2;
            const g = document.createElementNS(svgNS, 'g');
            g.setAttribute('class', 'dag-edge');
            g.setAttribute('aria-hidden', 'true');
            const path = document.createElementNS(svgNS, 'path');
            path.setAttribute('d', `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2 - 4}`);
            path.setAttribute('fill', 'none');
            path.setAttribute('stroke', '#9bb0ba');
            path.setAttribute('stroke-width', '1.5');
            const arrow = document.createElementNS(svgNS, 'path');
            arrow.setAttribute('d', `M ${x2 - 3.5} ${y2 - 4} L ${x2 + 3.5} ${y2 - 4} L ${x2} ${y2 - 0.5} Z`);
            arrow.setAttribute('fill', '#9bb0ba');
            g.append(path, arrow);
            svg.appendChild(g);
        }

        for (const [id, p] of pos) {
            const m = meta.get(id);
            const statusColor = m.status.color;
            const label = m.latest === null || m.latest === undefined ? 'nessuna misura' : `${fmt(m.latest)} m`;
            const g = document.createElementNS(svgNS, 'g');
            g.setAttribute('class', 'dag-node');
            g.setAttribute('data-sensor', id);
            g.setAttribute('tabindex', '0');
            g.setAttribute('role', 'button');
            g.setAttribute('aria-label', `${id}: ${label} · ${m.status.label}. Vai alla scheda della stazione.`);
            const title = document.createElementNS(svgNS, 'title');
            title.textContent = `${id} · ${label} · ${m.status.label}`;
            const rect = document.createElementNS(svgNS, 'rect');
            rect.setAttribute('x', p.x); rect.setAttribute('y', p.y);
            rect.setAttribute('width', p.w); rect.setAttribute('height', p.h);
            rect.setAttribute('rx', p.h / 2);
            rect.setAttribute('fill', '#ffffff');
            rect.setAttribute('stroke', '#d8e2e6');
            rect.setAttribute('stroke-width', '1.25');
            const textX = p.x + PADX + DOT;
            const lines = nameLines.get(id) || [id];
            const nameCys = lines.length === 1 ? [p.y + 16] : [p.y + 10, p.y + 21];
            const dot = document.createElementNS(svgNS, 'circle');
            dot.setAttribute('cx', p.x + PADX + 3.5);
            dot.setAttribute('cy', nameCys[0]);
            dot.setAttribute('r', '4');
            dot.setAttribute('fill', statusColor);
            const texts = lines.map((line, i) => {
                const name = document.createElementNS(svgNS, 'text');
                name.setAttribute('x', textX);
                name.setAttribute('y', nameCys[i]);
                name.setAttribute('dominant-baseline', 'central');
                name.setAttribute('font-size', nameFontPx);
                name.setAttribute('font-weight', '600');
                name.setAttribute('fill', '#173747');
                name.textContent = line;
                return name;
            });
            const value = document.createElementNS(svgNS, 'text');
            value.setAttribute('x', textX);
            value.setAttribute('y', p.y + H - 9);
            value.setAttribute('dominant-baseline', 'central');
            value.setAttribute('font-size', Math.max(9, nameFontPx - 1));
            value.setAttribute('font-weight', '600');
            value.setAttribute('fill', m.latest === null || m.latest === undefined ? '#94a3b8' : statusColor);
            value.textContent = valueText(id);
            g.append(title, rect, dot, ...texts, value);
            svg.appendChild(g);
        }
        container.replaceChildren(svg);
    }

    const api = { DAG_COLUMN, renderStationDag };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.RenoStationDag = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
