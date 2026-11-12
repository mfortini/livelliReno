(function(root) {
    'use strict';
    const HOUR = 3600000;
    const MINUTE = 60000;
    const DEFAULT_STEP_MS = 30 * MINUTE;
    const WEAK_R = 0.35;
    const FORK_K = 1.0; // ±1 RMSE ≈ forchetta empirica min/max attorno alla mediana
    const MIN_STEP_ABS = 0.05; // m per mezz’ora, pavimento del rate-limit
    const SKILL_KEEP = 0.10; // allineato a tools/fit-transfer.py
    const BIAS_WINDOW_HOURS = 48;
    const BIAS_EMA_ALPHA = 0.08;
    const RAIN6_STEPS = 12;
    const RAIN24_STEPS = 48;
    const finite = value => typeof value === 'number' && Number.isFinite(value);
    const edgeKey = (upstream, downstream) => `${upstream}->${downstream}`;

    function latestFinite(series) {
        if (!series?.length) return null;
        for (let i = series.length - 1; i >= 0; i--) {
            if (finite(series[i]?.value)) return series[i];
        }
        return null;
    }

    // Max |Δy| tipico per step (p95 storico, riscalato alla griglia), per evitare salti impossibili.
    function maxStepFromSeries(series, stepMs = DEFAULT_STEP_MS) {
        const deltas = [];
        for (let i = 1; i < (series?.length || 0); i++) {
            const previous = series[i - 1], current = series[i];
            if (!finite(previous?.value) || !finite(current?.value)) continue;
            const dt = +current.time - +previous.time;
            if (dt <= 0 || dt > stepMs * 2.5) continue;
            deltas.push(Math.abs(current.value - previous.value) * (stepMs / dt));
        }
        if (!deltas.length) return null;
        deltas.sort((a, b) => a - b);
        const p95 = deltas[Math.min(deltas.length - 1, Math.floor(deltas.length * 0.95))];
        return Math.max(MIN_STEP_ABS, p95 * 1.5);
    }

    function buildGraph(config) {
        const nodes = Object.keys(config?.sensors || {});
        const edges = [];
        for (const [downstream, sensor] of Object.entries(config?.sensors || {})) {
            const deps = Array.isArray(sensor?.dependencies) ? sensor.dependencies : [];
            for (const upstream of deps) {
                if (!config.sensors[upstream]) continue;
                edges.push({ upstream, downstream, id: edgeKey(upstream, downstream) });
            }
        }
        return { nodes, edges };
    }

    function rankNodes(config) {
        const { nodes, edges } = buildGraph(config);
        const followers = Object.fromEntries(nodes.map(id => [id, []]));
        for (const edge of edges) followers[edge.upstream].push(edge.downstream);

        const ranks = Object.fromEntries(nodes.map(id => [id, 0]));
        const visiting = new Set();
        const visited = new Set();

        function dfs(sensor) {
            if (visiting.has(sensor)) return ranks[sensor] || 0;
            if (visited.has(sensor)) return ranks[sensor] || 0;
            visiting.add(sensor);
            let best = 0;
            for (const follower of followers[sensor] || []) {
                best = Math.max(best, dfs(follower) + 1);
            }
            visiting.delete(sensor);
            visited.add(sensor);
            ranks[sensor] = best;
            return best;
        }

        for (const sensor of nodes) dfs(sensor);
        // ranks: higher = more upstream (leaves downstream = 0), matching previous card layout
        return ranks;
    }

    // Le serie sono ordinate per tempo (parseSensorData le ordina): ricerca binaria del primo punto con t >= target.
    function sampleAt(series, targetMs) {
        const n = series?.length || 0;
        if (!n) return null;
        let left = 0;
        let right = n;
        while (left < right) {
            const mid = (left + right) >> 1;
            if (+series[mid].time < targetMs) left = mid + 1;
            else right = mid;
        }
        let hi = null;
        for (let i = left; i < n; i++) {
            if (finite(series[i]?.value)) { hi = series[i]; break; }
        }
        let lo = hi && +hi.time === targetMs ? hi : null;
        for (let i = left - 1; !lo && i >= 0; i--) {
            if (finite(series[i]?.value)) lo = series[i];
        }
        if (lo && hi) {
            const t0 = +lo.time;
            const t1 = +hi.time;
            if (t1 === t0) return lo.value;
            if (t1 - t0 > DEFAULT_STEP_MS * 2.5) {
                const nearest = Math.abs(t0 - targetMs) <= Math.abs(t1 - targetMs) ? lo : hi;
                return Math.abs(+nearest.time - targetMs) <= DEFAULT_STEP_MS / 2 + 1 ? nearest.value : null;
            }
            const w = (targetMs - t0) / (t1 - t0);
            return lo.value * (1 - w) + hi.value * w;
        }
        const only = lo || hi;
        if (!only) return null;
        return Math.abs(+only.time - targetMs) <= DEFAULT_STEP_MS / 2 + 1 ? only.value : null;
    }

    function alignSeries(upstream, downstream, hours, now = new Date(), stepMs = DEFAULT_STEP_MS) {
        const end = +now;
        const start = end - hours * HOUR;
        const xs = [];
        const ys = [];
        const times = [];
        for (let t = start; t <= end; t += stepMs) {
            const x = sampleAt(upstream?.timeSeries, t);
            const y = sampleAt(downstream?.timeSeries, t);
            if (!finite(x) || !finite(y)) continue;
            times.push(new Date(t));
            xs.push(x);
            ys.push(y);
        }
        return { times, xs, ys };
    }

    function pearson(xs, ys) {
        const n = Math.min(xs.length, ys.length);
        if (n < 3) return null;
        let sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;
        for (let i = 0; i < n; i++) {
            const x = xs[i], y = ys[i];
            sumX += x; sumY += y;
            sumXX += x * x; sumYY += y * y;
            sumXY += x * y;
        }
        const denom = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
        if (!finite(denom) || denom === 0) return null;
        return (n * sumXY - sumX * sumY) / denom;
    }

    function estimateLag(xs, ys, maxLagSteps) {
        let bestLag = 0;
        let bestR = null;
        const limit = Math.max(0, Math.min(maxLagSteps, Math.floor(xs.length / 3)));
        for (let lag = 0; lag <= limit; lag++) {
            const xSlice = xs.slice(0, xs.length - lag);
            const ySlice = ys.slice(lag);
            const r = pearson(xSlice, ySlice);
            if (r === null) continue;
            if (bestR === null || Math.abs(r) > Math.abs(bestR) || (Math.abs(r) === Math.abs(bestR) && lag < bestLag)) {
                bestR = r;
                bestLag = lag;
            }
        }
        return { lagSteps: bestLag, r: bestR };
    }

    function fitAffine(xs, ys) {
        const multi = fitMultiAffine([xs], ys);
        return { a: multi.a?.[0] ?? null, b: multi.b };
    }

    // Ordinary least squares: y ≈ b + a0·x0 + a1·x1 + …
    function fitMultiAffine(columns, ys) {
        const k = columns?.length || 0;
        const n = ys?.length || 0;
        if (!k || n < k + 2) return { a: null, b: null };
        for (const col of columns) if (col.length < n) return { a: null, b: null };
        const m = k + 1;
        const ata = Array.from({ length: m }, () => Array(m).fill(0));
        const aty = Array(m).fill(0);
        for (let i = 0; i < n; i++) {
            const row = [1];
            for (let j = 0; j < k; j++) row.push(columns[j][i]);
            for (let r = 0; r < m; r++) {
                aty[r] += row[r] * ys[i];
                for (let c = 0; c < m; c++) ata[r][c] += row[r] * row[c];
            }
        }
        const beta = solveSymmetric(ata, aty);
        if (!beta || beta.some(v => !finite(v))) return { a: null, b: null };
        return { b: beta[0], a: beta.slice(1) };
    }

    function solveSymmetric(matrix, vector) {
        const n = vector.length;
        const a = matrix.map((row, i) => row.slice().concat(vector[i]));
        for (let col = 0; col < n; col++) {
            let pivot = col;
            for (let row = col + 1; row < n; row++) {
                if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
            }
            if (Math.abs(a[pivot][col]) < 1e-12) return null;
            if (pivot !== col) { const tmp = a[col]; a[col] = a[pivot]; a[pivot] = tmp; }
            const div = a[col][col];
            for (let j = col; j <= n; j++) a[col][j] /= div;
            for (let row = 0; row < n; row++) {
                if (row === col) continue;
                const factor = a[row][col];
                for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
            }
        }
        return a.map(row => row[n]);
    }

    function applyAffine(x, model) {
        if (!finite(x) || !finite(model?.a) || !finite(model?.b)) return null;
        return model.a * x + model.b;
    }

    function applyMulti(xs, model, dxs = null) {
        if (!finite(model?.b) || !Array.isArray(model?.a) || !Array.isArray(xs)) return null;
        if (model.a.length !== xs.length) return null;
        let sum = model.b;
        for (let i = 0; i < xs.length; i++) {
            if (!finite(xs[i]) || !finite(model.a[i])) return null;
            sum += model.a[i] * xs[i];
        }
        if (Array.isArray(model.c) && Array.isArray(dxs) && model.c.length === dxs.length) {
            for (let i = 0; i < dxs.length; i++) {
                if (!finite(dxs[i]) || !finite(model.c[i])) return null;
                sum += model.c[i] * dxs[i];
            }
        }
        return sum;
    }

    function forkAround(med, spreadLow, spreadHigh = spreadLow) {
        if (!finite(med)) return null;
        const lo = finite(spreadLow) && spreadLow > 0 ? spreadLow : 0;
        const hi = finite(spreadHigh) && spreadHigh > 0 ? spreadHigh : lo;
        return { min: med - lo, med, max: med + hi };
    }

    function withFork(points, spreadLow, spreadHigh = spreadLow) {
        return points.map(point => {
            const fork = forkAround(point.value, spreadLow, spreadHigh);
            return fork
                ? { time: point.time, value: fork.med, min: fork.min, med: fork.med, max: fork.max }
                : { time: point.time, value: null, min: null, med: null, max: null };
        });
    }

    function pickRegime(model, driver) {
        const regimes = model?.regimes;
        if (!regimes?.base) return model;
        const thr = Number(model.highThreshold);
        const useHigh = finite(thr) && finite(driver) && driver > thr && regimes.high;
        const chosen = useHigh ? regimes.high : regimes.base;
        return {
            ...model,
            a: chosen.a,
            b: chosen.b,
            c: chosen.c,
            rmse: chosen.rmse ?? model.rmse,
            spreadLow: chosen.spreadLow ?? model.spreadLow,
            spreadHigh: chosen.spreadHigh ?? model.spreadHigh,
            activeRegime: useHigh ? 'high' : 'base'
        };
    }

    function residuals(observed, predicted) {
        const errors = [];
        for (let i = 0; i < observed.length; i++) {
            if (finite(observed[i]) && finite(predicted[i])) errors.push(predicted[i] - observed[i]);
        }
        if (!errors.length) return { mae: null, rmse: null };
        const mae = errors.reduce((s, e) => s + Math.abs(e), 0) / errors.length;
        const rmse = Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / errors.length);
        return { mae, rmse };
    }

    function parseRegimeBlock(block) {
        if (!block) return null;
        const a = (block.a || []).map(Number);
        const c = Array.isArray(block.c) ? block.c.map(Number) : a.map(() => 0);
        const b = Number(block.b);
        if (!a.length || a.some(v => !finite(v)) || !finite(b)) return null;
        if (c.length !== a.length || c.some(v => !finite(v))) return null;
        return {
            a,
            b,
            c,
            r: finite(Number(block.r)) ? Number(block.r) : null,
            rmse: finite(Number(block.rmse)) ? Number(block.rmse) : null,
            spreadLow: finite(Number(block.spreadLow)) ? Number(block.spreadLow) : null,
            spreadHigh: finite(Number(block.spreadHigh)) ? Number(block.spreadHigh) : null,
            samples: Number.isFinite(Number(block.samples)) ? Number(block.samples) : null
        };
    }

    function parseHorizonBlock(block, k) {
        if (!block) return null;
        const wLocal = Number(block.wLocal);
        const b = Number(block.b);
        const wWindow = Array.isArray(block.wWindow) ? block.wWindow.map(Number) : null;
        const wRate = Array.isArray(block.wRate) ? block.wRate.map(Number) : null;
        if (!finite(wLocal) || !finite(b) || !wWindow || !wRate) return null;
        if (wWindow.length !== k || wRate.length !== k) return null;
        if (wWindow.some(v => !finite(v)) || wRate.some(v => !finite(v))) return null;
        // Optional second lag (tau+h); default 0 for older coeffs
        let wWindow2 = Array.isArray(block.wWindow2) ? block.wWindow2.map(Number) : wWindow.map(() => 0);
        let wRate2 = Array.isArray(block.wRate2) ? block.wRate2.map(Number) : wRate.map(() => 0);
        if (wWindow2.length !== k || wRate2.length !== k) return null;
        if (wWindow2.some(v => !finite(v)) || wRate2.some(v => !finite(v))) return null;
        const wRain6 = finite(Number(block.wRain6)) ? Number(block.wRain6) : 0;
        const wRain24 = finite(Number(block.wRain24)) ? Number(block.wRain24) : 0;
        const val = block.validateEvents || block.validate || {};
        const spreadLow = finite(Number(block.spreadLow)) ? Number(block.spreadLow)
            : (finite(Number(val.spreadLow)) ? Number(val.spreadLow) : null);
        const spreadHigh = finite(Number(block.spreadHigh)) ? Number(block.spreadHigh)
            : (finite(Number(val.spreadHigh)) ? Number(val.spreadHigh) : null);
        const spreadLowHigh = finite(Number(block.spreadLowHigh)) ? Number(block.spreadLowHigh) : spreadLow;
        const spreadHighHigh = finite(Number(block.spreadHighHigh)) ? Number(block.spreadHighHigh) : spreadHigh;
        return {
            b,
            wLocal,
            wWindow,
            wRate,
            wWindow2,
            wRate2,
            wRain6,
            wRain24,
            rmse: finite(Number(val.rmse)) ? Number(val.rmse) : (finite(Number(block.rmse)) ? Number(block.rmse) : null),
            skill: finite(Number(val.skill)) ? Number(val.skill) : null,
            spreadLow,
            spreadHigh,
            spreadLowHigh,
            spreadHighHigh,
            coverage: finite(Number(val.coverage)) ? Number(val.coverage) : null
        };
    }

    function rainCumulAt(rainSeries, targetMs, steps, stepMs = DEFAULT_STEP_MS) {
        if (!rainSeries?.length || steps <= 0) return 0;
        let sum = 0;
        let n = 0;
        for (let i = 0; i < steps; i++) {
            const v = sampleAt(rainSeries, targetMs - i * stepMs);
            if (!finite(v)) continue;
            sum += v;
            n++;
        }
        return n ? sum : 0;
    }

    function basinRainAt(rainSensors, targetMs, stepMs = DEFAULT_STEP_MS) {
        const list = (rainSensors || []).filter(Boolean);
        if (!list.length) return { rain6h: 0, rain24h: 0 };
        let r6 = 0, r24 = 0;
        for (const sensor of list) {
            r6 = Math.max(r6, rainCumulAt(sensor.timeSeries, targetMs, RAIN6_STEPS, stepMs));
            r24 = Math.max(r24, rainCumulAt(sensor.timeSeries, targetMs, RAIN24_STEPS, stepMs));
        }
        return { rain6h: r6, rain24h: r24 };
    }

    function driverIsHigh(upstreams, model, nowMs, stepMs = DEFAULT_STEP_MS) {
        const thresholds = model?.thresholds;
        if (!thresholds || !Array.isArray(model?.lagMinutes)) return false;
        for (let i = 0; i < upstreams.length; i++) {
            const name = model.upstreams?.[i];
            const thr = name ? thresholds[name] : null;
            if (!thr || !finite(thr.mean) || !finite(thr.std) || !finite(thr.z85) || thr.std === 0) continue;
            const x = sampleAt(upstreams[i]?.timeSeries, nowMs - model.lagMinutes[i] * MINUTE);
            if (!finite(x)) continue;
            if ((x - thr.mean) / thr.std > thr.z85) return true;
        }
        return false;
    }

    function estimateRecentBias(upstreams, downstream, model, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        if (!model?.horizons?.['30'] || !downstream?.timeSeries?.length) return 0;
        const block = model.horizons['30'];
        const nowMs = +now;
        const start = nowMs - BIAS_WINDOW_HOURS * HOUR;
        // Subsample every 2 h: enough for slow drift, cheap enough for backtests.
        const stride = Math.max(stepMs, 2 * HOUR);
        let ema = 0;
        let count = 0;
        for (let t = start; t <= nowMs - stepMs; t += stride) {
            const y0 = sampleAt(downstream.timeSeries, t);
            const yFut = sampleAt(downstream.timeSeries, t + stepMs);
            if (!finite(y0) || !finite(yFut)) continue;
            const yPrev = sampleAt(downstream.timeSeries, t - stepMs);
            const dyLocal = finite(yPrev) ? y0 - yPrev : 0;
            let ok = true;
            const dxw = [], dxr = [], dxw2 = [], dxr2 = [];
            for (let i = 0; i < upstreams.length; i++) {
                const lagMs = model.lagMinutes[i] * MINUTE;
                const xNow = sampleAt(upstreams[i]?.timeSeries, t - lagMs);
                const xWin = sampleAt(upstreams[i]?.timeSeries, t - lagMs - stepMs);
                const xPrev = sampleAt(upstreams[i]?.timeSeries, t - lagMs - stepMs);
                const xNow2 = sampleAt(upstreams[i]?.timeSeries, t - lagMs - stepMs);
                const xWin2 = sampleAt(upstreams[i]?.timeSeries, t - lagMs - 2 * stepMs);
                const xPrev2 = sampleAt(upstreams[i]?.timeSeries, t - lagMs - 2 * stepMs);
                if (!finite(xNow) || !finite(xWin) || !finite(xNow2) || !finite(xWin2)) { ok = false; break; }
                dxw.push(xNow - xWin);
                dxr.push(finite(xPrev) ? xNow - xPrev : 0);
                dxw2.push(xNow2 - xWin2);
                dxr2.push(finite(xPrev2) ? xNow2 - xPrev2 : 0);
            }
            if (!ok) continue;
            const rain = basinRainAt(options.rainSensors, t, stepMs);
            let delta = block.b + block.wLocal * dyLocal;
            for (let i = 0; i < dxw.length; i++) {
                delta += block.wWindow[i] * dxw[i] + block.wRate[i] * dxr[i];
                delta += (block.wWindow2?.[i] || 0) * dxw2[i] + (block.wRate2?.[i] || 0) * dxr2[i];
            }
            delta += (block.wRain6 || 0) * rain.rain6h + (block.wRain24 || 0) * rain.rain24h;
            const resid = (y0 + delta) - yFut;
            ema = count === 0 ? resid : (1 - BIAS_EMA_ALPHA) * ema + BIAS_EMA_ALPHA * resid;
            count++;
        }
        return count >= 12 ? ema : 0;
    }

    function parseOfflineEdge(coeffs, upstream, downstream) {
        const edge = coeffs?.edges?.[edgeKey(upstream, downstream)];
        if (!edge) return null;
        // v3 edges are metadata only — prefer node horizons
        if (edge.kind === 'direct_horizon_delta_ridge' || (!finite(Number(edge.a)) && edge.reliable != null)) {
            return null;
        }
        const lagMinutes = Number(edge.lagMinutes);
        const a = Number(edge.a);
        const b = Number(edge.b);
        const c = finite(Number(edge.c)) ? Number(edge.c) : 0;
        if (!finite(lagMinutes) || lagMinutes < 0 || !finite(a) || !finite(b)) return null;
        return {
            upstreams: [upstream],
            lagMinutes: [lagMinutes],
            a: [a],
            b,
            c: [c],
            r: finite(Number(edge.r)) ? Number(edge.r) : null,
            rmse: finite(Number(edge.rmse)) ? Number(edge.rmse) : null,
            spreadLow: finite(Number(edge.spreadLow)) ? Number(edge.spreadLow) : null,
            spreadHigh: finite(Number(edge.spreadHigh)) ? Number(edge.spreadHigh) : null,
            kind: typeof edge.kind === 'string' ? edge.kind : null,
            source: 'offline',
            trainedAt: typeof edge.trainedAt === 'string' ? edge.trainedAt : null
        };
    }

    function parseOfflineNode(coeffs, downstreamId, upstreamIds) {
        const node = coeffs?.nodes?.[downstreamId];
        if (!node || !Array.isArray(node.upstreams) || !Array.isArray(node.lagMinutes)) return null;
        if (node.upstreams.length !== upstreamIds.length) return null;
        if (!upstreamIds.every((id, i) => id === node.upstreams[i])) return null;
        const lagMinutes = node.lagMinutes.map(Number);
        if (lagMinutes.some(v => !finite(v) || v < 0)) return null;
        const k = upstreamIds.length;

        // v3 direct-horizon delta ridge
        if (node.horizons && (node.kind === 'direct_horizon_delta_ridge' || !node.regimes)) {
            const horizons = {};
            for (const [key, block] of Object.entries(node.horizons)) {
                const parsed = parseHorizonBlock(block, k);
                if (parsed) horizons[key] = parsed;
            }
            if (!Object.keys(horizons).length) return null;
            const h120 = horizons['120'] || Object.values(horizons).at(-1);
            return {
                upstreams: node.upstreams.slice(),
                lagMinutes,
                horizons,
                reliable: node.reliable !== false && (node.skill120 == null || node.skill120 >= SKILL_KEEP),
                skill120: finite(Number(node.skill120)) ? Number(node.skill120) : (h120?.skill ?? null),
                thresholds: node.thresholds || null,
                useRain: node.useRain === true,
                r: h120?.skill ?? null,
                rmse: h120?.rmse ?? null,
                spreadLow: h120?.spreadLow ?? null,
                spreadHigh: h120?.spreadHigh ?? null,
                // compat fields for older UI paths
                a: lagMinutes.map(() => 0),
                b: 0,
                c: lagMinutes.map(() => 0),
                kind: 'direct_horizon_delta_ridge',
                source: 'offline',
                trainedAt: typeof node.trainedAt === 'string' ? node.trainedAt : null
            };
        }

        // v2 dual-regime
        if (node.regimes?.base) {
            const base = parseRegimeBlock(node.regimes.base);
            const high = parseRegimeBlock(node.regimes.high) || base;
            if (!base) return null;
            return {
                upstreams: node.upstreams.slice(),
                lagMinutes,
                a: base.a,
                b: base.b,
                c: base.c,
                regimes: { base, high },
                highThreshold: finite(Number(node.highThreshold)) ? Number(node.highThreshold) : null,
                thresholdOn: node.thresholdOn || 'max_upstream_lagged',
                reliable: true,
                r: finite(Number(node.r)) ? Number(node.r) : base.r,
                rmse: finite(Number(node.rmse)) ? Number(node.rmse) : base.rmse,
                spreadLow: finite(Number(node.spreadLow)) ? Number(node.spreadLow) : base.spreadLow,
                spreadHigh: finite(Number(node.spreadHigh)) ? Number(node.spreadHigh) : base.spreadHigh,
                kind: node.kind || 'dual_regime_multi_lag_ridge',
                source: 'offline',
                trainedAt: typeof node.trainedAt === 'string' ? node.trainedAt : null
            };
        }

        // v1 flat multi-input
        if (!Array.isArray(node.a)) return null;
        const a = node.a.map(Number);
        const b = Number(node.b);
        const c = Array.isArray(node.c) ? node.c.map(Number) : a.map(() => 0);
        if (a.length !== lagMinutes.length || a.some(v => !finite(v)) || !finite(b)) return null;
        if (c.length !== a.length || c.some(v => !finite(v))) return null;
        return {
            upstreams: node.upstreams.slice(),
            lagMinutes,
            a,
            b,
            c,
            reliable: true,
            r: finite(Number(node.r)) ? Number(node.r) : null,
            rmse: finite(Number(node.rmse)) ? Number(node.rmse) : null,
            spreadLow: finite(Number(node.spreadLow)) ? Number(node.spreadLow) : null,
            spreadHigh: finite(Number(node.spreadHigh)) ? Number(node.spreadHigh) : null,
            kind: typeof node.kind === 'string' ? node.kind : null,
            source: 'offline',
            trainedAt: typeof node.trainedAt === 'string' ? node.trainedAt : null
        };
    }

    // ŷ(t+h) = y(t) + b + wLocal·Δy_local + Σ wWindow·Δx_window + Σ wRate·Δx_rate
    //        + Σ wWindow2·Δx_window(τ+h) + Σ wRate2·Δx_rate(τ+h) + rain − biasEMA
    function forecastDirectHorizon(upstreams, model, horizonHours, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        const anchor = options.anchor;
        if (!finite(anchor?.value) || !model?.horizons) return Object.assign([], { bias: null });
        const nowMs = +now;
        const end = nowMs + horizonHours * HOUR;
        const y0 = anchor.value;
        const yPrev = sampleAt(options.downstream?.timeSeries, +anchor.time - stepMs);
        const dyLocal = finite(yPrev) ? y0 - yPrev : 0;
        const bias = finite(options.bias) ? options.bias : 0;
        const high = driverIsHigh(upstreams, model, nowMs, stepMs);
        const rain = basinRainAt(options.rainSensors, nowMs, stepMs);
        const points = [];
        if (anchor?.time) {
            points.push({
                time: anchor.time instanceof Date ? anchor.time : new Date(anchor.time),
                value: y0,
                regime: 'anchor',
                spreadLow: 0,
                spreadHigh: 0
            });
        }
        let previous = y0;
        const maxStep = options.maxStep;
        const horizonKeys = Object.keys(model.horizons).map(Number).filter(Number.isFinite).sort((a, b) => a - b)
            .filter(minutes => minutes * MINUTE <= horizonHours * HOUR + 1);
        for (const blockMinutes of horizonKeys) {
            const block = model.horizons[String(blockMinutes)];
            if (!block) continue;
            const target = Math.ceil((nowMs + blockMinutes * MINUTE) / stepMs) * stepMs;
            if (target > end + 1) continue;
            let ok = true;
            const dxw = [];
            const dxr = [];
            const dxw2 = [];
            const dxr2 = [];
            const blockMs = blockMinutes * MINUTE;
            for (let i = 0; i < upstreams.length; i++) {
                const lagMs = model.lagMinutes[i] * MINUTE;
                const xNow = sampleAt(upstreams[i]?.timeSeries, nowMs - lagMs);
                const xWin = sampleAt(upstreams[i]?.timeSeries, nowMs - lagMs - blockMs);
                const xPrev = sampleAt(upstreams[i]?.timeSeries, nowMs - lagMs - stepMs);
                const xNow2 = sampleAt(upstreams[i]?.timeSeries, nowMs - lagMs - blockMs);
                const xWin2 = sampleAt(upstreams[i]?.timeSeries, nowMs - lagMs - 2 * blockMs);
                const xPrev2 = sampleAt(upstreams[i]?.timeSeries, nowMs - lagMs - blockMs - stepMs);
                if (!finite(xNow) || !finite(xWin)) { ok = false; break; }
                dxw.push(xNow - xWin);
                dxr.push(finite(xPrev) ? xNow - xPrev : 0);
                if (finite(xNow2) && finite(xWin2)) {
                    dxw2.push(xNow2 - xWin2);
                    dxr2.push(finite(xPrev2) ? xNow2 - xPrev2 : 0);
                } else {
                    dxw2.push(0);
                    dxr2.push(0);
                }
            }
            if (!ok) continue;
            let delta = block.b + block.wLocal * dyLocal;
            for (let i = 0; i < dxw.length; i++) {
                delta += block.wWindow[i] * dxw[i] + block.wRate[i] * dxr[i];
                delta += (block.wWindow2?.[i] || 0) * dxw2[i] + (block.wRate2?.[i] || 0) * dxr2[i];
            }
            delta += (block.wRain6 || 0) * rain.rain6h + (block.wRain24 || 0) * rain.rain24h;
            delta -= bias;
            let value = y0 + delta;
            if (finite(maxStep) && finite(previous)) {
                const steps = Math.max(1, Math.round((target - (points.at(-1) ? +points.at(-1).time : nowMs)) / stepMs));
                const d = value - previous;
                if (Math.abs(d) > maxStep * steps) value = previous + Math.sign(d || 1) * maxStep * steps;
            }
            // High fork when upstream is elevated, a move is predicted, or the
            // downstream has already shifted recently (local events without upstream z85).
            let recentMove = Math.abs(dyLocal) >= 0.05;
            if (!recentMove && options.downstream?.timeSeries) {
                const yRecent = sampleAt(options.downstream.timeSeries, nowMs - 2 * HOUR);
                if (finite(yRecent)) recentMove = Math.abs(y0 - yRecent) >= 0.05;
            }
            const useHigh = high || Math.abs(delta) >= 0.05 || recentMove;
            const spreadLow = useHigh
                ? (finite(block.spreadLowHigh) ? block.spreadLowHigh : block.spreadLow)
                : block.spreadLow;
            const spreadHigh = useHigh
                ? (finite(block.spreadHighHigh) ? block.spreadHighHigh : block.spreadHigh)
                : block.spreadHigh;
            points.push({
                time: new Date(target),
                value,
                regime: useHigh ? 'high' : 'base',
                spreadLow,
                spreadHigh
            });
            previous = value;
        }
        points.bias = bias;
        return points;
    }

    function alignMultiRaw(upstreams, downstream, hours, now = new Date(), stepMs = DEFAULT_STEP_MS) {
        const end = +now;
        const start = end - hours * HOUR;
        const times = [];
        const ys = [];
        const xs = upstreams.map(() => []);
        for (let t = start; t <= end; t += stepMs) {
            const y = sampleAt(downstream?.timeSeries, t);
            const vals = upstreams.map(up => sampleAt(up?.timeSeries, t));
            if (!finite(y) || vals.some(v => !finite(v))) continue;
            times.push(new Date(t));
            ys.push(y);
            vals.forEach((v, j) => xs[j].push(v));
        }
        return { times, ys, xs };
    }

    function buildLaggedDesign(raw, lagStepsList) {
        const maxLag = Math.max(0, ...lagStepsList.map(l => l + 1));
        const levelCols = lagStepsList.map(() => []);
        const rateCols = lagStepsList.map(() => []);
        const drivers = [];
        const ys = [];
        const times = [];
        for (let i = maxLag; i < raw.ys.length; i++) {
            let ok = true;
            const levels = [];
            const rates = [];
            for (let j = 0; j < lagStepsList.length; j++) {
                const idx = i - lagStepsList[j];
                const value = raw.xs[j][idx];
                const prev = raw.xs[j][idx - 1];
                if (!finite(value) || !finite(prev)) { ok = false; break; }
                levels.push(value);
                rates.push(value - prev);
            }
            if (!ok) continue;
            levels.forEach((v, j) => levelCols[j].push(v));
            rates.forEach((v, j) => rateCols[j].push(v));
            drivers.push(Math.max(...levels));
            ys.push(raw.ys[i]);
            times.push(raw.times[i]);
        }
        return { columns: levelCols, rateColumns: rateCols, drivers, ys, times };
    }

    function flattenSingle(result) {
        if (!result?.ok) return result;
        return {
            ...result,
            lagMinutes: Array.isArray(result.lagMinutes) ? result.lagMinutes[0] : result.lagMinutes,
            a: Array.isArray(result.a) ? result.a[0] : result.a,
            upstreams: result.upstreams
        };
    }

    function predictEdge(upstream, downstream, options = {}) {
        return flattenSingle(predictNode(downstream, [upstream], options));
    }

    // Predizione a valle: preferisce modello offline direct-horizon se affidabile.
    function predictNode(downstream, upstreams, options = {}) {
        const list = (upstreams || []).filter(Boolean);
        if (!list.length) return { ok: false, reason: 'Nessuna stazione a monte.' };
        const hours = options.hours || 48;
        const now = options.now || new Date();
        const stepMs = options.stepMs || DEFAULT_STEP_MS;
        const maxLagHours = options.maxLagHours || Math.min(12, hours / 2);
        const holdoutFrac = options.holdoutFrac ?? 0.2;
        const horizonHours = options.horizonHours ?? 2;
        const upstreamIds = list.map(up => up.id || options.upstreamId);
        const offlineNode = parseOfflineNode(options.coeffs, downstream?.id || options.downstreamId, upstreamIds)
            || (list.length === 1 ? parseOfflineEdge(options.coeffs, upstreamIds[0], downstream?.id || options.downstreamId) : null);

        const anchor = latestFinite(downstream?.timeSeries);
        const maxStep = maxStepFromSeries(downstream?.timeSeries, stepMs);

        // Offline direct-horizon: usa solo se reliable (skill validato ≥ soglia).
        if (offlineNode?.kind === 'direct_horizon_delta_ridge') {
            if (offlineNode.reliable === false) {
                return {
                    ok: false,
                    reason: 'Nessuna stima affidabile: il trasferimento monte→valle non batte la persistenza in validazione.',
                    source: 'offline',
                    kind: offlineNode.kind,
                    skill120: offlineNode.skill120,
                    reliable: false
                };
            }
            if (!finite(anchor?.value)) {
                return { ok: false, reason: 'Manca l’ultima misura a valle per ancorare la previsione.' };
            }
            const rainSensors = options.rainSensors || [];
            const bias = options.biasCorrection === false
                ? 0
                : estimateRecentBias(list, downstream, offlineNode, now, stepMs, { rainSensors });
            const rawForecast = forecastDirectHorizon(list, offlineNode, horizonHours, now, stepMs, {
                anchor, maxStep, downstream, bias, rainSensors
            });
            const forecast = rawForecast.map(point => {
                const lo = finite(point.spreadLow) ? point.spreadLow : offlineNode.spreadLow;
                const hi = finite(point.spreadHigh) ? point.spreadHigh : offlineNode.spreadHigh;
                const fork = forkAround(point.value, lo, hi);
                return fork
                    ? { time: point.time, value: fork.med, min: fork.min, med: fork.med, max: fork.max }
                    : { time: point.time, value: point.value, min: point.value, med: point.value, max: point.value };
            });
            const spreadLow = offlineNode.spreadLow;
            const spreadHigh = offlineNode.spreadHigh;
            return {
                ok: true,
                source: 'offline',
                kind: 'direct_horizon_delta_ridge',
                reliable: true,
                upstreams: offlineNode.upstreams,
                lagMinutes: offlineNode.lagMinutes,
                a: offlineNode.a,
                b: offlineNode.b,
                c: offlineNode.c,
                r: offlineNode.r,
                skill120: offlineNode.skill120,
                mae: null,
                rmse: offlineNode.rmse,
                forkSpread: spreadLow,
                spreadLow,
                spreadHigh,
                bias,
                maxStep,
                predicted: [],
                observed: [],
                times: [],
                forecast,
                next: nextHalfHourForks(forecast, Math.max(6, Math.round(horizonHours * 2)), now),
                reason: null
            };
        }

        const raw = alignMultiRaw(list, downstream, hours, now, stepMs);
        const maxLagSteps = Math.floor(maxLagHours * HOUR / stepMs);

        if (raw.ys.length < 8 + list.length) {
            if (offlineNode) {
                const spreadLow = finite(offlineNode.spreadLow) ? offlineNode.spreadLow
                    : (finite(offlineNode.rmse) ? offlineNode.rmse * FORK_K : null);
                const spreadHigh = finite(offlineNode.spreadHigh) ? offlineNode.spreadHigh : spreadLow;
                const rawForecast = forecastForwardMulti(list, offlineNode, horizonHours, now, stepMs, { anchor, maxStep });
                const forecast = withFork(rawForecast, spreadLow, spreadHigh);
                return {
                    ok: true,
                    source: 'offline',
                    kind: offlineNode.kind || 'dual_regime_multi_lag_ridge',
                    upstreams: offlineNode.upstreams,
                    lagMinutes: offlineNode.lagMinutes,
                    a: offlineNode.a,
                    b: offlineNode.b,
                    c: offlineNode.c,
                    highThreshold: offlineNode.highThreshold ?? null,
                    r: offlineNode.r,
                    mae: null,
                    rmse: offlineNode.rmse,
                    forkSpread: spreadLow,
                    spreadLow,
                    spreadHigh,
                    bias: rawForecast.bias ?? null,
                    maxStep,
                    predicted: [],
                    observed: [],
                    times: [],
                    forecast,
                    next: nextHalfHourForks(forecast, Math.max(6, Math.round(horizonHours * 2)), now),
                    reason: 'Serie allineate insufficienti: uso coefficienti offline.'
                };
            }
            return { ok: false, reason: 'Serie allineate insufficienti per stimare il trasferimento multi-ingresso.' };
        }

        const lagStepsList = list.map((_, j) => estimateLag(raw.xs[j], raw.ys, maxLagSteps).lagSteps);
        const design = buildLaggedDesign(raw, lagStepsList);
        if (design.ys.length < 8 + list.length) {
            return { ok: false, reason: 'Campioni ritardati insufficienti per il fit multi-ingresso.' };
        }

        const split = Math.max(list.length + 2, Math.floor(design.ys.length * (1 - holdoutFrac)));
        const trainLevel = design.columns.map(col => col.slice(0, split));
        const trainY = design.ys.slice(0, split);
        const liveFit = fitMultiAffine(trainLevel, trainY);
        const liveA = liveFit.a;
        const trainPred = trainY.map((_, i) => applyMulti(trainLevel.map(col => col[i]), { a: liveA, b: liveFit.b }));
        const liveR = pearson(trainPred, trainY);
        const lagSaturated = lagStepsList.some(steps => steps >= maxLagSteps);
        // Con coeffs v3 offline affidabile ha già fatto return. Qui: offline legacy o live.
        const preferOffline = offlineNode && (
            liveR === null || Math.abs(liveR) < WEAK_R || !liveA || lagSaturated
        );

        const model = preferOffline
            ? { ...offlineNode, source: 'offline' }
            : {
                upstreams: upstreamIds,
                lagMinutes: lagStepsList.map(steps => steps * stepMs / MINUTE),
                a: liveA,
                b: liveFit.b,
                c: liveA ? liveA.map(() => 0) : null,
                r: liveR,
                source: 'live',
                kind: 'multi_lag_ridge'
            };

        if (!Array.isArray(model.a) || !finite(model.b)) {
            return { ok: false, reason: 'Impossibile stimare i coefficienti multi-ingresso.' };
        }

        const predictedValues = design.ys.map((_, i) => {
            const levels = design.columns.map(col => col[i]);
            const active = pickRegime(model, design.drivers[i]);
            return applyMulti(levels, { a: active.a, b: active.b });
        });
        const holdY = design.ys.slice(split);
        const holdPred = holdY.length
            ? holdY.map((_, i) => {
                const idx = split + i;
                const active = pickRegime(model, design.drivers[idx]);
                return applyMulti(design.columns.map(col => col[idx]), { a: active.a, b: active.b });
            })
            : predictedValues;
        const metrics = residuals(holdY.length ? holdY : design.ys, holdPred);
        const rmse = finite(metrics.rmse) ? metrics.rmse : (finite(model.rmse) ? model.rmse : null);
        const spreadLow = finite(model.spreadLow) ? model.spreadLow : (finite(rmse) ? rmse * FORK_K : null);
        const spreadHigh = finite(model.spreadHigh) ? model.spreadHigh : spreadLow;
        const predicted = withFork(design.times.map((time, i) => ({ time, value: predictedValues[i] })), spreadLow, spreadHigh);
        const rawForecast = forecastForwardMulti(list, model, horizonHours, now, stepMs, { anchor, maxStep });
        const forecast = withFork(rawForecast, spreadLow, spreadHigh);

        return {
            ok: true,
            source: model.source,
            kind: model.kind || (model.regimes ? 'dual_regime_multi_lag_ridge' : 'multi_lag_ridge'),
            upstreams: model.upstreams,
            lagMinutes: model.lagMinutes,
            a: model.a,
            b: model.b,
            c: model.c,
            highThreshold: model.highThreshold ?? null,
            r: model.r,
            mae: metrics.mae,
            rmse,
            forkSpread: spreadLow,
            spreadLow,
            spreadHigh,
            bias: rawForecast.bias ?? null,
            maxStep,
            predicted,
            observed: design.times.map((time, i) => ({ time, value: design.ys[i] })),
            times: design.times,
            forecast,
            next: nextHalfHourForks(forecast, Math.max(6, Math.round(horizonHours * 2)), now),
            reason: preferOffline
                ? (lagSaturated
                    ? 'Lag live al tetto della finestra: uso coefficienti offline.'
                    : 'Correlazione live debole: uso coefficienti offline.')
                : null
        };
    }

    function forecastForward(upstream, model, horizonHours, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        const multiModel = Array.isArray(model?.a)
            ? model
            : { a: [model?.a], b: model?.b, lagMinutes: [model?.lagMinutes] };
        return forecastForwardMulti([upstream], multiModel, horizonHours, now, stepMs, options);
    }

    function forecastForwardMulti(upstreams, model, horizonHours, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        if (!Array.isArray(model?.lagMinutes) || model.lagMinutes.length !== upstreams.length) {
            return Object.assign([], { bias: null });
        }
        const nowMs = +now;
        const first = Math.ceil((nowMs + 1) / stepMs) * stepMs;
        const end = nowMs + horizonHours * HOUR;
        const raw = [];
        for (let t = first; t <= end; t += stepMs) {
            const xs = [];
            let ok = true;
            for (let i = 0; i < upstreams.length; i++) {
                const lagMs = model.lagMinutes[i] * MINUTE;
                // Solo livelli ritardati: i termini di rate (Δx) in open-loop amplificano il rumore
                // a 30′ e producono zigzag poco idraulici (c spesso |c|≫1).
                const x = sampleAt(upstreams[i]?.timeSeries, t - lagMs);
                if (!finite(x)) { ok = false; break; }
                xs.push(x);
            }
            if (!ok) continue;
            const driver = Math.max(...xs);
            const active = pickRegime(model, driver);
            if (!Array.isArray(active?.a) || !finite(active?.b) || active.a.length !== xs.length) continue;
            const value = applyMulti(xs, { a: active.a, b: active.b });
            if (!finite(value)) continue;
            raw.push({ time: new Date(t), value, regime: active.activeRegime || 'base' });
        }
        if (!raw.length) return Object.assign([], { bias: null });

        const anchor = options.anchor;
        const maxStep = options.maxStep;
        let bias = 0;
        if (finite(anchor?.value)) bias = anchor.value - raw[0].value;

        const points = [];
        // Punto di raccordo sull’ultima osservazione a valle (continuità temporale/valore).
        if (finite(anchor?.value) && anchor?.time) {
            points.push({ time: anchor.time instanceof Date ? anchor.time : new Date(anchor.time), value: anchor.value, regime: 'anchor' });
        }
        // EMA sul target ancorato: evita spezzate da rumore dei monti campionati a 30′.
        const smooth = finite(options.smooth) ? Math.min(0.9, Math.max(0, options.smooth)) : 0.65;
        let previous = finite(anchor?.value) ? anchor.value : raw[0].value + bias;
        for (const point of raw) {
            let value = smooth * previous + (1 - smooth) * (point.value + bias);
            if (finite(maxStep) && finite(previous)) {
                const delta = value - previous;
                if (Math.abs(delta) > maxStep) value = previous + Math.sign(delta || 1) * maxStep;
            }
            points.push({ time: point.time, value, regime: point.regime });
            previous = value;
        }
        points.bias = bias;
        return points;
    }

    function nextHalfHourForks(forecast, count = 6, now = null) {
        const nowMs = now ? +now : Date.now();
        return (forecast || [])
            .filter(point => +point.time > nowMs)
            .slice(0, count)
            .map(point => ({
                time: point.time,
                min: point.min,
                med: point.med ?? point.value,
                max: point.max
            }));
    }

    async function loadCoeffs(fetcher, url = 'models/transfer-coeffs.json') {
        try {
            const response = await fetcher(url, { cache: 'no-store' });
            if (!response?.ok) return null;
            const data = await response.json();
            if (data?.edges || data?.nodes) return data;
            return null;
        } catch {
            return null;
        }
    }

    const api = {
        edgeKey, buildGraph, rankNodes, alignSeries, estimateLag, fitAffine, fitMultiAffine, applyAffine, applyMulti,
        pickRegime, latestFinite, maxStepFromSeries, predictEdge, predictNode, forecastForward, forecastForwardMulti,
        forecastDirectHorizon, parseOfflineNode, nextHalfHourForks, forkAround, withFork, loadCoeffs, pearson, residuals,
        estimateRecentBias, driverIsHigh, basinRainAt, DEFAULT_STEP_MS, FORK_K, SKILL_KEEP
    };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.RenoPredict = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
