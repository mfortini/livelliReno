(function(root) {
    'use strict';
    const TemporalSampling = typeof require === 'function'
        ? require('./temporal-sampling.js')
        : (root.TemporalSampling || globalThis.TemporalSampling);
    const HOUR = 3600000;
    const MINUTE = 60000;
    const DEFAULT_STEP_MS = 30 * MINUTE;
    const WEAK_R = 0.35;
    const FORK_K = 1.0; // ±1 RMSE ≈ forchetta empirica min/max attorno alla mediana
    const MIN_STEP_ABS = 0.05; // m per mezz’ora, pavimento del rate-limit
    const SKILL_KEEP = 0.10; // allineato a tools/fit-transfer.py (soglia enter)
    const SKILL_ENTER = 0.10; // isteresi: entra reliable se skillLo ≥ enter
    const SKILL_EXIT = 0.05;  // isteresi: esce reliable solo se skillLo < exit
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

    function horizonBlockSkill(block) {
        if (!block) return null;
        if (finite(Number(block.validateEvents?.skill))) return Number(block.validateEvents.skill);
        if (finite(Number(block.validate?.skill))) return Number(block.validate.skill);
        if (finite(Number(block.skill))) return Number(block.skill);
        return null;
    }

    // Isteresi sul limite inferiore CI (o skill puntuale): enter 0.10 / exit 0.05.
    function reliableFromSkill(skillLo, previousReliable) {
        if (!finite(skillLo)) return false;
        if (previousReliable === true) return skillLo >= SKILL_EXIT;
        return skillLo >= SKILL_ENTER;
    }

    function skillLoOf(node) {
        const lo = Number(node?.skill120CI?.lo);
        if (finite(lo)) return lo;
        const s = Number(node?.skill120);
        return finite(s) ? s : null;
    }

    // Longest horizon with event/validate skill ≥ SKILL_KEEP, clamped to [min, cap].
    function packedHorizonEntries(model) {
        const direct = model?.horizons;
        if (direct && Object.keys(direct).length) return Object.entries(direct);
        const meta = model?.packedWeights?.horizonMeta;
        if (meta && Object.keys(meta).length) return Object.entries(meta);
        return [];
    }

    function maxReliableHorizonMinutes(model, options = {}) {
        const cap = finite(Number(options.capMinutes)) ? Number(options.capMinutes) : 720;
        const min = finite(Number(options.minMinutes)) ? Number(options.minMinutes) : 120;
        let best = min;
        const entries = packedHorizonEntries(model);
        if (!entries.length) return Math.min(cap, Math.max(min, best));
        for (const [key, block] of entries) {
            const minutes = Number(key);
            if (!Number.isFinite(minutes) || minutes > cap || minutes < 0) continue;
            const skill = horizonBlockSkill(block);
            if (skill == null || skill < SKILL_KEEP) continue;
            if (minutes > best) best = minutes;
        }
        return Math.min(cap, Math.max(min, best));
    }

    // Last knot available (for fade band beyond reliable cut).
    function maxHorizonMinutesAvailable(model, options = {}) {
        const cap = finite(Number(options.capMinutes)) ? Number(options.capMinutes) : 720;
        const entries = packedHorizonEntries(model);
        if (!entries.length) return maxReliableHorizonMinutes(model, options);
        let best = 0;
        for (const [key] of entries) {
            const minutes = Number(key);
            if (!Number.isFinite(minutes) || minutes > cap || minutes < 0) continue;
            if (minutes > best) best = minutes;
        }
        return Math.max(best, maxReliableHorizonMinutes(model, options));
    }

    // Attach pinball Q10/Q90 + conformal onto prod MLP (median layers unchanged).
    function mergeQuantileBands(node, bands) {
        if (!node || node.kind !== 'window_delta_mlp' || !bands?.horizons) return node;
        const out = { ...node, horizons: { ...node.horizons } };
        for (const [key, src] of Object.entries(bands.horizons)) {
            const block = out.horizons[key];
            if (!block || !Array.isArray(src.layersQ10) || !Array.isArray(src.layersQ90)) continue;
            out.horizons[key] = {
                ...block,
                layersQ10: src.layersQ10,
                layersQ50: block.layersQ50 || block.layers,
                layersQ90: src.layersQ90,
                conformalQhat: finite(Number(src.conformalQhat)) ? Number(src.conformalQhat) : block.conformalQhat
            };
        }
        out.quantiles = Array.isArray(bands.quantiles) ? bands.quantiles.slice() : [0.1, 0.5, 0.9];
        out.bandSource = typeof bands.source === 'string' ? bands.source : 'cento-dnn-bands';
        return out;
    }

    // Soft-attach crossing-time / hazard export. Warp is opt-in via adopt / useCrossingTime.
    function mergeCrossingTime(node, crossing) {
        if (!node || node.kind !== 'window_delta_mlp' || !crossing?.thresholds) return node;
        const okKind = !crossing.kind
            || crossing.kind === 'cento_crossing_time'
            || crossing.kind === 'cento_crossing_hazard';
        if (!okKind) return node;
        return { ...node, crossingTime: crossing };
    }

    function predictCrossingTau(crossing, sensorMap, rainSensors, nowMs, stepMs = DEFAULT_STEP_MS) {
        if (!crossing?.thresholds || !crossing.featureNames) return null;
        const primaryNum = Number(crossing.primaryOffsetM ?? 1);
        const block = crossing.thresholds[String(primaryNum)]
            || crossing.thresholds[primaryNum.toFixed(1)]
            || crossing.thresholds['1']
            || crossing.thresholds['1.0'];
        if (!block) return null;
        const raw = buildWindowFeatureVector(
            sensorMap,
            crossing.stations || [],
            crossing.lookbackSteps || 48,
            nowMs,
            stepMs,
            rainSensors,
            crossing.useRain !== false,
            crossing.featureRecipe || {},
            0
        );
        const x = standardizeWindowFeatures(raw, crossing.featureNames, crossing.mean, crossing.std);
        if (!x) return null;
        const censor = finite(block.censorMin) ? block.censorMin : 360;
        const offsetM = Number(block.offsetM) || primaryNum || 1;
        const gateP = finite(Number(crossing.decodeGateP)) ? Number(crossing.decodeGateP) : 0.35;

        // WP2 hazard: multi-logit discrete-time hazards → mode τ + P(crossing).
        if (crossing.kind === 'cento_crossing_hazard' || block.kind === 'discrete_hazard') {
            const layers = block.layers;
            if (!layers) return null;
            const logits = mlpForwardVec(x, layers);
            if (!logits?.length) return null;
            const hazards = logits.map(z => 1 / (1 + Math.exp(-z)));
            const nBins = hazards.length;
            const pmf = new Array(nBins);
            let surv = 1;
            let pCross = 0;
            let bestP = -1;
            let modeBin = nBins;
            for (let j = 0; j < nBins; j++) {
                const p = hazards[j] * surv;
                pmf[j] = p;
                pCross += p;
                if (p > bestP) {
                    bestP = p;
                    modeBin = j + 1;
                }
                surv *= (1 - hazards[j]);
            }
            const pCens = Math.max(0, 1 - pCross);
            let modeMin = modeBin * (block.stepMinutes || 30);
            if (pCens >= bestP) modeMin = censor;
            return {
                offsetM,
                censorMin: censor,
                modeMin,
                pCross,
                pCens,
                decodeGateP: gateP,
                allowDecode: pCross >= gateP && modeMin < censor - 1,
                q10: null,
                q50: modeMin,
                q90: null,
                kind: 'hazard'
            };
        }

        if (!block?.models) return null;
        const clipTau = tau => Math.max(30, Math.min(censor, tau));
        const runQ = key => {
            const layers = block.models[key]?.layers;
            if (!layers) return null;
            const tau = mlpForward(x, layers);
            return finite(tau) ? clipTau(tau) : null;
        };
        const q10 = runQ('0.1');
        const q50 = runQ('0.5');
        const q90 = runQ('0.9');
        if (!finite(q50)) return null;
        return {
            offsetM,
            censorMin: censor,
            q10,
            q50,
            q90,
            modeMin: q50,
            pCross: q50 < censor - 1 ? 1 : 0,
            allowDecode: q50 < censor - 1,
            kind: 'pinball'
        };
    }

    function pathRiseTauMin(forecast, y0, offsetM, nowMs) {
        if (!finite(y0) || !finite(offsetM) || !forecast?.length) return null;
        const thr = y0 + offsetM;
        for (const p of forecast) {
            if (p.regime === 'anchor') continue;
            const v = p.med ?? p.value;
            if (!finite(v) || !p.time) continue;
            if (v >= thr) return Math.max(30, (+p.time - nowMs) / MINUTE);
        }
        return null;
    }

    // Align level fan to crossing-time mode (most probable): warp if path already rises,
    // otherwise lift the rising limb so y0+offset is hit at modeTau (MLP often never reaches it).
    function warpForecastByCrossing(forecast, crossingPred, y0, nowMs) {
        if (!forecast?.length || !crossingPred || !finite(crossingPred.modeMin)) {
            return { forecast, warped: false, scale: 1, pathTau: null };
        }
        const censor = finite(crossingPred.censorMin) ? crossingPred.censorMin : 360;
        const offsetM = finite(crossingPred.offsetM) ? crossingPred.offsetM : 1;
        if (crossingPred.modeMin >= censor - 1) {
            return { forecast, warped: false, scale: 1, pathTau: null, stable: true };
        }
        const thr = y0 + offsetM;
        const modeMs = crossingPred.modeMin * MINUTE;
        const pathTau = pathRiseTauMin(forecast, y0, offsetM, nowMs);

        if (finite(pathTau) && pathTau >= 15) {
            const scale = crossingPred.modeMin / pathTau;
            if (scale > 0.25 && scale < 4 && Math.abs(scale - 1) > 0.05) {
                const warped = forecast.map(p => {
                    if (p.regime === 'anchor') return p;
                    const dt = +p.time - nowMs;
                    return { ...p, time: new Date(nowMs + dt * scale), probable: true };
                });
                return {
                    forecast: warped,
                    warped: true,
                    method: 'time_warp',
                    scale,
                    pathTau,
                    modeTau: crossingPred.modeMin,
                    q10: crossingPred.q10,
                    q90: crossingPred.q90
                };
            }
        }

        // Constructive lift: force med/min/max to reach thr at modeTau, then keep MLP shape after.
        const lifted = forecast.map(p => {
            if (p.regime === 'anchor') return { ...p, probable: true };
            const dt = +p.time - nowMs;
            const med0 = finite(p.med) ? p.med : p.value;
            let med = med0;
            if (dt <= modeMs) {
                const frac = modeMs > 0 ? Math.max(0, Math.min(1, dt / modeMs)) : 1;
                const ramp = y0 + (thr - y0) * frac;
                med = Math.max(med0, ramp);
            } else {
                // After crossing: keep MLP increment relative to path-at-mode if available.
                const pathAtMode = pathTau
                    ? null
                    : null;
                void pathAtMode;
                med = Math.max(med0, thr + Math.max(0, med0 - thr) * 0.25);
                if (med0 < thr) med = Math.max(med0, thr);
            }
            const lift = med - med0;
            const min = finite(p.min) ? p.min + lift : med;
            const max = finite(p.max) ? Math.max(p.max + lift, med) : med;
            return {
                ...p,
                value: med,
                med,
                min: Math.min(min, med),
                max: Math.max(max, med),
                probable: true
            };
        });
        return {
            forecast: lifted,
            warped: true,
            method: 'level_lift',
            scale: 1,
            pathTau,
            modeTau: crossingPred.modeMin,
            q10: crossingPred.q10,
            q90: crossingPred.q90
        };
    }

    function applySkillGates(data, gates) {
        if (!data?.nodes || !gates?.stations) return data;
        const enter = finite(Number(gates.skillEnter)) ? Number(gates.skillEnter) : SKILL_ENTER;
        const exit = finite(Number(gates.skillExit)) ? Number(gates.skillExit) : SKILL_EXIT;
        for (const [name, gate] of Object.entries(gates.stations)) {
            const node = data.nodes[name];
            if (!node) continue;
            const prev = node.reliable === true;
            const lo = finite(Number(gate.skill120CI?.lo)) ? Number(gate.skill120CI.lo)
                : (finite(Number(gate.skill120)) ? Number(gate.skill120) : skillLoOf(node));
            let next = prev;
            if (finite(lo)) {
                next = prev ? lo >= exit : lo >= enter;
            }
            data.nodes[name] = {
                ...node,
                skill120: finite(Number(gate.skill120)) ? Number(gate.skill120) : node.skill120,
                skill120CI: gate.skill120CI || node.skill120CI || null,
                reliable: next,
                reliableNote: `isteresi enter=${enter}/exit=${exit} on skillLo=${finite(lo) ? lo.toFixed(3) : 'n/d'}`
            };
        }
        return data;
    }

    // Upstream freshness: names without a finite reading in the last staleMinutes.
    function missingUpstreams(upstreams, now = new Date(), staleMinutes = 90) {
        const nowMs = +now;
        const staleMs = Math.max(0, Number(staleMinutes) || 0) * MINUTE;
        const missing = [];
        for (const up of upstreams || []) {
            const last = latestFinite(up?.timeSeries);
            if (!last || (staleMs > 0 && nowMs - +last.time > staleMs)) {
                missing.push(up?.id || up?.name || '?');
            }
        }
        return missing;
    }

    function widenForecastBand(forecast, factor = 1.5) {
        if (!Array.isArray(forecast) || !(factor > 1)) return forecast;
        return forecast.map(p => {
            if (p?.regime === 'anchor') return p;
            const med = p.med ?? p.value;
            if (!finite(med)) return p;
            const halfLo = finite(p.min) ? med - p.min : null;
            const halfHi = finite(p.max) ? p.max - med : null;
            return {
                ...p,
                min: finite(halfLo) ? med - halfLo * factor : p.min,
                max: finite(halfHi) ? med + halfHi * factor : p.max
            };
        });
    }

    // Cross-correlate predicted[i] vs observed[i+d]; positive lagMin ⇒ predicted lags observed.
    function estimatePredictedLag(points, stepMs = DEFAULT_STEP_MS) {
        if (!Array.isArray(points) || points.length < 10) return null;
        const pred = [];
        const obs = [];
        for (const p of points) {
            if (!finite(p?.predicted) || !finite(p?.observed)) continue;
            pred.push(p.predicted);
            obs.push(p.observed);
        }
        if (pred.length < 10) return null;
        let sumP = 0, sumO = 0, sumPP = 0, sumOO = 0;
        for (let i = 0; i < pred.length; i++) {
            sumP += pred[i]; sumO += obs[i];
            sumPP += pred[i] * pred[i]; sumOO += obs[i] * obs[i];
        }
        const n = pred.length;
        const varP = n * sumPP - sumP * sumP;
        const varO = n * sumOO - sumO * sumO;
        if (!(varP > 0) || !(varO > 0)) return null;
        const stepMin = (finite(stepMs) ? stepMs : DEFAULT_STEP_MS) / MINUTE;
        let bestD = 0;
        let bestR = null;
        for (let d = -4; d <= 4; d++) {
            const xs = [];
            const ys = [];
            for (let i = 0; i < n; i++) {
                const j = i + d;
                if (j < 0 || j >= n) continue;
                xs.push(pred[i]);
                ys.push(obs[j]);
            }
            const r = pearson(xs, ys);
            if (r === null) continue;
            if (
                bestR === null
                || Math.abs(r) > Math.abs(bestR)
                || (Math.abs(r) === Math.abs(bestR) && Math.abs(d) < Math.abs(bestD))
            ) {
                bestR = r;
                bestD = d;
            }
        }
        if (bestR === null) return null;
        return { lagSteps: -bestD, lagMin: -bestD * stepMin, r: bestR };
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

    function rainSeriesStepMs(series, fallback = DEFAULT_STEP_MS) {
        const gaps = [];
        for (let i = 1; i < (series?.length || 0) && gaps.length < 40; i++) {
            const dt = +series[i].time - +series[i - 1].time;
            if (dt > 0 && dt < 6 * HOUR) gaps.push(dt);
        }
        if (!gaps.length) return fallback;
        gaps.sort((a, b) => a - b);
        const median = gaps[Math.floor(gaps.length / 2)];
        // Live ARPAE precip is typically 1 h cumulata; storico is denser.
        return median >= 45 * MINUTE ? HOUR : fallback;
    }

    function basinRainAt(rainSensors, targetMs, stepMs = DEFAULT_STEP_MS) {
        const list = (rainSensors || []).filter(Boolean);
        if (!list.length) return { rain6h: 0, rain24h: 0 };
        let r6 = 0, r24 = 0;
        for (const sensor of list) {
            const sampleMs = rainSeriesStepMs(sensor.timeSeries, stepMs);
            const steps6 = Math.max(1, Math.round(6 * HOUR / sampleMs));
            const steps24 = Math.max(1, Math.round(24 * HOUR / sampleMs));
            r6 = Math.max(r6, rainCumulAt(sensor.timeSeries, targetMs, steps6, sampleMs));
            r24 = Math.max(r24, rainCumulAt(sensor.timeSeries, targetMs, steps24, sampleMs));
        }
        return { rain6h: r6, rain24h: r24 };
    }

    function mlpForwardVec(x, layers) {
        let cur = x.slice();
        for (const layer of layers || []) {
            const W = layer.W, b = layer.b;
            const arrayLike = value => Array.isArray(value) || ArrayBuffer.isView(value);
            if (!arrayLike(W) || !arrayLike(b) || W.length !== b.length) return null;
            const next = new Array(b.length);
            for (let o = 0; o < b.length; o++) {
                let sum = b[o];
                const row = W[o];
                if (!arrayLike(row) || row.length !== cur.length) return null;
                for (let i = 0; i < cur.length; i++) sum += row[i] * cur[i];
                next[o] = layer.activation === 'relu' ? Math.max(0, sum) : sum;
            }
            cur = next;
        }
        return cur;
    }

    function mlpForward(x, layers) {
        const cur = mlpForwardVec(x, layers);
        return cur && cur.length === 1 ? cur[0] : null;
    }

    function safeFeatureName(station) {
        return String(station).replace(/ /g, '_');
    }

    function lin3Extrapolate(series, endMs, stepMs, hSteps, nPoints = 3) {
        // OLS linear on last nPoints (oldest→newest); predict hSteps after the last sample.
        // Fallback: last finite level (persistence) when the window is incomplete.
        const n = Math.max(2, Math.round(Number(nPoints) || 3));
        const y = [];
        for (let i = 0; i < n; i++) {
            const v = sampleAt(series, endMs - (n - 1 - i) * stepMs);
            if (!finite(v)) {
                const y0 = sampleAt(series, endMs);
                return finite(y0) ? y0 : null;
            }
            y.push(v);
        }
        let sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (let i = 0; i < n; i++) {
            sx += i; sy += y[i]; sxx += i * i; sxy += i * y[i];
        }
        const det = n * sxx - sx * sx;
        if (!finite(det) || Math.abs(det) < 1e-12) return y[n - 1];
        const a0 = (sy * sxx - sx * sxy) / det;
        const a1 = (n * sxy - sx * sy) / det;
        const xh = (n - 1) + (finite(Number(hSteps)) ? Number(hSteps) : 0);
        return a0 + a1 * xh;
    }

    function poly2Extrapolate(series, endMs, stepMs, hSteps, nPoints = 6) {
        // OLS quadratic on last nPoints; predict hSteps after the last sample.
        // Fallback: persistence, then 3-point poly2 if the full window is incomplete.
        const tryNs = [Math.max(3, Math.round(Number(nPoints) || 6)), 3];
        const y0 = sampleAt(series, endMs);
        for (const n of tryNs) {
            const y = [];
            let complete = true;
            for (let i = 0; i < n; i++) {
                const v = sampleAt(series, endMs - (n - 1 - i) * stepMs);
                if (!finite(v)) { complete = false; break; }
                y.push(v);
            }
            if (!complete) continue;
            let sx = 0, sx2 = 0, sx3 = 0, sx4 = 0, sy = 0, sxy = 0, sx2y = 0;
            for (let i = 0; i < n; i++) {
                const xi = i, yi = y[i], xi2 = xi * xi;
                sx += xi; sx2 += xi2; sx3 += xi2 * xi; sx4 += xi2 * xi2;
                sy += yi; sxy += xi * yi; sx2y += xi2 * yi;
            }
            const M = [
                [n, sx, sx2, sy],
                [sx, sx2, sx3, sxy],
                [sx2, sx3, sx4, sx2y]
            ];
            let singular = false;
            for (let col = 0; col < 3; col++) {
                let piv = col;
                for (let r = col + 1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
                [M[col], M[piv]] = [M[piv], M[col]];
                const div = M[col][col];
                if (!finite(div) || Math.abs(div) < 1e-12) { singular = true; break; }
                for (let c = col; c < 4; c++) M[col][c] /= div;
                for (let r = 0; r < 3; r++) {
                    if (r === col) continue;
                    const f = M[r][col];
                    for (let c = col; c < 4; c++) M[r][c] -= f * M[col][c];
                }
            }
            if (singular) continue;
            const a0 = M[0][3], a1 = M[1][3], a2 = M[2][3];
            const xh = (n - 1) + (finite(Number(hSteps)) ? Number(hSteps) : 0);
            return a0 + a1 * xh + a2 * xh * xh;
        }
        return finite(y0) ? y0 : null;
    }

    function localPolyDerivatives(series, endMs, stepMs, nPoints = 4, options = {}) {
        // OLS y ≈ a0 + a1 x + a2 x² with x = -(n-1)..0 → d1=a1, d2=2 a2 (per step).
        const sampleFn = options.sampleFn || ((s, t) => sampleAt(s, t));
        const tryN = [nPoints, 3];
        for (const n of tryN) {
            if (n < 3) continue;
            const y = [];
            let complete = true;
            for (let i = 0; i < n; i++) {
                const v = sampleFn(series, endMs - (n - 1 - i) * stepMs);
                if (!finite(v)) { complete = false; break; }
                y.push(v);
            }
            if (!complete) continue;
            const x = [];
            for (let i = 0; i < n; i++) x.push(i - (n - 1));
            let sx = 0, sx2 = 0, sx3 = 0, sx4 = 0, sy = 0, sxy = 0, sx2y = 0;
            for (let i = 0; i < n; i++) {
                const xi = x[i], yi = y[i], xi2 = xi * xi;
                sx += xi; sx2 += xi2; sx3 += xi2 * xi; sx4 += xi2 * xi2;
                sy += yi; sxy += xi * yi; sx2y += xi2 * yi;
            }
            const M = [
                [n, sx, sx2, sy],
                [sx, sx2, sx3, sxy],
                [sx2, sx3, sx4, sx2y]
            ];
            let singular = false;
            for (let col = 0; col < 3; col++) {
                let piv = col;
                for (let r = col + 1; r < 3; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
                [M[col], M[piv]] = [M[piv], M[col]];
                const div = M[col][col];
                if (!finite(div) || Math.abs(div) < 1e-12) { singular = true; break; }
                for (let c = col; c < 4; c++) M[col][c] /= div;
                for (let r = 0; r < 3; r++) {
                    if (r === col) continue;
                    const f = M[r][col];
                    for (let c = col; c < 4; c++) M[r][c] -= f * M[col][c];
                }
            }
            if (singular) continue;
            return { d1: M[1][3], d2: 2 * M[2][3] };
        }
        return { d1: null, d2: null };
    }

    function localLin3Slope(series, endMs, stepMs, nPoints = 3, options = {}) {
        // OLS y ≈ a0 + a1 x with x = -(n-1)..0 → d1=a1 (per step); no d2.
        const sampleFn = options.sampleFn || ((s, t) => sampleAt(s, t));
        const n = Math.max(2, Math.round(Number(nPoints) || 3));
        const y = [];
        for (let i = 0; i < n; i++) {
            const v = sampleFn(series, endMs - (n - 1 - i) * stepMs);
            if (!finite(v)) return { d1: null, d2: null };
            y.push(v);
        }
        let sx = 0, sx2 = 0, sy = 0, sxy = 0;
        for (let i = 0; i < n; i++) {
            const xi = i - (n - 1);
            sx += xi; sx2 += xi * xi; sy += y[i]; sxy += xi * y[i];
        }
        const det = n * sx2 - sx * sx;
        if (!finite(det) || Math.abs(det) < 1e-12) return { d1: null, d2: null };
        return { d1: (n * sxy - sx * sy) / det, d2: null };
    }

    // upstreamLagAlign: upstream samples are re-indexed around the wave arrival.
    // mode 'shift': lag k reads x(t + h - tau_i - k·step) (window moved back).
    // mode 'augment': flat window untouched + a* block reads x(t + h - tau_i - j·step).
    // For h ≤ tau_i the arriving wave is already observed; negative shifts are
    // masked as miss=1 so the model never peeks at the future.
    function upstreamLagOffsetSteps(recipe, station, horizonMinutes, stepMinutes) {
        if (recipe?.upstreamLagAlign !== true) return 0;
        const lagMin = Number(recipe?.upstreamLagMinutes?.[station]);
        if (!finite(lagMin)) return 0;
        const hSteps = Math.round((finite(Number(horizonMinutes)) ? Number(horizonMinutes) : 0) / stepMinutes);
        return Math.round(lagMin / stepMinutes) - hSteps;
    }

    function stationWindowLagKeys(recipe, station, lookback) {
        const sparse = recipe?.stationLagSteps?.[station];
        if (Array.isArray(sparse) && sparse.length) {
            return sparse.map((s) => Math.round(Number(s))).filter((s) => Number.isFinite(s) && s >= 0);
        }
        const per = Number(recipe?.stationLookbackSteps?.[station]);
        // Per-station L_i may exceed the node lookback default when lookback-τ cap is raised.
        const lb = Number.isFinite(per) && per > 0 ? Math.round(per) : lookback;
        return Array.from({ length: Math.max(1, lb) }, (_, k) => k);
    }

    function buildWindowFeatureVector(sensorMap, stations, lookback, nowMs, stepMs, rainSensors, useRain, recipe = null, horizonMinutes = 0) {
        const values = {};
        const sogliaLevels = recipe?.sogliaLevels || {};
        const useHeadroom = recipe?.useSogliaHeadroom === true;
        const useRatio = recipe?.useSogliaRatio === true;
        const usePolyD = recipe?.usePolyDerivatives !== false;
        const derivMode = recipe?.derivMode === 'lin3' ? 'lin3' : 'poly';
        const polyPoints = Number(recipe?.polyDerivPoints) > 0
            ? Number(recipe.polyDerivPoints)
            : (derivMode === 'lin3' ? 3 : 6);
        const sogliaKind = recipe?.sogliaKind || 'yellow';
        const levelPrefix = useHeadroom ? `headroom_${sogliaKind}` : 'level';
        const ratioPrefix = `ratio_${sogliaKind}`;
        const stepMinutes = stepMs / MINUTE;
        const temporal = TemporalSampling?.temporalContract
            ? TemporalSampling.temporalContract(recipe)
            : { inputStepMinutes: stepMinutes, method: 'raw', maxGapMinutes: 60, derivWindowMinutes: 150 };
        const temporalOn = temporal.inputStepMinutes !== stepMinutes || temporal.method !== 'raw';
        const featureStepMs = temporalOn ? temporal.inputStepMinutes * MINUTE : stepMs;
        const featureStepMinutes = featureStepMs / MINUTE;
        const derivativeStepMs = temporalOn ? DEFAULT_STEP_MS : stepMs;
        const temporalViews = new Map();
        const temporalViewFor = (series) => {
            let view = temporalViews.get(series);
            if (!view) {
                view = TemporalSampling.createTemporalView(series, nowMs, {
                    maxGapMinutes: temporal.maxGapMinutes,
                    method: temporal.method
                });
                temporalViews.set(series, view);
            }
            return view;
        };
        const sampleFeature = temporalOn
            ? (series, t) => temporalViewFor(series).sample(t)
            : (series, t) => sampleAt(series, t);
        const tauFrac = recipe?.tauFrac === true;
        const dropLevelAll = recipe?.featureSignal === 'delta';
        const dropLevelSet = new Set(
            Array.isArray(recipe?.dropLevelStations) ? recipe.dropLevelStations : []
        );
        const dropLevelUpsTau = tauFrac && recipe?.tauFracSignal === 'delta';
        for (const station of stations) {
            const series = sensorMap?.[station]?.timeSeries;
            const tag = safeFeatureName(station);
            const ref = sogliaLevels[station];
            const sparseKeys = Array.isArray(recipe?.stationLagSteps?.[station]);
            const dropLevel = dropLevelAll
                || dropLevelSet.has(station)
                || (dropLevelUpsTau && sparseKeys);
            const head = !dropLevel && useHeadroom && finite(ref) && ref > 0;
            const ratioOn = !dropLevel && useRatio && finite(ref) && ref > 0;
            const lagMode = recipe?.upstreamLagAlign === true ? (recipe?.upstreamLagMode || 'shift') : null;
            const lagOff = lagMode === 'shift'
                ? upstreamLagOffsetSteps(recipe, station, horizonMinutes, featureStepMinutes)
                : 0;
            const augmentSteps = lagMode === 'augment'
                ? upstreamLagOffsetSteps(recipe, station, horizonMinutes, featureStepMinutes)
                : null;
            const lagKeys = stationWindowLagKeys(recipe, station, lookback);
            for (const k of lagKeys) {
                const kEff = k + lagOff;
                const t = kEff >= 0 ? nowMs - kEff * featureStepMs : null;
                const level = t === null ? null : sampleFeature(series, t);
                const prev = t === null ? null : sampleFeature(series, t - featureStepMs);
                const miss = finite(level) ? 0 : 1;
                const signal = head && finite(level) ? ref - level : level;
                const signalPrev = head && finite(prev) ? ref - prev : prev;
                const delta = finite(signal) && finite(signalPrev) ? signal - signalPrev : null;
                values[`delta_${tag}_lag${k}`] = delta;
                values[`miss_${tag}_lag${k}`] = miss;
                if (!dropLevel) {
                    values[`${head ? levelPrefix : 'level'}_${tag}_lag${k}`] = finite(signal) ? signal : null;
                }
                if (usePolyD) {
                    const { d1, d2 } = t === null
                        ? { d1: null, d2: null }
                        : (derivMode === 'lin3'
                            ? localLin3Slope(series, t, derivativeStepMs, polyPoints, { sampleFn: sampleFeature })
                            : localPolyDerivatives(series, t, derivativeStepMs, polyPoints, { sampleFn: sampleFeature }));
                    values[`d1_${tag}_lag${k}`] = d1;
                    if (derivMode !== 'lin3') values[`d2_${tag}_lag${k}`] = d2;
                }
                if (ratioOn) {
                    const r = finite(level) ? level / ref : null;
                    const rp = finite(prev) ? prev / ref : null;
                    values[`${ratioPrefix}_${tag}_lag${k}`] = r;
                    values[`ratio_delta_${tag}_lag${k}`] = finite(r) && finite(rp) ? r - rp : null;
                }
            }
            const spectral = recipe?.spectral;
            if (spectral && spectral.enabled !== false && TemporalSampling?.spectralFeatures) {
                const W = Math.max(4, Math.min(lookback,
                    Math.round(Number(spectral.windowSteps) || lookback)));
                const fftBins = Math.max(0, Math.round(Number(spectral.fftBins ?? 12)));
                const waveletScales = Math.max(0, Math.round(Number(spectral.waveletScales ?? 5)));
                const win = new Array(W);
                for (let k = 0; k < W; k++) {
                    const v = sampleFeature(series, nowMs - k * featureStepMs);
                    win[k] = finite(v) ? v : NaN;
                }
                const feats = TemporalSampling.spectralFeatures(win, fftBins, waveletScales);
                const names = TemporalSampling.spectralFeatureNames(tag, fftBins, waveletScales);
                for (let i = 0; i < names.length; i++) values[names[i]] = feats[i];
            }
            const alignWindow = Math.max(1, Math.round(Number(recipe?.upstreamLagAlignWindow) || 16));
            for (let j = 0; j < (augmentSteps === null ? 0 : alignWindow); j++) {
                const s = augmentSteps + j;
                const t = s >= 0 ? nowMs - s * featureStepMs : null;
                const level = t === null ? null : sampleFeature(series, t);
                const prev = t === null ? null : sampleFeature(series, t - featureStepMs);
                const miss = finite(level) ? 0 : 1;
                const signal = head && finite(level) ? ref - level : level;
                const signalPrev = head && finite(prev) ? ref - prev : prev;
                const delta = finite(signal) && finite(signalPrev) ? signal - signalPrev : null;
                values[`alevel_${tag}_lag${j}`] = finite(signal) ? signal : null;
                values[`adelta_${tag}_lag${j}`] = delta;
                values[`amiss_${tag}_lag${j}`] = miss;
                if (usePolyD) {
                    const { d1, d2 } = t === null
                        ? { d1: null, d2: null }
                        : (derivMode === 'lin3'
                            ? localLin3Slope(series, t, derivativeStepMs, polyPoints, { sampleFn: sampleFeature })
                            : localPolyDerivatives(series, t, derivativeStepMs, polyPoints, { sampleFn: sampleFeature }));
                    values[`ad1_${tag}_lag${j}`] = d1;
                    if (derivMode !== 'lin3') values[`ad2_${tag}_lag${j}`] = d2;
                }
            }
        }
        const pairs = Array.isArray(recipe?.chainPairs) ? recipe.chainPairs : [];
        for (const pair of pairs) {
            if (!Array.isArray(pair) || pair.length < 2) continue;
            const [a, b] = pair;
            const tag = `${safeFeatureName(a)}__${safeFeatureName(b)}`;
            const seriesA = sensorMap?.[a]?.timeSeries;
            const seriesB = sensorMap?.[b]?.timeSeries;
            for (let k = 0; k < lookback; k++) {
                const t = nowMs - k * featureStepMs;
                const va = sampleFeature(seriesA, t);
                const vb = sampleFeature(seriesB, t);
                const miss = (finite(va) && finite(vb)) ? 0 : 1;
                values[`chain_${tag}_lag${k}`] = miss ? null : va - vb;
                values[`chain_miss_${tag}_lag${k}`] = miss;
            }
        }
        if (useRain) {
            const rain = basinRainAt(rainSensors, nowMs, DEFAULT_STEP_MS);
            values.rain6h = rain.rain6h;
            values.rain24h = rain.rain24h;
        }
        // Residual baselines: Cento ŷ(t+h) − y0 as an explicit feature (matches fit-ml-cento).
        if (recipe?.targetMode === 'residual_lin3' || recipe?.targetMode === 'residual_poly2') {
            const down = stations[0];
            const tag = safeFeatureName(down);
            const series = sensorMap?.[down]?.timeSeries;
            const residualStepMs = temporalOn ? DEFAULT_STEP_MS : stepMs;
            const residualStepMinutes = residualStepMs / MINUTE;
            const hSteps = Math.round((finite(Number(horizonMinutes)) ? Number(horizonMinutes) : 0) / residualStepMinutes);
            const y0 = sampleFeature(series, nowMs);
            if (recipe.targetMode === 'residual_lin3') {
                const nPts = Math.max(2, Math.round(Number(recipe.lin3Points) || 3));
                const lin3 = lin3Extrapolate(series, nowMs, residualStepMs, hSteps, nPts);
                values[`lin3_delta_${tag}`] = finite(lin3) && finite(y0) ? lin3 - y0 : 0;
            } else {
                const nPts = Math.max(3, Math.round(Number(recipe.poly2Points) || 6));
                const poly2 = poly2Extrapolate(series, nowMs, residualStepMs, hSteps, nPts);
                values[`poly2_delta_${tag}`] = finite(poly2) && finite(y0) ? poly2 - y0 : 0;
            }
        }
        return values;
    }

    function standardizeWindowFeatures(values, featureNames, mean, std) {
        if (!Array.isArray(featureNames) || !Array.isArray(mean) || !Array.isArray(std)) return null;
        if (featureNames.length !== mean.length || featureNames.length !== std.length) return null;
        const out = new Array(featureNames.length);
        for (let i = 0; i < featureNames.length; i++) {
            const name = featureNames[i];
            const raw = values[name];
            if (name.startsWith('miss_')) {
                out[i] = finite(raw) ? raw : 1;
                continue;
            }
            const mu = mean[i], sd = std[i] || 1;
            if (!finite(raw)) {
                out[i] = 0;
                continue;
            }
            out[i] = (raw - mu) / (sd < 1e-6 ? 1 : sd);
        }
        return out;
    }

    function sigmoid(x) {
        return 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, x))));
    }

    function gruCell(x, h, weightIh, weightHh, biasIh, biasHh) {
        // PyTorch GRU: gates ordered reset, update, new.
        const hidden = h.length;
        const r = new Array(hidden);
        const z = new Array(hidden);
        const n = new Array(hidden);
        const out = new Array(hidden);
        for (let i = 0; i < hidden; i++) {
            let ir = biasIh[i], iz = biasIh[hidden + i], inew = biasIh[2 * hidden + i];
            let hr = biasHh[i], hz = biasHh[hidden + i], hnew = biasHh[2 * hidden + i];
            for (let j = 0; j < x.length; j++) {
                ir += weightIh[i][j] * x[j];
                iz += weightIh[hidden + i][j] * x[j];
                inew += weightIh[2 * hidden + i][j] * x[j];
            }
            for (let j = 0; j < hidden; j++) {
                hr += weightHh[i][j] * h[j];
                hz += weightHh[hidden + i][j] * h[j];
                hnew += weightHh[2 * hidden + i][j] * h[j];
            }
            r[i] = sigmoid(ir + hr);
            z[i] = sigmoid(iz + hz);
            n[i] = Math.tanh(inew + r[i] * hnew);
            out[i] = (1 - z[i]) * n[i] + z[i] * h[i];
        }
        return out;
    }

    function gruForward(seq, gru) {
        let h = new Array(gru.hiddenSize).fill(0);
        for (const x of seq) h = gruCell(x, h, gru.weightIh, gru.weightHh, gru.biasIh, gru.biasHh);
        return h;
    }

    function decodeKX(rawK, rawX, dt = 1, xMin = 0.05, xMax = 0.45) {
        // Stable box: 2·K·X ≤ dt ≤ 2·K·(1−X) ⇒ non-negative Muskingum coeffs.
        const x = xMin + (xMax - xMin) * sigmoid(rawX);
        const kMin = (0.5 * dt) / Math.max(1e-3, 1 - x);
        const kMax = (0.5 * dt) / Math.max(1e-3, x);
        const k = kMin + (kMax - kMin) * sigmoid(rawK);
        return { K: k, X: x };
    }

    function decodeMuskingumTheta(raw, options = {}) {
        const s = i => sigmoid(raw[i]);
        const stable = options.stableKX === true || options.paramDecode?.stableKX === true;
        if (stable) {
            const dt = finite(Number(options.paramDecode?.dt)) ? Number(options.paramDecode.dt) : 1;
            const xMin = finite(Number(options.paramDecode?.xMin)) ? Number(options.paramDecode.xMin) : 0.05;
            const xMax = finite(Number(options.paramDecode?.xMax)) ? Number(options.paramDecode.xMax) : 0.45;
            const f = decodeKX(raw[0], raw[1], dt, xMin, xMax);
            const sReach = decodeKX(raw[2], raw[3], dt, xMin, xMax);
            return {
                K_f: f.K, X_f: f.X,
                K_s: sReach.K, X_s: sReach.X,
                g_f: 0.05 + 0.95 * s(4),
                g_s: 0.05 + 0.95 * s(5),
                lam: 0.05 + 0.9 * s(6),
                alpha: 0.5 * s(7),
                spread: 0.01 + 0.25 * s(8)
            };
        }
        // Legacy unbounded decode (kept for older weight files).
        return {
            K_f: 1 + 23 * s(0),
            X_f: 0.05 + 0.4 * s(1),
            K_s: 1 + 23 * s(2),
            X_s: 0.05 + 0.4 * s(3),
            g_f: 0.05 + 0.95 * s(4),
            g_s: 0.05 + 0.95 * s(5),
            lam: 0.05 + 0.9 * s(6),
            alpha: 0.5 * s(7),
            spread: 0.01 + 0.25 * s(8)
        };
    }

    function muskingumCoeffs(K, X, dt = 1) {
        const denom = Math.max(1e-3, K - K * X + 0.5 * dt);
        return {
            c0: (-K * X + 0.5 * dt) / denom,
            c1: (K * X + 0.5 * dt) / denom,
            c2: (K - K * X - 0.5 * dt) / denom
        };
    }

    function muskingumStep(I2, I1, O1, K, X, dt = 1) {
        const { c0, c1, c2 } = muskingumCoeffs(K, X, dt);
        return c0 * I2 + c1 * I1 + c2 * O1;
    }

    function muskingumWarmup(Iseq, K, X, dt = 1) {
        if (!Iseq?.length) return { I_prev: 0, O: 0 };
        let I_prev = Iseq[0];
        let O = Iseq[0];
        for (let k = 1; k < Iseq.length; k++) {
            const I = Iseq[k];
            O = muskingumStep(I, I_prev, O, K, X, dt);
            I_prev = I;
        }
        return { I_prev, O };
    }

    function sampleNormLevel(series, tMs, mu, sd, stepMs) {
        const v = sampleAt(series, tMs);
        if (!finite(v) || !finite(mu) || !finite(sd) || sd < 1e-6) return null;
        return (v - mu) / sd;
    }

    function parseMuskingumGruNode(node) {
        if (!node || node.kind !== 'muskingum_gru') return null;
        if (!node.gru || !node.head || !node.means || !node.stds) return null;
        const lookbackSteps = Number(node.lookbackSteps);
        if (!finite(lookbackSteps) || lookbackSteps < 1) return null;
        if (!Array.isArray(node.gru.weightIh) || !Array.isArray(node.head.W)) return null;
        return {
            kind: 'muskingum_gru',
            source: 'offline',
            downstream: node.downstream || 'Cento',
            upstreams: Array.isArray(node.upstreams) ? node.upstreams.slice() : ['Forcelli', 'Sostegno Reno'],
            upstreamAlt: node.upstreamAlt || 'Bonconvento',
            lookbackSteps,
            stepMinutes: finite(Number(node.stepMinutes)) ? Number(node.stepMinutes) : 30,
            featPerStep: finite(Number(node.featPerStep)) ? Number(node.featPerStep) : 10,
            means: node.means,
            stds: node.stds,
            lags: node.lags || {},
            gru: node.gru,
            head: node.head,
            spreadLow: finite(Number(node.spreadLow)) ? Number(node.spreadLow) : 0.05,
            spreadHigh: finite(Number(node.spreadHigh)) ? Number(node.spreadHigh) : 0.05,
            reliable: node.reliable === true,
            skill120: finite(Number(node.skill120)) ? Number(node.skill120) : null,
            paramDecode: node.paramDecode || null,
            lagMinutes: [
                Number(node.lags?.Forcelli?.minutes) || 0,
                Number(node.lags?.['Sostegno Reno']?.minutes) || 0
            ],
            a: [],
            b: 0,
            c: [],
            r: finite(Number(node.skill120)) ? Number(node.skill120) : null,
            rmse: null
        };
    }

    function buildMuskingumSeq(model, sensorMap, rainSensors, nowMs, stepMs) {
        const L = model.lookbackSteps;
        const mu = model.means;
        const sd = model.stds;
        const lagF = (Number(model.lags?.Forcelli?.steps) || 0) * stepMs;
        const lagS = (Number(model.lags?.['Sostegno Reno']?.steps) || 0) * stepMs;
        const down = sensorMap[model.downstream];
        const upF = sensorMap['Forcelli'];
        const upS = sensorMap['Sostegno Reno'] || sensorMap[model.upstreamAlt];
        const seq = [];
        // Finite previous only (matches Python: missing prev ⇒ delta 0, do not carry filled 0).
        let prevC = null, prevF = null, prevS = null;
        for (let k = L - 1; k >= 0; k--) {
            const t = nowMs - k * stepMs;
            let zc = sampleNormLevel(down?.timeSeries, t, mu.Cento ?? mu[model.downstream], sd.Cento ?? sd[model.downstream], stepMs);
            let zf = sampleNormLevel(upF?.timeSeries, t - lagF, mu.Forcelli, sd.Forcelli, stepMs);
            let zs = sampleNormLevel(upS?.timeSeries, t - lagS, mu['Sostegno Reno'], sd['Sostegno Reno'], stepMs);
            if (zs == null && sensorMap[model.upstreamAlt]) {
                zs = sampleNormLevel(
                    sensorMap[model.upstreamAlt].timeSeries, t - lagS,
                    mu.Bonconvento ?? mu[model.upstreamAlt],
                    sd.Bonconvento ?? sd[model.upstreamAlt],
                    stepMs
                );
            }
            const missC = zc == null ? 1 : 0;
            const missF = zf == null ? 1 : 0;
            const missS = zs == null ? 1 : 0;
            const dzc = (zc != null && prevC != null) ? zc - prevC : 0;
            const dzf = (zf != null && prevF != null) ? zf - prevF : 0;
            const dzs = (zs != null && prevS != null) ? zs - prevS : 0;
            if (zc != null) prevC = zc;
            if (zf != null) prevF = zf;
            if (zs != null) prevS = zs;
            if (zc == null) zc = 0;
            if (zf == null) zf = 0;
            if (zs == null) zs = 0;
            let rain = 0;
            if (rainSensors?.length && finite(mu.rain) && finite(sd.rain) && sd.rain > 1e-6) {
                // Basin step rain: max across gauges (same as Python load_rain_step).
                let peak = null;
                for (const g of rainSensors) {
                    const v = sampleAt(g?.timeSeries, t);
                    if (!finite(v)) continue;
                    peak = peak == null ? v : Math.max(peak, v);
                }
                if (peak != null) rain = (peak - mu.rain) / sd.rain;
            }
            seq.push([zc, missC, dzc, zf, missF, dzf, zs, missS, dzs, rain]);
        }
        return seq;
    }

    function forecastMuskingumGru(model, sensorMap, rainSensors, horizonHours, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        const anchor = options.anchor;
        if (!finite(anchor?.value)) return [];
        const nowMs = +now;
        const steps = Math.max(1, Math.round(horizonHours * HOUR / stepMs));
        const seq = buildMuskingumSeq(model, sensorMap, rainSensors, nowMs, stepMs);
        if (!seq.length) return [];
        const h = gruForward(seq, model.gru);
        const raw = [];
        for (let i = 0; i < model.head.W.length; i++) {
            let s = model.head.b[i] || 0;
            for (let j = 0; j < h.length; j++) s += model.head.W[i][j] * h[j];
            raw.push(s);
        }
        const theta = decodeMuskingumTheta(raw, { paramDecode: model.paramDecode, stableKX: model.paramDecode?.stableKX });
        const muC = model.means.Cento;
        const sdC = model.stds.Cento;
        let z = (anchor.value - muC) / (sdC < 1e-6 ? 1 : sdC);
        const lagF = (Number(model.lags?.Forcelli?.steps) || 0) * stepMs;
        const lagS = (Number(model.lags?.['Sostegno Reno']?.steps) || 0) * stepMs;
        const upF = sensorMap.Forcelli;
        const upS = sensorMap['Sostegno Reno'] || sensorMap[model.upstreamAlt];
        const L = model.lookbackSteps;
        const warmF = [];
        const warmS = [];
        for (let k = L - 1; k >= 0; k--) {
            const t = nowMs - k * stepMs;
            warmF.push(sampleNormLevel(upF?.timeSeries, t - lagF, model.means.Forcelli, model.stds.Forcelli, stepMs) ?? z);
            let zs = sampleNormLevel(upS?.timeSeries, t - lagS, model.means['Sostegno Reno'], model.stds['Sostegno Reno'], stepMs);
            if (zs == null && sensorMap[model.upstreamAlt]) {
                zs = sampleNormLevel(
                    sensorMap[model.upstreamAlt].timeSeries, t - lagS,
                    model.means.Bonconvento ?? model.means[model.upstreamAlt],
                    model.stds.Bonconvento ?? model.stds[model.upstreamAlt],
                    stepMs
                );
            }
            warmS.push(zs ?? z);
        }
        const warmFState = muskingumWarmup(warmF, theta.K_f, theta.X_f);
        const warmSState = muskingumWarmup(warmS, theta.K_s, theta.X_s);
        let I_f_prev = warmFState.I_prev, O_f = warmFState.O;
        let I_s_prev = warmSState.I_prev, O_s = warmSState.O;
        const rain0 = seq.at(-1)?.[9] || 0;
        const points = [{
            time: anchor.time instanceof Date ? anchor.time : new Date(anchor.time),
            value: anchor.value,
            regime: 'anchor',
            spreadLow: 0,
            spreadHigh: 0
        }];
        for (let k = 1; k <= steps; k++) {
            const t = nowMs + k * stepMs;
            // Lagged upstream at t is still observed when lag ≥ horizon.
            let zf = sampleNormLevel(upF?.timeSeries, t - lagF, model.means.Forcelli, model.stds.Forcelli, stepMs);
            let zs = sampleNormLevel(upS?.timeSeries, t - lagS, model.means['Sostegno Reno'], model.stds['Sostegno Reno'], stepMs);
            if (zf == null) zf = I_f_prev;
            if (zs == null) zs = I_s_prev;
            O_f = muskingumStep(zf, I_f_prev, O_f, theta.K_f, theta.X_f);
            O_s = muskingumStep(zs, I_s_prev, O_s, theta.K_s, theta.X_s);
            const routed = theta.g_f * O_f + theta.g_s * O_s;
            z = (1 - theta.lam) * z + theta.lam * routed + theta.alpha * rain0;
            I_f_prev = zf;
            I_s_prev = zs;
            const value = z * sdC + muC;
            points.push({
                time: new Date(t),
                value,
                regime: 'muskingum',
                spreadLow: theta.spread,
                spreadHigh: theta.spread
            });
        }
        return points;
    }

    function parseWindowMlpNode(node) {
        if (!node || node.kind !== 'window_delta_mlp') return null;
        if (!Array.isArray(node.stations) || !Array.isArray(node.featureNames) || !node.horizons) return null;
        const lookbackSteps = Number(node.lookbackSteps);
        if (!finite(lookbackSteps) || lookbackSteps < 1) return null;
        const horizons = {};
        for (const [key, block] of Object.entries(node.horizons)) {
            if (!block?.layers?.length || !Array.isArray(block.mean) || !Array.isArray(block.std)) continue;
            if (block.mean.length !== node.featureNames.length || block.std.length !== node.featureNames.length) continue;
            horizons[key] = {
                layers: block.layers,
                mean: block.mean,
                std: block.std,
                layersQ10: Array.isArray(block.layersQ10) ? block.layersQ10 : null,
                layersQ50: Array.isArray(block.layersQ50) ? block.layersQ50 : null,
                layersQ90: Array.isArray(block.layersQ90) ? block.layersQ90 : null,
                conformalQhat: finite(Number(block.conformalQhat)) ? Number(block.conformalQhat) : null,
                spreadLow: finite(Number(block.spreadLow)) ? Number(block.spreadLow) : null,
                spreadHigh: finite(Number(block.spreadHigh)) ? Number(block.spreadHigh) : null,
                spreadLowHigh: finite(Number(block.spreadLowHigh)) ? Number(block.spreadLowHigh) : null,
                spreadHighHigh: finite(Number(block.spreadHighHigh)) ? Number(block.spreadHighHigh) : null,
                skill: finite(Number(block.validateEvents?.skill)) ? Number(block.validateEvents.skill)
                    : (finite(Number(block.validate?.skill)) ? Number(block.validate.skill) : null),
                rmse: finite(Number(block.validateEvents?.rmse)) ? Number(block.validateEvents.rmse)
                    : (finite(Number(block.validate?.rmse)) ? Number(block.validate.rmse) : null)
            };
        }
        if (!Object.keys(horizons).length) return null;
        const h120 = horizons['120'] || Object.values(horizons).at(-1);
        return {
            kind: 'window_delta_mlp',
            source: 'offline',
            downstream: node.downstream || 'Cento',
            stations: node.stations.slice(),
            upstreams: Array.isArray(node.upstreams) ? node.upstreams.slice() : node.stations.filter(s => s !== (node.downstream || 'Cento')),
            acceptors: Array.isArray(node.acceptors) ? node.acceptors.slice() : [],
            featureNames: node.featureNames.slice(),
            featureRecipe: node.featureRecipe && typeof node.featureRecipe === 'object' ? node.featureRecipe : null,
            lookbackSteps,
            stepMinutes: finite(Number(node.stepMinutes)) ? Number(node.stepMinutes) : 30,
            useRain: node.useRain === true,
            rainGauges: Array.isArray(node.rainGauges) ? node.rainGauges.slice() : [],
            horizons,
            reliable: node.reliable !== false && (node.skill120 == null || node.skill120 >= SKILL_KEEP),
            skill120: finite(Number(node.skill120)) ? Number(node.skill120) : (h120?.skill ?? null),
            r: h120?.skill ?? null,
            rmse: h120?.rmse ?? null,
            spreadLow: h120?.spreadLow ?? null,
            spreadHigh: h120?.spreadHigh ?? null,
            lagMinutes: Array.isArray(node.upstreams)
                ? node.upstreams.map(u => {
                    const lag = Number(node.featureRecipe?.upstreamLagMinutes?.[u]);
                    return node.featureRecipe?.upstreamLagAlign === true && finite(lag)
                        ? lag
                        : node.lookbackSteps * 30;
                })
                : [],
            a: [],
            b: 0,
            c: [],
            trainedAt: typeof node.trainedAt === 'string' ? node.trainedAt : null,
            crossingTime: node.crossingTime && typeof node.crossingTime === 'object' ? node.crossingTime : null,
            quantiles: Array.isArray(node.quantiles) ? node.quantiles.slice() : null,
            bandSource: typeof node.bandSource === 'string' ? node.bandSource : null,
            risingBand: node.risingBand && typeof node.risingBand === 'object' ? {
                upperInflate: Number(node.risingBand.upperInflate),
                lowerInflate: Number(node.risingBand.lowerInflate),
                targetCoverage: node.risingBand.targetCoverage,
                source: node.risingBand.source
            } : null,
            moeRising: node.moeRising && typeof node.moeRising === 'object' ? node.moeRising : null
        };
    }

    function buildSeqChannelMatrix(sensorMap, stations, lookback, nowMs, stepMs, rainSensors, useRain) {
        // Oldest→newest, L rows × C channels (level/delta/miss per station + optional rain).
        const rows = [];
        for (let i = 0; i < lookback; i++) {
            const t = nowMs - (lookback - 1 - i) * stepMs;
            const row = [];
            for (const station of stations) {
                const series = sensorMap?.[station]?.timeSeries;
                const level = sampleAt(series, t);
                const prev = sampleAt(series, t - stepMs);
                const miss = finite(level) ? 0 : 1;
                row.push(finite(level) ? level : null);
                row.push(finite(level) && finite(prev) ? level - prev : null);
                row.push(miss);
            }
            if (useRain) {
                const rain = basinRainAt(rainSensors, t, stepMs);
                row.push(rain.rain6h);
                row.push(rain.rain24h);
            }
            rows.push(row);
        }
        return rows;
    }

    function standardizeSeqMatrix(rows, channelNames, mean, std) {
        if (!Array.isArray(rows) || !Array.isArray(channelNames) || !Array.isArray(mean) || !Array.isArray(std)) return null;
        if (channelNames.length !== mean.length || channelNames.length !== std.length) return null;
        const out = [];
        for (const row of rows) {
            if (!Array.isArray(row) || row.length !== channelNames.length) return null;
            const stdRow = new Array(channelNames.length);
            for (let j = 0; j < channelNames.length; j++) {
                const name = channelNames[j];
                const raw = row[j];
                if (name.startsWith('miss_')) {
                    stdRow[j] = finite(raw) ? raw : 1;
                    continue;
                }
                const mu = mean[j], sd = std[j] || 1;
                if (!finite(raw)) {
                    stdRow[j] = 0;
                    continue;
                }
                stdRow[j] = (raw - mu) / (sd < 1e-6 ? 1 : sd);
            }
            out.push(stdRow);
        }
        return out;
    }

    function conv1dForward(seqLC, convLayers) {
        // seqLC: L × C → treat as channels-first C × L through each Conv1d+ReLU.
        if (!Array.isArray(seqLC) || !seqLC.length || !Array.isArray(convLayers) || !convLayers.length) return null;
        const L = seqLC.length;
        const C0 = seqLC[0].length;
        let cur = new Array(C0);
        for (let c = 0; c < C0; c++) {
            cur[c] = new Array(L);
            for (let t = 0; t < L; t++) cur[c][t] = seqLC[t][c];
        }
        for (const layer of convLayers) {
            const W = layer.W, b = layer.b;
            if (!Array.isArray(W) || !Array.isArray(b) || W.length !== b.length) return null;
            const kSize = Number(layer.kernelSize) || (W[0]?.[0]?.length || 0);
            const pad = Number.isFinite(Number(layer.padding)) ? Number(layer.padding) : Math.floor(kSize / 2);
            const inC = cur.length;
            const next = new Array(b.length);
            for (let o = 0; o < b.length; o++) {
                const outRow = new Array(L);
                const wOut = W[o];
                if (!Array.isArray(wOut) || wOut.length !== inC) return null;
                for (let t = 0; t < L; t++) {
                    let sum = b[o];
                    for (let ic = 0; ic < inC; ic++) {
                        const ker = wOut[ic];
                        if (!Array.isArray(ker) || ker.length !== kSize) return null;
                        for (let k = 0; k < kSize; k++) {
                            const src = t - pad + k;
                            const v = (src >= 0 && src < L) ? cur[ic][src] : 0;
                            sum += ker[k] * v;
                        }
                    }
                    outRow[t] = layer.activation === 'relu' ? Math.max(0, sum) : sum;
                }
                next[o] = outRow;
            }
            cur = next;
        }
        // Last timestep features for the head.
        return cur.map(ch => ch[L - 1]);
    }

    function layerNormVec(x, norm) {
        if (!Array.isArray(x) || !norm || !Array.isArray(norm.weight) || !Array.isArray(norm.bias)) return null;
        const d = x.length;
        if (norm.weight.length !== d || norm.bias.length !== d) return null;
        const eps = finite(Number(norm.eps)) ? Number(norm.eps) : 1e-5;
        let mean = 0;
        for (let i = 0; i < d; i++) mean += x[i];
        mean /= d;
        let var_ = 0;
        for (let i = 0; i < d; i++) {
            const z = x[i] - mean;
            var_ += z * z;
        }
        var_ /= d;
        const inv = 1 / Math.sqrt(var_ + eps);
        const out = new Array(d);
        for (let i = 0; i < d; i++) out[i] = (x[i] - mean) * inv * norm.weight[i] + norm.bias[i];
        return out;
    }

    function linearVec(x, layer) {
        if (!Array.isArray(x) || !layer || !Array.isArray(layer.W) || !Array.isArray(layer.b)) return null;
        const out = new Array(layer.b.length);
        for (let o = 0; o < layer.b.length; o++) {
            const row = layer.W[o];
            if (!Array.isArray(row) || row.length !== x.length) return null;
            let s = layer.b[o];
            for (let j = 0; j < x.length; j++) s += row[j] * x[j];
            out[o] = s;
        }
        return out;
    }

    function softmaxRow(logits) {
        let m = -Infinity;
        for (const v of logits) if (v > m) m = v;
        const ex = new Array(logits.length);
        let sum = 0;
        for (let i = 0; i < logits.length; i++) {
            const e = Math.exp(logits[i] - m);
            ex[i] = e;
            sum += e;
        }
        const inv = sum > 0 ? 1 / sum : 0;
        for (let i = 0; i < ex.length; i++) ex[i] *= inv;
        return ex;
    }

    /** Multi-head self-attention matching PyTorch MHA (batch_first, no mask). seq: L×d */
    function multiHeadSelfAttn(seq, attn) {
        if (!Array.isArray(seq) || !seq.length || !attn) return null;
        const L = seq.length;
        const d = Number(attn.dModel);
        const nhead = Number(attn.nhead);
        if (!finite(d) || !finite(nhead) || nhead < 1 || d % nhead !== 0) return null;
        const dh = d / nhead;
        const Wp = attn.inProjWeight;
        const bp = attn.inProjBias;
        const Wo = attn.outProjWeight;
        const bo = attn.outProjBias;
        if (!Array.isArray(Wp) || Wp.length !== 3 * d || !Array.isArray(bp) || bp.length !== 3 * d) return null;
        if (!Array.isArray(Wo) || Wo.length !== d || !Array.isArray(bo) || bo.length !== d) return null;
        const Q = new Array(L);
        const K = new Array(L);
        const V = new Array(L);
        for (let t = 0; t < L; t++) {
            const x = seq[t];
            if (!Array.isArray(x) || x.length !== d) return null;
            const q = new Array(d);
            const k = new Array(d);
            const v = new Array(d);
            for (let o = 0; o < d; o++) {
                let sq = bp[o], sk = bp[d + o], sv = bp[2 * d + o];
                const wq = Wp[o], wk = Wp[d + o], wv = Wp[2 * d + o];
                if (!Array.isArray(wq) || wq.length !== d) return null;
                for (let j = 0; j < d; j++) {
                    sq += wq[j] * x[j];
                    sk += wk[j] * x[j];
                    sv += wv[j] * x[j];
                }
                q[o] = sq;
                k[o] = sk;
                v[o] = sv;
            }
            Q[t] = q;
            K[t] = k;
            V[t] = v;
        }
        const scale = 1 / Math.sqrt(dh);
        const heads = new Array(L);
        for (let t = 0; t < L; t++) heads[t] = new Array(d);
        for (let h = 0; h < nhead; h++) {
            const off = h * dh;
            for (let qi = 0; qi < L; qi++) {
                const logits = new Array(L);
                for (let kj = 0; kj < L; kj++) {
                    let s = 0;
                    for (let c = 0; c < dh; c++) s += Q[qi][off + c] * K[kj][off + c];
                    logits[kj] = s * scale;
                }
                const w = softmaxRow(logits);
                for (let c = 0; c < dh; c++) {
                    let s = 0;
                    for (let kj = 0; kj < L; kj++) s += w[kj] * V[kj][off + c];
                    heads[qi][off + c] = s;
                }
            }
        }
        const out = new Array(L);
        for (let t = 0; t < L; t++) {
            const y = new Array(d);
            for (let o = 0; o < d; o++) {
                const row = Wo[o];
                if (!Array.isArray(row) || row.length !== d) return null;
                let s = bo[o];
                for (let j = 0; j < d; j++) s += row[j] * heads[t][j];
                y[o] = s;
            }
            out[t] = y;
        }
        return out;
    }

    function transformerForward(seqLC, tf) {
        if (!Array.isArray(seqLC) || !seqLC.length || !tf) return null;
        const L = seqLC.length;
        const d = Number(tf.dModel);
        const inProj = tf.inProj;
        if (!finite(d) || !inProj || !Array.isArray(tf.pos) || !Array.isArray(tf.layers) || !tf.norm) return null;
        if (tf.pos.length < L) return null;
        let cur = new Array(L);
        for (let t = 0; t < L; t++) {
            const proj = linearVec(seqLC[t], inProj);
            if (!proj || proj.length !== d) return null;
            const pos = tf.pos[t];
            if (!Array.isArray(pos) || pos.length !== d) return null;
            const row = new Array(d);
            for (let j = 0; j < d; j++) row[j] = proj[j] + pos[j];
            cur[t] = row;
        }
        for (const layer of tf.layers) {
            if (!layer?.attn || !layer.norm1 || !layer.norm2 || !layer.linear1 || !layer.linear2) return null;
            const n1 = new Array(L);
            for (let t = 0; t < L; t++) {
                n1[t] = layerNormVec(cur[t], layer.norm1);
                if (!n1[t]) return null;
            }
            const attnOut = multiHeadSelfAttn(n1, layer.attn);
            if (!attnOut) return null;
            const mid = new Array(L);
            for (let t = 0; t < L; t++) {
                const row = new Array(d);
                for (let j = 0; j < d; j++) row[j] = cur[t][j] + attnOut[t][j];
                mid[t] = row;
            }
            const n2 = new Array(L);
            for (let t = 0; t < L; t++) {
                n2[t] = layerNormVec(mid[t], layer.norm2);
                if (!n2[t]) return null;
            }
            const next = new Array(L);
            for (let t = 0; t < L; t++) {
                const h1 = linearVec(n2[t], layer.linear1);
                if (!h1) return null;
                for (let j = 0; j < h1.length; j++) h1[j] = Math.max(0, h1[j]);
                const h2 = linearVec(h1, layer.linear2);
                if (!h2 || h2.length !== d) return null;
                const row = new Array(d);
                for (let j = 0; j < d; j++) row[j] = mid[t][j] + h2[j];
                next[t] = row;
            }
            cur = next;
        }
        const last = layerNormVec(cur[L - 1], tf.norm);
        return last;
    }

    function seqDeltaForward(seqStd, block, kind) {
        if (!seqStd || !block) return null;
        if (kind === 'window_delta_gru') {
            if (!block.gru || !Array.isArray(block.headLayers)) return null;
            const h = gruForward(seqStd, block.gru);
            return mlpForward(h, block.headLayers);
        }
        if (kind === 'window_delta_cnn') {
            if (!Array.isArray(block.convLayers) || !Array.isArray(block.headLayers)) return null;
            const feat = conv1dForward(seqStd, block.convLayers);
            if (!feat) return null;
            return mlpForward(feat, block.headLayers);
        }
        if (kind === 'window_delta_transformer') {
            if (!block.transformer || !Array.isArray(block.headLayers)) return null;
            const feat = transformerForward(seqStd, block.transformer);
            if (!feat) return null;
            return mlpForward(feat, block.headLayers);
        }
        return null;
    }

    function parseWindowSeqNode(node) {
        const kind = node?.kind;
        if (kind !== 'window_delta_gru' && kind !== 'window_delta_cnn' && kind !== 'window_delta_transformer') return null;
        if (!Array.isArray(node.stations) || !Array.isArray(node.channelNames) || !node.horizons) return null;
        const lookbackSteps = Number(node.lookbackSteps);
        if (!finite(lookbackSteps) || lookbackSteps < 1) return null;
        const nC = node.channelNames.length;
        const horizons = {};
        for (const [key, block] of Object.entries(node.horizons)) {
            if (!Array.isArray(block?.mean) || !Array.isArray(block?.std)) continue;
            if (block.mean.length !== nC || block.std.length !== nC) continue;
            if (kind === 'window_delta_gru' && (!block.gru || !Array.isArray(block.headLayers))) continue;
            if (kind === 'window_delta_cnn' && (!Array.isArray(block.convLayers) || !Array.isArray(block.headLayers))) continue;
            if (kind === 'window_delta_transformer' && (!block.transformer || !Array.isArray(block.headLayers))) continue;
            horizons[key] = {
                mean: block.mean,
                std: block.std,
                gru: block.gru || null,
                convLayers: block.convLayers || null,
                transformer: block.transformer || null,
                headLayers: block.headLayers,
                spreadLow: finite(Number(block.spreadLow)) ? Number(block.spreadLow) : null,
                spreadHigh: finite(Number(block.spreadHigh)) ? Number(block.spreadHigh) : null,
                spreadLowHigh: finite(Number(block.spreadLowHigh)) ? Number(block.spreadLowHigh) : null,
                spreadHighHigh: finite(Number(block.spreadHighHigh)) ? Number(block.spreadHighHigh) : null,
                skill: finite(Number(block.validateEvents?.skill)) ? Number(block.validateEvents.skill)
                    : (finite(Number(block.validate?.skill)) ? Number(block.validate.skill) : null),
                rmse: finite(Number(block.validateEvents?.rmse)) ? Number(block.validateEvents.rmse)
                    : (finite(Number(block.validate?.rmse)) ? Number(block.validate.rmse) : null)
            };
        }
        if (!Object.keys(horizons).length) return null;
        const h120 = horizons['120'] || Object.values(horizons).at(-1);
        return {
            kind,
            source: 'offline',
            downstream: node.downstream || 'Cento',
            stations: node.stations.slice(),
            upstreams: Array.isArray(node.upstreams) ? node.upstreams.slice() : node.stations.filter(s => s !== (node.downstream || 'Cento')),
            channelNames: node.channelNames.slice(),
            lookbackSteps,
            stepMinutes: finite(Number(node.stepMinutes)) ? Number(node.stepMinutes) : 30,
            useRain: node.useRain === true,
            rainGauges: Array.isArray(node.rainGauges) ? node.rainGauges.slice() : [],
            horizons,
            reliable: node.reliable !== false && (node.skill120 == null || node.skill120 >= SKILL_KEEP),
            skill120: finite(Number(node.skill120)) ? Number(node.skill120) : (h120?.skill ?? null),
            r: h120?.skill ?? null,
            rmse: h120?.rmse ?? null,
            spreadLow: h120?.spreadLow ?? null,
            spreadHigh: h120?.spreadHigh ?? null,
            lagMinutes: Array.isArray(node.upstreams) ? node.upstreams.map(() => lookbackSteps * 30) : [],
            a: [],
            b: 0,
            c: [],
            trainedAt: typeof node.trainedAt === 'string' ? node.trainedAt : null
        };
    }

    function forecastWindowSeq(model, sensorMap, rainSensors, horizonHours, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        const anchor = options.anchor;
        if (!finite(anchor?.value) || !model?.horizons) return Object.assign([], { bias: null });
        const nowMs = +now;
        const end = nowMs + horizonHours * HOUR;
        const y0 = anchor.value;
        const yPrev = sampleAt(options.downstream?.timeSeries, +anchor.time - stepMs);
        const dyLocal = finite(yPrev) ? y0 - yPrev : 0;
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
        let maxStep = options.maxStep;
        if (finite(maxStep) && finite(dyLocal)) {
            maxStep = Math.max(maxStep, Math.abs(dyLocal));
        }
        const smooth = finite(options.smooth) ? Math.min(0.9, Math.max(0, options.smooth)) : 0;
        const rawSeq = buildSeqChannelMatrix(
            sensorMap, model.stations, model.lookbackSteps, nowMs, stepMs, rainSensors, model.useRain
        );
        const horizonKeys = Object.keys(model.horizons).map(Number).filter(Number.isFinite).sort((a, b) => a - b)
            .filter(minutes => minutes * MINUTE <= horizonHours * HOUR + 1);
        for (const blockMinutes of horizonKeys) {
            const block = model.horizons[String(blockMinutes)];
            if (!block) continue;
            const target = Math.ceil((nowMs + blockMinutes * MINUTE) / stepMs) * stepMs;
            if (target > end + 1) continue;
            const seqStd = standardizeSeqMatrix(rawSeq, model.channelNames, block.mean, block.std);
            const delta = seqDeltaForward(seqStd, block, model.kind);
            if (!finite(delta)) continue;
            let value = smooth * previous + (1 - smooth) * (y0 + delta);
            if (finite(maxStep) && finite(previous)) {
                const steps = Math.max(1, Math.round((target - (points.at(-1) ? +points.at(-1).time : nowMs)) / stepMs));
                const d = value - previous;
                if (Math.abs(d) > maxStep * steps) value = previous + Math.sign(d || 1) * maxStep * steps;
            }
            let recentMove = Math.abs(dyLocal) >= 0.05;
            if (!recentMove && options.downstream?.timeSeries) {
                const yRecent = sampleAt(options.downstream.timeSeries, nowMs - 2 * HOUR);
                if (finite(yRecent)) recentMove = Math.abs(y0 - yRecent) >= 0.05;
            }
            const useHigh = Math.abs(delta) >= 0.05 || recentMove;
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
        points.bias = null;
        return points;
    }

    // M3: GRU short + ridge long with linear weight in [blendStartMin, blendEndMin].
    function forecastBlendGruRidge(gruModel, ridgeModel, upstreams, sensorMap, rainSensors, horizonHours, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        const blendStart = finite(Number(options.blendStartMin)) ? Number(options.blendStartMin) : 90;
        const blendEnd = finite(Number(options.blendEndMin)) ? Number(options.blendEndMin) : 150;
        const span = Math.max(1, blendEnd - blendStart);
        const gruHorizon = Math.min(horizonHours, blendEnd / 60);
        const gruPts = forecastWindowSeq(gruModel, sensorMap, rainSensors, gruHorizon, now, stepMs, options);
        // Long leg: ridge by default; options.mlpModel switches to the window MLP (blend_gru_mlp).
        const ridgePts = options.mlpModel
            ? forecastWindowMlp(options.mlpModel, sensorMap, rainSensors, horizonHours, now, stepMs, options)
            : forecastDirectHorizon(upstreams, ridgeModel, horizonHours, now, stepMs, options);
        const longRegime = options.mlpModel ? 'mlp' : 'ridge';
        const nowMs = +now;
        const gruByT = new Map();
        for (const p of gruPts) {
            if (p.regime === 'anchor') continue;
            gruByT.set(+p.time, p);
        }
        const ridgeByT = new Map();
        for (const p of ridgePts) {
            if (p.regime === 'anchor') continue;
            ridgeByT.set(+p.time, p);
        }
        const times = [...new Set([...gruByT.keys(), ...ridgeByT.keys()])].sort((a, b) => a - b);
        const points = [];
        if (options.anchor?.time) {
            points.push({
                time: options.anchor.time instanceof Date ? options.anchor.time : new Date(options.anchor.time),
                value: options.anchor.value,
                regime: 'anchor',
                spreadLow: 0,
                spreadHigh: 0
            });
        }
        for (const t of times) {
            if (t > nowMs + horizonHours * HOUR + 1) continue;
            const minutes = (t - nowMs) / MINUTE;
            const g = gruByT.get(t);
            const r = ridgeByT.get(t);
            let value = null;
            let spreadLow = null;
            let spreadHigh = null;
            let regime = 'blend';
            if (minutes <= blendStart && g && finite(g.value)) {
                value = g.value;
                spreadLow = g.spreadLow;
                spreadHigh = g.spreadHigh;
                regime = 'gru';
            } else if (minutes >= blendEnd && r && finite(r.value)) {
                value = r.value;
                spreadLow = r.spreadLow;
                spreadHigh = r.spreadHigh;
                regime = longRegime;
            } else if (g && r && finite(g.value) && finite(r.value)) {
                const w = Math.min(1, Math.max(0, (minutes - blendStart) / span));
                value = (1 - w) * g.value + w * r.value;
                const loG = finite(g.spreadLow) ? g.value - g.spreadLow : g.value;
                const hiG = finite(g.spreadHigh) ? g.value + g.spreadHigh : g.value;
                const loR = finite(r.spreadLow) ? r.value - r.spreadLow : r.value;
                const hiR = finite(r.spreadHigh) ? r.value + r.spreadHigh : r.value;
                const lo = (1 - w) * loG + w * loR;
                const hi = (1 - w) * hiG + w * hiR;
                spreadLow = Math.max(0, value - Math.min(lo, hi));
                spreadHigh = Math.max(0, Math.max(lo, hi) - value);
                regime = 'blend';
            } else if (g && finite(g.value)) {
                value = g.value;
                spreadLow = g.spreadLow;
                spreadHigh = g.spreadHigh;
                regime = 'gru';
            } else if (r && finite(r.value)) {
                value = r.value;
                spreadLow = r.spreadLow;
                spreadHigh = r.spreadHigh;
                regime = longRegime;
            }
            if (!finite(value)) continue;
            points.push({ time: new Date(t), value, regime, spreadLow, spreadHigh });
        }
        points.bias = null;
        return points;
    }

    function forecastWindowMlp(model, sensorMap, rainSensors, horizonHours, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        const anchor = options.anchor;
        if (!finite(anchor?.value) || !model?.horizons) return Object.assign([], { bias: null });
        const nowMs = +now;
        const end = nowMs + horizonHours * HOUR;
        const y0 = anchor.value;
        const yPrev = sampleAt(options.downstream?.timeSeries, +anchor.time - stepMs);
        const dyLocal = finite(yPrev) ? y0 - yPrev : 0;
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
        // Feature vector is independent of h unless lag-align or residual baselines use it.
        const tModeRecipe = model.featureRecipe?.targetMode;
        const featuresDependOnHorizon = model.featureRecipe?.upstreamLagAlign === true
            || tModeRecipe === 'residual_lin3'
            || tModeRecipe === 'residual_poly2';
        const reuseFeatures = !featuresDependOnHorizon;
        let sharedRaw = null;
        // Floor maxStep with the last observed 30′ move so quiet lookback p95 cannot
        // cancel an already-rising limb (causal; does not invent peaks).
        let maxStep = options.maxStep;
        if (finite(maxStep) && finite(dyLocal)) {
            maxStep = Math.max(maxStep, Math.abs(dyLocal));
        }
        // Independent horizon MLPs already target y0+Δh. Default smooth=0 (raw knots).
        // Optional EMA (legacy 0.65) shrinks the first increment to (1−smooth)·Δ — see tests.
        const smooth = finite(options.smooth) ? Math.min(0.9, Math.max(0, options.smooth)) : 0;
        const horizonKeys = Object.keys(model.horizons).map(Number).filter(Number.isFinite).sort((a, b) => a - b)
            .filter(minutes => minutes * MINUTE <= horizonHours * HOUR + 1);
        for (const blockMinutes of horizonKeys) {
            const block = model.horizons[String(blockMinutes)];
            if (!block) continue;
            const target = Math.ceil((nowMs + blockMinutes * MINUTE) / stepMs) * stepMs;
            if (target > end + 1) continue;
            const raw = reuseFeatures && sharedRaw
                ? sharedRaw
                : buildWindowFeatureVector(
                    sensorMap, model.stations, model.lookbackSteps, nowMs,
                    (finite(Number(model.stepMinutes)) ? Number(model.stepMinutes) : 30) * MINUTE,
                    rainSensors, model.useRain,
                    model.featureRecipe, reuseFeatures ? 0 : blockMinutes
                );
            if (reuseFeatures && !sharedRaw) sharedRaw = raw;
            const x = standardizeWindowFeatures(raw, model.featureNames, block.mean, block.std);
            if (!x) continue;
            const layersMed = block.layersQ50 || block.layers;
            const mlpOut = mlpForward(x, layersMed);
            if (!finite(mlpOut)) continue;
            // Residual modes: MLP predicts r̂; absolute Δ = baseline_delta + r̂ (ŷ = y0 + Δ).
            const tMode = model.featureRecipe?.targetMode;
            const residualMode = tMode === 'residual_lin3' || tMode === 'residual_poly2';
            const downTag = safeFeatureName(model.downstream || model.stations[0] || 'Cento');
            const baseFeat = tMode === 'residual_poly2'
                ? `poly2_delta_${downTag}`
                : `lin3_delta_${downTag}`;
            const baseDelta = residualMode && finite(raw[baseFeat]) ? raw[baseFeat] : 0;
            const delta = residualMode ? mlpOut + baseDelta : mlpOut;
            let value = smooth * previous + (1 - smooth) * (y0 + delta);
            if (finite(maxStep) && finite(previous)) {
                const steps = Math.max(1, Math.round((target - (points.at(-1) ? +points.at(-1).time : nowMs)) / stepMs));
                const d = value - previous;
                if (Math.abs(d) > maxStep * steps) value = previous + Math.sign(d || 1) * maxStep * steps;
            }
            let recentMove = Math.abs(dyLocal) >= 0.05;
            if (!recentMove && options.downstream?.timeSeries) {
                const yRecent = sampleAt(options.downstream.timeSeries, nowMs - 2 * HOUR);
                if (finite(yRecent)) recentMove = Math.abs(y0 - yRecent) >= 0.05;
            }
            const useHigh = Math.abs(delta) >= 0.05 || recentMove;
            let spreadLow = useHigh
                ? (finite(block.spreadLowHigh) ? block.spreadLowHigh : block.spreadLow)
                : block.spreadLow;
            let spreadHigh = useHigh
                ? (finite(block.spreadHighHigh) ? block.spreadHighHigh : block.spreadHigh)
                : block.spreadHigh;
            // M1: pinball q10/q90 band (+ optional conformal inflate on Δ)
            if (block.layersQ10 && block.layersQ90) {
                const d10raw = mlpForward(x, block.layersQ10);
                const d90raw = mlpForward(x, block.layersQ90);
                const d10 = residualMode && finite(d10raw) ? d10raw + baseDelta : d10raw;
                const d90 = residualMode && finite(d90raw) ? d90raw + baseDelta : d90raw;
                const qhat = finite(block.conformalQhat) ? block.conformalQhat : 0;
                if (finite(d10) && finite(d90)) {
                    const lo = y0 + Math.min(d10, d90) - qhat;
                    const hi = y0 + Math.max(d10, d90) + qhat;
                    spreadLow = Math.max(0, value - lo);
                    spreadHigh = Math.max(0, hi - value);
                }
            }
            points.push({
                time: new Date(target),
                value,
                regime: useHigh ? 'high' : 'base',
                spreadLow,
                spreadHigh
            });
            previous = value;
        }
        points.bias = null;
        return points;
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

    // Online phase + level bias for window_delta_mlp (A/B; default off in predictNode).
    // Reuses forecastWindowMlp on a short causal window — never calls predictNode (no recursion).
    const ONLINE_PHASE_MAX_MIN = 60; // gated ±2 steps @30′
    const ONLINE_CORR_MIN_BIAS = 4;
    const ONLINE_CORR_MIN_LAG = 10;
    const ONLINE_CORR_MIN_R = 0.5;

    function estimateMlpOnlineCorrection(model, sensorMap, rainSensors, downstream, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        if (!model?.horizons || !downstream?.timeSeries?.length) {
            return { bias: 0, phaseShiftMin: 0, lag: null, nBias: 0, nLag: 0 };
        }
        const nowMs = +now;
        const windowHours = finite(options.onlineWindowHours) ? options.onlineWindowHours : 12;
        const horizonHours = 2;
        const stride = Math.max(stepMs, HOUR);
        const start = nowMs - windowHours * HOUR;
        const lagPoints = [];
        let ema = 0;
        let count = 0;
        for (let origin = start; origin <= nowMs - horizonHours * HOUR + 1; origin += stride) {
            const downT = truncateSensor(downstream, origin);
            const anchor = latestFinite(downT?.timeSeries);
            if (!anchor || origin - +anchor.time > stepMs) continue;
            const mapT = {};
            for (const [key, sensor] of Object.entries(sensorMap || {})) {
                mapT[key] = truncateSensor(sensor, origin);
            }
            if (downT?.id) mapT[downT.id] = downT;
            const rainT = (rainSensors || []).map(sensor => truncateSensor(sensor, origin));
            const forecast = forecastWindowMlp(model, mapT, rainT, horizonHours, new Date(origin), stepMs, {
                anchor,
                downstream: downT,
                maxStep: options.maxStep,
                smooth: 0
            });
            let firstKnot = null;
            for (const point of forecast) {
                if (point.regime === 'anchor') continue;
                if (+point.time > nowMs) continue;
                const observed = sampleAt(downstream.timeSeries, +point.time);
                if (!finite(observed) || !finite(point.value)) continue;
                lagPoints.push({ predicted: point.value, observed });
                if (!firstKnot) firstKnot = point;
            }
            if (firstKnot) {
                const observed = sampleAt(downstream.timeSeries, +firstKnot.time);
                if (finite(observed)) {
                    const resid = firstKnot.value - observed;
                    ema = count === 0 ? resid : (1 - BIAS_EMA_ALPHA) * ema + BIAS_EMA_ALPHA * resid;
                    count++;
                }
            }
        }
        const lag = estimatePredictedLag(
            lagPoints.map(point => ({ predicted: point.predicted, observed: point.observed })),
            stepMs
        );
        let phaseShiftMin = 0;
        if (lag && lagPoints.length >= ONLINE_CORR_MIN_LAG && finite(lag.lagMin) && Math.abs(lag.r) >= ONLINE_CORR_MIN_R) {
            const steps = Math.round(lag.lagMin / 30);
            phaseShiftMin = Math.max(-ONLINE_PHASE_MAX_MIN, Math.min(ONLINE_PHASE_MAX_MIN, steps * 30));
        }
        return {
            bias: count >= ONLINE_CORR_MIN_BIAS ? ema : 0,
            phaseShiftMin,
            lag,
            nBias: count,
            nLag: lagPoints.length
        };
    }

    function applyMlpOnlineCorrection(forecast, correction) {
        if (!Array.isArray(forecast) || !correction) return forecast;
        const bias = finite(correction.bias) ? correction.bias : 0;
        const phaseMs = finite(correction.phaseShiftMin) ? correction.phaseShiftMin * MINUTE : 0;
        if (!bias && !phaseMs) return forecast;
        return forecast.map(point => {
            if (point.regime === 'anchor') return point;
            const next = { ...point };
            if (bias) {
                next.value = point.value - bias;
                if (finite(point.med)) next.med = point.med - bias;
                if (finite(point.min)) next.min = point.min - bias;
                if (finite(point.max)) next.max = point.max - bias;
            }
            if (phaseMs) next.time = new Date(+point.time - phaseMs);
            return next;
        });
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

    function softplus(x, beta = 5) {
        // Stable softplus
        const bx = beta * x;
        if (bx > 20) return x;
        if (bx < -20) return Math.exp(bx) / beta;
        return Math.log1p(Math.exp(bx)) / beta;
    }

    function decodeRcTheta(raw, paramDecode = {}) {
        const gMin = finite(Number(paramDecode.gMin)) ? Number(paramDecode.gMin) : 0.02;
        const gMax = finite(Number(paramDecode.gMax)) ? Number(paramDecode.gMax) : 2.5;
        const cMin = finite(Number(paramDecode.cMin)) ? Number(paramDecode.cMin) : 0.5;
        const cMax = finite(Number(paramDecode.cMax)) ? Number(paramDecode.cMax) : 12;
        const pos = (i, lo, hi) => lo + (hi - lo) * sigmoid(raw[i]);
        return {
            G_f: pos(0, gMin, gMax),
            G_s: pos(1, gMin, gMax),
            G_d: pos(2, gMin, gMax),
            G_p: pos(3, gMin, gMax),
            G_2: pos(4, gMin, gMax),
            G_g: pos(5, gMin, gMax),
            C_c: pos(6, cMin, cMax),
            alpha: 0.5 * sigmoid(raw[7]),
            spread: 0.01 + 0.25 * sigmoid(raw[8])
        };
    }

    function rcStep(z, zf, zs, zAcc, rain, theta, dt = 1) {
        const gAcc = [theta.G_d, theta.G_p, theta.G_2, theta.G_g];
        let drain = 0;
        for (let i = 0; i < 4; i++) drain += gAcc[i] * softplus(z - (zAcc[i] ?? 0));
        const inflow = theta.G_f * (zf - z) + theta.G_s * (zs - z);
        const cc = Math.max(theta.C_c, 0.5);
        return z + (inflow - drain + theta.alpha * rain) * (dt / cc);
    }

    function parseRcGruNode(node) {
        if (!node || node.kind !== 'rc_gru') return null;
        if (!node.gru || !node.head || !node.means || !node.stds) return null;
        const lookbackSteps = Number(node.lookbackSteps);
        if (!finite(lookbackSteps) || lookbackSteps < 1) return null;
        if (!Array.isArray(node.gru.weightIh) || !Array.isArray(node.head.W)) return null;
        return {
            kind: 'rc_gru',
            source: 'offline',
            downstream: node.downstream || 'Cento',
            upstreams: Array.isArray(node.upstreams) ? node.upstreams.slice() : ['Forcelli', 'Sostegno Reno'],
            upstreamAlt: node.upstreamAlt || 'Bonconvento',
            acceptors: Array.isArray(node.acceptors) ? node.acceptors.slice()
                : ['Dosso', 'Opera Reno Panfilia', 'Secondo Salto', 'Gallo'],
            lookbackSteps,
            stepMinutes: finite(Number(node.stepMinutes)) ? Number(node.stepMinutes) : 30,
            featPerStep: finite(Number(node.featPerStep)) ? Number(node.featPerStep) : 22,
            means: node.means,
            stds: node.stds,
            lags: node.lags || {},
            gru: node.gru,
            head: node.head,
            paramDecode: node.paramDecode || null,
            spreadLow: finite(Number(node.spreadLow)) ? Number(node.spreadLow) : 0.05,
            spreadHigh: finite(Number(node.spreadHigh)) ? Number(node.spreadHigh) : 0.05,
            reliable: node.reliable === true,
            skill120: finite(Number(node.skill120)) ? Number(node.skill120) : null,
            lagMinutes: [
                Number(node.lags?.Forcelli?.minutes) || 0,
                Number(node.lags?.['Sostegno Reno']?.minutes) || 0
            ],
            a: [],
            b: 0,
            c: [],
            r: finite(Number(node.skill120)) ? Number(node.skill120) : null,
            rmse: null
        };
    }

    function buildRcSeq(model, sensorMap, rainSensors, nowMs, stepMs) {
        const L = model.lookbackSteps;
        const mu = model.means;
        const sd = model.stds;
        const lagF = (Number(model.lags?.Forcelli?.steps) || 0) * stepMs;
        const lagS = (Number(model.lags?.['Sostegno Reno']?.steps) || 0) * stepMs;
        const acceptors = model.acceptors || [];
        const down = sensorMap[model.downstream];
        const upF = sensorMap.Forcelli;
        const upS = sensorMap['Sostegno Reno'] || sensorMap[model.upstreamAlt];
        const seq = [];
        let prevC = null, prevF = null, prevS = null;
        const prevA = acceptors.map(() => null);
        for (let k = L - 1; k >= 0; k--) {
            const t = nowMs - k * stepMs;
            let zc = sampleNormLevel(down?.timeSeries, t, mu.Cento ?? mu[model.downstream], sd.Cento ?? sd[model.downstream], stepMs);
            let zf = sampleNormLevel(upF?.timeSeries, t - lagF, mu.Forcelli, sd.Forcelli, stepMs);
            let zs = sampleNormLevel(upS?.timeSeries, t - lagS, mu['Sostegno Reno'], sd['Sostegno Reno'], stepMs);
            if (zs == null && sensorMap[model.upstreamAlt]) {
                zs = sampleNormLevel(
                    sensorMap[model.upstreamAlt].timeSeries, t - lagS,
                    mu.Bonconvento ?? mu[model.upstreamAlt],
                    sd.Bonconvento ?? sd[model.upstreamAlt],
                    stepMs
                );
            }
            const row = [];
            const pushSeries = (z, prev) => {
                const miss = z == null ? 1 : 0;
                const dz = (z != null && prev != null) ? z - prev : 0;
                const nextPrev = z != null ? z : prev;
                row.push(z == null ? 0 : z, miss, dz);
                return nextPrev;
            };
            prevC = pushSeries(zc, prevC);
            prevF = pushSeries(zf, prevF);
            prevS = pushSeries(zs, prevS);
            for (let ai = 0; ai < 4; ai++) {
                const name = acceptors[ai];
                let za = null;
                if (name) za = sampleNormLevel(sensorMap[name]?.timeSeries, t, mu[name], sd[name], stepMs);
                prevA[ai] = pushSeries(za, prevA[ai]);
            }
            let rain = 0;
            if (rainSensors?.length && finite(mu.rain) && finite(sd.rain) && sd.rain > 1e-6) {
                let peak = null;
                for (const g of rainSensors) {
                    const v = sampleAt(g?.timeSeries, t);
                    if (!finite(v)) continue;
                    peak = peak == null ? v : Math.max(peak, v);
                }
                if (peak != null) rain = (peak - mu.rain) / sd.rain;
            }
            row.push(rain);
            seq.push(row);
        }
        return seq;
    }

    function forecastRcGru(model, sensorMap, rainSensors, horizonHours, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        const anchor = options.anchor;
        if (!finite(anchor?.value)) return [];
        const nowMs = +now;
        const steps = Math.max(1, Math.round(horizonHours * HOUR / stepMs));
        const seq = buildRcSeq(model, sensorMap, rainSensors, nowMs, stepMs);
        if (!seq.length) return [];
        const h = gruForward(seq, model.gru);
        const raw = [];
        for (let i = 0; i < model.head.W.length; i++) {
            let s = model.head.b[i] || 0;
            for (let j = 0; j < h.length; j++) s += model.head.W[i][j] * h[j];
            raw.push(s);
        }
        const theta = decodeRcTheta(raw, model.paramDecode || {});
        const muC = model.means.Cento;
        const sdC = model.stds.Cento;
        let z = (anchor.value - muC) / (sdC < 1e-6 ? 1 : sdC);
        const lagF = (Number(model.lags?.Forcelli?.steps) || 0) * stepMs;
        const lagS = (Number(model.lags?.['Sostegno Reno']?.steps) || 0) * stepMs;
        const upF = sensorMap.Forcelli;
        const upS = sensorMap['Sostegno Reno'] || sensorMap[model.upstreamAlt];
        const acceptors = model.acceptors || [];
        // Freeze acceptors at last observed (backwater persistence).
        const zAcc0 = [];
        for (let ai = 0; ai < 4; ai++) {
            const name = acceptors[ai];
            let za = name
                ? sampleNormLevel(sensorMap[name]?.timeSeries, nowMs, model.means[name], model.stds[name], stepMs)
                : null;
            zAcc0.push(za == null ? z : za);
        }
        const rain0 = seq.at(-1)?.at(-1) || 0;
        // Warm lookback
        let zWarm = seq[0][0];
        for (let k = 1; k < seq.length; k++) {
            const zf = seq[k][3];
            const zs = seq[k][6];
            const zAcc = [seq[k][9], seq[k][12], seq[k][15], seq[k][18]];
            const rain = seq[k][21] || 0;
            zWarm = rcStep(zWarm, zf, zs, zAcc, rain, theta);
        }
        z = (anchor.value - muC) / (sdC < 1e-6 ? 1 : sdC);
        const points = [{
            time: anchor.time instanceof Date ? anchor.time : new Date(anchor.time),
            value: anchor.value,
            regime: 'anchor',
            spreadLow: 0,
            spreadHigh: 0
        }];
        for (let k = 1; k <= steps; k++) {
            const t = nowMs + k * stepMs;
            let zf = sampleNormLevel(upF?.timeSeries, t - lagF, model.means.Forcelli, model.stds.Forcelli, stepMs);
            let zs = sampleNormLevel(upS?.timeSeries, t - lagS, model.means['Sostegno Reno'], model.stds['Sostegno Reno'], stepMs);
            if (zf == null) zf = z;
            if (zs == null) zs = z;
            z = rcStep(z, zf, zs, zAcc0, rain0, theta);
            points.push({
                time: new Date(t),
                value: z * sdC + muC,
                regime: 'rc',
                spreadLow: theta.spread,
                spreadHigh: theta.spread
            });
        }
        return points;
    }

    function parseBlendGruRidgeNode(node) {
        if (!node || node.kind !== 'blend_gru_ridge') return null;
        const gru = parseWindowSeqNode(node.gru || node.seq);
        const ridgeRaw = node.ridge || node.direct;
        if (!gru || !ridgeRaw) return null;
        const ups = Array.isArray(ridgeRaw.upstreams) ? ridgeRaw.upstreams : (Array.isArray(node.upstreams) ? node.upstreams : []);
        const ridge = parseOfflineNode(
            { nodes: { [node.downstream || gru.downstream || '_']: { ...ridgeRaw, kind: ridgeRaw.kind || 'direct_horizon_delta_ridge' } } },
            node.downstream || gru.downstream || '_',
            ups
        );
        if (!ridge || ridge.kind !== 'direct_horizon_delta_ridge') return null;
        const skill120 = finite(Number(node.skill120)) ? Number(node.skill120)
            : (finite(gru.skill120) ? gru.skill120 : ridge.skill120);
        return {
            kind: 'blend_gru_ridge',
            source: 'offline',
            downstream: node.downstream || gru.downstream,
            gru,
            ridge,
            blendStartMin: finite(Number(node.blendStartMin)) ? Number(node.blendStartMin) : 90,
            blendEndMin: finite(Number(node.blendEndMin)) ? Number(node.blendEndMin) : 150,
            upstreams: ridge.upstreams.slice(),
            lagMinutes: ridge.lagMinutes.slice(),
            horizons: { ...(ridge.horizons || {}), ...(gru.horizons || {}) },
            reliable: node.reliable !== false && (skill120 == null || skill120 >= SKILL_KEEP),
            skill120,
            r: skill120,
            rmse: ridge.rmse ?? gru.rmse ?? null,
            spreadLow: gru.spreadLow ?? ridge.spreadLow,
            spreadHigh: gru.spreadHigh ?? ridge.spreadHigh,
            a: ridge.a || [],
            b: ridge.b || 0,
            c: ridge.c || [],
            trainedAt: typeof node.trainedAt === 'string' ? node.trainedAt : null
        };
    }

    function parseBlendGruMlpNode(node) {
        if (!node || node.kind !== 'blend_gru_mlp') return null;
        const gru = parseWindowSeqNode(node.gru || node.seq);
        const mlp = parseWindowMlpNode(node.mlp);
        if (!gru || !mlp) return null;
        const skill120 = finite(Number(node.skill120)) ? Number(node.skill120)
            : (finite(gru.skill120) ? gru.skill120 : mlp.skill120);
        const upstreams = [...new Set([...(gru.upstreams || []), ...(mlp.upstreams || [])])];
        return {
            kind: 'blend_gru_mlp',
            source: 'offline',
            downstream: node.downstream || gru.downstream || mlp.downstream,
            gru,
            mlp,
            blendStartMin: finite(Number(node.blendStartMin)) ? Number(node.blendStartMin) : 60,
            blendEndMin: finite(Number(node.blendEndMin)) ? Number(node.blendEndMin) : 120,
            upstreams,
            lagMinutes: upstreams.map(() => (mlp.lookbackSteps || 0) * (mlp.stepMinutes || 30)),
            horizons: { ...(mlp.horizons || {}), ...(gru.horizons || {}) },
            reliable: node.reliable !== false && (skill120 == null || skill120 >= SKILL_KEEP),
            skill120,
            r: skill120,
            rmse: mlp.rmse ?? gru.rmse ?? null,
            spreadLow: gru.spreadLow ?? mlp.spreadLow,
            spreadHigh: gru.spreadHigh ?? mlp.spreadHigh,
            a: [],
            b: 0,
            c: [],
            trainedAt: typeof node.trainedAt === 'string' ? node.trainedAt : null
        };
    }

    function parseOfflineNode(coeffs, downstreamId, upstreamIds) {
        const node = coeffs?.nodes?.[downstreamId];
        if (!node) return null;
        if (node.kind === 'rc_gru') return parseRcGruNode(node);
        if (node.kind === 'muskingum_gru') return parseMuskingumGruNode(node);
        if (node.kind === 'blend_gru_ridge') return parseBlendGruRidgeNode(node);
        if (node.kind === 'blend_gru_mlp') return parseBlendGruMlpNode(node);
        if (node.kind === 'window_delta_mlp') return parseWindowMlpNode(node);
        if (node.kind === 'window_delta_gru' || node.kind === 'window_delta_cnn' || node.kind === 'window_delta_transformer') {
            return parseWindowSeqNode(node);
        }
        if (!Array.isArray(node.upstreams) || !Array.isArray(node.lagMinutes)) return null;
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
    // Knots are independent per horizon; densify at stepMs with linear Δ interp + maxStep + light EMA.
    function forecastDirectHorizon(upstreams, model, horizonHours, now = new Date(), stepMs = DEFAULT_STEP_MS, options = {}) {
        const anchor = options.anchor;
        if (!finite(anchor?.value) || !model?.horizons) return Object.assign([], { bias: null });
        const nowMs = +now;
        const t0 = +anchor.time || nowMs;
        const end = Math.max(nowMs, t0) + horizonHours * HOUR;
        const y0 = anchor.value;
        const yPrev = sampleAt(options.downstream?.timeSeries, t0 - stepMs);
        const dyLocal = finite(yPrev) ? y0 - yPrev : 0;
        const bias = finite(options.bias) ? options.bias : 0;
        const high = driverIsHigh(upstreams, model, nowMs, stepMs);
        const rain = basinRainAt(options.rainSensors, nowMs, stepMs);
        let maxStep = options.maxStep;
        if (finite(maxStep) && finite(dyLocal)) {
            maxStep = Math.max(maxStep, Math.abs(dyLocal));
        }
        // Knots are independent per horizon; heavy EMA only delays the curve.
        const smooth = finite(options.smooth) ? Math.min(0.9, Math.max(0, options.smooth)) : 0.15;

        let recentMove = Math.abs(dyLocal) >= 0.05;
        if (!recentMove && options.downstream?.timeSeries) {
            const yRecent = sampleAt(options.downstream.timeSeries, nowMs - 2 * HOUR);
            if (finite(yRecent)) recentMove = Math.abs(y0 - yRecent) >= 0.05;
        }

        const horizonKeys = Object.keys(model.horizons).map(Number).filter(Number.isFinite).sort((a, b) => a - b)
            .filter(minutes => minutes * MINUTE <= horizonHours * HOUR + 1);
        const knots = [{ time: t0, value: y0, regime: 'anchor', spreadLow: 0, spreadHigh: 0, delta: 0 }];
        for (const blockMinutes of horizonKeys) {
            const block = model.horizons[String(blockMinutes)];
            if (!block) continue;
            const target = t0 + blockMinutes * MINUTE;
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
            const useHigh = high || Math.abs(delta) >= 0.05 || recentMove;
            knots.push({
                time: target,
                value: y0 + delta,
                delta,
                regime: useHigh ? 'high' : 'base',
                spreadLow: useHigh
                    ? (finite(block.spreadLowHigh) ? block.spreadLowHigh : block.spreadLow)
                    : block.spreadLow,
                spreadHigh: useHigh
                    ? (finite(block.spreadHighHigh) ? block.spreadHighHigh : block.spreadHigh)
                    : block.spreadHigh
            });
        }

        function interpKnot(t) {
            if (knots.length === 1) return knots[0];
            if (t <= knots[0].time) return knots[0];
            if (t >= knots.at(-1).time) return knots.at(-1);
            let i = 1;
            while (i < knots.length && knots[i].time < t) i++;
            const a = knots[i - 1];
            const b = knots[i];
            const span = b.time - a.time;
            const w = span > 0 ? (t - a.time) / span : 0;
            return {
                time: t,
                value: a.value + w * (b.value - a.value),
                regime: w < 0.5 ? a.regime : b.regime,
                spreadLow: (a.spreadLow || 0) + w * ((b.spreadLow || 0) - (a.spreadLow || 0)),
                spreadHigh: (a.spreadHigh || 0) + w * ((b.spreadHigh || 0) - (a.spreadHigh || 0))
            };
        }

        const points = [{
            time: new Date(t0),
            value: y0,
            regime: 'anchor',
            spreadLow: 0,
            spreadHigh: 0
        }];
        let previous = y0;
        const lastKnot = knots.at(-1)?.time || t0;
        const denseEnd = Math.min(end, Math.max(lastKnot, nowMs + horizonHours * HOUR));
        for (let t = t0 + stepMs; t <= denseEnd + 1; t += stepMs) {
            const raw = interpKnot(t);
            let value = smooth * previous + (1 - smooth) * raw.value;
            if (finite(maxStep) && finite(previous)) {
                const d = value - previous;
                if (Math.abs(d) > maxStep) value = previous + Math.sign(d || 1) * maxStep;
            }
            points.push({
                time: new Date(t),
                value,
                regime: raw.regime,
                spreadLow: raw.spreadLow,
                spreadHigh: raw.spreadHigh
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
        const maxHorizonMinutes = maxReliableHorizonMinutes(offlineNode);

        const anchor = latestFinite(downstream?.timeSeries);
        const maxStep = maxStepFromSeries(downstream?.timeSeries, stepMs);

        // Offline RC+GRU (passive circuit with valle acceptors); only when reliable.
        if (offlineNode?.kind === 'rc_gru') {
            if (offlineNode.reliable !== true) {
                return {
                    ok: false,
                    maxHorizonMinutes,
                    reason: 'Nessuna stima affidabile: RC+GRU sotto soglia vs MLP; resta il modello precedente.',
                    source: 'offline',
                    kind: offlineNode.kind,
                    skill120: offlineNode.skill120,
                    reliable: false
                };
            }
            if (!finite(anchor?.value)) {
                return { ok: false, reason: 'Manca l’ultima misura a valle per ancorare la previsione.' };
            }
            const sensorMap = { ...(options.sensorMap || {}) };
            if (downstream?.id) sensorMap[downstream.id] = downstream;
            for (const up of list) {
                if (up?.id) sensorMap[up.id] = up;
            }
            for (const name of offlineNode.acceptors || []) {
                if (!sensorMap[name] && options.sensorMap?.[name]) sensorMap[name] = options.sensorMap[name];
            }
            const rainSensors = options.rainSensors || [];
            const rawForecast = forecastRcGru(offlineNode, sensorMap, rainSensors, horizonHours, now, stepMs, {
                anchor, maxStep, downstream
            });
            if (!rawForecast.length) {
                return { ok: false, reason: 'Rollout RC incompleto.' };
            }
            const forecast = rawForecast.map(point => {
                const lo = finite(point.spreadLow) ? point.spreadLow : offlineNode.spreadLow;
                const hi = finite(point.spreadHigh) ? point.spreadHigh : offlineNode.spreadHigh;
                const fork = forkAround(point.value, lo, hi);
                return fork
                    ? { time: point.time, value: fork.med, min: fork.min, med: fork.med, max: fork.max }
                    : { time: point.time, value: point.value, min: point.value, med: point.value, max: point.value };
            });
            return {
                ok: true,
                maxHorizonMinutes,
                source: 'offline',
                kind: 'rc_gru',
                reliable: true,
                upstreams: offlineNode.upstreams,
                acceptors: offlineNode.acceptors,
                lagMinutes: offlineNode.lagMinutes,
                a: [],
                b: 0,
                c: [],
                r: offlineNode.skill120,
                skill120: offlineNode.skill120,
                mae: null,
                rmse: null,
                forkSpread: offlineNode.spreadLow,
                spreadLow: offlineNode.spreadLow,
                spreadHigh: offlineNode.spreadHigh,
                bias: null,
                maxStep,
                predicted: [],
                observed: [],
                times: [],
                forecast,
                next: nextHalfHourForks(forecast, Math.max(6, Math.round(horizonHours * 2)), now),
                reason: null
            };
        }

        // Offline Muskingum+GRU (dense physics rollout); only when reliable/adopted.
        if (offlineNode?.kind === 'muskingum_gru') {
            if (offlineNode.reliable !== true) {
                return {
                    ok: false,
                    maxHorizonMinutes,
                    reason: 'Nessuna stima affidabile: Muskingum+GRU sotto soglia vs MLP; resta il modello precedente.',
                    source: 'offline',
                    kind: offlineNode.kind,
                    skill120: offlineNode.skill120,
                    reliable: false
                };
            }
            if (!finite(anchor?.value)) {
                return { ok: false, reason: 'Manca l’ultima misura a valle per ancorare la previsione.' };
            }
            const sensorMap = { ...(options.sensorMap || {}) };
            if (downstream?.id) sensorMap[downstream.id] = downstream;
            for (const up of list) {
                if (up?.id) sensorMap[up.id] = up;
            }
            const rainSensors = options.rainSensors || [];
            const rawForecast = forecastMuskingumGru(offlineNode, sensorMap, rainSensors, horizonHours, now, stepMs, {
                anchor, maxStep, downstream
            });
            if (!rawForecast.length) {
                return { ok: false, reason: 'Rollout Muskingum incompleto.' };
            }
            const forecast = rawForecast.map(point => {
                const lo = finite(point.spreadLow) ? point.spreadLow : offlineNode.spreadLow;
                const hi = finite(point.spreadHigh) ? point.spreadHigh : offlineNode.spreadHigh;
                const fork = forkAround(point.value, lo, hi);
                return fork
                    ? { time: point.time, value: fork.med, min: fork.min, med: fork.med, max: fork.max }
                    : { time: point.time, value: point.value, min: point.value, med: point.value, max: point.value };
            });
            return {
                ok: true,
                maxHorizonMinutes,
                source: 'offline',
                kind: 'muskingum_gru',
                reliable: true,
                upstreams: offlineNode.upstreams,
                lagMinutes: offlineNode.lagMinutes,
                a: [],
                b: 0,
                c: [],
                r: offlineNode.skill120,
                skill120: offlineNode.skill120,
                mae: null,
                rmse: null,
                forkSpread: offlineNode.spreadLow,
                spreadLow: offlineNode.spreadLow,
                spreadHigh: offlineNode.spreadHigh,
                bias: null,
                maxStep,
                predicted: [],
                observed: [],
                times: [],
                forecast,
                next: nextHalfHourForks(forecast, Math.max(6, Math.round(horizonHours * 2)), now),
                reason: null
            };
        }

        // M3: GRU≤2h + ridge long with weighted junction (o gamba MLP per blend_gru_mlp).
        if (offlineNode?.kind === 'blend_gru_ridge' || offlineNode?.kind === 'blend_gru_mlp') {
            const longLabel = offlineNode.kind === 'blend_gru_mlp' ? 'MLP' : 'ridge';
            if (offlineNode.reliable === false) {
                return {
                    ok: false,
                    maxHorizonMinutes,
                    reason: `Nessuna stima affidabile: blend GRU+${longLabel} sotto soglia.`,
                    source: 'offline',
                    kind: offlineNode.kind,
                    skill120: offlineNode.skill120,
                    reliable: false
                };
            }
            if (!finite(anchor?.value)) {
                return { ok: false, reason: 'Manca l’ultima misura a valle per ancorare la previsione.' };
            }
            const sensorMap = { ...(options.sensorMap || {}) };
            if (downstream?.id) sensorMap[downstream.id] = downstream;
            for (const up of list) {
                if (up?.id) sensorMap[up.id] = up;
            }
            const rainSensors = options.rainSensors || [];
            const blendUps = (offlineNode.upstreams || []).map(id => sensorMap[id]).filter(Boolean);
            if (blendUps.length !== (offlineNode.upstreams || []).length) {
                return { ok: false, reason: `Serie a monte incomplete per il tratto ${longLabel} del blend.` };
            }
            const rawForecast = forecastBlendGruRidge(
                offlineNode.gru, offlineNode.ridge, blendUps, sensorMap, rainSensors, horizonHours, now, stepMs, {
                    anchor,
                    maxStep: options.maxStep === null ? null : maxStep,
                    downstream,
                    rainSensors,
                    smooth: options.smooth,
                    blendStartMin: offlineNode.blendStartMin,
                    blendEndMin: offlineNode.blendEndMin,
                    mlpModel: offlineNode.kind === 'blend_gru_mlp' ? offlineNode.mlp : null
                }
            );
            if (!rawForecast.length) {
                return { ok: false, reason: `Blend GRU+${longLabel} incompleto sulla finestra richiesta.` };
            }
            const forecast = rawForecast.map(point => {
                const lo = finite(point.spreadLow) ? point.spreadLow : offlineNode.spreadLow;
                const hi = finite(point.spreadHigh) ? point.spreadHigh : offlineNode.spreadHigh;
                const fork = forkAround(point.value, lo, hi);
                return fork
                    ? { time: point.time, value: fork.med, min: fork.min, med: fork.med, max: fork.max, regime: point.regime }
                    : { time: point.time, value: point.value, min: point.value, med: point.value, max: point.value, regime: point.regime };
            });
            return {
                ok: true,
                maxHorizonMinutes,
                source: 'offline',
                kind: offlineNode.kind,
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
                forkSpread: offlineNode.spreadLow,
                spreadLow: offlineNode.spreadLow,
                spreadHigh: offlineNode.spreadHigh,
                bias: null,
                maxStep,
                predicted: [],
                observed: [],
                times: [],
                forecast,
                next: nextHalfHourForks(forecast, Math.max(6, Math.round(horizonHours * 2)), now),
                reason: null
            };
        }

        // Offline window MLP / seq GRU / seq CNN (Cento): no upstream-id match required.
        if (
            offlineNode?.kind === 'window_delta_mlp'
            || offlineNode?.kind === 'window_delta_gru'
            || offlineNode?.kind === 'window_delta_cnn'
            || offlineNode?.kind === 'window_delta_transformer'
        ) {
            if (offlineNode.reliable === false) {
                return {
                    ok: false,
                    maxHorizonMinutes,
                    reason: 'Nessuna stima affidabile: il modello a finestra non batte la persistenza in validazione.',
                    source: 'offline',
                    kind: offlineNode.kind,
                    skill120: offlineNode.skill120,
                    reliable: false
                };
            }
            if (!finite(anchor?.value)) {
                return { ok: false, reason: 'Manca l’ultima misura a valle per ancorare la previsione.' };
            }
            const sensorMap = { ...(options.sensorMap || {}) };
            if (downstream?.id) sensorMap[downstream.id] = downstream;
            for (const up of list) {
                if (up?.id) sensorMap[up.id] = up;
            }
            const rainSensors = options.rainSensors || [];
            const forecastFn = offlineNode.kind === 'window_delta_mlp' ? forecastWindowMlp : forecastWindowSeq;
            let rawForecast = forecastFn(offlineNode, sensorMap, rainSensors, horizonHours, now, stepMs, {
                anchor,
                maxStep: options.maxStep === null ? null : maxStep,
                downstream,
                smooth: options.smooth
            });
            if (!rawForecast.length) {
                return { ok: false, reason: 'Finestra feature incompleta per il modello a finestra.' };
            }
            // M8 online phase+bias: opt-in only (A/B); default off until promotion.
            let onlineCorr = null;
            if (options.onlineCorrection === true && offlineNode.kind === 'window_delta_mlp') {
                onlineCorr = estimateMlpOnlineCorrection(
                    offlineNode, sensorMap, rainSensors, downstream, now, stepMs,
                    { maxStep: options.maxStep === null ? null : maxStep, onlineWindowHours: options.onlineWindowHours }
                );
                rawForecast = applyMlpOnlineCorrection(rawForecast, onlineCorr);
            }
            let forecast = rawForecast.map(point => {
                const lo = finite(point.spreadLow) ? point.spreadLow : offlineNode.spreadLow;
                const hi = finite(point.spreadHigh) ? point.spreadHigh : offlineNode.spreadHigh;
                const fork = forkAround(point.value, lo, hi);
                return fork
                    ? { time: point.time, value: fork.med, min: fork.min, med: fork.med, max: fork.max }
                    : { time: point.time, value: point.value, min: point.value, med: point.value, max: point.value };
            });
            // Crossing-time / hazard warp: soft-loaded; adopt or useCrossingTime.
            // WP2.3: hazard decode only when allowDecode (P(crossing) ≥ decodeGateP).
            let crossingMeta = null;
            const crossing = offlineNode.crossingTime;
            const useCross = options.useCrossingTime === true
                || (options.useCrossingTime !== false && crossing?.adopt === true);
            if (useCross && crossing && offlineNode.kind === 'window_delta_mlp') {
                const tauPred = predictCrossingTau(crossing, sensorMap, rainSensors, +now, stepMs);
                const gated = tauPred && tauPred.allowDecode !== false;
                const warped = gated
                    ? warpForecastByCrossing(forecast, tauPred, anchor.value, +now)
                    : { forecast, warped: false, scale: 1, pathTau: null, gatedOff: true };
                forecast = warped.forecast;
                crossingMeta = {
                    ...tauPred,
                    warped: warped.warped,
                    scale: warped.scale,
                    pathTau: warped.pathTau,
                    stable: warped.stable === true,
                    gatedOff: warped.gatedOff === true || !gated
                };
            }
            // M3 MoE: if a rising-only expert is attached and limb is rising, swap forecast.
            let moeMeta = null;
            const moeRaw = options.coeffs?.nodes?.[downstream?.id || options.downstreamId]?.moeRising
                || offlineNode.moeRising;
            const risingGate = detectRisingRegime(downstream, list, now, { stepMs });
            if (offlineNode.kind === 'window_delta_mlp' && moeRaw && risingGate.active) {
                const moeModel = parseWindowMlpNode({ ...moeRaw, reliable: true });
                if (moeModel) {
                    const moeRawForecast = forecastWindowMlp(moeModel, sensorMap, rainSensors, horizonHours, now, stepMs, {
                        anchor, maxStep, downstream, smooth: options.smooth
                    });
                    if (moeRawForecast.length) {
                        forecast = moeRawForecast.map(point => {
                            const lo = finite(point.spreadLow) ? point.spreadLow : moeModel.spreadLow;
                            const hi = finite(point.spreadHigh) ? point.spreadHigh : moeModel.spreadHigh;
                            const fork = forkAround(point.value, lo, hi);
                            return fork
                                ? { time: point.time, value: fork.med, min: fork.min, med: fork.med, max: fork.max }
                                : { time: point.time, value: point.value, min: point.value, med: point.value, max: point.value };
                        });
                        moeMeta = { active: true, source: 'moeRising', downD1: risingGate.downD1 };
                    }
                }
            }
            // M2: widen upper band only while rising (calibrated coverage ≥85%).
            let risingBandMeta = null;
            const risingInflate = offlineNode.risingBand?.upperInflate;
            const risingLower = offlineNode.risingBand?.lowerInflate;
            if (offlineNode.kind === 'window_delta_mlp' && (risingInflate > 1 || risingLower > 1) && risingGate.active) {
                forecast = applyRisingUpperInflate(forecast, risingInflate || 1, risingLower || 1);
                risingBandMeta = {
                    active: true,
                    upperInflate: risingInflate || 1,
                    lowerInflate: risingLower || 1,
                    downD1: risingGate.downD1,
                    upstreamRising: risingGate.upstreamRising
                };
            }
            // Peak-within-h upstream amplitude prior: lift mediana while rising (opt-in / adopt).
            let upstreamAmpMeta = null;
            const ampDoc = options.coeffs?.nodes?.[downstream?.id || options.downstreamId]?.upstreamAmplitude
                || offlineNode.upstreamAmplitude;
            const useAmp = ampDoc
                && (options.useUpstreamAmplitude === true
                    || (options.useUpstreamAmplitude !== false && ampDoc.adopt === true));
            if (offlineNode.kind === 'window_delta_mlp' && useAmp && risingGate.active && finite(anchor?.value)) {
                const lifted = applyUpstreamAmplitudeLift(
                    forecast, ampDoc, sensorMap, anchor.value, +now, stepMs
                );
                forecast = lifted.forecast;
                upstreamAmpMeta = {
                    active: lifted.nLifted > 0,
                    nLifted: lifted.nLifted,
                    source: 'upstreamAmplitude',
                    downD1: risingGate.downD1
                };
            }
            return {
                ok: true,
                maxHorizonMinutes,
                source: 'offline',
                kind: offlineNode.kind,
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
                forkSpread: offlineNode.spreadLow,
                spreadLow: offlineNode.spreadLow,
                spreadHigh: offlineNode.spreadHigh,
                bias: onlineCorr ? onlineCorr.bias : null,
                phaseShiftMin: onlineCorr ? onlineCorr.phaseShiftMin : null,
                crossing: crossingMeta,
                risingBand: risingBandMeta,
                upstreamAmplitude: upstreamAmpMeta,
                moeRising: moeMeta,
                probableLabel: crossingMeta?.warped ? 'più probabile' : null,
                maxStep,
                predicted: [],
                observed: [],
                times: [],
                forecast,
                next: nextHalfHourForks(forecast, Math.max(6, Math.round(horizonHours * 2)), now),
                reason: null
            };
        }

        // Offline direct-horizon: usa solo se reliable (skill validato ≥ soglia).
        if (offlineNode?.kind === 'direct_horizon_delta_ridge') {
            if (offlineNode.reliable === false) {
                return {
                    ok: false,
                    maxHorizonMinutes,
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
                anchor, maxStep, downstream, bias, rainSensors, smooth: options.smooth
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
                maxHorizonMinutes,
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
                    maxHorizonMinutes,
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
            maxHorizonMinutes,
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

    function seqGruUrl(downstream) {
        const slug = String(downstream || '').toLowerCase().replace(/\s+/g, '-');
        return slug ? `models/seq-${slug}-gru.json` : null;
    }

    // Soft-adopt only where a seq export is expected (MLP nodes + known ridge→GRU).
    // Avoids N×404 over every ridge station without a seq-* file.
    const SEQ_SOFT_ADOPT_EXTRA = ['Casalecchio chiusa', 'Gallo'];

    function seqSoftAdoptCandidates(data) {
        const names = new Set();
        for (const [name, node] of Object.entries(data?.nodes || {})) {
            const kind = String(node?.kind || '');
            if (kind.startsWith('window_delta_') && kind !== 'window_delta_gru') names.add(name);
        }
        for (const name of SEQ_SOFT_ADOPT_EXTRA) {
            if (data?.nodes?.[name]) names.add(name);
        }
        return [...names];
    }

    async function softAdoptSeqGru(fetcher, data, downstream) {
        const url = seqGruUrl(downstream);
        if (!url) return;
        try {
            const seqRes = await fetcher(url);
            if (!seqRes?.ok) return;
            const seq = await seqRes.json();
            if (seq?.kind === 'window_delta_gru' && seq.reliable === true && seq.downstream) {
                data.nodes = data.nodes || {};
                data.nodes[seq.downstream] = seq;
            }
        } catch {
            /* keep transfer-coeffs node */
        }
    }

    function seriesUntil(series, untilMs) {
        const n = series?.length || 0;
        if (!n) return [];
        let lo = 0;
        let hi = n;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (+series[mid].time <= untilMs) lo = mid + 1;
            else hi = mid;
        }
        return series.slice(0, lo);
    }

    function truncateSensor(sensor, untilMs) {
        if (!sensor) return sensor;
        return Object.assign({}, sensor, { timeSeries: seriesUntil(sensor.timeSeries, untilMs) });
    }

    // Replay the offline model over the visible past: at each origin t, forecast t+H
    // using only observations ≤ t (no peeking). Returns points at the target times.
    function hindcastSeries(downstream, upstreams, options = {}) {
        const now = options.now || new Date();
        const hours = options.hours || 48;
        const horizonHours = options.horizonHours ?? 2;
        const stepMs = options.stepMs || DEFAULT_STEP_MS;
        const horizonMs = horizonHours * HOUR;
        const endMs = +now;
        const startMs = endMs - hours * HOUR;
        const rainSensors = options.rainSensors || [];
        const points = [];
        for (let origin = startMs; origin <= endMs - horizonMs + 1; origin += stepMs) {
            const targetMs = origin + horizonMs;
            const downT = truncateSensor(downstream, origin);
            const upsT = (upstreams || []).map(up => truncateSensor(up, origin));
            const rainT = rainSensors.map(r => truncateSensor(r, origin));
            const sensorMap = { ...(options.sensorMap || {}) };
            if (downT?.id) sensorMap[downT.id] = downT;
            for (const up of upsT) {
                if (up?.id) sensorMap[up.id] = up;
            }
            for (const r of rainT) {
                if (r?.id) sensorMap[r.id] = r;
            }
            const result = predictNode(downT, upsT, {
                hours: Math.max(hours, 24),
                now: new Date(origin),
                coeffs: options.coeffs,
                horizonHours,
                downstreamId: downstream?.id || options.downstreamId,
                sensorMap,
                rainSensors: rainT,
                biasCorrection: options.biasCorrection,
                onlineCorrection: options.onlineCorrection,
                onlineWindowHours: options.onlineWindowHours
            });
            if (!result?.ok) continue;
            const forecast = (result.forecast || []).filter(p => p.regime !== 'anchor');
            let best = null;
            let bestDist = Infinity;
            for (const p of forecast) {
                const dist = Math.abs(+p.time - targetMs);
                if (dist < bestDist) {
                    best = p;
                    bestDist = dist;
                }
            }
            if (!best || bestDist > stepMs) continue;
            const observed = sampleAt(downstream?.timeSeries, targetMs);
            if (!finite(observed) || !finite(best.med ?? best.value)) continue;
            const predicted = best.med ?? best.value;
            points.push({
                time: new Date(targetMs),
                origin: new Date(origin),
                observed,
                predicted,
                residual: predicted - observed,
                min: best.min,
                max: best.max
            });
        }
        return {
            ok: points.length > 0,
            horizonHours,
            points,
            lag: estimatePredictedLag(points, stepMs),
            reason: points.length ? null : 'Hindcast incompleto sul intervallo visibile.'
        };
    }

    async function softMergeCentoBands(fetcher, data) {
        try {
            const manifest = data?.sidecars?.bands;
            if (manifest && manifest.compatible !== true) return false;
            const res = await fetcher(manifest?.url || 'models/cento-dnn-bands.json');
            if (!res?.ok) return false;
            const bands = await res.json();
            if (!bands?.horizons || !data.nodes?.Cento) return false;
            if (data.nodes.Cento.kind !== 'window_delta_mlp') return false;
            data.nodes.Cento = mergeQuantileBands(data.nodes.Cento, bands);
            return true;
        } catch {
            return false;
        }
    }

    async function softMergeCentoRisingMoe(fetcher, data) {
        try {
            const manifest = data?.sidecars?.risingMoe;
            if (manifest && manifest.reliable !== true) return false;
            const res = await fetcher(manifest?.url || 'models/cento-dnn-rising.json');
            if (!res?.ok) return false;
            const doc = await res.json();
            if (!doc?.horizons || doc.kind !== 'window_delta_mlp') return false;
            if (doc.reliable !== true) return false;
            if (!data.nodes?.Cento || data.nodes.Cento.kind !== 'window_delta_mlp') return false;
            data.nodes.Cento = { ...data.nodes.Cento, moeRising: doc };
            return true;
        } catch {
            return false;
        }
    }

    async function softMergeCentoRisingInflate(fetcher, data) {
        try {
            const res = await fetcher('models/cento-rising-band-inflate.json');
            if (!res?.ok) return false;
            const doc = await res.json();
            const inflate = Number(doc?.upperInflate);
            if (!(inflate > 1) || !data.nodes?.Cento) return false;
            if (data.nodes.Cento.kind !== 'window_delta_mlp') return false;
            const lower = Number(doc?.lowerInflate);
            data.nodes.Cento = {
                ...data.nodes.Cento,
                risingBand: {
                    upperInflate: inflate,
                    lowerInflate: lower > 1 ? lower : 1,
                    targetCoverage: doc.targetCoverage ?? 0.85,
                    source: 'cento-rising-band-inflate'
                }
            };
            return true;
        } catch {
            return false;
        }
    }

    async function softMergeCentoUpstreamAmplitude(fetcher, data) {
        try {
            const res = await fetcher('models/cento-upstream-amplitude.json');
            if (!res?.ok) return false;
            const doc = await res.json();
            if (doc?.kind !== 'cento_upstream_amplitude' || !doc.horizons) return false;
            if (!data.nodes?.Cento || data.nodes.Cento.kind !== 'window_delta_mlp') return false;
            data.nodes.Cento = { ...data.nodes.Cento, upstreamAmplitude: doc };
            return true;
        } catch {
            return false;
        }
    }

    function seriesFromMap(sensorMap, name) {
        const entry = sensorMap?.[name];
        if (Array.isArray(entry)) return entry;
        return entry?.timeSeries || null;
    }

    function buildUpstreamAmplitudeFeatures(sensorMap, ampDoc, nowMs) {
        const names = ampDoc?.featureNames;
        const ups = ampDoc?.upstreams;
        if (!Array.isArray(names) || !Array.isArray(ups)) return null;
        const centoSeries = seriesFromMap(sensorMap, 'Cento');
        const y0 = sampleAt(centoSeries, nowMs);
        if (!finite(y0)) return null;
        const y2 = sampleAt(centoSeries, nowMs - 2 * HOUR);
        const y6 = sampleAt(centoSeries, nowMs - 6 * HOUR);
        const feat = {
            bias: 1,
            cento_y0: y0,
            cento_d2h: finite(y2) ? y0 - y2 : 0,
            cento_d6h: finite(y6) ? y0 - y6 : 0
        };
        for (const u of ups) {
            const tag = String(u).replace(/\s+/g, '_');
            const series = seriesFromMap(sensorMap, u);
            const yu = sampleAt(series, nowMs);
            if (!finite(yu)) return null;
            const u2 = sampleAt(series, nowMs - 2 * HOUR);
            const u6 = sampleAt(series, nowMs - 6 * HOUR);
            feat[`${tag}_y`] = yu;
            feat[`${tag}_d2h`] = finite(u2) ? yu - u2 : 0;
            feat[`${tag}_d6h`] = finite(u6) ? yu - u6 : 0;
        }
        return names.map(n => finite(feat[n]) ? feat[n] : 0);
    }

    function predictUpstreamPeakDelta(ampDoc, sensorMap, nowMs, horizonMinutes) {
        const block = ampDoc?.horizons?.[String(horizonMinutes)];
        if (!block?.weights || !Array.isArray(block.weights)) return null;
        const x = buildUpstreamAmplitudeFeatures(sensorMap, ampDoc, nowMs);
        if (!x || x.length !== block.weights.length) return null;
        let s = 0;
        for (let i = 0; i < x.length; i++) s += x[i] * block.weights[i];
        return s;
    }

    function applyUpstreamAmplitudeLift(forecast, ampDoc, sensorMap, y0, nowMs, stepMs = DEFAULT_STEP_MS) {
        if (!Array.isArray(forecast) || !finite(y0)) return { forecast, nLifted: 0 };
        const keys = Object.keys(ampDoc?.horizons || {}).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
        if (!keys.length) return { forecast, nLifted: 0 };
        let nLifted = 0;
        const out = forecast.map(p => {
            if (!p || p.regime === 'anchor') return p;
            const minutes = Math.round((+p.time - nowMs) / MINUTE);
            if (!(minutes > 0)) return p;
            // nearest horizon block ≤ minutes, else smallest
            let h = keys[0];
            for (const k of keys) {
                if (k <= minutes) h = k;
                else break;
            }
            if (minutes - h > 45 && keys.find(k => k >= minutes)) {
                h = keys.find(k => k >= minutes);
            }
            const dHat = predictUpstreamPeakDelta(ampDoc, sensorMap, nowMs, h);
            if (!finite(dHat)) return p;
            const prior = y0 + Math.max(0, dHat);
            const med = p.med ?? p.value;
            if (!finite(med) || !(prior > med + 0.01)) return p;
            nLifted++;
            const hiSpread = finite(p.max) ? Math.max(0, p.max - med) : 0;
            const loSpread = finite(p.min) ? Math.max(0, med - p.min) : 0;
            return {
                ...p,
                value: prior,
                med: prior,
                max: prior + hiSpread,
                min: prior - loSpread,
                regime: 'amp_lift'
            };
        });
        return { forecast: out, nLifted };
    }

    function applyRisingUpperInflate(forecast, upperInflate, lowerInflate = 1) {
        const ku = Number(upperInflate);
        const kl = Number(lowerInflate);
        if ((!Number.isFinite(ku) || ku <= 1) && (!Number.isFinite(kl) || kl <= 1)) return forecast;
        if (!Array.isArray(forecast)) return forecast;
        return forecast.map(p => {
            if (!p || p.regime === 'anchor') return p;
            const med = p.med ?? p.value;
            if (!finite(med)) return p;
            let max = p.max;
            let min = p.min;
            if (finite(max) && ku > 1) max = med + (max - med) * ku;
            if (finite(min) && kl > 1) min = med - (med - min) * kl;
            return { ...p, max, min };
        });
    }

    async function softMergeCentoCrossing(fetcher, data) {
        try {
            // Prefer discrete hazard (WP2); fall back to M9 pinball export.
            let crossing = null;
            for (const path of ['models/cento-crossing-hazard.json', 'models/cento-crossing-time.json']) {
                const res = await fetcher(path);
                if (!res?.ok) continue;
                const doc = await res.json();
                if (doc?.thresholds) {
                    crossing = doc;
                    break;
                }
            }
            if (!crossing?.thresholds || !data.nodes?.Cento) return false;
            if (data.nodes.Cento.kind !== 'window_delta_mlp') return false;
            data.nodes.Cento = mergeCrossingTime(data.nodes.Cento, crossing);
            return true;
        } catch {
            return false;
        }
    }

    async function fetchSkillGates(fetcher) {
        try {
            const res = await fetcher('models/skill-gates.json');
            if (!res?.ok) return null;
            return await res.json();
        } catch {
            return null;
        }
    }

    async function softApplySkillGates(fetcher, data) {
        const gates = await fetchSkillGates(fetcher);
        if (!gates) return false;
        applySkillGates(data, gates);
        return true;
    }

    /** Causal rising-limb gate: local d1>0 and at least one upstream d1>0. */
    function detectRisingRegime(downstream, upstreams, now = new Date(), options = {}) {
        const stepMs = options.stepMs || DEFAULT_STEP_MS;
        const nPoints = options.polyPoints || 6;
        const nowMs = +now;
        const downSeries = downstream?.timeSeries;
        const downD = localPolyDerivatives(downSeries, nowMs, stepMs, nPoints);
        const centoRising = finite(downD.d1) && downD.d1 > 0;
        const ups = Array.isArray(upstreams) ? upstreams : [];
        let upstreamRising = 0;
        const details = [];
        for (const up of ups) {
            const d = localPolyDerivatives(up?.timeSeries, nowMs, stepMs, nPoints);
            const rising = finite(d.d1) && d.d1 > 0;
            if (rising) upstreamRising++;
            details.push({ id: up?.id || up?.name || null, d1: d.d1, rising });
        }
        const active = centoRising && upstreamRising > 0;
        return {
            active,
            downD1: downD.d1,
            upstreamRising,
            upstreamTotal: ups.length,
            details
        };
    }

    function normalCdf(z) {
        // Abramowitz–Stegun 26.2.17 approximation.
        const x = Number(z);
        if (!Number.isFinite(x)) return null;
        const t = 1 / (1 + 0.2316419 * Math.abs(x));
        const d = 0.3989422804014327 * Math.exp(-0.5 * x * x);
        const p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        return x >= 0 ? 1 - p : p;
    }

    /**
     * P(Y > thr) from forecast band at absolute time targetMs.
     * Uses med ± (max−min)/(2·z90) as Normal, with z90 = Φ⁻¹(0.9).
     */
    function exceedanceFromBand(forecast, targetMs, thr, stepMs = DEFAULT_STEP_MS) {
        if (!Number.isFinite(thr) || !Array.isArray(forecast)) return null;
        let best = null;
        let bestDist = Infinity;
        for (const p of forecast) {
            if (!p || p.regime === 'anchor') continue;
            const dist = Math.abs(+p.time - targetMs);
            if (dist < bestDist) {
                best = p;
                bestDist = dist;
            }
        }
        if (!best || bestDist > stepMs) return null;
        const med = best.med ?? best.value;
        const lo = best.min;
        const hi = best.max;
        if (!finite(med) || !finite(lo) || !finite(hi)) return null;
        if (hi <= lo + 1e-9) return med > thr ? 1 : 0;
        const z90 = 1.2815515655446004;
        const sigma = (hi - lo) / (2 * z90);
        if (!(sigma > 0)) return med > thr ? 1 : 0;
        const p = 1 - normalCdf((thr - med) / sigma);
        return Math.max(0, Math.min(1, p));
    }

    function risingExceedanceSummary(forecast, thresholds, now = new Date(), horizonsHours = [2, 4, 6]) {
        const nowMs = +now;
        const out = [];
        const thrEntries = Object.entries(thresholds || {}).filter(([, v]) => Number.isFinite(v));
        for (const h of horizonsHours) {
            const targetMs = nowMs + h * HOUR;
            const row = { horizonHours: h, probs: {} };
            for (const [color, thr] of thrEntries) {
                row.probs[color] = exceedanceFromBand(forecast, targetMs, thr);
            }
            out.push(row);
        }
        return out;
    }


    function decodePackedMlp(input) {
        const buffer = input instanceof ArrayBuffer
            ? input
            : (ArrayBuffer.isView(input) ? input.buffer.slice(input.byteOffset, input.byteOffset + input.byteLength) : null);
        if (!buffer || buffer.byteLength < 12) return null;
        const dv = new DataView(buffer);
        const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
        if (magic !== 'RMLQ' || dv.getUint16(4, true) !== 1) return null;
        const horizonCount = dv.getUint16(6, true);
        const inputSize = dv.getUint32(8, true);
        const end = buffer.byteLength;
        let off = 12;
        const need = n => {
            if (off + n > end) throw new Error('packed MLP truncated');
        };
        const f32Array = n => {
            need(4 * n);
            const out = new Array(n);
            for (let i = 0; i < n; i++, off += 4) out[i] = dv.getFloat32(off, true);
            return out;
        };
        const activations = ['linear', 'relu'];
        const horizons = {};
        try {
            for (let hIndex = 0; hIndex < horizonCount; hIndex++) {
                need(8);
                const horizon = dv.getUint32(off, true); off += 4;
                const layerCount = dv.getUint16(off, true); off += 2;
                off += 2;
                const mean = f32Array(inputSize);
                const std = f32Array(inputSize);
                const layers = [];
                for (let li = 0; li < layerCount; li++) {
                    need(12);
                    const outSize = dv.getUint32(off, true); off += 4;
                    const inSize = dv.getUint32(off, true); off += 4;
                    const activation = activations[dv.getUint8(off)] || 'linear'; off += 4;
                    if (!outSize || !inSize || outSize > 4096 || inSize > 8192) throw new Error('packed MLP invalid layer shape');
                    const b = f32Array(outSize);
                    const scales = f32Array(outSize);
                    need(2 * outSize * inSize);
                    const W = new Array(outSize);
                    for (let r = 0; r < outSize; r++) {
                        const row = new Float32Array(inSize);
                        const scale = scales[r] || 1;
                        for (let c = 0; c < inSize; c++, off += 2) row[c] = dv.getInt16(off, true) * scale;
                        W[r] = row;
                    }
                    layers.push({ W, b, activation });
                }
                horizons[String(horizon)] = { mean, std, layers };
            }
            if (off !== end) return null;
            return { inputSize, horizons };
        } catch {
            return null;
        }
    }

    async function hydratePackedMlpNode(fetcher, data, station = 'Cento', options = {}) {
        try {
            const node = data?.nodes?.[station];
            // Blend nodes keep the MLP leg nested: hydrate it instead of the top node.
            const target = node?.packedWeights ? node : (node?.mlp?.packedWeights ? node.mlp : node);
            const packed = target?.packedWeights;
            if (!packed?.chunks) return false;
            const part = options.part || 'initial';
            const parts = part === 'all' ? Object.keys(packed.chunks) : [part];
            let changed = false;
            target.horizons = target.horizons || {};
            for (const key of parts) {
                const chunk = packed.chunks[key];
                if (!chunk?.url || !Array.isArray(chunk.horizons)) continue;
                if (chunk.horizons.every(h => target.horizons[String(h)]?.layers)) continue;
                const res = await fetcher(chunk.url);
                if (!res?.ok || typeof res.arrayBuffer !== 'function') continue;
                const decoded = decodePackedMlp(await res.arrayBuffer());
                if (!decoded || decoded.inputSize !== packed.inputSize) continue;
                for (const h of chunk.horizons) {
                    const block = decoded.horizons[String(h)];
                    if (!block) continue;
                    target.horizons[String(h)] = {
                        ...(packed.horizonMeta?.[String(h)] || {}),
                        ...block
                    };
                    changed = true;
                }
            }
            return changed;
        } catch {
            return false;
        }
    }

    async function loadCoeffs(fetcher, url = 'models/transfer-coeffs-runtime.json') {
        try {
            let response = await fetcher(url);
            // Fallback to full training export when runtime pack is absent.
            if (!response?.ok && /transfer-coeffs-runtime\.json$/.test(url)) {
                response = await fetcher('models/transfer-coeffs.json');
            }
            if (!response?.ok) return null;
            const data = await response.json();
            if (!(data?.edges || data?.nodes)) return null;
            // Skill gates first: they can make a station unusable regardless of model kind.
            const gates = await fetchSkillGates(fetcher);
            if (gates) applySkillGates(data, gates);
            // Soft-adopt only seq exports marked reliable by the runtime pack. Without packed
            // metadata (legacy runtime), retain the old probe behavior.
            const candidates = seqSoftAdoptCandidates(data).filter(name => {
                if (!data.seqAdopt) return true;
                if (data.seqAdopt[name]?.reliable !== true) return false;
                if (!gates) return true;
                const gate = gates.stations?.[name];
                const lo = finite(Number(gate?.skill120CI?.lo)) ? Number(gate.skill120CI.lo)
                    : (finite(Number(gate?.skill120)) ? Number(gate.skill120) : null);
                if (!finite(lo)) return true;
                const exit = finite(Number(gates.skillExit)) ? Number(gates.skillExit) : SKILL_EXIT;
                return lo >= exit;
            });
            await Promise.all(candidates.map(name => softAdoptSeqGru(fetcher, data, name)));
            if (gates && candidates.length) {
                for (const name of candidates) {
                    const gated = applySkillGates({ nodes: { [name]: data.nodes[name] } }, gates).nodes[name];
                    data.nodes[name] = gated;
                }
            }
            // Hybrid RC / Muskingum: both current exports are reliable:false — skip probe.
            // Re-enable when a reliable hybrid is shipped alongside transfer-coeffs.
            // M2 rising inflate is tiny — merge before first paint so salita UI/band work without waiting for q10/q90.
            await Promise.all([
                softMergeCentoRisingInflate(fetcher, data),
                softMergeCentoUpstreamAmplitude(fetcher, data),
                hydratePackedMlpNode(fetcher, data, 'Cento', { part: 'initial' })
            ]);
            return data;
        } catch {
            return null;
        }
    }

    const api = {
        edgeKey, buildGraph, rankNodes, alignSeries, estimateLag, fitAffine, fitMultiAffine, applyAffine, applyMulti,
        pickRegime, latestFinite, maxStepFromSeries, predictEdge, predictNode, forecastForward, forecastForwardMulti,
        forecastDirectHorizon, forecastWindowMlp, forecastWindowSeq, forecastBlendGruRidge, forecastMuskingumGru, forecastRcGru, parseOfflineNode,
        parseWindowMlpNode, parseWindowSeqNode, parseBlendGruRidgeNode, parseMuskingumGruNode, parseRcGruNode, gruForward, conv1dForward, transformerForward, muskingumStep, muskingumCoeffs,
        muskingumWarmup, decodeMuskingumTheta, decodeKX, decodeRcTheta, rcStep, softplus, mlpForward, mlpForwardVec, seqDeltaForward,
        sampleAt, sampleNormLevel, buildMuskingumSeq, buildRcSeq, buildSeqChannelMatrix, standardizeSeqMatrix,
        buildWindowFeatureVector, lin3Extrapolate, poly2Extrapolate, localPolyDerivatives, localLin3Slope, standardizeWindowFeatures, nextHalfHourForks, forkAround, withFork, loadCoeffs,
        softMergeCentoBands, softMergeCentoRisingInflate, softMergeCentoRisingMoe, softMergeCentoCrossing,
        softMergeCentoUpstreamAmplitude, softApplySkillGates, decodePackedMlp, hydratePackedMlpNode,
        applyRisingUpperInflate, applyUpstreamAmplitudeLift, predictUpstreamPeakDelta,
        seqGruUrl, seqSoftAdoptCandidates, pearson, residuals, estimateRecentBias, estimateMlpOnlineCorrection, applyMlpOnlineCorrection,
        driverIsHigh, basinRainAt, hindcastSeries,
        truncateSensor, seriesUntil, maxReliableHorizonMinutes, maxHorizonMinutesAvailable, estimatePredictedLag,
        reliableFromSkill, skillLoOf, mergeQuantileBands, mergeCrossingTime, predictCrossingTau, warpForecastByCrossing,
        applySkillGates, missingUpstreams, widenForecastBand,
        detectRisingRegime, exceedanceFromBand, risingExceedanceSummary, normalCdf,
        TemporalSampling,
        DEFAULT_STEP_MS, FORK_K, SKILL_KEEP, SKILL_ENTER, SKILL_EXIT
    };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.RenoPredict = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
