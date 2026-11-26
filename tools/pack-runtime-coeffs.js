#!/usr/bin/env node
// Pack browser-facing transfer coeffs: compact JSON, strip bulky train/holdout blobs.
// window_delta_mlp weights are quantized row-wise to int16 and split into binary chunks.
// Usage: node tools/pack-runtime-coeffs.js
//   [--in models/transfer-coeffs.json] [--out models/transfer-coeffs-runtime.json] [--check]

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const root = path.resolve(__dirname, '..');
function opt(name, fallback) {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const hasFlag = name => process.argv.includes(`--${name}`);

const STRIP_KEYS = new Set(['train', 'holdout', 'holdoutEvents', 'earlyStopping']);
const ACT = { linear: 0, relu: 1 };

function stripHorizonBlobs(node) {
    if (!node?.horizons || typeof node.horizons !== 'object') return node;
    const horizons = {};
    for (const [h, block] of Object.entries(node.horizons)) {
        if (!block || typeof block !== 'object') {
            horizons[h] = block;
            continue;
        }
        const slim = { ...block };
        for (const key of STRIP_KEYS) delete slim[key];
        horizons[h] = slim;
    }
    return { ...node, horizons };
}

function round(v, digits = 6) {
    return Number.isFinite(Number(v)) ? Number(Number(v).toFixed(digits)) : v ?? null;
}

function horizonMeta(block) {
    const out = {};
    for (const key of ['validate', 'validateEvents', 'spreadLow', 'spreadHigh', 'spreadLowHigh', 'spreadHighHigh', 'rmse', 'skill', 'conformalQhat']) {
        if (block?.[key] !== undefined) out[key] = block[key];
    }
    return out;
}

function layerMeta(layers) {
    return (layers || []).map(layer => ({
        out: layer.W?.length || 0,
        in: layer.W?.[0]?.length || 0,
        activation: layer.activation || 'linear'
    }));
}

function encodeMlpChunk(node, horizons) {
    const inputSize = node.featureNames.length;
    let bytes = 12;
    for (const h of horizons) {
        const block = node.horizons[String(h)];
        bytes += 8 + 8 * inputSize;
        for (const layer of block.layers) {
            const out = layer.W.length;
            const input = layer.W[0].length;
            bytes += 12 + 8 * out + 2 * out * input;
        }
    }
    const buf = Buffer.alloc(bytes);
    let o = 0;
    buf.write('RMLQ', o, 'ascii'); o += 4;
    buf.writeUInt16LE(1, o); o += 2;
    buf.writeUInt16LE(horizons.length, o); o += 2;
    buf.writeUInt32LE(inputSize, o); o += 4;
    for (const h of horizons) {
        const block = node.horizons[String(h)];
        buf.writeUInt32LE(h, o); o += 4;
        buf.writeUInt16LE(block.layers.length, o); o += 2;
        buf.writeUInt16LE(0, o); o += 2;
        for (const v of block.mean) { buf.writeFloatLE(Number(v) || 0, o); o += 4; }
        for (const v of block.std) { buf.writeFloatLE(Number(v) || 0, o); o += 4; }
        for (const layer of block.layers) {
            const rows = layer.W;
            const out = rows.length;
            const input = rows[0].length;
            buf.writeUInt32LE(out, o); o += 4;
            buf.writeUInt32LE(input, o); o += 4;
            buf.writeUInt8(ACT[layer.activation || 'linear'] ?? 0, o); o += 1;
            buf.writeUInt8(0, o); buf.writeUInt8(0, o); buf.writeUInt8(0, o); o += 3;
            for (const v of layer.b) { buf.writeFloatLE(Number(v) || 0, o); o += 4; }
            const scales = rows.map(row => Math.max(1e-12, Math.max(...row.map(v => Math.abs(Number(v) || 0))) / 32767));
            for (const v of scales) { buf.writeFloatLE(v, o); o += 4; }
            for (let r = 0; r < out; r++) {
                const scale = scales[r];
                for (const raw of rows[r]) {
                    const q = Math.max(-32768, Math.min(32767, Math.round((Number(raw) || 0) / scale)));
                    buf.writeInt16LE(q, o); o += 2;
                }
            }
        }
    }
    if (o !== bytes) throw new Error(`Packed size mismatch: wrote ${o}, expected ${bytes}`);
    return buf;
}

function slug(name) {
    return String(name).toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

function packMlpNode(name, node) {
    const inputSize = node.featureNames?.length || 0;
    const horizons = Object.keys(node.horizons || {}).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
    if (!inputSize || !horizons.length) return node;
    const initial = horizons.filter(h => h <= 120);
    const tail = horizons.filter(h => h > 120);
    const base = slug(name);
    const chunks = {};
    const writeChunk = (part, list) => {
        if (!list.length) return null;
        const label = `${String(list[0]).padStart(3, '0')}-${String(list.at(-1)).padStart(3, '0')}`;
        const rel = `models/${base}-mlp-h${label}.bin`;
        const file = path.resolve(root, rel);
        const buf = encodeMlpChunk(node, list);
        fs.writeFileSync(file, buf);
        return { url: rel, bytes: buf.length, horizons: list };
    };
    const first = writeChunk('initial', initial);
    const rest = writeChunk('tail', tail);
    if (first) chunks.initial = first;
    if (rest) chunks.tail = rest;
    const meta = {};
    for (const h of horizons) meta[String(h)] = horizonMeta(node.horizons[String(h)]);
    const firstBlock = node.horizons[String(horizons[0])];
    const slim = stripHorizonBlobs({ ...node, horizons: {} });
    slim.packedWeights = {
        format: 'RMLQ',
        version: 1,
        inputSize,
        horizons,
        chunks,
        layers: layerMeta(firstBlock.layers),
        horizonMeta: meta
    };
    return slim;
}

function scanSeqAdopt() {
    const dir = path.resolve(root, 'models');
    const out = {};
    if (!fs.existsSync(dir)) return out;
    for (const file of fs.readdirSync(dir).filter(f => /^seq-.+-gru\.json$/.test(f))) {
        try {
            const data = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
            if (data?.downstream) out[data.downstream] = {
                url: `models/${file}`,
                reliable: data.reliable === true,
                skill120: round(data.skill120, 4)
            };
        } catch {}
    }
    return out;
}

function sidecarManifest(cento) {
    const input = cento?.featureNames?.length || null;
    const out = {};
    const bandsPath = path.resolve(root, 'models/cento-dnn-bands.json');
    if (fs.existsSync(bandsPath)) {
        try {
            const bands = JSON.parse(fs.readFileSync(bandsPath, 'utf8'));
            const first = Object.values(bands.horizons || {})[0];
            const size = first?.layersQ10?.[0]?.W?.[0]?.length || null;
            out.bands = {
                url: 'models/cento-dnn-bands.json',
                available: true,
                compatible: !!(input && size === input),
                inputSize: size,
                reason: size === input ? 'compatible' : `input ${size || 'n/d'} != ${input || 'n/d'}`
            };
        } catch { out.bands = { available: false, compatible: false, reason: 'parse error' }; }
    } else out.bands = { available: false, compatible: false, reason: 'missing' };
    const risingPath = path.resolve(root, 'models/cento-dnn-rising.json');
    if (fs.existsSync(risingPath)) {
        try {
            const rising = JSON.parse(fs.readFileSync(risingPath, 'utf8'));
            out.risingMoe = {
                url: 'models/cento-dnn-rising.json', available: true, reliable: rising.reliable === true,
                reason: rising.reliable === true ? 'reliable' : 'reliable:false'
            };
        } catch { out.risingMoe = { available: false, reliable: false, reason: 'parse error' }; }
    } else out.risingMoe = { available: false, reliable: false, reason: 'missing' };
    return out;
}

function pack(data, sourceSha) {
    const out = { ...data, sourceSha256: sourceSha, seqAdopt: scanSeqAdopt() };
    if (data.nodes && typeof data.nodes === 'object') {
        out.nodes = {};
        for (const [name, node] of Object.entries(data.nodes)) {
            out.nodes[name] = node?.kind === 'window_delta_mlp' && node?.horizons
                ? packMlpNode(name, node)
                : stripHorizonBlobs(node);
        }
        out.sidecars = sidecarManifest(data.nodes.Cento);
    }
    return out;
}

const inPath = path.resolve(root, opt('in', 'models/transfer-coeffs.json'));
const outPath = path.resolve(root, opt('out', 'models/transfer-coeffs-runtime.json'));
if (!fs.existsSync(inPath)) {
    console.error(`Missing input: ${inPath}`);
    process.exit(1);
}
const raw = fs.readFileSync(inPath, 'utf8');
const sourceSha = crypto.createHash('sha256').update(raw).digest('hex');

if (hasFlag('check')) {
    const runtime = JSON.parse(fs.readFileSync(outPath, 'utf8'));
    const missing = [];
    for (const node of Object.values(runtime.nodes || {})) {
        for (const chunk of Object.values(node.packedWeights?.chunks || {})) {
            if (!fs.existsSync(path.resolve(root, chunk.url))) missing.push(chunk.url);
        }
    }
    if (runtime.sourceSha256 !== sourceSha || missing.length) {
        console.error(`Runtime pack stale. hash=${runtime.sourceSha256 !== sourceSha} missing=${missing.join(',') || 'none'}`);
        process.exit(1);
    }
    console.log('Runtime pack up to date');
    process.exit(0);
}

const data = JSON.parse(raw);
const packed = pack(data, sourceSha);
const body = JSON.stringify(packed);
fs.writeFileSync(outPath, body);
const inMb = (Buffer.byteLength(raw) / 1e6).toFixed(2);
const outMb = (Buffer.byteLength(body) / 1e6).toFixed(2);
console.log(`Packed ${path.relative(root, inPath)} (${inMb} MB) → ${path.relative(root, outPath)} (${outMb} MB)`);
