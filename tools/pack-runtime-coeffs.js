#!/usr/bin/env node
// Pack browser-facing transfer coeffs: compact JSON, strip bulky train/holdout blobs.
// Keeps validate / validateEvents (needed for maxReliableHorizonMinutes / UI fade cut).
// Usage: node tools/pack-runtime-coeffs.js
//   [--in models/transfer-coeffs.json] [--out models/transfer-coeffs-runtime.json]

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
function opt(name, fallback) {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

// Keep validate / validateEvents: maxReliableHorizonMinutes and UI skill cuts read them.
const STRIP_KEYS = new Set([
    'train', 'holdout', 'holdoutEvents', 'earlyStopping'
]);

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

function pack(data) {
    const out = { ...data };
    if (data.nodes && typeof data.nodes === 'object') {
        out.nodes = {};
        for (const [name, node] of Object.entries(data.nodes)) {
            out.nodes[name] = stripHorizonBlobs(node);
        }
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
const data = JSON.parse(raw);
const packed = pack(data);
const body = JSON.stringify(packed);
fs.writeFileSync(outPath, body);
const inMb = (Buffer.byteLength(raw) / 1e6).toFixed(2);
const outMb = (Buffer.byteLength(body) / 1e6).toFixed(2);
console.log(`Packed ${path.relative(root, inPath)} (${inMb} MB) → ${path.relative(root, outPath)} (${outMb} MB)`);
