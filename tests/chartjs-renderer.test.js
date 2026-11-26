const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { Script } = require('node:vm');

function read(path) {
    return readFileSync(new URL('../' + path, `file://${__filename}`), 'utf8');
}

test('site pages use the shared Chart.js renderer and no Plotly runtime', () => {
    for (const path of ['index.html', 'predizioni.html', 'piene.html']) {
        const html = read(path);
        assert.match(html, /<script src="chartjs-renderer\.js"><\/script>/, path);
        assert.doesNotMatch(html, /plotly/i, path);
        for (const match of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) {
            assert.doesNotThrow(() => new Script(match[1]), path);
        }
    }
    const loader = read('asset-loader.js');
    assert.match(loader, /chart\.js@4\.5\.1/);
    assert.match(loader, /integrity: 'sha384-/);
    assert.doesNotMatch(loader, /plotly/i);
    const renderer = read('chartjs-renderer.js');
    assert.doesNotThrow(() => new Script(renderer));
    assert.match(renderer, /renderTraceChart/);
    assert.match(renderer, /timeZone: 'Europe\/Rome'/);
});
