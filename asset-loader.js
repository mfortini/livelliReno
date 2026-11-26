(function(root) {
    'use strict';
    const promises = new Map();

    function loadScript(key, src, attrs = {}) {
        if (promises.has(key)) return promises.get(key);
        const promise = new Promise((resolve, reject) => {
            const existing = document.querySelector(`script[data-asset="${key}"]`);
            if (existing) {
                existing.addEventListener('load', resolve, { once: true });
                existing.addEventListener('error', () => reject(new Error(`Asset non disponibile: ${key}`)), { once: true });
                return;
            }
            const script = document.createElement('script');
            script.src = src;
            script.async = true;
            script.dataset.asset = key;
            for (const [name, value] of Object.entries(attrs)) script.setAttribute(name, value);
            script.onload = resolve;
            script.onerror = () => reject(new Error(`Asset non disponibile: ${key}`));
            document.head.appendChild(script);
        });
        promises.set(key, promise);
        return promise;
    }

    function loadCss(key, href, attrs = {}) {
        if (document.querySelector(`link[data-asset="${key}"]`)) return;
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = href;
        link.dataset.asset = key;
        for (const [name, value] of Object.entries(attrs)) link.setAttribute(name, value);
        document.head.appendChild(link);
    }

    function loadChartJs() {
        if (window.Chart) return Promise.resolve(window.Chart);
        return loadScript('chartjs', 'https://unpkg.com/chart.js@4.5.1/dist/chart.umd.min.js', {
            integrity: 'sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ',
            crossorigin: ''
        }).then(() => window.Chart);
    }

    function loadLeaflet() {
        if (window.L) return Promise.resolve(window.L);
        loadCss('leaflet-css', 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css', {
            integrity: 'sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=',
            crossorigin: ''
        });
        loadCss('leaflet-gesture-css', 'https://unpkg.com/leaflet-gesture-handling@1.2.2/dist/leaflet-gesture-handling.min.css');
        return loadScript('leaflet', 'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js', {
            integrity: 'sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=',
            crossorigin: ''
        }).then(() => loadScript('leaflet-gesture', 'https://unpkg.com/leaflet-gesture-handling@1.2.2/dist/leaflet-gesture-handling.min.js').catch(() => null))
            .then(() => window.L);
    }

    root.RenoAssets = { loadChartJs, loadLeaflet };
})(typeof globalThis !== 'undefined' ? globalThis : this);
