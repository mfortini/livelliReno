(function(root) {
    'use strict';

    const timeFmt = new Intl.DateTimeFormat('it-IT', {
        timeZone: 'Europe/Rome', day: '2-digit', month: 'short',
        hour: '2-digit', minute: '2-digit', hourCycle: 'h23'
    });
    const tooltipTimeFmt = new Intl.DateTimeFormat('it-IT', {
        timeZone: 'Europe/Rome', day: '2-digit', month: '2-digit', year: 'numeric',
        hour: '2-digit', minute: '2-digit', hourCycle: 'h23'
    });
    const axisNumberFmt = new Intl.NumberFormat('it-IT', { maximumFractionDigits: 2 });
    const valueFmt = new Intl.NumberFormat('it-IT', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    function drawLine(ctx, x0, y0, x1, y1, color, width, dash) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = width || 1;
        ctx.setLineDash(dash || []);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.restore();
    }

    function dashArray(style) {
        if (style === 'dash') return [5, 4];
        if (style === 'dot') return [2, 3];
        return [];
    }

    function colorWithOpacity(color, opacity) {
        if (opacity == null || opacity >= 1 || !/^#[0-9a-f]{6}$/i.test(color)) return color;
        const n = parseInt(color.slice(1), 16);
        return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${opacity})`;
    }

    function titleText(title) {
        if (!title) return '';
        return typeof title === 'string' ? title : title.text || '';
    }

    const shapesPlugin = {
        id: 'renoShapes',
        beforeDatasetsDraw(chart, args, options) {
            const area = chart.chartArea;
            const xScale = chart.scales.x;
            const yScale = chart.scales.y;
            if (!area || !xScale || !yScale || !options) return;
            const ctx = chart.ctx;
            ctx.save();
            ctx.beginPath();
            ctx.rect(area.left, area.top, area.right - area.left, area.bottom - area.top);
            ctx.clip();
            for (const rect of options.rects || []) {
                const x0 = Math.max(area.left, xScale.getPixelForValue(rect.x0));
                const x1 = Math.min(area.right, xScale.getPixelForValue(rect.x1));
                if (x1 > x0) {
                    ctx.fillStyle = rect.color;
                    ctx.fillRect(x0, area.top, x1 - x0, area.bottom - area.top);
                }
            }
            for (const line of options.horizontal || []) {
                if (!Number.isFinite(line.y)) continue;
                const y = yScale.getPixelForValue(line.y);
                if (y >= area.top && y <= area.bottom) {
                    drawLine(ctx, area.left, y, area.right, y, line.color, line.width, line.dash);
                }
            }
            for (const line of options.vertical || []) {
                if (!Number.isFinite(line.x)) continue;
                const x = xScale.getPixelForValue(line.x);
                if (x >= area.left && x <= area.right) {
                    drawLine(ctx, x, area.top, x, area.bottom, line.color, line.width, line.dash);
                }
            }
            ctx.restore();
        }
    };

    function dataset(input) {
        return {
            label: input.label,
            data: input.points,
            borderColor: input.color,
            backgroundColor: input.fillColor || input.color,
            borderWidth: input.width == null ? 2 : input.width,
            borderDash: input.dash || [],
            pointRadius: input.pointRadius || 0,
            pointHoverRadius: input.pointRadius || 0,
            pointBackgroundColor: input.color,
            pointBorderColor: input.color,
            tension: 0,
            spanGaps: false,
            fill: input.fillTarget ? { target: input.fillTarget, above: input.fillColor } : false,
            order: input.order == null ? 10 : input.order,
            hidden: !!input.hidden,
            renoShowLegend: input.showLegend !== false
        };
    }

    function shapesFromLayout(layout) {
        const out = { horizontal: [], vertical: [], rects: [] };
        for (const shape of layout?.shapes || []) {
            if (shape.type === 'rect') {
                out.rects.push({ x0: shape.x0, x1: shape.x1, color: shape.fillcolor });
            } else if (shape.type === 'line' && shape.xref === 'paper' && shape.y0 === shape.y1) {
                out.horizontal.push({
                    y: shape.y0, color: shape.line?.color || '#64748b',
                    width: shape.line?.width || 1, dash: dashArray(shape.line?.dash)
                });
            } else if (shape.type === 'line' && shape.x0 === shape.x1) {
                out.vertical.push({
                    x: shape.x0, color: shape.line?.color || '#9bb0ba',
                    width: shape.line?.width || 1, dash: dashArray(shape.line?.dash)
                });
            }
        }
        if (layout?.yaxis?.zeroline) {
            out.horizontal.push({ y: 0, color: layout.yaxis.zerolinecolor || '#9bb0ba', width: 1, dash: [] });
        }
        if (layout?.xaxis?.zeroline) {
            out.vertical.push({ x: 0, color: layout.xaxis.zerolinecolor || '#9bb0ba', width: 1, dash: [] });
        }
        return out;
    }

    function datasetFromTrace(trace) {
        const color = trace.line?.color || trace.marker?.color || '#536b78';
        return {
            label: trace.name || '',
            points: (trace.y || []).map((y, i) => ({ x: trace.x[i], y })),
            color: colorWithOpacity(color, trace.opacity),
            width: trace.line?.width == null ? 2 : trace.line.width,
            dash: dashArray(trace.line?.dash),
            fillTarget: trace.fill === 'tonexty' ? '-1' : null,
            fillColor: trace.fillcolor,
            pointRadius: trace.mode?.includes('markers') ? Math.max(1.5, (trace.marker?.size || 5) / 2) : 0,
            showLegend: trace.showlegend !== false && !!trace.name
        };
    }

    function xAxisTick(cfg) {
        if (cfg.xTickFormatter) return cfg.xTickFormatter;
        return cfg.xType === 'time'
            ? value => timeFmt.format(new Date(value))
            : value => axisNumberFmt.format(value);
    }

    function xTooltipTitle(cfg) {
        if (cfg.xTooltipFormatter) return cfg.xTooltipFormatter;
        return cfg.xType === 'time'
            ? value => tooltipTimeFmt.format(new Date(value))
            : value => axisNumberFmt.format(value);
    }

    function renderLineChart(el, cfg) {
        if (!el || !root.Chart) return null;
        if (el._renoChart) {
            el._renoChart.destroy();
            el._renoChart = null;
        }
        el.textContent = '';
        const canvas = document.createElement('canvas');
        canvas.setAttribute('aria-hidden', 'true');
        el.appendChild(canvas);
        const interactive = !!cfg.interactive;
        const tooltipTitle = xTooltipTitle(cfg);
        const chart = new root.Chart(canvas, {
            type: 'line',
            data: { datasets: cfg.datasets.map(dataset) },
            options: {
                animation: false,
                responsive: true,
                maintainAspectRatio: false,
                events: cfg.events || (interactive ? ['mousemove', 'mouseout', 'click', 'touchstart', 'touchmove'] : []),
                normalized: true,
                interaction: { mode: cfg.tooltipMode || 'index', intersect: false },
                hover: { mode: cfg.tooltipMode || 'index', intersect: false },
                font: { family: 'system-ui, sans-serif', size: 10, color: '#536b78' },
                layout: { padding: { top: cfg.legend && cfg.legendPosition !== 'bottom' ? 0 : 4, right: 8, bottom: 0, left: 0 } },
                scales: {
                    x: {
                        type: 'linear',
                        min: cfg.xMin,
                        max: cfg.xMax,
                        title: { display: !!cfg.xTitle, text: cfg.xTitle || '', color: '#536b78', font: { size: 10 } },
                        grid: { color: '#e7eef2' },
                        border: { display: false },
                        ticks: {
                            maxTicksLimit: cfg.tickCount || 4,
                            maxRotation: 0,
                            color: '#536b78',
                            font: { size: 10 },
                            callback: xAxisTick(cfg)
                        }
                    },
                    y: {
                        min: cfg.yMin,
                        max: cfg.yMax,
                        title: { display: !!cfg.yTitle, text: cfg.yTitle || '', color: '#536b78', font: { size: 10 } },
                        grid: { color: '#e7eef2' },
                        border: { display: false },
                        ticks: {
                            color: '#536b78', font: { size: 10 }, maxTicksLimit: 6,
                            callback: value => axisNumberFmt.format(value)
                        },
                        afterFit(scale) { scale.width = cfg.yAxisWidth || 48; }
                    }
                },
                plugins: {
                    legend: {
                        display: !!cfg.legend,
                        position: cfg.legendPosition || 'top',
                        align: 'start',
                        labels: {
                            boxWidth: 18, boxHeight: 2, color: '#536b78', font: { size: 10 }, padding: 10,
                            filter: (item, data) => data.datasets[item.datasetIndex]?.renoShowLegend !== false
                        }
                    },
                    title: {
                        display: !!cfg.title,
                        text: cfg.title || '',
                        align: 'start',
                        color: '#536b78',
                        font: { size: 11, weight: 'normal' },
                        padding: { bottom: 4 }
                    },
                    tooltip: {
                        enabled: cfg.tooltip == null ? interactive : !!cfg.tooltip,
                        callbacks: {
                            title: items => items.length ? tooltipTitle(items[0].parsed.x) : '',
                            label: item => {
                                const label = item.dataset.label ? item.dataset.label + ': ' : '';
                                return `${label}${valueFmt.format(item.parsed.y)}${cfg.yUnit == null ? ' m' : cfg.yUnit}`;
                            }
                        }
                    },
                    renoShapes: cfg.shapes || {}
                }
            },
            plugins: [shapesPlugin]
        });
        el._renoChart = chart;
        return chart;
    }

    function renderTraceChart(el, traces, layout, opts = {}) {
        const xr = layout?.xaxis?.range || [];
        const yr = layout?.yaxis?.range || [];
        const legendY = layout?.legend?.y;
        return renderLineChart(el, {
            datasets: (traces || []).map(datasetFromTrace),
            xMin: xr[0], xMax: xr[1], yMin: yr[0], yMax: yr[1],
            xType: opts.xType || (layout?.xaxis?.type === 'date' ? 'time' : 'linear'),
            xTitle: opts.xTitle == null ? titleText(layout?.xaxis?.title) : opts.xTitle,
            yTitle: opts.yTitle == null ? titleText(layout?.yaxis?.title) : opts.yTitle,
            legend: opts.legend == null ? layout?.showlegend !== false : !!opts.legend,
            legendPosition: opts.legendPosition || (Number.isFinite(legendY) && legendY < 0 ? 'bottom' : 'top'),
            title: opts.title == null ? titleText(layout?.title) : opts.title,
            shapes: opts.shapes || shapesFromLayout(layout),
            tickCount: opts.tickCount || layout?.xaxis?.tickvals?.length || 4,
            yAxisWidth: opts.yAxisWidth || 48,
            interactive: !!opts.interactive,
            tooltipMode: opts.tooltipMode,
            xTickFormatter: opts.xTickFormatter,
            xTooltipFormatter: opts.xTooltipFormatter,
            yUnit: opts.yUnit
        });
    }

    function destroyChart(el) {
        if (!el) return;
        if (el._renoChart) {
            el._renoChart.destroy();
            el._renoChart = null;
        }
        el.replaceChildren();
    }

    root.RenoChartjs = { renderLineChart, renderTraceChart, destroyChart };
})(typeof globalThis !== 'undefined' ? globalThis : this);
