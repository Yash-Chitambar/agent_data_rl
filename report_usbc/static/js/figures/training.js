/**
 * Training curves with one x grid per seed (runs stop at different step counts).
 * Data (static/data/curves.json): { xLabel, yLabel, panels: [{title, keys}], series: { key: {label, color, seeds: [{name, x, y, steps, to90}]} } }
 */
import { s, h, svg, responsive, linear, axes, line, area, drawOn, tween, onVisible,
  tooltip, tipRow, pointer, controls, legend, cvar, pct, fmt } from './kit.js';

function interp(xs, ys, x) {
  if (x < xs[0] || x > xs[xs.length - 1]) return null;
  let i = 1;
  while (i < xs.length && xs[i] < x) i++;
  if (i >= xs.length) return ys[ys.length - 1];
  const t = (x - xs[i - 1]) / ((xs[i] - xs[i - 1]) || 1);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

export default async function (mount, ctx) {
  const data = await ctx.data('curves.json');
  let mode = 'band', logx = false, animate = false;
  const ui = controls(mount);
  ui.seg('Seeds', [['band', 'Mean and range'], ['each', 'Each seed']], mode, (v) => { mode = v; redraw(); });
  ui.seg('Steps', [[false, 'Linear'], [true, 'Log']], logx, (v) => { logx = v; redraw(); });
  const row = h('div', { class: 'kit-row' }, mount);
  const used = [...new Set(data.panels.flatMap((p) => p.keys))].filter((k) => data.series[k]);
  legend(mount, used.map((k) => {
    const sr = data.series[k];
    return { label: `${sr.label} (${sr.seeds.length} seed${sr.seeds.length > 1 ? 's' : ''})`, color: cvar(sr.color), kind: 'line' };
  }).concat([{ label: '90% line', color: 'var(--muted)', kind: 'line' }]));

  const redraw = () => { animate = true; draw(row.clientWidth); animate = false; };
  const draw = (width) => {
    row.innerHTML = '';
    const cols = width < 560 ? 1 : data.panels.length;
    const pw = Math.floor((width - 24 * (cols - 1)) / cols);
    data.panels.forEach((p) => panel(h('div', {}, row), p, pw));
  };

  function panel(parent, p, W) {
    const keys = p.keys.filter((k) => data.series[k]);
    const H = Math.round(Math.min(280, Math.max(220, W * 0.66)));
    const m = { t: 28, r: 12, b: 44, l: 58 }, w = W - m.l - m.r, ht = H - m.t - m.b;
    const el = svg(parent, W, H);
    const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
    const xmax = Math.max(...keys.flatMap((k) => data.series[k].seeds.map((sd) => sd.x[sd.x.length - 1])));
    const xmin = logx ? 1 : 0;
    const x = logx
      ? (() => { const l0 = 0, l1 = Math.log10(xmax); const f = (v) => ((Math.log10(Math.max(v, 1)) - l0) / (l1 - l0)) * w; f.invert = (px) => 10 ** (l0 + (px / w) * (l1 - l0)); f.ticks = () => [1, 3, 10, 30, 100, 300].filter((v) => v <= xmax); return f; })()
      : linear([xmin, xmax], [0, w]);
    const y = linear([0, 1], [ht, 0]);
    axes(g, { x, y, w, h: ht, yFormat: (v) => pct(v), xFormat: (v) => `${fmt(v)}`, xLabel: data.xLabel, yLabel: data.yLabel, title: p.title });
    s('line', { x1: 0, x2: w, y1: y(0.9), y2: y(0.9), stroke: 'var(--muted)', 'stroke-dasharray': '2 4' }, g);

    // Common grid per method: mean and min-max over the seeds that reach each x.
    const agg = {};
    for (const k of keys) {
      const seeds = data.series[k].seeds;
      const end = Math.min(...seeds.map((sd) => sd.x[sd.x.length - 1]));
      const start = Math.max(...seeds.map((sd) => sd.x[0]));
      const grid = Array.from({ length: 160 }, (_, i) => Math.min(end, start + (i / 159) * (end - start)));
      const vals = grid.map((gx) => seeds.map((sd) => interp(sd.x, sd.y, Math.min(gx, sd.x[sd.x.length - 1]))).filter((v) => v != null));
      agg[k] = { grid, mean: vals.map((v) => v.reduce((a, b) => a + b, 0) / v.length),
        lo: vals.map((v) => Math.min(...v)), hi: vals.map((v) => Math.max(...v)) };
    }
    const paths = [];
    for (const k of keys) {
      const sr = data.series[k], col = cvar(sr.color), a = agg[k];
      if (mode === 'band') {
        if (sr.seeds.length > 1) {
          const b = s('path', { d: area(a.grid.map(x), a.lo.map(y), a.hi.map(y)), fill: col, opacity: 0.16 }, g);
          if (animate) { b.style.opacity = 0; tween(600, (e) => (b.style.opacity = 0.16 * e)); }
        }
        paths.push(s('path', { d: line(a.grid.map((gx, i) => [x(gx), y(a.mean[i])])), fill: 'none', stroke: col, 'stroke-width': 2, 'stroke-linejoin': 'round' }, g));
      } else {
        sr.seeds.forEach((sd, j) => paths.push(s('path', {
          d: line(sd.x.map((sx, i) => [x(Math.max(sx, xmin)), y(sd.y[i])])), fill: 'none', stroke: col,
          'stroke-width': 1.6, opacity: 0.55 + 0.45 * (j === 0), 'stroke-dasharray': j ? '5 3' : null,
        }, g)));
      }
    }
    if (animate) paths.forEach((pa, i) => drawOn(pa, 1000, i * 120));

    const hover = s('line', { class: 'hover-line', y1: 0, y2: ht, visibility: 'hidden' }, g);
    s('rect', {
      width: w, height: ht, fill: 'transparent',
      onpointermove: (ev) => {
        const [px] = pointer(ev, el);
        const xv = Math.max(xmin, x.invert(px - m.l));
        hover.setAttribute('x1', x(xv)); hover.setAttribute('x2', x(xv)); hover.setAttribute('visibility', 'visible');
        let rows = `<b>${p.title}</b><div class="num">${xv.toFixed(1)}M steps</div>`;
        for (const k of keys) {
          const sr = data.series[k];
          sr.seeds.forEach((sd) => {
            const v = interp(sd.x, sd.y, xv);
            rows += tipRow(cvar(sr.color), `${sr.label} · ${sd.name.split('-').pop()}`, v == null ? 'run ended' : pct(v, 1));
          });
        }
        tooltip.show(rows, ev);
      },
      onpointerleave: () => { hover.setAttribute('visibility', 'hidden'); tooltip.hide(); },
    }, g);
  }

  responsive(row, draw);
  onVisible(mount, () => { animate = true; draw(row.clientWidth); animate = false; });
}
