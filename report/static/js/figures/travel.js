/**
 * Lateral distance moved before seating, on matched starts where every policy succeeds.
 * Step histograms on a log x axis, median lines, hover per bin. Data: travel.json.
 */
import { s, h, svg, responsive, linear, log, axes, line, drawOn, onVisible, tooltip, tipRow,
  pointer, legend, cvar } from './kit.js';

const META = { blind: ['Blind (ours)', 0], visual: ['Visual (ours)', 1], spiral: ['Spiral search, privileged', 2] };

export default async function (mount, ctx) {
  const data = await ctx.data('travel.json');
  const keys = Object.keys(data.series);
  const holder = h('div', {}, mount);
  legend(mount, keys.map((k) => ({ label: `${META[k][0]}: median ${data.series[k].median.toFixed(0)} mm`, color: cvar(META[k][1]), kind: 'line' })));
  let animate = false;

  function draw(W) {
    holder.innerHTML = '';
    const H = Math.round(Math.min(300, Math.max(230, W * 0.36)));
    const m = { t: 16, r: 12, b: 44, l: 52 }, w = W - m.l - m.r, ht = H - m.t - m.b;
    const el = svg(holder, W, H);
    const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
    const E = data.edges;
    const x = log([E[0], E[E.length - 1]], [0, w]);
    x.ticks = () => [3, 10, 30, 100, 300, 1000];
    const ymax = Math.max(...keys.flatMap((k) => data.series[k].counts));
    const y = linear([0, ymax * 1.08], [ht, 0]);
    axes(g, { x, y, w, h: ht, xFormat: (v) => `${v}`, xLabel: `lateral travel before seating (mm, log scale) · n = ${data.n} matched starts`, yLabel: 'episodes' });
    const paths = [];
    keys.forEach((k) => {
      const c = data.series[k].counts;
      const pts = [[x(E[0]), y(0)]];
      c.forEach((v, i) => { pts.push([x(E[i]), y(v)]); pts.push([x(E[i + 1]), y(v)]); });
      pts.push([x(E[E.length - 1]), y(0)]);
      s('path', { d: line(pts) + 'Z', fill: cvar(META[k][1]), opacity: 0.12 }, g);
      paths.push(s('path', { d: line(pts), fill: 'none', stroke: cvar(META[k][1]), 'stroke-width': 1.8 }, g));
      const md = data.series[k].median;
      s('line', { x1: x(md), x2: x(md), y1: 0, y2: ht, stroke: cvar(META[k][1]), 'stroke-width': 1.5, 'stroke-dasharray': '5 3' }, g);
    });
    if (animate) paths.forEach((p, i) => drawOn(p, 900, i * 150));
    const hover = s('rect', { y: 0, height: ht, fill: 'var(--ink)', opacity: 0.06, visibility: 'hidden' }, g);
    s('rect', {
      width: w, height: ht, fill: 'transparent',
      onpointermove: (ev) => {
        const [px] = pointer(ev, el);
        const xv = x.invert(px - m.l);
        let i = E.findIndex((e, j) => j < E.length - 1 && xv >= e && xv < E[j + 1]);
        if (i < 0) i = xv < E[0] ? 0 : E.length - 2;
        hover.setAttribute('x', x(E[i])); hover.setAttribute('width', x(E[i + 1]) - x(E[i])); hover.setAttribute('visibility', 'visible');
        let rows = `<b>${E[i].toFixed(0)}–${E[i + 1].toFixed(0)} mm</b>`;
        keys.forEach((k) => { rows += tipRow(cvar(META[k][1]), META[k][0], `${data.series[k].counts[i]} episodes`); });
        keys.slice(1).forEach((k) => {
          const vs = data.series[k].vs_blind;
          if (vs) rows += `<div>Blind moves less than ${META[k][0].split(',')[0].toLowerCase()} on <span class="num">${(100 * (1 - vs.frac_blind_longer)).toFixed(0)}%</span> of starts</div>`;
        });
        tooltip.show(rows, ev);
      },
      onpointerleave: () => { hover.setAttribute('visibility', 'hidden'); tooltip.hide(); },
    }, g);
  }
  responsive(holder, draw);
  onVisible(mount, () => { animate = true; draw(holder.clientWidth); animate = false; });
}
