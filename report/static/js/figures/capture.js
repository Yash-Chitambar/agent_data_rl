/**
 * Capture radius (plan E2): success against radial start distance, 0.5 mm bins with Wilson CIs,
 * the logistic fit and the 50% radius with its bootstrap CI. Data: capture.json (E2 part).
 */
import { s, h, svg, responsive, linear, axes, line, drawOn, onVisible, tooltip, tipRow, legend, cvar, pct } from './kit.js';

export default async function (mount, ctx) {
  const data = (await ctx.data('capture.json')).E2;
  const holder = h('div', {}, mount);
  legend(mount, [
    { label: 'Measured success (0.5 mm bins, 95% Wilson interval)', color: cvar(0), kind: 'box' },
    { label: 'Logistic fit', color: cvar(1), kind: 'line' },
    { label: `50% radius ${data.r50_mm.toFixed(2)} mm (CI ${data.r50_ci[0].toFixed(2)}–${data.r50_ci[1].toFixed(2)})`, color: 'var(--ink2)', kind: 'line' },
  ]);
  let animate = false;
  function draw(W) {
    holder.innerHTML = '';
    const H = Math.round(Math.min(320, Math.max(240, W * 0.42)));
    const m = { t: 16, r: 14, b: 44, l: 58 }, w = W - m.l - m.r, ht = H - m.t - m.b;
    const el = svg(holder, W, H);
    const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
    const xmax = Math.ceil(Math.max(...data.bins.map((b) => b.r_hi)));
    const x = linear([0, xmax], [0, w]), y = linear([0, 1], [ht, 0]);
    axes(g, { x, y, w, h: ht, yFormat: (v) => pct(v), xFormat: (v) => `${v}`, xLabel: `radial start offset (mm) · clearance ${data.clearance_mm} mm · n = ${data.episodes.toLocaleString()}`, yLabel: 'success' });
    const fit = [];
    for (let i = 0; i <= 120; i++) { const r = (i / 120) * xmax; fit.push([x(r), y(1 / (1 + Math.exp(-(data.a - data.b * r))))]); }
    const fp = s('path', { d: line(fit), fill: 'none', stroke: cvar(1), 'stroke-width': 2 }, g);
    s('rect', { x: x(data.r50_ci[0]), width: Math.max(1, x(data.r50_ci[1]) - x(data.r50_ci[0])), y: 0, height: ht, fill: 'var(--ink)', opacity: 0.07 }, g);
    s('line', { x1: x(data.r50_mm), x2: x(data.r50_mm), y1: 0, y2: ht, stroke: 'var(--ink2)', 'stroke-dasharray': '4 3' }, g);
    s('line', { x1: 0, x2: w, y1: y(0.5), y2: y(0.5), stroke: 'var(--muted)', 'stroke-dasharray': '2 4' }, g);
    data.bins.forEach((b) => {
      const cx = x((b.r_lo + b.r_hi) / 2);
      s('path', { d: `M${cx},${y(b.ci[0])}V${y(b.ci[1])}`, stroke: cvar(0), 'stroke-width': 1.4 }, g);
      const dot = s('circle', { cx, cy: y(b.rate), r: 4, fill: cvar(0) }, g);
      dot.addEventListener('pointermove', (ev) => tooltip.show(`<b>${b.r_lo.toFixed(1)}–${b.r_hi.toFixed(1)} mm</b>` +
        tipRow(cvar(0), 'success', `${pct(b.rate, 1)} (${b.k}/${b.n})`) + `<div class="num">95% CI ${pct(b.ci[0], 1)}–${pct(b.ci[1], 1)}</div>` +
        Object.entries(b.failures).map(([k, v]) => `<div>${k.replace(/_/g, ' ')}: <span class="num">${v}</span></div>`).join(''), ev));
      dot.addEventListener('pointerleave', () => tooltip.hide());
    });
    if (animate) drawOn(fp, 1000);
  }
  responsive(holder, draw);
  onVisible(mount, () => { animate = true; draw(holder.clientWidth); animate = false; });
}
