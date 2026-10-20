/**
 * Horizontal bars with a 95% Wilson interval, one row per condition, value label at the right.
 * The figure element names its data file: <figure class="viz" data-fig="hbars" data-src="sweep.json">.
 * Data: { title, xlabel, rows: [{label, color, mean, lo, hi, k, n, note?, slip?}], extra?: "slip" }
 */
import { s, h, svg, responsive, linear, tween, onVisible, ease, tooltip, tipRow, cvar, pct } from './kit.js';

export default async function (mount, ctx) {
  const src = mount.closest('figure').dataset.src;
  const data = await ctx.data(src);
  const holder = h('div', {}, mount);
  let animate = false;
  function draw(W) {
    holder.innerHTML = '';
    const narrow = W < 600;
    const lw = narrow ? 0 : Math.min(330, W * 0.42), rowH = narrow ? 48 : 32;
    const m = { t: 6, r: 70, b: 30, l: lw + 10 }, w = W - m.l - m.r;
    const H = m.t + m.b + rowH * data.rows.length;
    const el = svg(holder, W, H);
    const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
    const x = linear([0, 1], [0, w]);
    const gg = s('g', { class: 'grid' }, g), tk = s('g', { class: 'tick' }, g);
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      s('line', { x1: x(v), x2: x(v), y1: 0, y2: rowH * data.rows.length }, gg);
      s('text', { x: x(v), y: rowH * data.rows.length + 16, 'text-anchor': 'middle', text: pct(v) }, tk);
    }
    data.rows.forEach((rw, i) => {
      const y0 = i * rowH + (narrow ? 20 : 6), bh = narrow ? 18 : rowH - 12;
      if (narrow) s('text', { x: 0, y: y0 - 5, text: rw.label, class: 'title' }, g);
      else s('text', { x: -10, y: y0 + bh / 2 + 4, 'text-anchor': 'end', text: rw.label, style: { fontSize: '12px', fill: 'var(--ink)' } }, g);
      const bar = s('rect', { x: 0, y: y0, height: bh, width: x(rw.mean), rx: 2, fill: cvar(rw.color) }, g);
      const err = s('path', { d: `M${x(rw.lo)},${y0 + bh / 2}H${x(rw.hi)}M${x(rw.lo)},${y0 + 3}V${y0 + bh - 3}M${x(rw.hi)},${y0 + 3}V${y0 + bh - 3}`, stroke: 'var(--ink2)', 'stroke-width': 1.2, fill: 'none' }, g);
      const dg = rw.mean > 0 && rw.mean < 0.01 ? 2 : 1;
      const val = s('text', { x: x(rw.hi) + 8, y: y0 + bh / 2 + 4, text: pct(rw.mean, dg), style: { fontVariantNumeric: 'tabular-nums', fill: 'var(--ink)' } }, g);
      if (animate) {
        err.style.opacity = 0; val.style.opacity = 0;
        tween(650 + i * 70, (e) => bar.setAttribute('width', x(rw.mean * e)), ease.out)
          .then(() => tween(200, (e) => { err.style.opacity = e; val.style.opacity = e; }));
      }
      s('rect', {
        x: -lw, y: i * rowH, width: w + lw + m.r, height: rowH, fill: 'transparent',
        onpointermove: (ev) => tooltip.show(`<b>${rw.label}</b>` + tipRow(cvar(rw.color), data.xlabel, `${pct(rw.mean, 1)} (${rw.k}/${rw.n})`) +
          `<div class="num">95% CI ${pct(rw.lo, 1)}–${pct(rw.hi, 1)}</div>` +
          (data.extra && rw[data.extra] ? `<div>Slipped <span class="num">${pct(rw[data.extra].mean, 1)} (${rw[data.extra].k})</span></div>` : '') +
          (rw.note ? `<div>${rw.note}</div>` : ''), ev),
        onpointerleave: () => tooltip.hide(),
      }, g);
    });
  }
  responsive(holder, draw);
  onVisible(mount, () => { animate = true; draw(holder.clientWidth); animate = false; });
}
