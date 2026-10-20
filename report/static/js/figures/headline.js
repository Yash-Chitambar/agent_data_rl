/**
 * Success rate per policy on the same 2,048 matched starts: horizontal bars, Wilson 95% CI,
 * paper value as a tick. Data: results.json (table, paper, paper_visual).
 */
import { s, h, svg, responsive, linear, tween, onVisible, ease, tooltip, tipRow, legend, cvar, pct } from './kit.js';

const ROWS = [
  ['blind', 'Blind (ours)', 0],
  ['visual', 'Visual, wrist RGB', 1],
  ['spiral', 'Spiral search, privileged', 2],
  ['random', 'Random actions', 3],
  ['zero', 'Zero action', 3],
];

export default async function (mount, ctx) {
  const data = await ctx.data('results.json');
  const rows = ROWS.filter(([k]) => data.table[k]);
  const paper = { blind: data.paper.success / 100, visual: data.paper_visual.success / 100 };
  const holder = h('div', {}, mount);
  legend(mount, [
    { label: 'Success, 95% Wilson interval', color: 'var(--ink2)', kind: 'line' },
    { label: 'Paper value', color: 'var(--ink)', kind: 'box' },
  ]);
  let animate = false;

  function draw(W) {
    holder.innerHTML = '';
    const narrow = W < 560;
    const lw = narrow ? 0 : 190, rowH = narrow ? 50 : 34;
    const m = { t: 8, r: 64, b: 30, l: lw + 8 }, w = W - m.l - m.r;
    const H = m.t + m.b + rowH * rows.length;
    const el = svg(holder, W, H);
    const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
    const x = linear([0, 1], [0, w]);
    const gg = s('g', { class: 'grid' }, g);
    const tk = s('g', { class: 'tick' }, g);
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      s('line', { x1: x(v), x2: x(v), y1: 0, y2: rowH * rows.length }, gg);
      s('text', { x: x(v), y: rowH * rows.length + 16, 'text-anchor': 'middle', text: pct(v) }, tk);
    }
    rows.forEach(([k, label, c], i) => {
      const v = data.table[k].success;
      const y0 = i * rowH + (narrow ? 20 : 6), bh = narrow ? 18 : rowH - 12;
      if (narrow) s('text', { x: 0, y: y0 - 6, text: label, class: 'title' }, g);
      else s('text', { x: -10, y: y0 + bh / 2 + 4, 'text-anchor': 'end', text: label, style: { fontSize: '12px', fill: 'var(--ink)' } }, g);
      const bar = s('rect', { x: 0, y: y0, height: bh, width: x(v.mean), rx: 2, fill: cvar(c), opacity: k === 'zero' ? 0.55 : 1 }, g);
      const err = s('path', { d: `M${x(v.lo)},${y0 + bh / 2}H${x(v.hi)}M${x(v.lo)},${y0 + 3}V${y0 + bh - 3}M${x(v.hi)},${y0 + 3}V${y0 + bh - 3}`, stroke: 'var(--ink2)', 'stroke-width': 1.2, fill: 'none' }, g);
      const val = s('text', { x: x(v.hi) + 8, y: y0 + bh / 2 + 4, text: pct(v.mean, v.mean >= 0.999 && v.mean < 1 ? 2 : 1), style: { fontVariantNumeric: 'tabular-nums', fill: 'var(--ink)' } }, g);
      if (paper[k] != null) s('rect', { x: x(paper[k]) - 1.5, y: y0 - 3, width: 3, height: bh + 6, fill: 'var(--ink)' }, g);
      if (animate) {
        err.style.opacity = 0; val.style.opacity = 0;
        tween(700 + i * 80, (e) => bar.setAttribute('width', x(v.mean * e)), ease.out)
          .then(() => tween(200, (e) => { err.style.opacity = e; val.style.opacity = e; }));
      }
      s('rect', {
        x: -lw, y: i * rowH, width: w + lw + m.r, height: rowH, fill: 'transparent',
        onpointermove: (ev) => {
          const t = data.table[k];
          tooltip.show(`<b>${label}</b>` + tipRow(cvar(c), 'Success', `${pct(v.mean, 2)} (${v.k}/${v.n})`) +
            `<div class="num">95% CI ${pct(v.lo, 1)}–${pct(v.hi, 1)}</div>` +
            (t.time != null ? `<div>Median time to seat <span class="num">${t.time.toFixed(2)} s</span></div>` : '') +
            (t.travel_mm != null ? `<div>Median lateral travel <span class="num">${t.travel_mm.toFixed(0)} mm</span></div>` : '') +
            (paper[k] != null ? `<div>Paper <span class="num">${pct(paper[k], 1)}</span></div>` : ''), ev);
        },
        onpointerleave: () => tooltip.hide(),
      }, g);
    });
  }
  responsive(holder, draw);
  onVisible(mount, () => { animate = true; draw(holder.clientWidth); animate = false; });
}
