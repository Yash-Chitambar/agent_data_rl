/**
 * Success on the training range and on six held-out part families (paper Table 6 bins).
 * Grouped bars (blind, visual) with Wilson CIs; dashes show the paper's values.
 * Data: results.json (ood, paper_ood, bins, bin_label).
 */
import { s, h, svg, responsive, linear, band, axes, tween, onVisible, ease, tooltip, tipRow,
  controls, legend, cvar, pct } from './kit.js';

export default async function (mount, ctx) {
  const data = await ctx.data('results.json');
  const pols = [['blind', 'Blind (ours)', 0], ['visual', 'Visual (ours)', 1]].filter(([k]) => data.ood[k]);
  const paperIdx = { blind: 0, visual: 1 };
  let showPaper = true, animate = false;
  controls(mount).seg('Paper values', [[true, 'Show'], [false, 'Hide']], true, (v) => { showPaper = v; animate = true; draw(holder.clientWidth); animate = false; });
  const holder = h('div', {}, mount);
  legend(mount, pols.map(([, l, c]) => ({ label: l, color: cvar(c), kind: 'box' }))
    .concat([{ label: 'Paper value (same policy type)', color: 'var(--ink)', kind: 'line' }]));

  function draw(W) {
    holder.innerHTML = '';
    const narrow = W < 560;
    const bins = data.bins;
    const H = narrow ? 300 : 320;
    const m = { t: 22, r: 8, b: narrow ? 70 : 48, l: 56 }, w = W - m.l - m.r, ht = H - m.t - m.b;
    const el = svg(holder, W, H);
    const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
    const labels = bins.map((b) => data.bin_label[b]);
    const gx = band(labels, [0, w], { inner: 0.25, outer: 0.08 });
    const bx = band(pols.map(([k]) => k), [0, gx.bw], { inner: 0.1, outer: 0 });
    const y = linear([0, 1], [ht, 0]);
    axes(g, { x: { ...gx, bw: gx.bw, keys: [] }, y, w, h: ht, yFormat: (v) => pct(v), yLabel: 'success' });
    // x labels (wrap/rotate on phones)
    labels.forEach((lb, i) => {
      const cx = gx(lb) + gx.bw / 2;
      if (narrow) s('text', { x: cx, y: ht + 12, 'text-anchor': 'end', transform: `rotate(-40 ${cx} ${ht + 12})`, text: lb, class: 'tick-l' }, g);
      else s('text', { x: cx, y: ht + 18, 'text-anchor': 'middle', text: lb }, g);
      void i;
    });
    const xd = gx(labels[1]) - (gx.step - gx.bw) / 2;
    s('line', { x1: xd, x2: xd, y1: -10, y2: ht, stroke: 'var(--muted)', 'stroke-dasharray': '4 4' }, g);
    if (!narrow) {
      s('text', { x: xd - 6, y: -8, 'text-anchor': 'end', text: 'training range' }, g);
      s('text', { x: xd + 6, y: -8, text: 'held-out part families' }, g);
    }
    bins.forEach((b) => {
      pols.forEach(([k, label, c]) => {
        const v = data.ood[k][b];
        const x0 = gx(data.bin_label[b]) + bx(k), bw = bx.bw;
        const paper = b === 'in_dist' ? (k === 'blind' ? data.paper.success : data.paper_visual.success)
          : (data.paper_ood[b] || [])[paperIdx[k]];
        if (!v) {
          s('text', { x: x0 + bw / 2, y: ht - 6, 'text-anchor': 'middle', text: '–' }, g);
          return;
        }
        const bar = s('rect', { x: x0, width: bw, y: y(v.mean), height: ht - y(v.mean), rx: 2, fill: cvar(c) }, g);
        const err = s('path', { d: `M${x0 + bw / 2},${y(v.lo)}V${y(v.hi)}M${x0 + bw / 2 - 3},${y(v.lo)}h6M${x0 + bw / 2 - 3},${y(v.hi)}h6`, stroke: 'var(--ink2)', 'stroke-width': 1.2, fill: 'none' }, g);
        if (showPaper && paper != null) s('line', { x1: x0 - 1, x2: x0 + bw + 1, y1: y(paper / 100), y2: y(paper / 100), stroke: 'var(--ink)', 'stroke-width': 2, 'stroke-dasharray': '4 2' }, g);
        if (animate) {
          err.style.opacity = 0;
          tween(700, (e) => { const yy = y(v.mean * e); bar.setAttribute('y', yy); bar.setAttribute('height', ht - yy); }, ease.out)
            .then(() => tween(200, (e) => (err.style.opacity = e)));
        }
        s('rect', {
          x: x0 - 2, width: bw + 4, y: 0, height: ht, fill: 'transparent',
          onpointermove: (ev) => tooltip.show(`<b>${data.bin_label[b]}</b>${b === 'in_dist' ? '' : ' · held out'}` +
            tipRow(cvar(c), label, `${pct(v.mean, 1)} (${v.k}/${v.n})`) +
            `<div class="num">95% CI ${pct(v.lo, 1)}–${pct(v.hi, 1)}</div>` +
            (paper != null ? `<div>Paper <span class="num">${paper.toFixed(1)}%</span></div>` : ''), ev),
          onpointerleave: () => tooltip.hide(),
        }, g);
      });
    });
  }
  responsive(holder, draw);
  onVisible(mount, () => { animate = true; draw(holder.clientWidth); animate = false; });
}
