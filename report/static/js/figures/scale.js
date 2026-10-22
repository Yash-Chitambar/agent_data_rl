/**
 * Phase D, 8 mm peg (RQ3): (a) test success against clearance for the scaled and the fixed
 * start regime, with Wilson CIs; (b) the two main failure types of the scaled regime.
 * Data: scale.json (scripts/phase_d_report.py).
 */
import { s, h, svg, responsive, linear, band, axes, tween, onVisible, ease, tooltip, tipRow,
  legend, cvar, pct } from './kit.js';

const REG = [['scaled', 'Scaled start (1.3 mm median)', 0], ['fixed', 'Fixed start (3 mm median)', 1]];
const FAIL = [['never_found', 'Did not find the hole', 2], ['wedged_partial', 'Wedged partly inside', 3]];

export default async function (mount, ctx) {
  const data = await ctx.data('scale.json');
  const cells = data.cells.filter((c) => c.opening_mm === 8);
  const cl = data.clearances_mm.filter((c) => cells.some((x) => x.clearance_mm === c));
  const get = (reg, c) => cells.find((x) => x.regime === reg && x.clearance_mm === c);
  const lab = (c) => `${c >= 1 ? c.toFixed(1) : c} mm`;
  const tip = (c) => `<b>8 mm peg, ${lab(c)} clearance</b>` + REG.filter(([r]) => get(r, c)).map(([r, l, ci]) => {
    const v = get(r, c);
    return tipRow(cvar(ci), l, `${pct(v.rate, 1)} (${v.k}/${v.n})`) +
      `<div class="num">CI ${pct(v.ci[0], 1)}–${pct(v.ci[1], 1)} · ${v.time_med.toFixed(2)} s · ${v.travel_mm.toFixed(0)} mm travel</div>`;
  }).join('');
  let animate = false;
  const row = h('div', { class: 'cols2' }, mount);
  const pa = h('div', {}, row), pb = h('div', {}, row);
  legend(pa, REG.map(([, l, c]) => ({ label: l, color: cvar(c), kind: 'line' })));
  legend(pb, FAIL.map(([, l, c]) => ({ label: `${l} (scaled start)`, color: cvar(c), kind: 'box' })));
  const ha = h('div', {}, pa), hb = h('div', {}, pb);

  function drawA(W) {
    ha.innerHTML = '';
    const H = 280, m = { t: 22, r: 12, b: 44, l: 52 }, w = W - m.l - m.r, ht = H - m.t - m.b;
    const g = s('g', { transform: `translate(${m.l},${m.t})` }, svg(ha, W, H));
    const x = band(cl.map(lab), [0, w], { inner: 0, outer: 0.3 });
    const y = linear([0.75, 1], [ht, 0]);
    axes(g, { x: { ...x, keys: [] }, y, w, h: ht, yTicks: [0.75, 0.8, 0.85, 0.9, 0.95, 1], yFormat: (v) => pct(v), yLabel: 'test success', xLabel: 'clearance per side' });
    s('text', { x: 0, y: -8, text: '(a) Success, 2,048 starts per point' }, g);
    cl.forEach((c) => s('text', { x: x(lab(c)) + x.bw / 2, y: ht + 16, 'text-anchor': 'middle', text: lab(c) }, g));
    REG.forEach(([r, , ci], i) => {
      const pts = cl.map((c) => [c, get(r, c)]).filter(([, v]) => v);
      const cx = (c) => x(lab(c)) + x.bw / 2 + (i ? 5 : -5);
      const path = s('path', { d: 'M' + pts.map(([c, v]) => `${cx(c)},${y(v.rate)}`).join('L'), fill: 'none', stroke: cvar(ci), 'stroke-width': 2 }, g);
      pts.forEach(([c, v]) => {
        s('path', { d: `M${cx(c)},${y(v.ci[0])}V${y(v.ci[1])}`, stroke: cvar(ci), 'stroke-width': 1.4 }, g);
        s('circle', { cx: cx(c), cy: y(v.rate), r: 4.5, fill: cvar(ci), stroke: 'var(--bg)', 'stroke-width': 2 }, g);
      });
      if (animate) { path.style.opacity = 0; tween(600, (e) => (path.style.opacity = e), ease.out); }
    });
    cl.forEach((c) => s('rect', { x: x(lab(c)), width: x.bw, y: 0, height: ht, fill: 'transparent',
      onpointermove: (ev) => tooltip.show(tip(c), ev), onpointerleave: () => tooltip.hide() }, g));
  }

  function drawB(W) {
    hb.innerHTML = '';
    const H = 280, m = { t: 22, r: 12, b: 44, l: 52 }, w = W - m.l - m.r, ht = H - m.t - m.b;
    const g = s('g', { transform: `translate(${m.l},${m.t})` }, svg(hb, W, H));
    const gx = band(cl.map(lab), [0, w], { inner: 0.3, outer: 0.15 });
    const bx = band(FAIL.map(([f]) => f), [0, gx.bw], { inner: 0.1, outer: 0 });
    const vmax = Math.max(...cl.map((c) => Math.max(...FAIL.map(([f]) => (get('scaled', c)?.failures[f] || 0) / 2048))));
    const y = linear([0, Math.ceil(vmax * 100) / 100], [ht, 0]);
    axes(g, { x: { ...gx, keys: [] }, y, w, h: ht, yFormat: (v) => pct(v, 1), yLabel: 'share of starts', xLabel: 'clearance per side' });
    s('text', { x: 0, y: -8, text: '(b) Failures, scaled start' }, g);
    cl.forEach((c) => {
      s('text', { x: gx(lab(c)) + gx.bw / 2, y: ht + 16, 'text-anchor': 'middle', text: lab(c) }, g);
      const v = get('scaled', c);
      if (!v) return;
      FAIL.forEach(([f, , ci]) => {
        const val = (v.failures[f] || 0) / v.n, x0 = gx(lab(c)) + bx(f);
        const bar = s('rect', { x: x0, width: bx.bw, y: y(val), height: ht - y(val), rx: 2, fill: cvar(ci) }, g);
        if (animate) tween(700, (e) => { const yy = y(val * e); bar.setAttribute('y', yy); bar.setAttribute('height', ht - yy); }, ease.out);
      });
      s('rect', { x: gx(lab(c)) - 3, width: gx.bw + 6, y: 0, height: ht, fill: 'transparent',
        onpointermove: (ev) => tooltip.show(`<b>8 mm peg, ${lab(c)}, scaled start</b>` +
          FAIL.map(([f, l, ci]) => tipRow(cvar(ci), l, `${v.failures[f] || 0} of ${v.n}`)).join(''), ev),
        onpointerleave: () => tooltip.hide() }, g);
    });
  }
  responsive(ha, drawA);
  responsive(hb, drawB);
  onVisible(mount, () => { animate = true; drawA(ha.clientWidth); drawB(hb.clientWidth); animate = false; });
}
