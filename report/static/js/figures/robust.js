/**
 * YAM robustness (Phase C): success under each test condition for the policy trained without
 * it, the policy trained with it, and (when present) the policy trained with all conditions
 * randomized. Grouped bars with Wilson CIs. Data: robust.json (scripts/report/phase_c.py).
 */
import { s, h, svg, responsive, linear, band, axes, tween, onVisible, ease, tooltip, tipRow,
  legend, cvar, pct } from './kit.js';

export default async function (mount, ctx) {
  const data = await ctx.data('robust.json');
  const colors = { base: 1, own: 0, randall: 2 };
  const pols = data.policies.map((p) => [p.key, p.label, colors[p.key] ?? 3]);
  let animate = false;
  const holder = h('div', {}, mount);
  legend(mount, pols.map(([, l, c]) => ({ label: l, color: cvar(c), kind: 'box' })));

  function draw(W) {
    holder.innerHTML = '';
    const narrow = W < 640;
    const conds = data.conditions;
    const H = narrow ? 330 : 330;
    const m = { t: 24, r: 8, b: narrow ? 92 : 60, l: 56 }, w = W - m.l - m.r, ht = H - m.t - m.b;
    const el = svg(holder, W, H);
    const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
    const labels = conds.map((c) => c.label);
    const gx = band(labels, [0, w], { inner: 0.22, outer: 0.06 });
    const bx = band(pols.map(([k]) => k), [0, gx.bw], { inner: 0.08, outer: 0 });
    const y = linear([0, 1], [ht, 0]);
    axes(g, { x: { ...gx, bw: gx.bw, keys: [] }, y, w, h: ht, yFormat: (v) => pct(v), yLabel: 'success' });
    labels.forEach((lb) => {
      const cx = gx(lb) + gx.bw / 2;
      if (narrow) s('text', { x: cx, y: ht + 12, 'text-anchor': 'end', transform: `rotate(-45 ${cx} ${ht + 12})`, text: lb }, g);
      else {
        const parts = lb.split(' ');
        const mid = Math.ceil(parts.length / 2);
        s('text', { x: cx, y: ht + 16, 'text-anchor': 'middle', text: parts.slice(0, mid).join(' ') }, g);
        if (parts.length > 1) s('text', { x: cx, y: ht + 30, 'text-anchor': 'middle', text: parts.slice(mid).join(' ') }, g);
      }
    });
    if (!narrow) s('text', { x: 0, y: -10, text: 'Condition in the test' }, g);
    conds.forEach((c) => {
      pols.forEach(([k, label, ci]) => {
        const v = c.values[k];
        const x0 = gx(c.label) + bx(k), bw = bx.bw;
        if (!v) return;
        const bar = s('rect', { x: x0, width: bw, y: y(v.mean), height: ht - y(v.mean), rx: 2, fill: cvar(ci) }, g);
        const err = s('path', { d: `M${x0 + bw / 2},${y(v.lo)}V${y(v.hi)}`, stroke: 'var(--ink2)', 'stroke-width': 1.2, fill: 'none' }, g);
        if (animate) {
          err.style.opacity = 0;
          tween(700, (e) => { const yy = y(v.mean * e); bar.setAttribute('y', yy); bar.setAttribute('height', ht - yy); }, ease.out)
            .then(() => tween(200, (e) => (err.style.opacity = e)));
        }
      });
      s('rect', {
        x: gx(c.label) - 3, width: gx.bw + 6, y: 0, height: ht, fill: 'transparent',
        onpointermove: (ev) => tooltip.show(`<b>Test: ${c.label}</b>` +
          pols.filter(([k]) => c.values[k]).map(([k, label, ci]) => {
            const v = c.values[k];
            const name = k === 'own' && c.key !== 'base' ? `${label} (${c.own_run})` : label;
            return tipRow(cvar(ci), name, `${pct(v.mean, 1)} (${v.k}/${v.n})`);
          }).join(''), ev),
        onpointerleave: () => tooltip.hide(),
      }, g);
    });
  }
  responsive(holder, draw);
  onVisible(mount, () => { animate = true; draw(holder.clientWidth); animate = false; });
}
