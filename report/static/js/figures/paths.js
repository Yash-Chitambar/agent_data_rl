/**
 * Search paths on matched starts: top view of the peg's bottom centre over the opening, and its
 * depth over time. A start selector and a timeline player. Data: paths.json.
 */
import { s, h, svg, responsive, linear, axes, line, drawOn, onVisible, tooltip, tipRow, controls,
  legend, player, cvar } from './kit.js';

const META = { blind: ['Blind (ours)', 0], visual: ['Visual (ours)', 1], spiral: ['Spiral search, privileged', 2] };

export default async function (mount, ctx) {
  const data = await ctx.data('paths.json');
  let wi = 0, t = 0, animate = false;
  const ui = controls(mount);
  ui.seg('Start', data.worlds.map((wd, i) => [i, `${i + 1}`]), 0, (v) => { wi = v; t = maxT() - 1; pl.set(t); animate = true; draw(row.clientWidth); animate = false; });
  const row = h('div', { class: 'kit-row' }, mount);
  const info = h('p', { class: 'viz-note' }, mount);
  const keysAll = [...new Set(data.worlds.flatMap((wd) => Object.keys(wd.paths)))];
  legend(mount, keysAll.map((k) => ({ label: META[k][0], color: cvar(META[k][1]), kind: 'line' }))
    .concat([{ label: 'Hole opening', color: 'var(--grid)', kind: 'box' }]));
  const maxT = () => Math.max(...Object.values(data.worlds[wi].paths).map((p) => p.z.length));
  const marks = [];

  function draw(W) {
    row.innerHTML = '';
    marks.length = 0;
    const wd = data.worlds[wi];
    const keys = Object.keys(wd.paths);
    const cols = W < 560 ? 1 : 2;
    const pw = Math.floor((W - 24 * (cols - 1)) / cols);
    // --- top view, equal aspect
    {
      const H = Math.round(Math.min(320, pw * 0.85));
      const m = { t: 26, r: 12, b: 40, l: 58 }, w = pw - m.l - m.r, ht = H - m.t - m.b;
      const el = svg(h('div', {}, row), pw, H);
      const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
      const pts = wd.hole.concat(...keys.map((k) => wd.paths[k].xy));
      let x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
      let y0 = Math.min(...pts.map((p) => p[1])), y1 = Math.max(...pts.map((p) => p[1]));
      const pad = 0.12 * Math.max(x1 - x0, y1 - y0);
      x0 -= pad; x1 += pad; y0 -= pad; y1 += pad;
      const k = Math.min(w / (x1 - x0), ht / (y1 - y0));
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const X = linear([cx - w / 2 / k, cx + w / 2 / k], [0, w]), Y = linear([cy - ht / 2 / k, cy + ht / 2 / k], [ht, 0]);
      axes(g, { x: X, y: Y, w, h: ht, xLabel: 'x (mm)', yLabel: 'y (mm)', title: '(a) Top view, fixture frame', grid: 'xy' });
      s('path', { d: line(wd.hole.map((p) => [X(p[0]), Y(p[1])])) + 'Z', fill: 'var(--grid)', stroke: 'var(--muted)', 'stroke-width': 1 }, g);
      const paths = [];
      keys.forEach((kk) => {
        const p = wd.paths[kk], col = cvar(META[kk][1]);
        paths.push(s('path', { d: line(p.xy.map((q) => [X(q[0]), Y(q[1])])), fill: 'none', stroke: col, 'stroke-width': 1.8, 'stroke-linejoin': 'round' }, g));
        s('circle', { cx: X(p.xy[0][0]), cy: Y(p.xy[0][1]), r: 4, fill: 'var(--surface)', stroke: col, 'stroke-width': 1.6 }, g);
        const foot = s('path', { fill: 'none', stroke: col, 'stroke-width': 1.4, 'stroke-dasharray': '4 3' }, g);
        const dot = s('circle', { r: 4.5, fill: col }, g);
        marks.push({ kind: 'xy', dot, foot, p, X, Y, hole: wd.hole });
      });
      if (animate) paths.forEach((pa, i) => drawOn(pa, 1100, i * 160));
    }
    // --- depth over time
    {
      const H = Math.round(Math.min(320, pw * 0.85));
      const m = { t: 26, r: 12, b: 40, l: 58 }, w = pw - m.l - m.r, ht = H - m.t - m.b;
      const el = svg(h('div', {}, row), pw, H);
      const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
      const T = maxT();
      const zs = keys.flatMap((kk) => wd.paths[kk].z);
      const X = linear([0, (T - 1) * data.dt], [0, w]);
      const Y = linear([Math.min(-wd.depth_mm - 2, ...zs), Math.max(4, ...zs)], [ht, 0]);
      axes(g, { x: X, y: Y, w, h: ht, xLabel: 'time (s)', yLabel: 'peg bottom height (mm)', title: '(b) Depth over time', xFormat: (v) => v.toFixed(1) });
      s('line', { x1: 0, x2: w, y1: Y(0), y2: Y(0), stroke: 'var(--muted)', 'stroke-width': 1 }, g);
      s('line', { x1: 0, x2: w, y1: Y(-wd.depth_mm), y2: Y(-wd.depth_mm), stroke: 'var(--muted)', 'stroke-dasharray': '4 3' }, g);
      s('text', { x: w - 2, y: Y(0) - 5, 'text-anchor': 'end', text: 'plate top' }, g);
      s('text', { x: w - 2, y: Y(-wd.depth_mm) - 5, 'text-anchor': 'end', text: 'hole floor' }, g);
      const paths = [];
      keys.forEach((kk) => {
        const p = wd.paths[kk], col = cvar(META[kk][1]);
        paths.push(s('path', { d: line(p.z.map((z, i) => [X(i * data.dt), Y(z)])), fill: 'none', stroke: col, 'stroke-width': 1.8 }, g));
        const dot = s('circle', { r: 4.5, fill: col }, g);
        marks.push({ kind: 'z', dot, p, X, Y });
      });
      const cur = s('line', { class: 'hover-line', y1: 0, y2: ht }, g);
      marks.push({ kind: 'cursor', cur, X });
      if (animate) paths.forEach((pa, i) => drawOn(pa, 1100, i * 160));
      s('rect', {
        width: w, height: ht, fill: 'transparent',
        onpointermove: (ev) => {
          const r = el.getBoundingClientRect();
          const tt = Math.round(X.invert(ev.clientX - r.left - m.l) / data.dt);
          let rows = `<b>Start ${wi + 1}</b> · t = ${(Math.max(0, tt) * data.dt).toFixed(2)} s`;
          keys.forEach((kk) => {
            const p = wd.paths[kk], i = Math.max(0, Math.min(p.z.length - 1, tt));
            rows += tipRow(cvar(META[kk][1]), META[kk][0], `${p.z[i].toFixed(1)} mm · ${(100 * p.inside[i]).toFixed(0)}% inside`);
          });
          tooltip.show(rows, ev);
        },
        onpointerleave: () => tooltip.hide(),
      }, g);
    }
    info.textContent = `Start ${wi + 1}: clearance ${wd.clearance_mm.toFixed(2)} mm, hole depth ${wd.depth_mm.toFixed(0)} mm. ` +
      keys.map((kk) => { const p = wd.paths[kk]; return `${META[kk][0]}: ${p.success ? `seated in ${p.time.toFixed(2)} s` : 'did not seat'}`; }).join('. ') + '.';
    place(t);
  }

  function place(i) {
    t = i;
    for (const mk of marks) {
      if (mk.kind === 'cursor') { mk.cur.setAttribute('x1', mk.X(i * data.dt)); mk.cur.setAttribute('x2', mk.X(i * data.dt)); continue; }
      const j = Math.min(i, mk.p.z.length - 1);
      if (mk.kind === 'xy') {
        const [px, py] = mk.p.xy[j];
        mk.dot.setAttribute('cx', mk.X(px)); mk.dot.setAttribute('cy', mk.Y(py));
        mk.foot.setAttribute('d', line(mk.hole.map((q) => [mk.X(q[0] + px), mk.Y(q[1] + py)])) + 'Z');
      }
      else { mk.dot.setAttribute('cx', mk.X(j * data.dt)); mk.dot.setAttribute('cy', mk.Y(mk.p.z[j])); }
    }
  }
  const N = Math.max(...data.worlds.map((wd) => Math.max(...Object.values(wd.paths).map((p) => p.z.length))));
  const pl = player(mount, { n: N, fps: 12.5, onFrame: (i) => place(Math.min(i, maxT() - 1)), label: (i) => `t = ${(i * data.dt).toFixed(2)} s (¼ speed)` });
  t = maxT() - 1;
  responsive(row, draw);
  pl.set(t);
  onVisible(mount, () => { animate = true; draw(row.clientWidth); animate = false; });
}
