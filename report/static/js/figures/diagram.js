/**
 * Animated architecture / pipeline diagram, drawn from data.
 * Pattern shown: a layout in a fixed logical coordinate space (scaled with viewBox, since
 * a diagram's geometry matters more than its text size), boxes that draw in on first
 * view, data "packets" flowing along edges while visible, and hover that explains a
 * block and highlights its connections.
 *
 * Data (static/data/diagram.json):
 *   { width: 900, height: 300,
 *     nodes: [ { id, label, sub?, x, y, w, h, color?: 0-4, kind?: 'data'|'op'|'loss', info?, train?: bool } ],
 *     edges: [ { from, to, label?, train?: bool } ] }
 *   x, y are box centers in the logical space. train: true draws dashed (training only).
 */
import { s, h, cvar, tooltip, onVisible, watchVisible, tween, ease, reducedMotion } from './kit.js';

export default async function (mount, ctx) {
  const data = await ctx.data('diagram.json');
  const W = data.width, H = data.height;
  // Below ~640 px a diagram's text gets unreadable, so it scrolls sideways instead of shrinking.
  const scroller = h('div', { style: { overflowX: 'auto', WebkitOverflowScrolling: 'touch' } }, mount);
  const el = s('svg', { class: 'kit-svg', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'architecture diagram' }, scroller);
  el.style.maxWidth = `${W}px`; el.style.minWidth = `${Math.min(W, data.minWidth || 640)}px`; el.style.margin = '0 auto';
  const defs = s('defs', {}, el);
  const arrow = s('marker', { id: 'arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
  s('path', { d: 'M0,0L10,5L0,10z', fill: 'var(--muted)' }, arrow);

  const byId = Object.fromEntries(data.nodes.map((n) => [n.id, n]));
  const gEdges = s('g', {}, el), gPackets = s('g', {}, el), gNodes = s('g', {}, el);

  // Edges: from the right side of one box to the left side of the next, with a smooth S-curve.
  const edges = data.edges.map((e) => {
    const a = byId[e.from], b = byId[e.to];
    const vertical = Math.abs(b.x - a.x) < (a.w + b.w) / 2;
    let x1, y1, x2, y2, d;
    if (vertical) {
      x1 = a.x; y1 = a.y + Math.sign(b.y - a.y) * a.h / 2;
      x2 = b.x; y2 = b.y - Math.sign(b.y - a.y) * b.h / 2;
      const my = (y1 + y2) / 2;
      d = `M${x1},${y1}C${x1},${my} ${x2},${my} ${x2},${y2}`;
    } else {
      x1 = a.x + Math.sign(b.x - a.x) * a.w / 2; y1 = a.y;
      x2 = b.x - Math.sign(b.x - a.x) * b.w / 2; y2 = b.y;
      const mx = (x1 + x2) / 2;
      d = `M${x1},${y1}C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
    }
    const path = s('path', {
      d, fill: 'none', stroke: 'var(--muted)', 'stroke-width': 1.4, 'marker-end': 'url(#arrow)',
      'stroke-dasharray': e.train ? '5 4' : null,
    }, gEdges);
    if (e.label) s('text', { x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 7, 'text-anchor': 'middle', text: e.label }, gEdges);
    return { ...e, path, len: path.getTotalLength() };
  });

  // Nodes.
  const nodeEls = data.nodes.map((n) => {
    const g = s('g', { transform: `translate(${n.x},${n.y})`, style: { cursor: n.info ? 'help' : 'default' } }, gNodes);
    const col = n.color != null ? cvar(n.color) : 'var(--ink2)';
    const rx = n.kind === 'data' ? n.h / 2 : 8;
    s('rect', {
      x: -n.w / 2, y: -n.h / 2, width: n.w, height: n.h, rx,
      fill: n.kind === 'loss' ? 'var(--bg)' : 'var(--surface)', stroke: col, 'stroke-width': n.kind === 'op' ? 2 : 1.4,
      'stroke-dasharray': n.train ? '5 4' : null,
    }, g);
    if (n.kind === 'op') s('rect', { x: -n.w / 2, y: -n.h / 2, width: n.w, height: n.h, rx, fill: col, opacity: 0.1 }, g);
    s('text', { class: 'title', y: n.sub ? -2 : 4, 'text-anchor': 'middle', text: n.label }, g);
    if (n.sub) s('text', { y: 13, 'text-anchor': 'middle', text: n.sub, style: { fontFamily: 'var(--mono)', fontSize: '10px' } }, g);
    g.addEventListener('pointerenter', (ev) => {
      focus(n.id);
      if (n.info) tooltip.show(`<b>${n.label}</b>${n.train ? ' · training only' : ''}<div>${n.info}</div>`, ev);
    });
    g.addEventListener('pointermove', (ev) => n.info && tooltip.show(`<b>${n.label}</b>${n.train ? ' · training only' : ''}<div>${n.info}</div>`, ev));
    g.addEventListener('pointerleave', () => { focus(null); tooltip.hide(); });
    return { n, g };
  });

  // Dim everything not connected to the hovered node.
  function focus(id) {
    const linked = new Set([id]);
    edges.forEach((e) => { if (e.from === id || e.to === id) { linked.add(e.from); linked.add(e.to); } });
    nodeEls.forEach(({ n, g }) => (g.style.opacity = id == null || linked.has(n.id) ? 1 : 0.25));
    edges.forEach((e) => (e.path.style.opacity = id == null || e.from === id || e.to === id ? 1 : 0.15));
  }

  // Entrance: nodes fade/rise in left-to-right, then edges draw.
  nodeEls.forEach(({ g }) => (g.style.opacity = 0));
  edges.forEach((e) => { e.path.style.strokeDasharray = e.train ? '5 4' : `${e.len} ${e.len}`; if (!e.train) e.path.style.strokeDashoffset = e.len; });
  onVisible(mount, async () => {
    const order = [...nodeEls].sort((a, b) => a.n.x - b.n.x);
    order.forEach(({ g, n }, i) => setTimeout(() => tween(400, (e) => {
      g.style.opacity = e;
      g.setAttribute('transform', `translate(${n.x},${n.y + 8 * (1 - e)})`);
    }, ease.out), i * 70));
    await new Promise((r) => setTimeout(r, order.length * 70 + 200));
    await Promise.all(edges.filter((e) => !e.train).map((e) => tween(600, (k) => (e.path.style.strokeDashoffset = e.len * (1 - k)), ease.inOut)));
    edges.forEach((e) => { if (!e.train) e.path.style.strokeDasharray = ''; });
    startPackets();
  });

  // Packets: small dots that travel each edge on a loop, while the figure is visible.
  function startPackets() {
    if (reducedMotion) return;
    const dots = edges.map((e, i) => ({
      e, phase: i * 0.37 % 1,
      c: s('circle', { r: 3, fill: e.train ? 'var(--c4)' : 'var(--c1)' }, gPackets),
    }));
    let visible = true, t0 = performance.now();
    watchVisible(mount, (v) => { visible = v; if (v) requestAnimationFrame(frame); });
    function frame(now) {
      if (!visible) return;
      const t = (now - t0) / 1800;
      for (const d of dots) {
        const p = d.e.path.getPointAtLength(((t + d.phase) % 1) * d.e.len);
        d.c.setAttribute('cx', p.x); d.c.setAttribute('cy', p.y);
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
}
