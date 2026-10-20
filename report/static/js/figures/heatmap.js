/**
 * Observation-history heatmap on <canvas> with a timeline player.
 * Pattern shown: canvas for dense data (thousands of cells redraw at 60 fps where SVG
 * would crawl), a sliding window that follows the playhead, and a synced line plot.
 *
 * Data (static/data/heatmap.json):
 *   { features: [names], window: 30, dt: 0.02, frames: [[f0, f1, ...], ...],   // T x F, z-scored
 *     signal?: { label, values: [T] } }
 */
import { h, s, svg, linear, line, player, cssColor, C } from './kit.js';

// Diverging colormap blue-white-orange, t in [-1, 1].
function diverge(t, bg) {
  const a = Math.min(1, Math.abs(t));
  const hex = t < 0 ? C.blue : C.orange;
  const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const b = bg.match(/\w\w/g).map((x) => parseInt(x, 16));
  return `rgb(${c.map((v, i) => Math.round(b[i] + (v - b[i]) * a)).join(',')})`;
}

export default async function (mount, ctx) {
  const data = await ctx.data('heatmap.json');
  const T = data.frames.length, F = data.features.length, Wn = data.window;

  const wrap = h('div', { style: { position: 'relative' } }, mount);
  const canvas = h('canvas', { style: { width: '100%', display: 'block', borderRadius: '6px' } }, wrap);
  const sigHolder = h('div', {}, mount);
  const g2 = canvas.getContext('2d');

  let cur = 0, labelW = 0;
  function drawHeat(t) {
    cur = t;
    const dpr = devicePixelRatio || 1;
    const W = wrap.clientWidth, rowH = Math.max(9, Math.min(16, 260 / F));
    labelW = Math.min(150, Math.max(64, W * 0.12));
    const H = rowH * F + 22;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    canvas.style.height = `${H}px`;
    g2.setTransform(dpr, 0, 0, dpr, 0, 0);
    const bgHex = (cssColor('--surface').match(/^#([0-9a-f]{6})$/i) || [, 'ffffff'])[1];
    g2.fillStyle = `#${bgHex}`; g2.fillRect(0, 0, W, H);
    const cw = (W - labelW) / Wn;
    g2.font = '10.5px Inter, system-ui, sans-serif'; g2.textBaseline = 'middle'; g2.textAlign = 'right';
    g2.fillStyle = cssColor('--ink2');
    data.features.forEach((name, f) => g2.fillText(name, labelW - 8, rowH * f + rowH / 2));
    for (let k = 0; k < Wn; k++) {
      const fi = t - (Wn - 1) + k;         // oldest on the left, current frame on the right
      if (fi < 0) continue;
      const row = data.frames[fi];
      for (let f = 0; f < F; f++) {
        g2.fillStyle = diverge(row[f] / 2.5, bgHex);
        g2.fillRect(labelW + k * cw, rowH * f, Math.ceil(cw), rowH - 1);
      }
    }
    g2.strokeStyle = cssColor('--ink'); g2.lineWidth = 1.5;
    g2.strokeRect(labelW + (Wn - 1) * cw + 0.75, 0.75, cw - 1.5, rowH * F - 1.5);
    g2.fillStyle = cssColor('--muted'); g2.textAlign = 'center';
    g2.fillText(`−${((Wn - 1) * data.dt).toFixed(2)} s`, labelW + cw / 2, H - 9);
    g2.fillText('now', labelW + (Wn - 0.5) * cw, H - 9);
    drawSignal(t);
  }

  function drawSignal(t) {
    if (!data.signal) return;
    sigHolder.innerHTML = '';
    // Same left margin as the heatmap's label column, so the two plots line up.
    const W = sigHolder.clientWidth || wrap.clientWidth, H = 90, m = { l: labelW, r: 0, t: 12, b: 16 };
    const el = svg(sigHolder, W, H);
    const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
    const w = W - m.l - m.r, ht = H - m.t - m.b;
    const x = linear([0, T - 1], [0, w]), y = linear([0, 1], [ht, 0]);
    s('path', { class: 'domain', d: `M0,${ht}H${w}` }, g);
    s('text', { x: -8, y: ht / 2 + 4, 'text-anchor': 'end', text: data.signal.label }, g);
    s('text', { x: 0, y: ht + 13, text: '0 s' }, g);
    s('text', { x: w, y: ht + 13, 'text-anchor': 'end', text: `${((T - 1) * data.dt).toFixed(1)} s` }, g);
    s('path', { d: line(data.signal.values.map((v, i) => [x(i), y(v)])), fill: 'none', stroke: 'var(--c1)', 'stroke-width': 1.6 }, g);
    s('rect', { x: x(Math.max(0, t - Wn + 1)), width: Math.max(1, x(t) - x(Math.max(0, t - Wn + 1))), y: 0, height: ht, fill: 'var(--c1)', opacity: 0.1 }, g);
    s('line', { x1: x(t), x2: x(t), y1: 0, y2: ht, stroke: 'var(--ink)', 'stroke-width': 1 }, g);
  }

  const pl = player(mount, { n: T, fps: 1 / data.dt, onFrame: drawHeat, label: (i) => `t = ${(i * data.dt).toFixed(2)} s` });
  new ResizeObserver(() => drawHeat(cur)).observe(wrap);
  pl.set(Wn - 1);
}
