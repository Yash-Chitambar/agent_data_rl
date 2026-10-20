/**
 * Linear probe on the policy's 32-d history latent (paper Fig. 10): (a) recall per contact mode for
 * three inputs; (b) latent swap: action change when one mode's latent replaces another's.
 * Data: probe.json.
 */
import { s, h, svg, responsive, linear, band, axes, tween, onVisible, ease, tooltip, tipRow, legend, cvar, pct } from './kit.js';

const INPUTS = [['latent', 'History latent z', 0], ['current_obs', 'Current observation', 3], ['random_encoder', 'Random encoder', 4]];
const MODE = { far: 'Far search', near: 'Near search', partial: 'Partly inside', moving: 'Inside, moving', stuck: 'Inside, stuck' };

export default async function (mount, ctx) {
  const data = await ctx.data('probe.json');
  const row = h('div', { class: 'kit-row' }, mount);
  legend(mount, INPUTS.map(([k, l, c]) => ({ label: `${l} (mean recall ${pct(data.bal[k])})`, color: cvar(c), kind: 'box' })));
  let animate = false;

  function draw(W) {
    row.innerHTML = '';
    const cols = W < 640 ? 1 : 2;
    const pw1 = cols === 1 ? W : Math.floor((W - 24) * 0.56), pw2 = cols === 1 ? W : W - 24 - pw1;
    // (a) recall bars
    {
      const H = 280, m = { t: 26, r: 8, b: 44, l: 58 }, w = pw1 - m.l - m.r, ht = H - m.t - m.b;
      const el = svg(h('div', {}, row), pw1, H);
      const g = s('g', { transform: `translate(${m.l},${m.t})` }, el);
      const labs = data.modes.map((md) => MODE[md]);
      const gx = band(labs, [0, w], { inner: 0.25, outer: 0.06 });
      const bx = band(INPUTS.map((i) => i[0]), [0, gx.bw], { inner: 0.08, outer: 0 });
      const y = linear([0, 1], [ht, 0]);
      axes(g, { x: gx, y, w, h: ht, yFormat: (v) => pct(v), yLabel: 'recall', title: '(a) Probe recall per contact mode' });
      if (w < 380) g.querySelectorAll('.tick text').forEach((tx) => { if (labs.includes(tx.textContent)) tx.textContent = tx.textContent.replace('Inside, ', '').replace(' search', '').replace('Partly inside', 'partial'); });
      data.modes.forEach((md, mi) => {
        INPUTS.forEach(([k, l, c]) => {
          const v = data.recall[k][mi];
          const x0 = gx(MODE[md]) + bx(k);
          const bar = s('rect', { x: x0, width: bx.bw, y: y(v), height: ht - y(v), rx: 1.5, fill: cvar(c) }, g);
          if (animate) tween(700, (e) => { const yy = y(v * e); bar.setAttribute('y', yy); bar.setAttribute('height', ht - yy); }, ease.out);
          s('rect', {
            x: x0 - 1, width: bx.bw + 2, y: 0, height: ht, fill: 'transparent',
            onpointermove: (ev) => tooltip.show(`<b>${MODE[md]}</b> · ${data.counts[md].toLocaleString()} states` + tipRow(cvar(c), l, pct(v, 1)), ev),
            onpointerleave: () => tooltip.hide(),
          }, g);
        });
      });
      s('line', { x1: 0, x2: w, y1: y(1 / 7), y2: y(1 / 7), stroke: 'var(--muted)', 'stroke-dasharray': '2 3' }, g);
      s('text', { x: w, y: y(1 / 7) - 4, 'text-anchor': 'end', text: 'chance (1/7)' }, g);
    }
    // (b) swap matrix
    {
      const n = data.modes.length;
      const lab = 92, size = Math.min(pw2 - lab - 12, 300), cell = size / n;
      const H = 26 + size + 64;
      const el = svg(h('div', {}, row), pw2, H);
      const g = s('g', { transform: `translate(${lab},26)` }, el);
      s('text', { class: 'title', x: -lab, y: -12, text: '(b) Latent swap: action change ratio' }, g);
      const vmax = Math.max(...data.swap.flat());
      data.swap.forEach((rw, i) => rw.forEach((v, j) => {
        const tcol = i === j ? 'var(--surface)' : `color-mix(in srgb, var(--c2) ${Math.round(100 * Math.min(1, (v - 0.8) / (vmax - 0.8)))}%, var(--surface))`;
        const r = s('rect', { x: j * cell, y: i * cell, width: cell - 2, height: cell - 2, rx: 3, fill: tcol, stroke: 'var(--rule)' }, g);
        const tx = s('text', { x: j * cell + cell / 2 - 1, y: i * cell + cell / 2 + 4, 'text-anchor': 'middle', text: i === j ? '–' : v.toFixed(2), style: { fill: 'var(--ink)', fontVariantNumeric: 'tabular-nums' } }, g);
        if (animate) { r.style.opacity = 0; tx.style.opacity = 0; setTimeout(() => tween(300, (e) => { r.style.opacity = e; tx.style.opacity = e; }), (i + j) * 60); }
        r.addEventListener('pointermove', (ev) => tooltip.show(i === j ? `<b>${MODE[data.modes[i]]}</b><div>Baseline: the latent swaps with another latent of the same mode.</div>` :
          `<b>${MODE[data.modes[i]]}</b> state, latent from <b>${MODE[data.modes[j]]}</b><div>The action changes <span class="num">${v.toFixed(2)}×</span> as much as a same-mode swap.</div>`, ev));
        r.addEventListener('pointerleave', () => tooltip.hide());
      }));
      data.modes.forEach((md, i) => {
        s('text', { x: -8, y: i * cell + cell / 2 + 4, 'text-anchor': 'end', text: MODE[md] }, g);
        s('text', { x: i * cell + cell / 2, y: size + 12, 'text-anchor': 'end', transform: `rotate(-35 ${i * cell + cell / 2} ${size + 12})`, text: MODE[md] }, g);
      });
    }
  }
  responsive(row, draw);
  onVisible(mount, () => { animate = true; draw(row.clientWidth); animate = false; });
}
