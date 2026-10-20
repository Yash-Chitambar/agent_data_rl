/**
 * kit.js: a small, dependency-free toolkit for paper figures on the web.
 *
 * Figures are remade from the paper's own data in the paper's own style: same palette
 * and encodings, light grid, no top/right spines, muted ticks, legend at the bottom.
 * On the web they also animate in when scrolled to, answer hover with exact values,
 * and expose the paper's choices (splits, scales, seeds) as toggles.
 *
 * SVG is drawn at the mount's real pixel width (see `responsive`), so text stays at
 * true size on every screen. Colors come from CSS classes and variables in style.css,
 * so dark mode is free; `C` mirrors them for canvas and three.js, which can't read CSS.
 */

// ------------------------------------------------------------------ palette
/** Series colors. Keep in sync with --c1…--c5 in style.css. */
export const C = {
  blue: '#2a78d6', orange: '#eb6834', aqua: '#1baf7a', yellow: '#eda100', magenta: '#e87ba4', violet: '#7e5bd6',
};
export const SERIES = [C.blue, C.orange, C.aqua, C.yellow, C.magenta, C.violet];
/** CSS variable for series i, for SVG fills/strokes that should follow the theme. */
export const cvar = (i) => `var(--c${(i % 6) + 1})`;
/** Read a CSS custom property (e.g. '--ink') as a color string, for canvas/three.js. */
export const cssColor = (name, el = document.documentElement) =>
  getComputedStyle(el).getPropertyValue(name).trim() || '#888';

export const reducedMotion =
  typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

// ------------------------------------------------------------------ DOM / SVG
const NS = 'http://www.w3.org/2000/svg';

function apply(node, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'class') node.setAttribute('class', v);
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  return node;
}
/** SVG element: s('rect', {x, y, width, height, fill}, parent). */
export function s(tag, attrs, parent) {
  const n = apply(document.createElementNS(NS, tag), attrs);
  if (parent) parent.appendChild(n);
  return n;
}
/** HTML element: h('div', {class: 'x', text: 'hi'}, parent). */
export function h(tag, attrs, parent) {
  const n = apply(document.createElement(tag), attrs);
  if (parent) parent.appendChild(n);
  return n;
}
/** A fresh <svg> of the given pixel size inside parent. */
export function svg(parent, width, height, attrs = {}) {
  return s('svg', { class: 'kit-svg', width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', ...attrs }, parent);
}

/**
 * Call draw(width) now and again whenever the mount's width changes (debounced).
 * draw should clear and rebuild its own SVG. Returns a function that stops watching.
 */
export function responsive(mount, draw) {
  let last = -1, timer = 0;
  const run = () => {
    const w = Math.round(mount.clientWidth);
    if (w > 0 && w !== last) { last = w; draw(w); }
  };
  const ro = new ResizeObserver(() => { clearTimeout(timer); timer = setTimeout(run, 60); });
  ro.observe(mount);
  run();
  return () => ro.disconnect();
}

// ------------------------------------------------------------------ scales
/** Linear scale. sc(v) maps domain->range; sc.invert(px); sc.ticks(n). */
export function linear([d0, d1], [r0, r1]) {
  const k = (r1 - r0) / ((d1 - d0) || 1);
  const sc = (v) => r0 + (v - d0) * k;
  sc.invert = (p) => d0 + (p - r0) / k;
  sc.domain = [d0, d1];
  sc.range = [r0, r1];
  sc.ticks = (n = 5) => niceTicks(d0, d1, n);
  return sc;
}
/** Log10 scale (domain must be > 0). */
export function log([d0, d1], [r0, r1]) {
  const l0 = Math.log10(d0), l1 = Math.log10(d1);
  const k = (r1 - r0) / ((l1 - l0) || 1);
  const sc = (v) => r0 + (Math.log10(Math.max(v, 1e-300)) - l0) * k;
  sc.invert = (p) => 10 ** (l0 + (p - r0) / k);
  sc.domain = [d0, d1];
  sc.range = [r0, r1];
  sc.ticks = () => {
    const out = [];
    for (let e = Math.floor(l0); e <= Math.ceil(l1); e++) {
      const v = 10 ** e;
      if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(v);
    }
    return out;
  };
  return sc;
}
/** Band scale for categories: sc(key) -> left px; sc.bw = band width. */
export function band(keys, [r0, r1], { inner = 0.2, outer = 0.1 } = {}) {
  const n = keys.length;
  const step = (r1 - r0) / Math.max(1, n - inner + 2 * outer);
  const bw = step * (1 - inner);
  const sc = (k) => r0 + step * (outer + keys.indexOf(k));
  sc.bw = bw;
  sc.step = step;
  sc.keys = keys;
  return sc;
}
/** ~n round tick values covering [lo, hi]: steps of 1, 2, 2.5 or 5 × 10^k. */
export function niceTicks(lo, hi, n = 5) {
  if (lo === hi) return [lo];
  if (lo > hi) [lo, hi] = [hi, lo];
  // A range that doesn't start on a round number can leave only 1-2 ticks: ask for more.
  for (let k = n; k <= n * 4; k *= 2) {
    const t = ticksOnce(lo, hi, k);
    if (t.length >= 3 || k * 2 > n * 4) return t;
  }
  return ticksOnce(lo, hi, n);
}
function ticksOnce(lo, hi, n) {
  const raw = (hi - lo) / Math.max(1, n);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => st >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) {
    out.push(+v.toPrecision(12));
  }
  return out;
}
/** Extent of values (ignores NaN/null), optionally padded by a fraction. */
export function extent(values, pad = 0) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) if (v != null && !Number.isNaN(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const p = (hi - lo) * pad;
  return [lo - p, hi + p];
}

// ------------------------------------------------------------------ axes
/**
 * Paper-style axes into group g: horizontal grid, left/bottom domain only, muted ticks.
 *   x, y        scales (x may be a band scale)
 *   w, h        plot size in px (the group is already translated to the plot origin)
 *   xTicks/yTicks, xFormat/yFormat, xLabel/yLabel, title, grid ('y' | 'x' | 'xy' | '')
 */
export function axes(g, { x, y, w, h: ht, xTicks, yTicks, xFormat = fmt, yFormat = fmt, xLabel, yLabel, title, grid = 'y' }) {
  const yt = yTicks || y.ticks(5);
  const gy = s('g', { class: 'grid' }, g);
  if (grid.includes('y')) for (const v of yt) s('line', { x1: 0, x2: w, y1: y(v), y2: y(v) }, gy);
  const isBand = typeof x.bw === 'number';
  const xt = isBand ? x.keys : (xTicks || x.ticks(Math.max(2, Math.floor(w / 80))));
  if (grid.includes('x') && !isBand) for (const v of xt) s('line', { x1: x(v), x2: x(v), y1: 0, y2: ht }, gy);

  s('path', { class: 'domain', d: `M0,0V${ht}H${w}` }, g);

  const tx = s('g', { class: 'tick' }, g);
  for (const v of xt) {
    const px = isBand ? x(v) + x.bw / 2 : x(v);
    s('text', { x: px, y: ht + 16, 'text-anchor': 'middle', text: isBand ? v : xFormat(v) }, tx);
  }
  const ty = s('g', { class: 'tick' }, g);
  for (const v of yt) s('text', { x: -8, y: y(v) + 3.5, 'text-anchor': 'end', text: yFormat(v) }, ty);

  if (xLabel) s('text', { class: 'axis-label', x: w / 2, y: ht + 36, 'text-anchor': 'middle', text: xLabel }, g);
  if (yLabel) s('text', { class: 'axis-label', transform: `translate(${-44},${ht / 2}) rotate(-90)`, 'text-anchor': 'middle', text: yLabel }, g);
  if (title) s('text', { class: 'title', x: 0, y: -12, text: title }, g);
}
/** Diagonal hatch pattern (e.g. to mark an ablation); returns `url(#id)`. */
export function hatch(svgEl, color, id = `h${Math.random().toString(36).slice(2, 8)}`) {
  const defs = svgEl.querySelector('defs') || s('defs', {}, svgEl);
  const p = s('pattern', { id, width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
  s('rect', { width: 6, height: 6, fill: color, opacity: 0.25 }, p);
  s('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: color, 'stroke-width': 2.5 }, p);
  return `url(#${id})`;
}

// ------------------------------------------------------------------ motion
export const ease = {
  linear: (t) => t,
  out: (t) => 1 - (1 - t) ** 3,
  inOut: (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2),
  back: (t) => 1 + 2.2 * (t - 1) ** 3 + 1.2 * (t - 1) ** 2,
};
/** Animate fn(e) for e in [0,1] over ms. Resolves when done. Instant under reduced motion. */
export function tween(ms, fn, easing = ease.out) {
  return new Promise((resolve) => {
    if (reducedMotion || ms <= 0) { fn(1); resolve(); return; }
    const t0 = performance.now();
    const tick = (now) => {
      // rAF timestamps are frame-start times and can precede t0: clamp to [0, 1].
      const t = Math.max(0, Math.min(1, (now - t0) / ms));
      fn(easing(t));
      if (t < 1) requestAnimationFrame(tick); else resolve();
    };
    requestAnimationFrame(tick);
  });
}
/** Draw a path on like a pen (stroke-dashoffset); returns the tween promise. */
export function drawOn(path, ms = 900, delay = 0) {
  const len = path.getTotalLength();
  path.style.strokeDasharray = `${len} ${len}`;
  path.style.strokeDashoffset = len;
  return new Promise((r) => setTimeout(r, delay)).then(() =>
    tween(ms, (e) => { path.style.strokeDashoffset = len * (1 - e); }, ease.inOut)
  ).then(() => { path.style.strokeDasharray = ''; path.style.strokeDashoffset = ''; });
}
/** Run fn once, the first time el is at least `threshold` visible. */
export function onVisible(el, fn, threshold = 0.35) {
  const io = new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
  }, { threshold });
  io.observe(el);
}
/** Call cb(true/false) as el enters/leaves view; use it to pause sims and render loops. */
export function watchVisible(el, cb, threshold = 0.05) {
  const io = new IntersectionObserver((es) => cb(es[es.length - 1].isIntersecting), { threshold });
  io.observe(el);
  return () => io.disconnect();
}

// ------------------------------------------------------------------ paths
/** SVG path through points [[x,y],...], skipping null y (gaps). */
export function line(points) {
  let d = '', pen = false;
  for (const [x, y] of points) {
    if (y == null || Number.isNaN(y)) { pen = false; continue; }
    d += `${pen ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  }
  return d;
}
/** Closed band between lower and upper curves: xs, lo[], hi[] already in px. */
export function area(xs, lo, hi) {
  let d = '';
  xs.forEach((x, i) => { d += `${i ? 'L' : 'M'}${x.toFixed(1)},${hi[i].toFixed(1)}`; });
  for (let i = xs.length - 1; i >= 0; i--) d += `L${xs[i].toFixed(1)},${lo[i].toFixed(1)}`;
  return d + 'Z';
}

// ------------------------------------------------------------------ interaction
let tipEl = null;
/** Shared tooltip: tooltip.show(html, event), tooltip.hide(). Content is trusted HTML you build. */
export const tooltip = {
  show(html, ev) {
    if (!tipEl) tipEl = h('div', { class: 'kit-tooltip', role: 'status' }, document.body);
    tipEl.innerHTML = html;
    tipEl.classList.add('on');
    const pad = 14, r = tipEl.getBoundingClientRect();
    let x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
    tipEl.style.left = `${Math.max(8, x)}px`;
    tipEl.style.top = `${Math.max(8, y)}px`;
  },
  hide() { tipEl?.classList.remove('on'); },
};
/** Tooltip row: colored swatch, label, right-aligned number. */
export const tipRow = (color, label, value) =>
  `<div><span class="sw" style="background:${color}"></span>${label} <span class="num">${value}</span></div>`;

/** Pointer position in an SVG element's user coordinates. */
export function pointer(ev, svgEl) {
  const r = svgEl.getBoundingClientRect();
  const vb = svgEl.viewBox.baseVal;
  return [((ev.clientX - r.left) / r.width) * vb.width, ((ev.clientY - r.top) / r.height) * vb.height];
}

/**
 * Control bar. Returns builders that append to it:
 *   seg(label, options, value, onChange)  segmented toggle; options = [[value, text], ...]
 *   button(text, onClick)
 *   slider(label, {min, max, step, value}, onInput)
 */
export function controls(parent) {
  const bar = h('div', { class: 'kit-controls' }, parent);
  return {
    el: bar,
    seg(label, options, value, onChange) {
      if (label) h('span', { text: label }, bar);
      const wrap = h('div', { class: 'kit-seg', role: 'group', 'aria-label': label || 'options' }, bar);
      const btns = options.map(([v, text]) => h('button', {
        type: 'button', text, 'aria-pressed': String(v === value),
        onclick: () => { btns.forEach((b) => b.setAttribute('aria-pressed', String(b === btn(v)))); onChange(v); },
      }, wrap));
      const btn = (v) => btns[options.findIndex(([o]) => o === v)];
      return wrap;
    },
    button(text, onClick) { return h('button', { type: 'button', class: 'kit-btn', text, onclick: onClick }, bar); },
    slider(label, { min = 0, max = 1, step = 0.01, value = 0 }, onInput) {
      const lab = h('label', { style: { display: 'inline-flex', gap: '8px', alignItems: 'center' } }, bar);
      h('span', { text: label }, lab);
      const inp = h('input', { type: 'range', min, max, step, value, oninput: (e) => onInput(+e.target.value) }, lab);
      return inp;
    },
  };
}

/** Legend at the bottom: items = [{label, color, kind: 'box' | 'line' | 'hatch'}]. */
export function legend(parent, items) {
  const el = h('div', { class: 'kit-legend' }, parent);
  for (const it of items) {
    const sp = h('span', {}, el);
    const sw = h('i', { class: it.kind === 'line' ? 'line' : '' }, sp);
    sw.style.background = it.kind === 'hatch'
      ? `repeating-linear-gradient(45deg, ${it.color} 0 2px, transparent 2px 5px)` : it.color;
    if (it.kind === 'hatch') sw.style.border = `1px solid ${it.color}`;
    h('span', { text: it.label }, sp);
  }
  return el;
}

/**
 * Timeline player: play/pause button, scrubber and frame readout.
 *   n          number of frames
 *   fps        playback rate
 *   onFrame(i) called for every frame change
 *   label(i)   readout text (default "i / n")
 * Returns { set(i), play(), pause(), el }.
 */
export function player(parent, { n, fps = 30, onFrame, label = (i) => `${i + 1} / ${n}`, loop = true }) {
  const el = h('div', { class: 'kit-player' }, parent);
  const btn = h('button', { type: 'button', class: 'kit-btn', text: 'Play' }, el);
  const rng = h('input', { type: 'range', min: 0, max: n - 1, step: 1, value: 0, 'aria-label': 'timeline' }, el);
  const out = h('span', { text: label(0) }, el);
  let i = 0, playing = false, raf = 0, last = 0;
  const set = (k) => {
    i = Math.max(0, Math.min(n - 1, k | 0));
    rng.value = i; out.textContent = label(i); onFrame(i);
  };
  const step = (now) => {
    if (!playing) return;
    if (now - last >= 1000 / fps) {
      last = now;
      if (i >= n - 1) { if (loop) set(0); else { pause(); return; } } else set(i + 1);
    }
    raf = requestAnimationFrame(step);
  };
  const play = () => { if (playing) return; playing = true; btn.textContent = 'Pause'; raf = requestAnimationFrame(step); };
  const pause = () => { playing = false; btn.textContent = 'Play'; cancelAnimationFrame(raf); };
  btn.onclick = () => (playing ? pause() : play());
  rng.oninput = () => { pause(); set(+rng.value); };
  set(0);
  return { set, play, pause, el, get index() { return i; } };
}

// ------------------------------------------------------------------ numbers
/** Compact general formatter: 12,000 -> 12k, 0.5 -> 0.5, 1e-4 -> 1e-4. */
export function fmt(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e9) return `${+(v / 1e9).toFixed(1)}B`;
  if (a >= 1e6) return `${+(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${+(v / 1e3).toFixed(1)}k`;
  if (a < 1e-3) return v.toExponential(0);
  return `${+v.toPrecision(3)}`;
}
/** 0.873 -> "87%" (digits = decimals). */
export const pct = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;
/** 95% Wilson score interval for k successes in n trials -> [lo, hi]. */
export function wilson(k, n, z = 1.96) {
  if (!n) return [0, 1];
  const p = k / n, z2 = z * z, den = 1 + z2 / n;
  const c = (p + z2 / (2 * n)) / den;
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / den;
  return [Math.max(0, c - half), Math.min(1, c + half)];
}
