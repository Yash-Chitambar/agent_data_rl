/**
 * demos.js: success and failure examples, one grid video per experiment.
 * Data: static/data/demos.json = {groups: [{name, items: [{id, label, title, video, poster,
 * n, rate, failures: {type: count}, text}]}]}. Buttons pick the group and the experiment.
 */
import { h } from './kit.js';

const pct = (x) => (100 * x >= 99.95 || 100 * x < 0.05 ? (100 * x).toFixed(0) : (100 * x).toFixed(1)) + '%';
const words = (s) => s.replace(/_/g, ' ');

export default async function (mount, ctx) {
  const data = await ctx.data('demos.json');
  const controls = h('div', { class: 'kit-controls' }, mount);
  const groupSeg = h('div', { class: 'kit-seg', role: 'group', 'aria-label': 'Task',
    style: { flexWrap: 'wrap', borderRadius: '16px' } }, controls);
  const itemRow = h('div', { class: 'kit-controls' }, mount);
  const video = h('video', {
    muted: true, loop: true, playsinline: true, autoplay: true, controls: true,
    style: { width: '100%', borderRadius: '8px', display: 'block', background: '#000' },
  }, mount);
  video.muted = true;
  const info = h('div', { style: { marginTop: '10px', fontSize: '0.9rem', lineHeight: '1.5' } }, mount);

  function show(item) {
    video.poster = ctx.asset(item.poster);
    video.src = ctx.asset(item.video);
    video.play().catch(() => {});
    const fails = Object.entries(item.failures || {}).sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${words(k)} ${v}`).join(', ');
    info.innerHTML = '';
    h('p', { style: { margin: '0 0 4px' }, html: `<b>${item.title}.</b> Success ${pct(item.rate)} of ${item.n.toLocaleString()} episodes` +
      (fails ? `. Failures: ${fails}.` : '.') }, info);
    if (item.text) h('p', { style: { margin: 0, color: 'var(--ink2)' }, text: item.text }, info);
  }

  function pickGroup(gi) {
    [...groupSeg.children].forEach((b, i) => b.setAttribute('aria-pressed', String(i === gi)));
    itemRow.innerHTML = '';
    const seg = h('div', { class: 'kit-seg', role: 'group', 'aria-label': 'Experiment',
      style: { flexWrap: 'wrap', borderRadius: '16px' } }, itemRow);
    const items = data.groups[gi].items;
    items.forEach((it, i) => h('button', {
      type: 'button', text: it.label, 'aria-pressed': String(i === 0),
      onclick: () => {
        [...seg.children].forEach((b, j) => b.setAttribute('aria-pressed', String(j === i)));
        show(it);
      },
    }, seg));
    show(items[0]);
  }

  data.groups.forEach((g, gi) => h('button', { type: 'button', text: g.name, onclick: () => pickGroup(gi) }, groupSeg));
  pickGroup(0);
}
