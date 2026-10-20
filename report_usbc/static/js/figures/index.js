/**
 * Figure loader. Every <figure class="viz" data-fig="NAME"> is mounted by the module
 * static/js/figures/NAME.js shortly before it scrolls into view. A figure module is:
 *
 *     export default async function (mount, ctx) { ... }
 *
 * `mount` is the figure's .viz-mount element. `ctx` gives figures their data without
 * hard-coding paths, so the page works from any URL prefix (GitHub Pages subpaths).
 * A figure that throws shows an inline error instead of breaking the page.
 */
const STATIC = new URL('../../', import.meta.url); // -> static/

export const ctx = {
  /** Absolute URL of a file under static/. */
  asset: (path) => new URL(path, STATIC).href,
  /** Parsed JSON from static/data/<name>. */
  data: async (name) => {
    const r = await fetch(new URL(`data/${name}`, STATIC));
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return r.json();
  },
  /** Raw bytes from static/data/<name>. */
  bytes: async (name) => {
    const r = await fetch(new URL(`data/${name}`, STATIC));
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  },
};

async function mountFigure(fig) {
  const name = fig.dataset.fig;
  const mount = fig.querySelector('.viz-mount');
  if (!mount || !/^[a-z0-9_]+$/.test(name)) return;
  mount.classList.add('loading');
  try {
    const mod = await import(`./${name}.js`);
    await mod.default(mount, ctx);
  } catch (err) {
    console.error(`figure "${name}":`, err);
    mount.innerHTML = `<p class="viz-error">Could not draw this figure: ${err.message}</p>`;
  } finally {
    mount.classList.remove('loading');
    fig.dataset.mounted = '1';
  }
}

const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    io.unobserve(e.target);
    mountFigure(e.target);
  }
}, { rootMargin: '400px 0px' });

document.querySelectorAll('figure.viz[data-fig]').forEach((f) => io.observe(f));
