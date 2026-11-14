"""Release probe of contact_sens.py per cell: of the worlds fully in at rest (jaws closed), the share still fully in 0.5 s after
the jaws open, and the depth change at the release. Reads runs/contact_sens/*.npz -> results/real_align/contact_sens/release.{md,json}.

    python scripts/eval/contact_sens_release.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEAT, FULL = 5.8, 0.25
rows = []
for f in sorted((ROOT / 'runs/contact_sens').glob('*.npz')):
    z = np.load(f, allow_pickle=True)
    k_ = 3 if f.stem.startswith('funnel_') else 4          # <tag>_<place>_<pad>_<plug>_<v1-v2-..> or funnel_<tag>_<place>_<g1-g2-..>
    parts = f.stem.split('_', k_)
    names = parts[-1].split('-')
    n = len(z['d_rest']) // len(names)
    for k, v in enumerate(names):
        s = slice(k * n, (k + 1) * n)
        rest, rel = z['d_rest'][s], z['d_rel'][s]
        full = rest >= SEAT - FULL
        both = full & np.isfinite(rel)
        stay = rel[both] >= SEAT - FULL
        drop = rest[both] - rel[both]
        rows.append(dict(file=f.stem, cell='_'.join(parts[:-1]), group=v, full_at_rest=int(both.sum()), stay_full=int(stay.sum()),
                         stay_frac=float(stay.mean()) if both.any() else None,
                         drop_mm_p50_p95_p99=np.percentile(drop, [50, 95, 99]).round(3).tolist() if both.any() else None))
out = ROOT / 'results/real_align/contact_sens'
(out / 'release.json').write_text(json.dumps(rows, indent=1))
lines = ['| Cell | Group | Fully in at rest | Still fully in 0.5 s after the jaws open | Depth drop p50 / p95 / p99 mm |', '|---|---|---|---|---|']
for r in rows:
    d = r['drop_mm_p50_p95_p99']
    lines.append(f"| {r['cell']} | {r['group']} | {r['full_at_rest']} | {r['stay_full']} ({100 * r['stay_frac']:.1f} %) | {d[0]:.3f} / {d[1]:.3f} / {d[2]:.3f} |")
(out / 'release.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
