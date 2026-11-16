"""Tables for the wide-motion dry-run grid (docs/real/WIDE_MOTION_20261116.md): scripts/eval/wide_grid_summary.one per run
folder (first failing node, seat at the end of the insertion = before the release, seat after the release) plus the
wide_motion events (hold choice, selection time, transport time) and a per-region table.

    PYTHONPATH=. python scripts/eval/wide_motion_grid.py --grid results/placement/wide_motion/grid --out results/placement/wide_motion/grid_summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'scripts/eval/{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def region(d):
    x, y, t = d
    if t <= -12:
        return 'yaw -15'
    if t <= -8:
        return 'yaw -8..-10'
    if t < 0:
        return 'yaw -5'
    if x < -12:
        return 'toward the bases (dx -25)'
    if x > 12 and y < -12:
        return '+x / -y corner'
    return 'rest of the box (yaw 0..+15)'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid', default=str(ROOT / 'results/placement/wide_motion/grid'))
    ap.add_argument('--out', default=str(ROOT / 'results/placement/wide_motion/grid_summary.json'))
    a = ap.parse_args()
    S, P = mod('wide_grid_summary'), mod('placement_grid')
    rows = []
    for d in sorted(Path(a.grid).iterdir()):
        if not (d / 'point.json').exists() or not (d / 'result.json').exists():
            continue
        r = S.one(d, P)
        evs = []
        for f in d.rglob('events.jsonl'):
            evs = [json.loads(x) for x in open(f)]
            break
        w = next((e for e in evs if e['kind'] == 'wide_hold_select'), {})
        t = next((e for e in evs if e['kind'] == 'wide_transport'), {})
        r.update(wide_ok=w.get('ok'), wide_choice=w.get('choice'), wide_select_s=w.get('seconds'), wide_fails=(w.get('best') or {}).get('fails'),
                 transport_joint_s=t.get('joint_s'), region=region(r['d']))
        if w and not w.get('ok'):
            r['node'] = 'hold: no candidate (stop before the hold)'
        r.pop('clearance', None)
        rows.append(r)
    Path(a.out).write_text(json.dumps(rows, indent=1, default=str))
    f = lambda v, n=1: '-' if v is None else f'{v:.{n}f}'  # noqa: E731
    print('| run | PC d (x, y mm; yaw deg) | reader d (x, y mm; yaw) | hold choice (along m, yaw deg) | fit err mm | node | align s | seated before release | seated after release |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in sorted(rows, key=lambda r: (r['policy'], r['d'][2], r['d'][0], r['d'][1])):
        ch = r.get('wide_choice')
        print(f"| {r['tag']} | {r['d']} | {r['rd']} | {'-' if ch is None else (ch[0], ch[1])} | {f(r.get('used_err_mm'), 2)} | {r['node']} | "
              f"{f(r.get('align_s'))} | {r.get('true_seated_end')} | {r.get('true_seated')} |")
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        agg[(r['policy'], r['region'])][r['node']] += 1
    print()
    for k in sorted(agg):
        v = agg[k]
        before = sum(1 for r in rows if (r['policy'], r['region']) == k and (r.get('true_seated_end') or (r.get('true_depth_end_mm') or 0) >= S.DEPTH_IN_MM))
        after = sum(1 for r in rows if (r['policy'], r['region']) == k and r.get('true_seated') is True)
        print(f"{k[0]} | {k[1]} | n = {sum(v.values())} | seated before release {before} | after release {after} | nodes {dict(v)}")


if __name__ == '__main__':
    main()
