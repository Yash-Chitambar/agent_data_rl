"""Table of the K7 evaluation set (docs/real/K7_20261114.md) from results/real_align/k7/<tag>_<group>.json, plus the acceptance check
for a rig candidate: >= 95 % in every cell of the step-2 placement box, the real-spot cell not below K4, no K4 acceptance cell more
than 2 points below K4, 10 N mating force (real_ho) >= 90 %.

    python scripts/eval/k7_table.py --tags k4,k6d,k4p,k7s0,k7ws1 --out results/real_align/k7/table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

D = Path(__file__).resolve().parents[2] / 'results/real_align/k7'
K2 = ['real_ho', 'ring3', 'ring4', 'ring5', 'roll3', 'roll4', 'pitch4', 'gap8', 'mix']


def load(tag, g):
    f = D / f'{tag}_{g}.json'
    return json.loads(f.read_text()) if f.exists() else []


def cell(r):
    return None if r is None else dict(k=r['k'], n=r['n'], rate=100 * r['rate'], lo=100 * r['wilson'][0], hi=100 * r['wilson'][1],
                                       t50=r.get('t_ok_p50'), t99=r.get('t_ok_p99'))


def collect(tag):
    rows = {}
    for g in ('k2', 'mate6', 'mate10', 'sig2', 'sig3'):
        for r in load(tag, g):
            rows[f'{g}:{r["cell"]}'] = cell(r)
    for r in load(tag, 'place'):
        c = cell(r); c['in_box'] = r['in_box']; c['pc'] = r['pc']
        rows[f'place:{r["pc_tag"]}'] = c
    for r in load(tag, 'replay'):
        rows[f'replay:{r["run"]}'] = cell(r)
    for m in ('0', '6', '10'):
        z = load(tag, f'depth{m}')
        if z:
            rows[f'depth{m}'] = z[0]
    return rows


def accept(R, tag, ref='k4'):
    a, b = R[tag], R[ref]
    out = {}
    box = [k for k, v in a.items() if k.startswith('place:') and v.get('in_box')]
    out['box_min'] = min(((a[k]['rate'], k) for k in box), default=None)
    out['box_ok'] = bool(box) and all(a[k]['rate'] >= 95 for k in box)
    out['real_ok'] = 'place:real' in a and 'place:real' in b and a['place:real']['rate'] >= b['place:real']['rate']
    d = {c: a[f'k2:{c}']['rate'] - b[f'k2:{c}']['rate'] for c in K2 if f'k2:{c}' in a and f'k2:{c}' in b}
    out['k2_worst'] = min(((v, c) for c, v in d.items()), default=None)
    out['k2_ok'] = len(d) == len(K2) and all(v >= -2 for v in d.values())
    out['mate10_ok'] = 'mate10:real_ho' in a and a['mate10:real_ho']['rate'] >= 90
    out['pass'] = all(out[k] for k in ('box_ok', 'real_ok', 'k2_ok', 'mate10_ok'))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='k4,k6d,k4p,k7s0,k7ws1')
    ap.add_argument('--out', default=str(D / 'table'))
    A = ap.parse_args()
    tags = A.tags.split(',')
    R = {t: collect(t) for t in tags}
    keys = []
    for t in tags:
        keys += [k for k in R[t] if k not in keys and not k.startswith('depth')]
    lines = ['| Cell | ' + ' | '.join(tags) + ' |', '|---|' + '---|' * len(tags)]
    for k in keys:
        vals = []
        for t in tags:
            c = R[t].get(k)
            vals.append('' if c is None else f"{c['rate']:.1f} [{c['lo']:.1f}, {c['hi']:.1f}] n={c['n']}")
        lines.append(f'| {k} | ' + ' | '.join(vals) + ' |')
    for m in ('0', '6', '10'):
        vals = []
        for t in tags:
            z = R[t].get(f'depth{m}')
            vals.append('' if z is None else f"seat {100 * z['sim_seat_rate']:.1f} %, t {z['t_seat_p50_p99']}, det depth {z['det_depth_mm_p5_p50_p95']}"
                        + (f", rest {z['rest_depth_mm_p5_p50_p95']} (seated {100 * z['rest_seated_frac']:.1f} %)" if z.get('rest_seated_frac') is not None else '') + f" n={z['n']}")
        lines.append(f'| depth, mate {m} N | ' + ' | '.join(vals) + ' |')
    acc = {t: accept(R, t) for t in tags if t != 'k4' and 'k4' in R}
    Path(A.out + '.md').write_text('\n'.join(lines) + '\n')
    Path(A.out + '.json').write_text(json.dumps(dict(cells=R, acceptance=acc), indent=1))
    print('\n'.join(lines))
    print(json.dumps(acc, indent=1))
