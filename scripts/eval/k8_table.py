"""K8 tables (docs/real/K8_20261115.md): the K7 evaluation set with each cell against K7 and K4, the contact grid, the K8 rule.

K4 / K7 rows come from results/real_align/k7 (the K7 doc runs, same scripts and n); K8 candidates from results/real_align/k8.
Contact grid: contact_sens.py raw summaries (K4 / K7: results/real_align/contact_sens/raw; K8: results/real_align/k8/contact).

K8 rule (set before the K8 results):
  1. the K7 rule against K4 (k7_table.accept: box >= 95 %, real spot >= K4, K2 cells >= K4 - 2, mate 10 N >= 90 %);
  2. no cell of the K7 set (K2, mate, sig, place) more than 2 points below K7;
  3. contact grid (real spot, nominal pads): base >= 98 % and port mu 0.35-0.60 >= 90 % seat (contact_sens pass rule);
  4. replay of real hand-overs: no run more than 5 points below K7 (n = 256 per run).

    python scripts/eval/k8_table.py --tags k8a,k8b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import scripts.eval.k7_table as K7T

ROOT = Path(__file__).resolve().parents[2]
D7, D8 = ROOT / 'results/real_align/k7', ROOT / 'results/real_align/k8'
CS7, CS8 = ROOT / 'results/real_align/contact_sens/raw', D8 / 'contact'
GRID = [('4x1.2', 'nom', 'base'), ('4x1.2', 'nom', 'pmu_lo'), ('4x1.2', 'nom', 'pmu_hi'), ('4x1.2', 'nom', 'pmu_vhi'),
        ('4x1.2', 'nom', 'ptc_2ms'), ('4x1.2', 'nom', 'ptc_8ms'), ('4x1.2', 'nom', 'pdamp_05'), ('4x1.2', 'nom', 'psolimp_soft'),
        ('4x1.2', 'nom', 'padmu_08'), ('4x1.2', 'nom', 'padmu_20'), ('4x1.2', 'nom', 'padtc_6ms'), ('4x1.2', 'nom', 'grip_06'),
        ('4x1.2', 'nom', 'grip_14'), ('2x1.2', 'nom', 'base'), ('2x1.2', 'nom', 'pmu_hi'), ('6x0.8', 'nom', 'base'),
        ('6x0.8', 'nom', 'pmu_hi'), ('4x1.2', 'tight', 'base'), ('4x1.2', 'tight', 'pmu_hi'), ('4x1.2', 'loose', 'base'),
        ('4x1.2', 'loose', 'pmu_hi')]
REPLAY_WEAK = ['approach_insert_171123_ai4', 'approach_insert_173731_ai5', 'full_175742_full_A2']


def load(tag, g):
    for d in (D8, D7):
        f = d / f'{tag}_{g}.json'
        if f.exists():
            return json.loads(f.read_text())
    return []


def grid(tag, place='real'):
    out = {}
    for pad, plug, v in GRID:
        for d in (CS8, CS7):
            f = d / f'{tag}_{place}_{pad}_{plug}.json'
            if f.exists():
                for r in json.loads(f.read_text()):
                    if r['variant'] == v:
                        out[f'{pad} {plug} {v}'] = dict(seat=100 * r['rate'], full=100 * r['full_depth_rate'], n=r['n'],
                                                        lo=100 * r['wilson'][0], hi=100 * r['wilson'][1], t=r['t_seat_p50_p90_p99'])
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='k8a')
    ap.add_argument('--out', default=str(D8 / 'table'))
    A = ap.parse_args()
    K7T.load = load
    cand = A.tags.split(',')
    tags = ['k4', 'k7'] + cand
    R = {t: K7T.collect(t) for t in tags}
    G = {t: grid(t) for t in tags}
    keys = [k for k in R['k7'] if not k.startswith('depth')]
    lines = ['## K7 evaluation set (seat %, n per cell as stated)', '',
             '| Cell | K4 | K7 | ' + ' | '.join(f'{t} (vs K7, vs K4)' for t in cand) + ' |', '|---|---|---|' + '---|' * len(cand)]
    f = lambda c: '' if c is None else f"{c['rate']:.1f}"   # noqa: E731
    for k in keys:
        a4, a7 = R['k4'].get(k), R['k7'].get(k)
        vals = []
        for t in cand:
            c = R[t].get(k)
            vals.append('' if c is None else f"{c['rate']:.1f} ({c['rate'] - a7['rate']:+.1f}, {c['rate'] - a4['rate']:+.1f}) n={c['n']}"
                        if a7 and a4 else f(c))
        lines.append(f'| {k} | {f(a4)} | {f(a7)} | ' + ' | '.join(vals) + ' |')
    lines += ['', '## Contact grid, rig spot (seat % / full-depth %, n = 1024 per cell)', '',
              '| Cell | K4 | K7 | ' + ' | '.join(cand) + ' |', '|---|---|---|' + '---|' * len(cand)]
    for pad, plug, v in GRID:
        k = f'{pad} {plug} {v}'
        lines.append(f'| {k} | ' + ' | '.join('' if k not in G[t] else f"{G[t][k]['seat']:.1f} / {G[t][k]['full']:.1f}" for t in tags) + ' |')
    acc = {}
    for t in cand:
        a = K7T.accept(R, t)
        d7 = {k: R[t][k]['rate'] - R['k7'][k]['rate'] for k in keys if k in R[t] and not k.startswith('replay:')}
        a['vs_k7_worst'] = min(((v, k) for k, v in d7.items()), default=None)
        a['vs_k7_ok'] = len(d7) == len([k for k in keys if not k.startswith('replay:')]) and all(v >= -2 for v in d7.values())
        g = G[t]
        a['grid_base'] = g.get('4x1.2 nom base', {}).get('seat'); a['grid_pmu_hi'] = g.get('4x1.2 nom pmu_hi', {}).get('seat')
        a['grid_ok'] = a['grid_base'] is not None and a['grid_pmu_hi'] is not None and a['grid_base'] >= 98 and a['grid_pmu_hi'] >= 90
        dr = {k: R[t][k]['rate'] - R['k7'][k]['rate'] for k in keys if k.startswith('replay:') and k in R[t]}
        a['replay_worst_vs_k7'] = min(((v, k) for k, v in dr.items()), default=None)
        a['replay_ok'] = bool(dr) and all(v >= -5 for v in dr.values())
        a['k8_pass'] = all(a[x] for x in ('pass', 'vs_k7_ok', 'grid_ok', 'replay_ok'))
        acc[t] = a
    Path(A.out + '.md').write_text('\n'.join(lines) + '\n')
    Path(A.out + '.json').write_text(json.dumps(dict(cells=R, grid=G, acceptance=acc), indent=1, default=str))
    print('\n'.join(lines))
    print(json.dumps(acc, indent=1, default=str))


if __name__ == '__main__':
    main()
