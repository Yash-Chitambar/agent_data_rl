"""Tables of docs/real/ZED_EDGEFIT_20261115.md from results/placement/zed_edgefit/replay.json (zed_edgefit_replay.py).

    PYTHONPATH=. python scripts/perception/zed_edgefit_summary.py results/placement/zed_edgefit/replay.json
"""
import json
import sys
from collections import defaultdict

import numpy as np


def stats(v):
    v = np.asarray(v, float)
    return f'{np.median(v):+.2f} [{v.min():+.2f}, {v.max():+.2f}] sd {v.std():.2f}' if len(v) else '-'


def main(path):
    d = json.load(open(path))
    fr = d['frames']
    print('n frames', len(fr))
    by = defaultdict(list)
    for r in fr:
        grp = r['date'] + (' maps' if r['run'].startswith('capture_17') and r['date'] == '2026-11-13' else '')
        by[grp].append(r)
    print('\n| Group | n | net gate ok (v1) | v5 gate ok | fit gate ok | both (used) | fit edges all/noL | rms px max | shift |fit - v1| mm p50 / max | |fit - v5| mm max | fit from v1 start: max |diff| mm |')
    for g in sorted(by):
        rs = by[g]
        sh = [r['fit']['shift_mm'] for r in rs]
        ds = [np.hypot(*r['fit']['d_start_mm']) for r in rs]
        dv = [np.hypot(*(np.subtract(r['fit']['fit_xy'], r['fit_from_v1']['fit_xy']) * 1e3)) for r in rs]
        print(f"| {g} | {len(rs)} | {sum(r['gate']['ok'] for r in rs)} | {sum(r['start']['ok'] for r in rs)} | {sum(r['fit']['ok'] for r in rs)} | "
              f"{sum(r['ok'] for r in rs)} | {sum(r['fit']['edges_used'] == 'all' for r in rs)}/{sum(r['fit']['edges_used'] == 'noL' for r in rs)} | "
              f"{max(r['fit']['inlier_rms_px'] for r in rs):.2f} | {np.median(sh):.1f} / {max(sh):.1f} | {max(ds):.1f} | {max(dv):.2f} |")
    for lim in (8., 10., 12.):
        n = sum(r['gate']['ok'] and not [w for w in r['fit']['why'] if 'gate net' not in w] and r['fit']['shift_mm'] <= lim for r in fr)
        print(f'combined gate pass with max_shift {lim} mm: {n} / {len(fr)}')
    arm = [r for r in fr if 'arm' in r]
    print('\nagainst the arm-derived port 2, n runs =', len(arm))
    for name, get in (('fit (used x, y)', lambda r: np.array(r['fit']['fit_xy']) * 1e3), ('v1', lambda r: np.array(r['gate']['xy_mm'])),
                      ('zed_v5', lambda r: np.array(r['start']['xy_mm'])), ('fit from v1 start', lambda r: np.array(r['fit_from_v1']['fit_xy']) * 1e3)):
        e = np.array([get(r) - np.array(r['arm']['xy_mm']) for r in arm])
        print(f'{name:18s} dx {stats(e[:, 0])}  dy {stats(e[:, 1])}  |dxy| p50 {np.median(np.hypot(*e.T)):.2f} max {np.hypot(*e.T).max():.2f}')
    for name, key in (('v1', 'gate'), ('zed_v5', 'start')):
        e = np.array([r[key]['yaw_deg'] - r['arm']['yaw_deg'] for r in arm])
        print(f'yaw {name} - arm: {stats(e)}')
    e = np.array([r['fit']['fit_yaw_deg'] - r['arm']['yaw_deg'] for r in arm])
    print(f'yaw fit - arm: {stats(e)}')
    for name, key in (('v1', 'gate'), ('zed_v5', 'start')):
        e = np.array([r['fit']['fit_yaw_deg'] - r[key]['yaw_deg'] for r in fr])
        print(f'yaw fit - {name}, all frames n={len(fr)}: {stats(e)}')
    # pose changes on the maps (from map_real)
    maps = {r['run'].split('_', 2)[-1]: r for r in fr if r['run'].startswith('capture_17')}
    if 'map_real' in maps:
        r0 = maps['map_real']
        print('\nyaw change from map_real (deg): fit / v1 / v5')
        for k, r in maps.items():
            print(k, round(r['fit']['fit_yaw_deg'] - r0['fit']['fit_yaw_deg'], 2), round(r['gate']['yaw_deg'] - r0['gate']['yaw_deg'], 2),
                  round(r['start']['yaw_deg'] - r0['start']['yaw_deg'], 2))
    t = [r['fit']['seconds'] for r in fr]
    print(f'\nfit time s: p50 {np.median(t):.2f} max {max(t):.2f}')


if __name__ == '__main__':
    main(sys.argv[1])
